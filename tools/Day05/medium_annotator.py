import urllib.request
import json
import base64
import io
import torch
import numpy as np
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from cvat_sdk import make_client
from cvat_sdk.models import LabeledDataRequest, LabeledShapeRequest, ShapeType
import rasterio.features
from shapely.geometry import shape as shapely_shape, Polygon, MultiPolygon

CVAT_HOST = "http://localhost:8080"
CVAT_USER = "hoap"
CVAT_PASS = "1toi9a"
TASK_ID = 15  # Change to your Medium Task ID
FRAMES_TO_ANNOTATE = None
MIN_AREA_POLYGON = 50.0
APPROX_EPSILON = 2.0

MEDIUM_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

def contour_to_polyline(cnt):
    ys = cnt[:, :, 1].flatten()
    xs = cnt[:, :, 0].flatten()
    min_y, max_y = int(np.min(ys)), int(np.max(ys))
    if max_y - min_y < 15: return None
    step = max(8, (max_y - min_y) // 8)
    line_pts = []
    for y_val in range(min_y, max_y + 1, step):
        mask_y = (ys >= y_val - step // 2) & (ys <= y_val + step // 2)
        if np.any(mask_y):
            avg_x = float(np.mean(xs[mask_y]))
            line_pts.extend([round(avg_x, 1), float(y_val)])
    bot_pt = cnt[cnt[:, :, 1].argmax()][0]
    if len(line_pts) >= 4:
        line_pts[-2] = float(bot_pt[0])
        line_pts[-1] = float(bot_pt[1])
        return line_pts
    return None

def run():
    print(f"Connecting to CVAT at {CVAT_HOST}...")
    with make_client(host=CVAT_HOST, credentials=(CVAT_USER, CVAT_PASS)) as client:
        task = client.tasks.retrieve(TASK_ID)
        labels = task.get_labels()
        name_to_id = {l.name: l.id for l in labels}
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        print("Loading Mask2Former (Instance - Medium Level)...")
        inst_model_name = "facebook/mask2former-swin-large-coco-instance"
        processor = Mask2FormerImageProcessor.from_pretrained(inst_model_name)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(inst_model_name).to(device).eval()
        
        id2label = model.config.id2label
        all_shapes = []
        
        frames_list = FRAMES_TO_ANNOTATE if FRAMES_TO_ANNOTATE is not None else list(range(task.size))
        for frame_idx in frames_list:
            print(f"\n--- Processing Frame {frame_idx} (Medium - Instance) ---")
            auth_header = "Basic " + base64.b64encode(f"{CVAT_USER}:{CVAT_PASS}".encode()).decode("ascii")
            req = urllib.request.Request(
                f"{CVAT_HOST}/api/tasks/{TASK_ID}/data?type=frame&number={frame_idx}",
                headers={"Authorization": auth_header},
            )
            with urllib.request.urlopen(req) as resp:
                img = Image.open(io.BytesIO(resp.read())).convert("RGB")

            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            
            results = processor.post_process_instance_segmentation(outputs, target_sizes=[img.size[::-1]])[0]
            segmentation = results["segmentation"].cpu().numpy().astype(np.int32)
            segments_info = results["segments_info"]

            poly_count = 0
            
            for segment in segments_info:
                class_idx = segment["label_id"]
                cls_name = id2label[class_idx]
                if cls_name == "motorbike": cls_name = "motorcycle"
                
                if cls_name not in MEDIUM_CLASSES or cls_name not in name_to_id:
                    continue

                label_id = name_to_id[cls_name]
                instance_id = segment["id"]
                instance_mask = (segmentation == instance_id).astype(np.uint8)
                
                geom_results = list(rasterio.features.shapes(instance_mask, mask=instance_mask.astype(bool), connectivity=4))
                
                for geom, val in geom_results:
                    if not val: continue
                    polygon = shapely_shape(geom)
                    if polygon.area < MIN_AREA_POLYGON: continue
                    polygon = polygon.simplify(APPROX_EPSILON, preserve_topology=True)
                    
                    polys = list(polygon.geoms) if polygon.geom_type == 'MultiPolygon' else [polygon] if polygon.geom_type == 'Polygon' else []
                    
                    for poly in polys:
                        coords = list(poly.exterior.coords)
                        if len(coords) < 3: continue
                        
                        if coords[0] == coords[-1]: coords = coords[:-1]
                        pts = [float(coord) for pt in coords for coord in pt]
                        if len(pts) >= 6:
                            shape = LabeledShapeRequest(
                                type=ShapeType("polygon"), frame=frame_idx, label_id=label_id,
                                points=pts, occluded=False, outside=False,
                            )
                            all_shapes.append(shape)
                            poly_count += 1
                            
            print(f"Frame {frame_idx}: created {poly_count} instance polygons.")

        print(f"\nClearing old annotations and uploading {len(all_shapes)} new shapes to Task {TASK_ID}...")
        task.set_annotations(LabeledDataRequest(shapes=all_shapes))
        print("Upload successful!")

if __name__ == "__main__":
    run()

