import urllib.request
import json
import base64
import io
import torch
import numpy as np
from PIL import Image
from transformers import EomtImageProcessor, EomtForUniversalSegmentation
from cvat_sdk import make_client
from cvat_sdk.models import LabeledDataRequest, LabeledShapeRequest, ShapeType
import rasterio.features
from shapely.geometry import shape as shapely_shape, Polygon, MultiPolygon
import topojson as tp

CVAT_HOST = "http://localhost:8080"
CVAT_USER = "hoap"
CVAT_PASS = "1toi9a"
TASK_ID = 14  # Change to your Easy Task ID
FRAMES_TO_ANNOTATE = None
MIN_AREA_POLYGON = 50.0
APPROX_EPSILON = 2.0

EASY_CLASSES = {"road", "sidewalk", "building", "vegetation", "sky"}

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

        print("Loading EoMT-DINOv3 (Semantic - Easy Level)...")
        sem_model_name = "tue-mps/cityscapes_semantic_eomt_large_1024"
        processor = EomtImageProcessor.from_pretrained(sem_model_name)
        model = EomtForUniversalSegmentation.from_pretrained(sem_model_name).to(device).eval()
        
        id2label = model.config.id2label
        all_shapes = []
        
        frames_list = FRAMES_TO_ANNOTATE if FRAMES_TO_ANNOTATE is not None else list(range(task.size))
        for frame_idx in frames_list:
            print(f"\n--- Processing Frame {frame_idx} (Easy - Semantic) ---")
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
            sem_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[img.size[::-1]])[0]
            sem_seg = sem_map.cpu().numpy().astype(np.int32)
            
            geom_results = list(rasterio.features.shapes(sem_seg, connectivity=4))
            
            features = []
            for geom, value in geom_results:
                class_idx = int(value)
                cls_name = id2label[class_idx]
                if cls_name not in EASY_CLASSES or cls_name not in name_to_id:
                    continue
                label_id = name_to_id[cls_name]
                
                if shapely_shape(geom).area < MIN_AREA_POLYGON:
                    continue
                    
                features.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"label_id": label_id, "cvat_label_name": cls_name}
                })
                
            poly_count = 0
            if features:
                topo = tp.Topology(features, prequantize=False)
                topo = topo.toposimplify(epsilon=APPROX_EPSILON)
                simplified_geojson_str = topo.to_geojson()
                simplified_features = json.loads(simplified_geojson_str)["features"]
                
                for feat in simplified_features:
                    label_id = feat["properties"]["label_id"]
                    cvat_label_name = feat["properties"]["cvat_label_name"]
                    geom_type = feat["geometry"]["type"]
                    coords = feat["geometry"]["coordinates"]
                    
                    polys_coords = [coords] if geom_type == "Polygon" else coords if geom_type == "MultiPolygon" else []
                        
                    for poly_coords in polys_coords:
                        if not poly_coords or len(poly_coords[0]) < 3: continue
                        exterior = poly_coords[0]
                        poly = Polygon(exterior)
                        bx, by, maxx, maxy = poly.bounds
                        bw, bh = maxx - bx, maxy - by
                        
                        if exterior[0] == exterior[-1]: exterior = exterior[:-1]
                        pts = [float(coord) for pt in exterior for coord in pt]
                        if len(pts) >= 6:
                            shape = LabeledShapeRequest(
                                type=ShapeType("polygon"), frame=frame_idx, label_id=label_id,
                                points=pts, occluded=False, outside=False,
                            )
                            all_shapes.append(shape)
                            poly_count += 1
                            
            print(f"Frame {frame_idx}: created {poly_count} gapless semantic polygons.")

        print(f"\nClearing old annotations and uploading {len(all_shapes)} new shapes to Task {TASK_ID}...")
        task.set_annotations(LabeledDataRequest(shapes=all_shapes))
        print("Upload successful!")

if __name__ == "__main__":
    run()

