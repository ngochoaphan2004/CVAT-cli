import urllib.request
import json
import base64
import io
import torch
import numpy as np
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from transformers import EomtImageProcessor, EomtForUniversalSegmentation
from cvat_sdk import make_client
from cvat_sdk.models import LabeledDataRequest, LabeledShapeRequest, ShapeType
import rasterio.features
from shapely.geometry import shape as shapely_shape, Polygon, MultiPolygon
import topojson as tp

CVAT_HOST = "http://localhost:8080"
CVAT_USER = "hoap"
CVAT_PASS = "1toi9a"
TASK_ID = 16  # Change to your Hard Task ID
FRAMES_TO_ANNOTATE = None
MIN_AREA_POLYGON = 50.0
APPROX_EPSILON = 2.0

STUFF_CLASSES = {"road", "sidewalk", "building", "vegetation", "sky"}
THING_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light"}

def run():
    print(f"Connecting to CVAT at {CVAT_HOST}...")
    with make_client(host=CVAT_HOST, credentials=(CVAT_USER, CVAT_PASS)) as client:
        task = client.tasks.retrieve(TASK_ID)
        labels = task.get_labels()
        name_to_id = {l.name: l.id for l in labels}
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        print("Loading EoMT-DINOv3 (Semantic - Hard Level)...")
        sem_model_name = "tue-mps/cityscapes_semantic_eomt_large_1024"
        sem_processor = EomtImageProcessor.from_pretrained(sem_model_name)
        sem_model = EomtForUniversalSegmentation.from_pretrained(sem_model_name).to(device).eval()
        
        print("Loading Mask2Former (Instance - Hard Level)...")
        inst_model_name = "facebook/mask2former-swin-large-coco-instance"
        inst_processor = Mask2FormerImageProcessor.from_pretrained(inst_model_name)
        inst_model = Mask2FormerForUniversalSegmentation.from_pretrained(inst_model_name).to(device).eval()

        sem_id2label = sem_model.config.id2label
        inst_id2label = inst_model.config.id2label
        
        all_shapes = []
        
        frames_list = FRAMES_TO_ANNOTATE if FRAMES_TO_ANNOTATE is not None else list(range(task.size))
        for frame_idx in frames_list:
            print(f"\n--- Processing Frame {frame_idx} (Hard - Panoptic Fusion) ---")
            auth_header = "Basic " + base64.b64encode(f"{CVAT_USER}:{CVAT_PASS}".encode()).decode("ascii")
            req = urllib.request.Request(
                f"{CVAT_HOST}/api/tasks/{TASK_ID}/data?type=frame&number={frame_idx}",
                headers={"Authorization": auth_header},
            )
            with urllib.request.urlopen(req) as resp:
                img = Image.open(io.BytesIO(resp.read())).convert("RGB")

            # 1. Semantic inference
            sem_inputs = sem_processor(images=img, return_tensors="pt")
            sem_inputs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in sem_inputs.items()}
            with torch.no_grad():
                sem_outputs = sem_model(**sem_inputs)
            sem_map = sem_processor.post_process_semantic_segmentation(sem_outputs, target_sizes=[img.size[::-1]])[0]
            sem_seg = sem_map.cpu().numpy().astype(np.int32)

            # 2. Instance inference
            inst_inputs = inst_processor(images=img, return_tensors="pt")
            inst_inputs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in inst_inputs.items()}
            with torch.no_grad():
                inst_outputs = inst_model(**inst_inputs)
            inst_results = inst_processor.post_process_instance_segmentation(inst_outputs, target_sizes=[img.size[::-1]])[0]
            inst_seg = inst_results["segmentation"].cpu().numpy().astype(np.int32)
            inst_info = inst_results["segments_info"]

            # 3. Panoptic fusion
            combined_map = np.zeros(img.size[::-1], dtype=np.int32)
            instance_to_cvat_label = {}
            
            for val in np.unique(sem_seg):
                cls_name = sem_id2label[val]
                if cls_name in STUFF_CLASSES and cls_name in name_to_id:
                    combined_map[sem_seg == val] = name_to_id[cls_name]
                    
            inst_idx = 1000
            for segment in inst_info:
                cls_name = inst_id2label[segment["label_id"]]
                if cls_name == "motorbike": cls_name = "motorcycle"
                
                if cls_name in THING_CLASSES and cls_name in name_to_id:
                    mask = (inst_seg == segment["id"])
                    combined_map[mask] = inst_idx
                    instance_to_cvat_label[inst_idx] = name_to_id[cls_name]
                    inst_idx += 1
                    
            # 4. Extract & Topologize
            geom_results = list(rasterio.features.shapes(combined_map, connectivity=4))
            
            features = []
            for geom, value in geom_results:
                val = int(value)
                if val == 0: continue
                
                label_id = val if val < 1000 else instance_to_cvat_label.get(val)
                if label_id is None: continue
                    
                if shapely_shape(geom).area < MIN_AREA_POLYGON: continue
                    
                features.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"label_id": label_id}
                })
                
            if features:
                topo = tp.Topology(features, prequantize=False)
                topo = topo.toposimplify(epsilon=APPROX_EPSILON)
                simplified_geojson_str = topo.to_geojson()
                simplified_features = json.loads(simplified_geojson_str)["features"]
                
                poly_count = 0
                for feat in simplified_features:
                    label_id = feat["properties"]["label_id"]
                    geom_type = feat["geometry"]["type"]
                    coords = feat["geometry"]["coordinates"]
                    
                    polys_coords = [coords] if geom_type == "Polygon" else coords if geom_type == "MultiPolygon" else []
                        
                    for poly_coords in polys_coords:
                        if not poly_coords or len(poly_coords[0]) < 3: continue
                        exterior = poly_coords[0]
                        if exterior[0] == exterior[-1]: exterior = exterior[:-1]
                        pts = [float(coord) for pt in exterior for coord in pt]
                        if len(pts) >= 6:
                            shape = LabeledShapeRequest(
                                type=ShapeType("polygon"), frame=frame_idx, label_id=label_id,
                                points=pts, occluded=False, outside=False,
                            )
                            all_shapes.append(shape)
                            poly_count += 1
                            
            print(f"Frame {frame_idx}: created {poly_count} gapless panoptic polygons.")

        print(f"\nClearing old annotations and uploading {len(all_shapes)} new shapes to Task {TASK_ID}...")
        task.set_annotations(LabeledDataRequest(shapes=all_shapes))
        print("Upload successful!")

if __name__ == "__main__":
    run()

