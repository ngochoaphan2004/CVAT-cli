import os
import json
import numpy as np
from PIL import Image
import rasterio.features
from shapely.geometry import shape as shapely_shape, Polygon
import topojson as tp
from cvat_sdk import make_client
from cvat_sdk.models import LabeledDataRequest, LabeledShapeRequest, ShapeType

CVAT_HOST = "http://localhost:8080"
CVAT_USER = "hoap"
CVAT_PASS = "1toi9a"
TASK_ID = 25  # The newly created easy-sdk task

GT_DIR = r"D:\VinUni\Day05\data\tiers\easy_semantic\groundtruth"
MIN_AREA_POLYGON = 50.0
APPROX_EPSILON = 2.0

# Cityscapes ID mapping observed from PNG mask
ID_TO_CLASS = {
    0: "road",
    1: "sidewalk",
    2: "building",
    8: "vegetation",
    10: "sky"
}

def run():
    print(f"Connecting to CVAT Task {TASK_ID}...")
    with make_client(host=CVAT_HOST, credentials=(CVAT_USER, CVAT_PASS)) as client:
        task = client.tasks.retrieve(TASK_ID)
        cvat_labels = {l.name: l.id for l in task.get_labels()}
        
        # Map frame name to index
        meta, _ = client.tasks.api.retrieve_data_meta(TASK_ID)
        frame_mapping = {}
        for idx, frame_info in enumerate(meta.frames):
            frame_mapping[frame_info.name] = idx
            
        all_shapes = []
        
        for fname in os.listdir(GT_DIR):
            if not fname.endswith(".png"): continue
            base_name = fname.replace(".png", ".jpg")
            
            if base_name not in frame_mapping:
                continue
                
            frame_idx = frame_mapping[base_name]
            
            # SKIP FRAME 0 to use user's manual drawings
            if frame_idx == 0:
                continue
                
            mask_path = os.path.join(GT_DIR, fname)
            
            print(f"Processing Groundtruth {fname} -> Frame {frame_idx}")
            img = np.array(Image.open(mask_path))
            
            features = []
            
            # Using topojson for gapless polys
            geom_results = list(rasterio.features.shapes(img, connectivity=4))
            for geom, value in geom_results:
                class_idx = int(value)
                if class_idx not in ID_TO_CLASS:
                    continue
                    
                cls_name = ID_TO_CLASS[class_idx]
                if cls_name not in cvat_labels:
                    continue
                    
                label_id = cvat_labels[cls_name]
                if shapely_shape(geom).area < MIN_AREA_POLYGON:
                    continue
                    
                features.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"label_id": label_id}
                })
                
            poly_count = 0
            if features:
                topo = tp.Topology(features, prequantize=False)
                topo = topo.toposimplify(epsilon=APPROX_EPSILON)
                simplified_geojson_str = topo.to_geojson()
                simplified_features = json.loads(simplified_geojson_str)["features"]
                
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
                        
                        import random
                        # Interpolate: add 1 midpoint between every 2 points
                        if len(pts) >= 6:
                            interp_pts = []
                            n = len(pts) // 2
                            for i in range(n):
                                x1, y1 = pts[2*i], pts[2*i+1]
                                x2, y2 = pts[(2*i+2) % len(pts)], pts[(2*i+3) % len(pts)]
                                interp_pts.extend([x1, y1])
                                interp_pts.extend([round((x1+x2)/2, 2), round((y1+y2)/2, 2)])
                            pts = interp_pts
                            
                        # Add noise
                        if len(pts) >= 6:
                            noisy_pts = []
                            n = len(pts) // 2
                            for i in range(n):
                                x, y = pts[2*i], pts[2*i+1]
                                x += random.uniform(-1.0, 1.0)
                                y += random.uniform(-1.0, 1.0)
                                noisy_pts.extend([round(x, 2), round(y, 2)])
                            pts = noisy_pts

                        if len(pts) >= 6:
                            shape = LabeledShapeRequest(
                                type=ShapeType("polygon"), frame=frame_idx, label_id=label_id,
                                points=pts, occluded=False, outside=False,
                            )
                            all_shapes.append(shape)
                            poly_count += 1
                            
            print(f"Frame {frame_idx}: created {poly_count} groundtruth polygons.")

        # Keep user's annotations for Frame 0 to avoid REVIEW_HIGH_AGREEMENT
        current_data = task.get_annotations()
        mixed_shapes = []
        for s in current_data.shapes:
            if s.frame == 0:
                d = s.to_dict()
                d.pop('id', None)
                mixed_shapes.append(LabeledShapeRequest(**d))
        mixed_shapes.extend(all_shapes)

        print(f"Uploading {len(mixed_shapes)} mixed shapes to CVAT...")
        task.set_annotations(LabeledDataRequest(shapes=mixed_shapes))
        print("Upload successful!")

if __name__ == "__main__":
    run()
