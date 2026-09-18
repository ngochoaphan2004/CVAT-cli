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
TASK_ID = 26  # Hard SDK Task

GT_JSON = r"D:\VinUni\Day05\data\tiers\hard_panoptic\groundtruth\panoptic.json"
GT_PNG_DIR = r"D:\VinUni\Day05\data\tiers\hard_panoptic\groundtruth\png"
MIN_AREA_POLYGON = 50.0
APPROX_EPSILON = 0.5

def run():
    print(f"Connecting to CVAT Task {TASK_ID}...")
    with make_client(host=CVAT_HOST, credentials=(CVAT_USER, CVAT_PASS)) as client:
        task = client.tasks.retrieve(TASK_ID)
        cvat_labels = {l.name: l.id for l in task.get_labels()}
        car_label_id = cvat_labels.get("car")
        
        # Map frame name to index
        meta, _ = client.tasks.api.retrieve_data_meta(TASK_ID)
        frame_mapping = {}
        for idx, frame_info in enumerate(meta.frames):
            frame_mapping[frame_info.name] = idx
            
        print("Reading Panoptic COCO groundtruth...")
        with open(GT_JSON, 'r') as f:
            coco = json.load(f)

        # Map category_id to category name
        cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
        
        # Map image_id to filename
        img_id_to_filename = {img['id']: os.path.basename(img['file_name']) for img in coco['images']}
        
        all_shapes = []
        
        for ann in coco['annotations']:
            img_id = ann['image_id']
            png_name = ann['file_name']
            
            # The original image name might be the same as the png name or mapped via images
            base_name = img_id_to_filename.get(img_id, png_name)
            if base_name not in frame_mapping:
                # try replacing png with jpg
                base_name = base_name.replace(".png", ".jpg")
                if base_name not in frame_mapping:
                    continue
                    
            frame_idx = frame_mapping[base_name]
            
            # Map segment id to cvat label id
            seg_id_to_cvat_label = {}
            NAME_MAPPING = {
                'sky-other-merged': 'sky',
                'tree-merged': 'vegetation',
                'building-other-merged': 'building',
                'pavement-merged': 'sidewalk'
            }
            
            for seg in ann['segments_info']:
                cat_name = cat_id_to_name.get(seg['category_id'])
                mapped_name = NAME_MAPPING.get(cat_name, cat_name)
                if mapped_name in cvat_labels:
                    seg_id_to_cvat_label[seg['id']] = cvat_labels[mapped_name]
            
            mask_path = os.path.join(GT_PNG_DIR, png_name)
            if not os.path.exists(mask_path):
                continue
                
            print(f"Processing Groundtruth {png_name} -> Frame {frame_idx}")
            img = np.array(Image.open(mask_path))
            
            # Convert RGB to ID: id = R + G*256 + B*256^2
            ids = img[:,:,0].astype(np.uint32) + img[:,:,1].astype(np.uint32)*256 + img[:,:,2].astype(np.uint32)*65536
            
            features = []
            
            # Using topojson for gapless polys
            geom_results = list(rasterio.features.shapes(ids.astype(np.int32), connectivity=4))
            for geom, value in geom_results:
                seg_id = int(value)
                if seg_id == 0 or seg_id not in seg_id_to_cvat_label:
                    continue
                    
                label_id = seg_id_to_cvat_label[seg_id]
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
                            
                        if len(pts) >= 6:
                            shape = LabeledShapeRequest(
                                type=ShapeType("polygon"), frame=frame_idx, label_id=label_id,
                                points=pts, occluded=False, outside=False,
                            )
                            all_shapes.append(shape)
                            poly_count += 1
                            
            print(f"Frame {frame_idx}: created {poly_count} groundtruth panoptic polygons.")

        # Keep user's annotations for Frame 0, but OVERWRITE 'car' with SDK cars
        current_data = task.get_annotations()
        mixed_shapes = []
        for s in current_data.shapes:
            if s.frame == 0 and s.label_id != car_label_id:
                d = s.to_dict()
                d.pop('id', None)
                mixed_shapes.append(LabeledShapeRequest(**d))
                
        for s in all_shapes:
            if s.frame != 0:
                mixed_shapes.append(s)
            elif s.frame == 0 and s.label_id == car_label_id:
                mixed_shapes.append(s)

        print(f"Uploading {len(mixed_shapes)} mixed shapes to CVAT...")
        task.set_annotations(LabeledDataRequest(shapes=mixed_shapes))
        print("Upload successful!")

if __name__ == "__main__":
    run()

