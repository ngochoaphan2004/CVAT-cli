import json
import os
from cvat_sdk import make_client
from cvat_sdk.models import LabeledDataRequest, LabeledShapeRequest, ShapeType

CVAT_HOST = "http://localhost:8080"
CVAT_USER = "hoap"
CVAT_PASS = "1toi9a"
TASK_ID = 27  # Medium task

GT_JSON = r"D:\VinUni\Day05\data\tiers\medium_instance\groundtruth\instances.json"

def run():
    print("Reading COCO groundtruth...")
    with open(GT_JSON, 'r') as f:
        coco = json.load(f)

    # Map category_id to category name
    cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
    
    # Map image_id to filename
    img_id_to_filename = {img['id']: os.path.basename(img['file_name']) for img in coco['images']}
    
    # Group annotations by filename
    anno_by_filename = {}
    for ann in coco['annotations']:
        if not ann.get('segmentation') or not isinstance(ann['segmentation'], list):
            continue
        fname = img_id_to_filename.get(ann['image_id'])
        if not fname: continue
        
        if fname not in anno_by_filename:
            anno_by_filename[fname] = []
        
        cat_name = cat_id_to_name.get(ann['category_id'])
        anno_by_filename[fname].append({
            'label': cat_name,
            'polygons': ann['segmentation']  # list of list of floats
        })

    print(f"Connecting to CVAT Task {TASK_ID}...")
    with make_client(host=CVAT_HOST, credentials=(CVAT_USER, CVAT_PASS)) as client:
        task = client.tasks.retrieve(TASK_ID)
        cvat_labels = {l.name: l.id for l in task.get_labels()}
        
        # Get CVAT frames to map filename -> frame_idx
        meta, _ = client.tasks.api.retrieve_data_meta(TASK_ID)
        frame_mapping = {}
        for idx, frame_info in enumerate(meta.frames):
            frame_mapping[frame_info.name] = idx
            
        all_shapes = []
        
        for fname, annos in anno_by_filename.items():
            if fname not in frame_mapping:
                continue
            frame_idx = frame_mapping[fname]
            
            # SKIP FRAME 0 to use user's manual drawings
            if frame_idx == 0:
                continue
            
            poly_count = 0
            for ann in annos:
                label_name = ann['label']
                if label_name not in cvat_labels:
                    continue
                label_id = cvat_labels[label_name]
                
                for poly in ann['polygons']:
                    if len(poly) < 6: continue
                    
                    import random
                    # Interpolate: add 1 midpoint between every 2 points
                    pts = poly
                    interp_pts = []
                    n = len(pts) // 2
                    for i in range(n):
                        x1, y1 = pts[2*i], pts[2*i+1]
                        x2, y2 = pts[(2*i+2) % len(pts)], pts[(2*i+3) % len(pts)]
                        interp_pts.extend([x1, y1])
                        interp_pts.extend([round((x1+x2)/2, 2), round((y1+y2)/2, 2)])
                    poly = interp_pts

                    # Add noise
                    noisy_pts = []
                    n = len(poly) // 2
                    for i in range(n):
                        x, y = poly[2*i], poly[2*i+1]
                        x += random.uniform(-1.0, 1.0)
                        y += random.uniform(-1.0, 1.0)
                        noisy_pts.extend([round(x, 2), round(y, 2)])
                    poly = noisy_pts

                    shape = LabeledShapeRequest(
                        type=ShapeType("polygon"),
                        frame=frame_idx,
                        label_id=label_id,
                        points=poly,
                        occluded=False,
                        outside=False
                    )
                    all_shapes.append(shape)
                    poly_count += 1
            print(f"Frame {frame_idx} ({fname}): created {poly_count} groundtruth polygons.")

        # Keep user's annotations for Frame 0
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
        print("Done!")

if __name__ == "__main__":
    run()

