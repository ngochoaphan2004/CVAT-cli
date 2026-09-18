import urllib.request
import json
import base64
import io
import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import EomtImageProcessor, EomtForUniversalSegmentation
from cvat_sdk import make_client
from cvat_sdk.models import LabeledDataRequest, LabeledShapeRequest, ShapeType
import rasterio.features
from shapely.geometry import shape as shapely_shape, Polygon, MultiPolygon
import topojson as tp

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Đọc cấu hình từ .env hoặc biến môi trường; nếu không có thì mặc định dùng localhost:8080
CVAT_HOST = os.getenv("CVAT_HOST", "http://localhost:8080")
CVAT_AUTH = os.getenv("CVAT_AUTH", "hoap:1toi9a")
if ":" in CVAT_AUTH:
    CVAT_USER, CVAT_PASS = CVAT_AUTH.split(":", 1)
else:
    CVAT_USER, CVAT_PASS = CVAT_AUTH, ""
CVAT_USER = os.getenv("CVAT_USER", CVAT_USER)
CVAT_PASS = os.getenv("CVAT_PASS", CVAT_PASS)

default_task_id = 38 if "localhost" in CVAT_HOST else 183
default_job_id = 0 if "localhost" in CVAT_HOST else 1581

TASK_ID = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.getenv("CVAT_TASK_ID", default_task_id))
JOB_ID = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.getenv("CVAT_JOB_ID", default_job_id))
MIN_AREA_POLYGON = 80.0
APPROX_EPSILON = 40.0

# Danh sách các frame cần phải vẽ (nếu để None hoặc [] thì sẽ vẽ tất cả các frame trừ COMPLETED_FRAMES)
# Ví dụ chỉ định cụ thể: TARGET_FRAMES = [1, 2, 3]
TARGET_FRAMES = None
# TARGET_FRAMES = [2]

# Danh sách các frame đã xử lý xong (sẽ bỏ qua không chạy các frame này)
COMPLETED_FRAMES = [
    0
]

# Chỉ gán nhãn những vật thể hình khối rõ ràng, dễ nhận dạng (như yêu cầu trước)
EASY_CLASSES = {
    "road", "car", "person", "building", "sky", 
    "bus", "truck", "motorcycle", "bicycle",
}

def contour_to_polyline(cnt):
    ys = cnt[:, :, 1].flatten()
    xs = cnt[:, :, 0].flatten()
    min_y, max_y = int(np.min(ys)), int(np.max(ys))
    if max_y - min_y < 15:
        return None

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
    target_desc = f"Job {JOB_ID} (Task {TASK_ID})" if JOB_ID else f"Task {TASK_ID}"
    print(f"Connecting to CVAT at {CVAT_HOST} for {target_desc}...")
    with make_client(host=CVAT_HOST, credentials=(CVAT_USER, CVAT_PASS)) as client:
        if JOB_ID:
            target = client.jobs.retrieve(JOB_ID)
            start_f = target.start_frame
            stop_f = target.stop_frame
        else:
            target = client.tasks.retrieve(TASK_ID)
            start_f = 0
            stop_f = target.size - 1

        labels = target.get_labels()
        name_to_id = {l.name: l.id for l in labels}

        # Tập hợp frame đã hoàn thành cần loại trừ
        completed_frames = set(COMPLETED_FRAMES)

        # Xác định danh sách frame cần vẽ
        if TARGET_FRAMES:
            candidate_frames = [f for f in TARGET_FRAMES if start_f <= f <= stop_f]
        else:
            candidate_frames = list(range(start_f, stop_f + 1))

        frames_to_process = [f for f in candidate_frames if f not in completed_frames]
        print(f"{target_desc} frames: {start_f} to {stop_f} (total {stop_f - start_f + 1})")
        if TARGET_FRAMES:
            print(f"Filter TARGET_FRAMES: {TARGET_FRAMES}")
        print(f"Skipped completed frames ({len(completed_frames)}): {sorted(list(completed_frames))}")
        print(f"Frames to process ({len(frames_to_process)}): {frames_to_process}")

        if not frames_to_process:
            print("No frames to process! Nothing to do.")
            return

        print("Loading EoMT-DINOv3 model (Cityscapes Large)...")
        model_name = "tue-mps/cityscapes_semantic_eomt_large_1024"
        processor = EomtImageProcessor.from_pretrained(model_name)
        model = EomtForUniversalSegmentation.from_pretrained(model_name)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)
        model.eval()

        id2label = model.config.id2label

        for frame_idx in frames_to_process:
            print(f"\n--- Processing Frame {frame_idx} (EoMT-DINOv3) ---")
            try:
                img = Image.open(target.get_frame(frame_idx)).convert("RGB")
            except Exception:
                auth_header = "Basic " + base64.b64encode(f"{CVAT_USER}:{CVAT_PASS}".encode()).decode("ascii")
                data_endpoint = f"jobs/{JOB_ID}" if JOB_ID else f"tasks/{TASK_ID}"
                req = urllib.request.Request(
                    f"{CVAT_HOST}/api/{data_endpoint}/data?type=frame&number={frame_idx}",
                    headers={"Authorization": auth_header},
                )
                with urllib.request.urlopen(req) as resp:
                    img = Image.open(io.BytesIO(resp.read())).convert("RGB")

            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            predicted_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[img.size[::-1]])[0]
            pred_seg = predicted_map.cpu().numpy().astype(np.int32)

            poly_count = 0
            polyline_count = 0
            frame_shapes = []
            
            # Use rasterio to extract gapless polygons
            geom_results = list(rasterio.features.shapes(pred_seg, connectivity=4))
            
            features = []
            for geom, value in geom_results:
                class_idx = int(value)
                class_name = id2label[class_idx]
                cvat_label_name = class_name
                if cvat_label_name == "traffic light": cvat_label_name = "traffic_light"
                elif cvat_label_name == "traffic sign": cvat_label_name = "traffic_sign"

                if cvat_label_name not in name_to_id or cvat_label_name not in EASY_CLASSES:
                    continue

                label_id = name_to_id[cvat_label_name]
                
                # Filter small areas before topology
                if shapely_shape(geom).area < MIN_AREA_POLYGON:
                    continue

                features.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"label_id": label_id, "cvat_label_name": cvat_label_name}
                })

            if features:
                # Topojson topology-preserving simplification
                topo = tp.Topology(features, prequantize=False)
                topo = topo.toposimplify(epsilon=APPROX_EPSILON)
                simplified_geojson_str = topo.to_geojson()
                simplified_features = json.loads(simplified_geojson_str)["features"]
                
                for feat in simplified_features:
                    label_id = feat["properties"]["label_id"]
                    cvat_label_name = feat["properties"]["cvat_label_name"]
                    geom_type = feat["geometry"]["type"]
                    coords = feat["geometry"]["coordinates"]
                    
                    polys_coords = []
                    if geom_type == "Polygon":
                        polys_coords = [coords]
                    elif geom_type == "MultiPolygon":
                        polys_coords = coords
                    else:
                        continue
                        
                    for poly_coords in polys_coords:
                        if not poly_coords or len(poly_coords[0]) < 3:
                            continue
                            
                        exterior = poly_coords[0]
                        poly = Polygon(exterior)
                        bx, by, maxx, maxy = poly.bounds
                        bw = maxx - bx
                        bh = maxy - by
                        
                        if cvat_label_name == "pole" and (bh >= 2.0 * bw or bw <= 16) and bh >= 20:
                            cnt = np.array(exterior).reshape(-1, 1, 2)
                            line_pts = contour_to_polyline(cnt)
                            if line_pts and len(line_pts) >= 4:
                                shape = LabeledShapeRequest(
                                    type=ShapeType("polyline"),
                                    frame=frame_idx,
                                    label_id=label_id,
                                    points=line_pts,
                                    occluded=False,
                                    outside=False,
                                )
                                frame_shapes.append(shape)
                                polyline_count += 1
                                continue
                                
                        if exterior[0] == exterior[-1]:
                            exterior = exterior[:-1]
                            
                        pts = [float(coord) for pt in exterior for coord in pt]
                        if len(pts) >= 6:
                            shape = LabeledShapeRequest(
                                type=ShapeType("polygon"),
                                frame=frame_idx,
                                label_id=label_id,
                                points=pts,
                                occluded=False,
                                outside=False,
                            )
                            frame_shapes.append(shape)
                            poly_count += 1

            print(f"Frame {frame_idx}: created {poly_count} polygons and {polyline_count} polylines.")

            # Incremental update: preserve existing annotations (especially Frame 0)
            current_data = target.get_annotations()
            updated_shapes = []
            for s in current_data.shapes:
                if s.frame != frame_idx:
                    d = s.to_dict()
                    d.pop('id', None)
                    updated_shapes.append(LabeledShapeRequest(**d))
            updated_shapes.extend(frame_shapes)
            target.set_annotations(LabeledDataRequest(shapes=updated_shapes))

            # Ghi nhận frame đã hoàn thành vào list trong code
            if frame_idx not in COMPLETED_FRAMES:
                COMPLETED_FRAMES.append(frame_idx)
            print(f"Frame {frame_idx} uploaded to CVAT and added to completed list.")

        print(f"\nAll specified frames for {target_desc} processed successfully!")

if __name__ == "__main__":
    run()

