import cv2
import json
from datetime import datetime
from OC_SORT.trackers.ocsort_tracker.ocsort import OCSort
import numpy as np
from ultralytics import YOLO
import time

model = YOLO('/content/yolo12s.onnx')
license_model = YOLO('/content/best.onnx')
tracker = OCSort(det_thresh=0.5)
cap = cv2.VideoCapture('/content/input_video.mp4')
width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('output_optimized.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
frame_idx = 0
log_data = []
saved_ids = set()
skip_rate = 2
start_time = time.time()
vehicle_buffer = {}
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % skip_rate != 0:
        frame_idx += 1
        continue

    vehicle_results = model.predict(frame, conf=0.6, verbose=False)[0]
    dets = []
    for box in vehicle_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        dets.append([x1, y1, x2, y2, conf])

    if len(dets) == 0:
        frame_idx += 1
        continue

    img_info = frame.shape[:2]
    img_size = (frame.shape[0], frame.shape[1])
    tracks = tracker.update(np.array(dets), img_info, img_size)

    for i, trk in enumerate(tracks):
        x1, y1, x2, y2, track_id = map(int, trk)
        cropped_vehicle = frame[y1:y2, x1:x2]
        lp_results = license_model.predict(cropped_vehicle, conf=0.5, verbose=False)[0]
        best_lp = None
        for j, lp_box in enumerate(lp_results.boxes):
            cls_id = int(lp_box.cls[0])
            lp_conf = float(lp_box.conf[0])
            if cls_id != 0 or lp_conf < 0.5:
                continue
            lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_box.xyxy[0])
            abs_x1 = max(x1 + lp_x1, 0)
            abs_y1 = max(y1 + lp_y1, 0)
            abs_x2 = min(x1 + lp_x2, frame.shape[1])
            abs_y2 = min(y1 + lp_y2, frame.shape[0])
            license_crop = frame[abs_y1:abs_y2, abs_x1:abs_x2].copy()
            cropped_vehicle_clean = frame[y1:y2, x1:x2].copy()
            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (255, 0, 0), 2)
            cv2.putText(frame, 'LP', (abs_x1, abs_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            best_lp = {
                "vehicle_crop": cropped_vehicle_clean,
                "license_crop": license_crop,
                "vehicle_file": f"vehicle_id{track_id}.jpg",
                "plate_file": f"plate_id{track_id}.jpg",
                "vehicle_bbox": [x1, y1, x2, y2],
                "plate_bbox": [abs_x1, abs_y1, abs_x2, abs_y2]
            }
            break

        if best_lp:
            if track_id not in saved_ids:
                cv2.imwrite(best_lp["vehicle_file"], best_lp["vehicle_crop"])
                cv2.imwrite(best_lp["plate_file"], best_lp["license_crop"])
                log_data.append({
                    "frame": frame_idx,
                    "timestamp": datetime.now().isoformat(),
                    "track_id": track_id,
                    "vehicle_file": best_lp["vehicle_file"],
                    "plate_file": best_lp["plate_file"],
                    "vehicle_bbox": best_lp["vehicle_bbox"],
                    "plate_bbox": best_lp["plate_bbox"]
                })
                saved_ids.add(track_id)
                if track_id in vehicle_buffer:
                    del vehicle_buffer[track_id]

        if track_id not in saved_ids:
            prev = vehicle_buffer.get(track_id)
            if not prev or prev['conf'] < trk[4]:
                vehicle_buffer[track_id] = {
                    "frame_idx": frame_idx,
                    "timestamp": datetime.now().isoformat(),
                    "conf": trk[4],
                    "vehicle_file": f"vehicle_id{track_id}.jpg",
                    "vehicle_crop": cropped_vehicle,
                    "vehicle_bbox": [x1, y1, x2, y2]
                }

    for trk in tracks:
        x1, y1, x2, y2, track_id = map(int, trk)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    frame_idx += 1

for track_id, data in vehicle_buffer.items():
    if track_id not in saved_ids:
        cv2.imwrite(data["vehicle_file"], data["vehicle_crop"])
        log_data.append({
            "frame": data["frame_idx"],
            "timestamp": data["timestamp"],
            "track_id": track_id,
            "vehicle_file": data["vehicle_file"],
            "plate_file": None,
            "vehicle_bbox": data["vehicle_bbox"],
            "plate_bbox": None,
            "note": "No license plate detected in any frame."
        })
        saved_ids.add(track_id)

cap.release()
out.release()
end_time = time.time()
avg_fps = frame_idx / (end_time - start_time)
print(f"Average FPS: {avg_fps:.2f}")
with open("detection_log.json", "w") as f:
    json.dump(log_data, f, indent=4)

# Average FPS: 18.78