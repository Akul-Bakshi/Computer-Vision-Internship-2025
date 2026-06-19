import cv2
import json
from datetime import datetime
from OC_SORT.trackers.ocsort_tracker.ocsort import OCSort
import numpy as np
from ultralytics import YOLO

model = YOLO('yolo12s.pt')
license_model = YOLO('/content/best.pt')

tracker = OCSort(det_thresh=0.5)

cap = cv2.VideoCapture('/content/input_video.mp4')
width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('output_detection.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_idx = 0
log_data = []

saved_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

        detected_plate = False

        for j, lp_box in enumerate(lp_results.boxes):
            cls_id = int(lp_box.cls[0])
            lp_conf = float(lp_box.conf[0])

            if cls_id != 0 or lp_conf < 0.5:
                continue

            lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_box.xyxy[0])
            margin = 15
            abs_x1 = max(x1 + lp_x1 - margin, 0)
            abs_y1 = max(y1 + lp_y1 - margin, 0)
            abs_x2 = min(x1 + lp_x2 + margin, frame.shape[1])
            abs_y2 = min(y1 + lp_y2 + margin, frame.shape[0])

            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (255, 0, 0), 2)
            cv2.putText(frame, 'LP', (abs_x1, abs_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            license_crop = frame[abs_y1:abs_y2, abs_x1:abs_x2]

            if track_id not in saved_ids:
                vehicle_filename = f"vehicle_id{track_id}.jpg"
                plate_filename = f"plate_id{track_id}.jpg"
                cv2.imwrite(vehicle_filename, cropped_vehicle)
                cv2.imwrite(plate_filename, license_crop)

                log_data.append({
                    "frame": frame_idx,
                    "timestamp": datetime.now().isoformat(),
                    "track_id": track_id,
                    "vehicle_file": vehicle_filename,
                    "plate_file": plate_filename,
                    "vehicle_bbox": [x1, y1, x2, y2],
                    "plate_bbox": [abs_x1, abs_y1, abs_x2, abs_y2]
                })
                saved_ids.add(track_id)

            detected_plate = True
            break

        if not detected_plate:
            if track_id not in saved_ids:
                vehicle_filename = f"vehicle_id{track_id}.jpg"
                cv2.imwrite(vehicle_filename, cropped_vehicle)
                log_data.append({
                    "frame": frame_idx,
                    "timestamp": datetime.now().isoformat(),
                    "track_id": track_id,
                    "vehicle_file": vehicle_filename,
                    "plate_file": None,
                    "vehicle_bbox": [x1, y1, x2, y2],
                    "plate_bbox": None,
                    "note": "No license plate detected with sufficient confidence."
                })
                saved_ids.add(track_id)
            
    for trk in tracks:
        x1, y1, x2, y2, track_id = map(int, trk)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    frame_idx += 1
cap.release()
out.release()
with open("detection_log.json", "w") as f:
    json.dump(log_data, f, indent=4)