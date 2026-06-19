import cv2
import torch
import numpy as np
import json
from trackers.ocsort_tracker.ocsort import OCSort
from ultralytics import YOLO

model = YOLO('yolo12s.pt')
names = model.names

tracker = OCSort(det_thresh=0.5)

cap = cv2.VideoCapture('/content/input_video.mp4')
width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('/content/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_idx = 0
track_log = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    dets = []
    results = model.predict(frame, conf=0.3, verbose=False)
    boxes = results[0].boxes

    for b in boxes:
        cls = int(b.cls)
        class_name = names[cls] if cls < len(names) else "unknown"
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        conf = float(b.conf)
        dets.append([x1, y1, x2, y2, conf])

    img_info = frame.shape[:2]
    img_size = (frame.shape[0], frame.shape[1])
    dets_np = np.array(dets)

    if dets_np.shape[0] == 0:
        out.write(frame)
        frame_idx += 1
        continue

    tracks = tracker.update(dets_np, img_info, img_size)

    for trk in tracks:
        x1, y1, x2, y2, track_id = map(int, trk)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        track_log.append({
            "frame": frame_idx,
            "track_id": track_id,
            "box": [x1, y1, x2, y2]
        })

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

with open('/content/tracking_log.json', 'w') as f:
    json.dump(track_log, f, indent=4)
