import json
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.utils.ops import process_mask
import cv2

car_model = YOLO("/content/yolov12/runs/detect/yolov12n_vehicle_training/weights/best.pt")
license_model = YOLO("/content/yolov12/runs/detect/yolov12n_license_training/weights/best.pt")

input_vid = "/content/train30.mp4"
cap = cv2.VideoCapture(input_vid)

results = car_model(source=input_vid, stream=True, conf=0.25)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_with_boxes.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
all_license_detections = []
confusion_matrix = ConfusionMatrix(task='detect', nc=2)

frame_idx = 0
for r in results:
    ret, frame = cap.read()
    if not ret:
        break

    car_boxes = r.boxes
    h, w, _ = frame.shape

    for box in car_boxes:
        cls_id = int(box.cls[0])
        print("Detected class name:", car_model.names[cls_id])
        if car_model.names[cls_id].lower() in ["car", "vehicle", "cars"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = float(box.conf[0])
            print(f"Drawing car box at {(x1, y1, x2, y2)} with confidence {confidence:.2f}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Car: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cropped_car = frame[y1:y2, x1:x2]
            lp_results = license_model.predict(cropped_car, conf=0.25, verbose=False)[0]
            for lp_box in lp_results.boxes:
                lp_cls_id = int(lp_box.cls[0])
                if lp_cls_id != 0:
                    continue

                lpx1, lpy1, lpx2, lpy2 = map(int, lp_box.xyxy[0])

                abs_x1 = x1 + lpx1
                abs_y1 = y1 + lpy1
                abs_x2 = x1 + lpx2
                abs_y2 = y1 + lpy2
                cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (255, 0, 0), 2)
                lp_conf = float(lp_box.conf[0])
                lp_label = "Plate: {:.2f}".format(lp_conf)
                cv2.putText(frame, lp_label, (abs_x1, abs_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                all_license_detections.append({
                    "frame": frame_idx,
                    "car_box": [x1, y1, x2, y2],
                    "plate_box": [abs_x1, abs_y1, abs_x2, abs_y2],
                    "confidence": float(lp_box.conf[0])
                })
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

with open("license_plate_detections.json", "w") as f:
    json.dump(all_license_detections, f, indent=2)

print("Evaluating License Plate Model (mAP)...")
license_map_results = license_model.val(data="/content/license_plates_dataset/data.yaml")
print("License Plate Model - mAP@0.5:", license_map_results.box.map50)
print("License Plate Model - mAP@0.5:0.95:", license_map_results.box.map)

print("Evaluating Vehicle Model (mAP)...")
vehicle_map_results = car_model.val(data="/content/vehicle_count_dataset/data.yaml")
print("Vehicle Model - mAP@0.5:", vehicle_map_results.box.map50)
print("Vehicle Model - mAP@0.5:0.95:", vehicle_map_results.box.map)
