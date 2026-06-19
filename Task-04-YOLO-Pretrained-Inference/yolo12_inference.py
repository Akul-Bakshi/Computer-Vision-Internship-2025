import json
input_vid = "/Users/akulbakshi/Desktop/ml notes/all open cv/Task 3/videotest.mp4"
yolo_version = input("Enter the YOLO version(v4,v8,v12): ")
if yolo_version == "v4":
    import cv2
    import numpy as np
    
    weights_path = "/Users/akulbakshi/Desktop/ml notes/all open cv/Task 3/yolov4-tiny.weights"
    config_path = "/Users/akulbakshi/Desktop/ml notes/all open cv/Task 3/yolov4-tiny.cfg"
    names_path = "/Users/akulbakshi/Desktop/ml notes/all open cv/Task 3/coco.names"

    net = cv2.dnn.readNet(weights_path, config_path)

    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    cap = cv2.VideoCapture(input_vid)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/Users/akulbakshi/Desktop/ml notes/all open cv/Task 3/output_video.mp4', fourcc, fps, (width, height))

    detections = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                bbox = [x, y, x + w, y + h]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                cv2.putText(frame, label, (x, max(y - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detection = {
                    "frame": int(frame_idx),
                    "class_id": int(class_ids[i]),
                    "confidence": float(confidences[i]),
                    "bbox": [int(coord) for coord in bbox]
                }
                detections.append(detection)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

elif yolo_version == "v8" or yolo_version == "v12":
    from ultralytics import YOLO

    if yolo_version == "v8":
        model = YOLO("yolov8n.pt")
    elif yolo_version == "v12":
        model = YOLO("yolov8x.pt")

    result = model(input_vid, show=False, save=True, save_txt=False, save_conf=True, conf=0.5, project="/Users/akulbakshi/Desktop/ml notes/all open cv/Task 3", name="output")

    detections = []
    for frame_idx, r in enumerate(result):
        for box in r.boxes:
            detection = {
                "frame": frame_idx,
                "class_id": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            }
            detections.append(detection)

with open("/Users/akulbakshi/Desktop/ml notes/all open cv/Task 3/detections.json", "w") as f:
    json.dump(detections, f, indent=4)