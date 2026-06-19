from flask import Flask, render_template, Response, url_for, redirect, request, jsonify
import cv2
import numpy as np
from ocsort import OCSort
import torch
from datetime import datetime
from pymongo import MongoClient
from pathlib import Path
import pandas as pd
from ultralytics import YOLO 
import face_recognition

#address
KNOWN_FACES_DIR = Path("/Users/akulbakshi/Desktop/ml notes/all open cv/Capstone_Project/known_faces")

reset_json = True

app = Flask(__name__)
camera = cv2.VideoCapture(0)
register_camera = cv2.VideoCapture(0)

attendance_active = False
attendance_start_frame = None
ATTENDANCE_DURATION_FRAMES = 90 

client = MongoClient('mongodb://localhost:27017/')
db = client['attendance_db']
attendance_collection = db['attendance']

#address
model = YOLO("/Users/akulbakshi/Desktop/ml notes/all open cv/Capstone_Project/yolov8n-face.pt") 

known_face_encodings = []
known_face_names = []

for img_path in KNOWN_FACES_DIR.iterdir():
    if img_path.suffix.lower() in [".jpg", ".png"]:
        img = face_recognition.load_image_file(str(img_path))
        encodings = face_recognition.face_encodings(img)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(img_path.stem)

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    record = attendance_collection.find_one({"name": name, "date": today})
    if not record:
        print(f"Marking attendance for {name}")
        attendance_collection.insert_one({
            "name": name,
            "date": today,
            "time": datetime.now().strftime("%H:%M:%S")
        })
        export_attendance_to_excel()

@app.route('/')
def home():
    today = datetime.now().strftime("%Y-%m-%d")
    present_records = attendance_collection.find({"date": today})
    present_names = {r["name"] for r in present_records}

    all_registered_names = [p.stem for p in KNOWN_FACES_DIR.iterdir() if p.suffix.lower() in [".jpg", ".png"]]

    attendance_list = [{"name": name, "present": name in present_names} for name in all_registered_names]
    return render_template('index.html', video_feed_url=url_for('video_feed'), attendance_list=attendance_list)

@app.route('/register')
def register():
    return render_template('register.html', video_feed_url=url_for('register_video_feed'))

@app.route('/save_face', methods=['POST'])
def save_face():
    name = request.form.get('name')
    success, frame = register_camera.read()
    if not success or not name:
        return "Failed to capture face or missing name", 400
    results = model(frame)
    if not results or not results[0].boxes:
        return "No face detected. Please try again.", 400

    box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    face_image = frame[y1:y2, x1:x2]
    filepath = KNOWN_FACES_DIR / f"{name}.jpg"
    cv2.imwrite(str(filepath), face_image)

    return redirect(url_for('register'))

def gen_register_frames():
    while True:
        success, frame = register_camera.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/attendance')
def attendance():
    return redirect(url_for('home'))

def gen_frames():
    tracker = OCSort(det_thresh=0.5)
    id_name_color_map = {}  # track_id -> (name, color)

    last_tracked_objects = []
    frame_count = 0

    global attendance_active, attendance_start_frame

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_count += 1
        run_detection = (frame_count % 3 == 0)

        if attendance_active:
            if attendance_start_frame is None:
                attendance_start_frame = frame_count
            elif frame_count - attendance_start_frame >= ATTENDANCE_DURATION_FRAMES:
                attendance_active = False
                attendance_start_frame = None

        if run_detection:
            results = model(frame, conf=0.15, verbose=False)
            detections = []
            if results and results[0].boxes:
                for box, conf in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = box.astype(int)
                    conf = float(conf)
                    detections.append([x1, y1, x2, y2, conf, 0])

            if detections:
                detections = np.array(detections)
                detections_tensor = torch.from_numpy(detections).float()
            else:
                detections_tensor = torch.empty((0, 6))

            tracked_objects = tracker.update(detections_tensor, frame.shape[:2])
            last_tracked_objects = tracked_objects.copy() if hasattr(tracked_objects, 'copy') else tracked_objects

            current_ids = set()

            for obj in tracked_objects:
                x1, y1, x2, y2, track_id, *rest = obj.astype(int)
                current_ids.add(track_id)

                h, w = frame.shape[:2]
                x1_clamped = max(0, x1)
                y1_clamped = max(0, y1)
                x2_clamped = min(w, x2)
                y2_clamped = min(h, y2)

                face_img = frame[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

                if face_img.size == 0:
                    if track_id not in id_name_color_map:
                        id_name_color_map[track_id] = ("UnRegistered", (0, 0, 255))
                else:
                    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_face)
                    if encodings:
                        face_encoding = encodings[0]
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.65)
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances) if face_distances.size > 0 else None
                        if best_match_index is not None and matches[best_match_index]:
                            matched_name = known_face_names[best_match_index]
                            color = (0, 255, 0)
                            if attendance_active:
                                mark_attendance(matched_name)
                            id_name_color_map[track_id] = (matched_name, color)
                        else:
                            # Only set unregistered if no cached name
                            if track_id not in id_name_color_map:
                                id_name_color_map[track_id] = ("UnRegistered", (0, 0, 255))
                    else:
                        if track_id not in id_name_color_map:
                            id_name_color_map[track_id] = ("UnRegistered", (0, 0, 255))

                name, color = id_name_color_map.get(track_id, ("UnRegistered", (0, 0, 255)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            lost_ids = set(id_name_color_map.keys()) - current_ids
            for lost_id in lost_ids:
                del id_name_color_map[lost_id]

        else:
            for obj in last_tracked_objects:
                x1, y1, x2, y2, track_id, *rest = obj.astype(int)
                name, color = id_name_color_map.get(track_id, ("UnRegistered", (0, 0, 255)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def export_attendance_to_excel():
    today = datetime.now().strftime("%Y-%m-%d")
    records = list(attendance_collection.find({"date": today}))
    if records:
        df = pd.DataFrame(records)
        #address
        record_dir = Path("/Users/akulbakshi/Desktop/ml notes/all open cv/Capstone_Project/records")
        record_dir.mkdir(exist_ok=True)
        file_path = record_dir / f"attendance_{today}.xlsx"
        df.to_excel(file_path, index=False)
        print(f"Attendance saved to {file_path}")
    else:
        print("No attendance records to export today")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register_video_feed')
def register_video_feed():
    return Response(gen_register_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_attendance', methods=['POST', 'GET'])
def start_attendance():
    global attendance_active, attendance_start_frame, reset_json
    today = datetime.now().strftime("%Y-%m-%d")
    attendance_active = True
    attendance_start_frame = None 
    reset_json = False 
    print("Attendance started and reset_json set to False")
    return jsonify({"message": "Attendance started and records reset"})

@app.route('/get_attendance')
def get_attendance():
    global reset_json
    today = datetime.now().strftime("%Y-%m-%d")
    all_registered_names = [p.stem for p in KNOWN_FACES_DIR.iterdir() if p.suffix.lower() in [".jpg", ".png"]]
    if reset_json:
        attendance_list = [{"name": name, "present": False} for name in all_registered_names]
    else:
        present_records = attendance_collection.find({"date": today})
        present_names = {r["name"] for r in present_records}
        attendance_list = [{"name": name, "present": name in present_names} for name in all_registered_names]
    return jsonify(attendance_list)

if __name__ == '__main__':
    reset_json = True
    app.run(debug=True)