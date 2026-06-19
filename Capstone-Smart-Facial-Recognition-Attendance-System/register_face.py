import cv2
from pathlib import Path
import face_recognition

#address
KNOWN_FACES_DIR = Path("/Users/akulbakshi/Desktop/ml notes/all open cv/Capstone_Project/known_faces")
KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)

known_encodings = []
known_names = []
for filename in KNOWN_FACES_DIR.iterdir():
    if filename.suffix.lower() in [".jpg", ".png"]:
        img = face_recognition.load_image_file(filename)
        enc = face_recognition.face_encodings(img)
        if enc:
            known_encodings.append(enc[0])
            known_names.append(filename.stem)

cap = cv2.VideoCapture(1)

latest_frame = None
latest_face_locations = []

def generate_frames_register():
    global latest_frame, latest_face_locations
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = frame_small[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        latest_frame = frame.copy()
        latest_face_locations = face_locations

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unregistered"
            color = (0, 0, 255) 

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
                color = (0, 255, 0)  

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(latest_frame, (left, top), (right, bottom), color, 2)
            cv2.putText(latest_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', latest_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def save_face(name):
    global latest_frame, latest_face_locations
    if not latest_face_locations or latest_frame is None or not name:
        return False
    top, right, bottom, left = [coord * 4 for coord in latest_face_locations[-1]]
    face_image = latest_frame[top:bottom, left:right]
    filepath = KNOWN_FACES_DIR / f"{name}.jpg"
    cv2.imwrite(str(filepath), face_image)
    return True

def release_resources():
    cap.release()
    cv2.destroyAllWindows()
