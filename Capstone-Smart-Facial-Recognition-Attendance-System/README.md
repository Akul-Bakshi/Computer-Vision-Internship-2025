# Smart Facial Recognition Attendance System

## Installation Instructions

1. Install required packages:

    pip install -r requirements.txt
    pip install openpyxl
    git clone https://github.com/noahcao/OC_SORT.git
    Then either copy the folder into your project or install with:
    pip install -e ./OC_SORT

2. Make sure your MongoDB server is installed and running locally before starting the app.

---

## Usage Instructions

### Setup file paths

- Before running, open the `app.py` and `register_face.py` file and update all file path with the comment `#address` to the correct paths on your system.

---

### Register new persons

- Before starting attendance, register new users by adding their face images to the `known_faces/` folder.  
- Each image filename should be the person's name (e.g., `John_Doe.jpg`).

---

### Running the system

1. Run the Flask app:

2. The terminal will show an address like:
    Running on http://127.0.0.1:5000

3. Open your web browser and navigate to the address shown in your terminal.

4. Click **Start Attendance** to begin attendance marking.

5. The attendance list updates live and attendance will stop automatically after about 10 seconds.

---

### Using CCTV or RTSP stream

- To use a CCTV feed instead of the webcam, update the camera source in `app.py` where it says `#address` with your camera's RTSP or IP URL.

---

## Notes

- Attendance records are saved in MongoDB and exported daily as Excel files in the `records/` folder.
- The system marks attendance once per person per day.

---

