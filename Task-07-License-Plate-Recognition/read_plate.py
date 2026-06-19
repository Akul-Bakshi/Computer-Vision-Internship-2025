import cv2
import pytesseract
import re
import json

with open('detection_log.json', 'r') as f:
    data = json.load(f)
    for entry in data:
        plate_file = entry.get('plate_file')
        if plate_file is None:
            continue  

        img_path = 'warped_' + plate_file  
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh)

        plate_number = re.sub(r'\W+', '', text).upper()
        entry['plate_number'] = plate_number

with open('detection_log.json', 'w') as f:
    json.dump(data, f, indent=4)