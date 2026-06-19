import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Image Transformation Options')
parser.add_argument('--clahe', action='store_true', help='Apply CLAHE histogram equalization')
parser.add_argument('--bilateral', action='store_true', help='Apply bilateral filtering')
parser.add_argument('--perspective', action='store_true', help='Apply perspective transformation')
args = parser.parse_args()

img = cv2.imread('image.jpg')
display_img = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

eroded = cv2.erode(binary, kernel, iterations=2)
dilated = cv2.dilate(binary,kernel, iterations=1)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

cv2.imwrite('eroded_image.jpg', eroded)
cv2.imwrite('dilated_image.jpg', dilated)
cv2.imwrite('opened_image.jpg', opened)
cv2.imwrite('closed_image.jpg', closed)

if args.bilateral:
    bilateral_filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    cv2.imwrite('bilateral_filtered_image.jpg', bilateral_filtered)

equalized = cv2.equalizeHist(gray)
cv2.imwrite('equalized_image.jpg', equalized)

if args.clahe:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)
    cv2.imwrite('clahe_image.jpg', clahe_image)

if args.perspective:
    points = []

    def mouse_callback(event,x,y,flag,param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) <4:
            points.append([x,y])
            cv2.circle(display_img,(x,y),5,(0,255,0),-1)
            cv2.imshow("Click corners in order: Top Left, Top Right, Bottom Right, Bottom Left",display_img)

    cv2.imshow("Click corners in order: Top Left, Top Right, Bottom Right, Bottom Left", display_img)
    cv2.setMouseCallback("Click corners in order: Top Left, Top Right, Bottom Right, Bottom Left", mouse_callback)

    while len(points)<4:
        cv2.waitKey(1)

    pts = np.array(points,dtype="float32")
    widthA = np.linalg.norm(pts[2]-pts[3])
    heightA = np.linalg.norm(pts[1]-pts[2])
    heightB = np.linalg.norm(pts[0] - pts[3])
    widthB = np.linalg.norm(pts[1] - pts[0])
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    cv2.imwrite('perspective_transformed_image.jpg', warped)
    cv2.destroyAllWindows()
