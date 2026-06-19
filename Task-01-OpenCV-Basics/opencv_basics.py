import cv2 

img = cv2.imread('image.jpg')
cap = cv2.VideoCapture(0)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite('rgb_image.jpg', rgb)
cv2.imwrite('gray_image.jpg', gray)
cv2.imwrite('hsv_image.jpg', hsv)

gaussian_blur = cv2.GaussianBlur(gray, (7, 7), 1)
median_blur = cv2.medianBlur(gray, 5)
cv2.imwrite('gaussian_blur.jpg', gaussian_blur)
cv2.imwrite('median_blur.jpg', median_blur)

sobel_x = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
cv2.imwrite('sobel_x.jpg', sobel_x)
cv2.imwrite('sobel_y.jpg', sobel_y)
cv2.imwrite('sobel_combined.jpg', sobel_combined)

edges = cv2.Canny(gaussian_blur, 70, 170)
cv2.imwrite('canny_edges.jpg', edges)
cv2.imwrite('sobel_x.jpg', sobel_x)
cv2.imwrite('sobel_y.jpg', sobel_y)
cv2.imwrite('sobel_combined.jpg', sobel_combined)
cv2.imwrite('canny_edges.jpg', edges)

contour, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = img.copy()
for c in contour:
    area = cv2.contourArea(c)
    if area > 250:
        cv2.drawContours(contour_image, [c], -1, (0, 255, 0), 3)
cv2.imwrite('Contours.jpg', contour_image)

saved = 0 
max_rois = 10

for i, c in enumerate(contour):
    area = cv2.contourArea(c)
    if area > 800:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h > 50:
            roi = img[y:y+h, x:x+w]
            cv2.imwrite(f'ROI_{i}.jpg', roi)
            saved += 1
            if saved >= max_rois:
                break

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_v = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_v = cv2.GaussianBlur(gray_v, (5, 5), 1)
    edges_v = cv2.Canny(blur_v, 50, 150)

    contours_v, _ = cv2.findContours(edges_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours_v:
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

    cv2.imshow("Video with contours", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.waitKey(1) & 0xFF == ord('q')
cv2.destroyAllWindows()


