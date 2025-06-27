import cv2

cap = cv2.VideoCapture("C:/Users/willi/Downloads/1.mp4")

while True:
    ret, image = cap.read()

    if not ret:
        break
    prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7)
    for point in p0:
        cords = (int(point[0,0]), int(point[0,1]))
        print(cords)
        cv2.circle(image, cords, 1, (0, 0, 225), 1)

    cv2.imshow("tracked_points", image)
    cv2.waitKey(100)