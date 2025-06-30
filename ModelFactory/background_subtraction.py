import cv2

cap = cv2.VideoCapture('C:/Users/willi/Downloads/1.mp4')  # or 0 for webcam
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)

    cv2.imshow('Foreground Mask', fgmask)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()