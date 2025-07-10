import cv2
import numpy as np

# flow = cv2.calcOpticalFlowFarneback(old_gray, new_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Meters per pixel of template
scale = 48.8 / 717


template_points = []
frame_points = []

def template_click_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        template_points.append((x, y))

def frame_click_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_points.append((x, y))

def update_template_points():
    frame_copy = template.copy()
    cv2.namedWindow("Select Template Points")
    cv2.setMouseCallback("Select Template Points", template_click_handler) # updates list

    while len(template_points) < 4:
        for point in template_points:
            cv2.circle(frame_copy, point, 1, (0, 225, 0), 2)
        cv2.imshow("Select Template Points", frame_copy)
        cv2.waitKey(1)
    cv2.destroyWindow("Select Template Points")

def update_frame_points(frame):
    frame_copy = frame.copy()
    cv2.namedWindow("Select Frame Points")
    cv2.setMouseCallback("Select Frame Points", frame_click_handler) # updates list

    while len(frame_points) < 4:
        for point in frame_points:
            cv2.circle(frame_copy, point, 1, (0, 225, 0), 2)
        cv2.imshow("Select Frame Points", frame_copy)
        cv2.waitKey(1)
    cv2.destroyWindow("Select Frame Points")


cap = cv2.VideoCapture("C:/Users/willi/Downloads/1.mp4")
template = cv2.imread("Field_Template.png")

ret, frame1 = cap.read()

while True:
    # Read new image
    ret, frame2 = cap.read()
    if not ret:
        break
    
    # Check to see if frame and template have at least 4 points.
    if len(frame_points) < 4:
        update_frame_points(frame1)
    if len(template_points) < 4:
        update_template_points()
    

    # Convert to NumPy float32 arrays
    frame_pts_np = np.array(frame_points, dtype=np.float32)
    template_pts_np = np.array(template_points, dtype=np.float32)
    
    p0 = np.array(frame_points, dtype=np.float32).reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), p0, None)

    print(p1)
    for i in range(len(p1)):
        cords = (int(p1[i][0,0]), int(p1[i][0,1]))
        cv2.circle(frame2, cords, 1, (0, 0, 225), 10)

    # Compute homography
    H, _ = cv2.findHomography(p1, template_pts_np, cv2.RANSAC)
    warped = cv2.warpPerspective(frame2, H, (template.shape[1], template.shape[0]))
    # cv2.perspectiveTransform() on masks
    cv2.imshow("Warped Frame", warped)
    cv2.waitKey(0)

    frame2 = frame1
    cv2.waitKey(1)