import math
import cv2
import numpy as np
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt

def dynamic_canny(gray_img, target_entropy = 0.15, entropy_threshold = 0.025, upper_z = 25, lower_z_ratio = 0.85):
    blur_kernal = 45
    lower_z = upper_z * lower_z_ratio

    try_number = 1
    while True:
        blured_img = cv2.GaussianBlur(gray_img, (blur_kernal, blur_kernal), 0, None, 0)

        med_brightness = np.median(blured_img)
        brightness_sdev = np.std(blured_img)
        upper_threashold = med_brightness + (upper_z * brightness_sdev)
        lower_threashold = med_brightness + (lower_z * brightness_sdev)

        """
        Color implementation
        color_chanels = cv2.split(blured_img)
        edges = [cv2.Canny(chanel, 30, 90) for chanel in color_chanels]
        combined_edges = cv2.bitwise_or(cv2.bitwise_or(edges[0], edges[1]), edges[2])
        """
        
        canny_edges = cv2.Canny(blured_img, lower_threashold, upper_threashold, apertureSize=7)

        # evaluation
        density = np.count_nonzero(canny_edges) / canny_edges.size
        entropy = shannon_entropy(canny_edges)
        print(f"k: {blur_kernal}, lower_z: {lower_z}, upper_z: {upper_z}, density: {density}, entropy: {entropy}")

        # display image
        cdist = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
        cv2.imshow(f"Density Adjust", cdist)
        cv2.waitKey(1)


        if entropy > (target_entropy + entropy_threshold) or entropy < (target_entropy - entropy_threshold):
            # If thresholds are too close, increase the upper
            upper_z += 500 * .5 ** (try_number - 1) * (entropy - target_entropy)
            lower_z = lower_z_ratio * upper_z

        else:
            return canny_edges, upper_z



cap = cv2.VideoCapture("C:/Users/willi/Downloads/1.mp4")
upper_z = 200
ret, old_image = cap.read()
old_gray_image = cv2.cvtColor(old_image, cv2.COLOR_BGR2GRAY)

new_tracking_points = []

while True:
    ret, image = cap.read()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if not ret:
        break

    edges, upper_z = dynamic_canny(image, upper_z=upper_z)

    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    linesP = cv2.HoughLinesP(edges, 50, np.pi / 180, 120, None, 150, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
            # cv2.circle(cdst, (l[0], l[1]), 1, (0,0,225), 3)
            # cv2.circle(cdst, (l[2], l[3]), 1, (0,0,225), 3)
            new_tracking_points.append((l[0], l[1]))
            new_tracking_points.append((l[2], l[3]))
        
        p0 = np.array(new_tracking_points, dtype=np.float32).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray_image, gray_image, p0, None)

        tracking_points2 = []
        for i in range(len(p1)):
            cords = (int(p1[i][0,0]), int(p1[i][0,1]))
            cv2.arrowedLine(image, new_tracking_points[i], cords, (0,0,225), 4)
            tracking_points2.append(cords)
        old_gray_image = gray_image
        new_tracking_points = tracking_points2
    cv2.imshow("Optical Flow", image)
    cv2.imshow("Detected Lines (in red) - Probabilistic Hough Transform", cdst)
    cv2.waitKey(1)

cap.release()


"""
src = cv2.imread("C:/Users/willi/OneDrive/Desktop/HITNET/data/datasets/Helmets/Compiled-Data/1.jpg")
dst = cv2.Canny(src, 50, 200, None, 3)

cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

lines = cv2.HoughLines(dst, 5, np.pi / 180, 850, None, 0, 0)


for i in range(0, len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    print(pt1, pt2)
    cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

# linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 200, None, 50, 10)
"""
"""
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
"""
        
# cv2.imshow("Source", src)
# cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
# cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

# cv2.waitKey()