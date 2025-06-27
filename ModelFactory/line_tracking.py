import math
import cv2
import numpy as np
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt

def dynamic_canny(gray_img, target_entropy = 0.185, entropy_threshold = 0.025, lower_z = 0.6, upper_z = 1.3):
    blur_kernal = 7

    while True:
        blurred_img = cv2.GaussianBlur(gray_img, (blur_kernal, blur_kernal), 0)

        med_brightness = np.median(blurred_img)
        brightness_sdev = np.std(blurred_img)
        upper_threashold = med_brightness + (upper_z * brightness_sdev)
        lower_threashold = med_brightness + (lower_z * brightness_sdev)
        
        canny_edges = cv2.Canny(blurred_img, lower_threashold, upper_threashold)

        # evaluation
        density = np.count_nonzero(canny_edges) / canny_edges.size
        entropy = shannon_entropy(canny_edges)
        print(f"k: {blur_kernal}, lower_z: {lower_z}, upper_z: {upper_z}, density: {density}, entropy: {entropy}")

        # display image
        cdist = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
        cv2.imshow(f"Density Adjust", cdist)
        cv2.waitKey(1)
        
        if entropy > (target_entropy + entropy_threshold):
            # If thresholds are too close, increase the upper
            if upper_z - lower_z < .25:
                upper_z += 0.03
            else:
                lower_z += 0.03

        elif entropy < (target_entropy - entropy_threshold):
            if upper_z - lower_z > 0.5:
                upper_z -= 0.03
            else:
                lower_z -= 0.03
        
        else:
            return canny_edges, lower_z, upper_z



cap = cv2.VideoCapture("C:/Users/willi/Downloads/1.mp4")
lower_z = 0.6
upper_z = 1.3

while True:
    print("===")
    ret, image = cap.read()

    if not ret:
        break
    dst, lower_z, upper_z = dynamic_canny(image, lower_z = lower_z, upper_z = upper_z)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 200, None, 150, 40)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    
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