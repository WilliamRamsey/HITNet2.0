import math
import cv2
import numpy as np
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt

def dynamic_canny(gray_img, target_entropy=0.4, target_density=0.10, entropy_threshold=0.1, density_threshold=0.025):
    blur_kernal = 11
    threashold_sigma = 0.33

    entropies = []
    densities = []

    while True:
        blurred_img = cv2.GaussianBlur(gray_img, (blur_kernal, blur_kernal), 0)

        med_brightness = np.median(blurred_img)
        lower_threashold = int(max(0, (1 - threashold_sigma) * med_brightness))
        upper_threashold = int(min(225, (1 + threashold_sigma) * med_brightness))
        
        canny_edges = cv2.Canny(blurred_img, lower_threashold, upper_threashold)

        # evaluation
        density = np.count_nonzero(canny_edges) / canny_edges.size
        entropy = shannon_entropy(canny_edges)
        print(f"k: {blur_kernal}, t1: {lower_threashold}, t2: {upper_threashold}, density: {density}, entropy: {entropy}")

        # display
        cdist = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
        cv2.imshow(f"Density Adjust", cdist)
        cv2.waitKey(1)

        # updation

        stop = True
        """
        if density > (target_density + density_threshold):
            threashold_sigma -= 0.05
            stop = False
        elif density < (target_density - density_threshold):
            threashold_sigma += 0.05
            stop = False

        if entropy > (target_entropy + entropy_threshold):
            blur_kernal += 2
            stop = False
        elif entropy < (target_entropy + entropy_threshold):
            blur_kernal = max(1, blur_kernal - 2)
            stop = False
            if blur_kernal == 1:
                stop = True
        """
        entropies.append(entropy)
        densities.append(density)

        if stop or lower_threashold == 0 or upper_threashold == 225:
            cv2.destroyAllWindows()
            print(len(entropies), len(entropies))
            plt.scatter(entropies, densities)
            plt.xlabel("Entropy")
            plt.ylabel("Densities")
            plt.title("Entropy vs density")
            plt.show()
            break

src = cv2.imread("C:/Users/willi/OneDrive/Desktop/HITNET/data/datasets/Helmets/Compiled-Data/1.jpg")
dynamic_canny(src)

"""
cap = cv2.VideoCapture("C:/Users/willi/OneDrive/Desktop/HITNET DATA/2024 Playoff/1.mp4")
while True:
    ret, image = cap.read()

    if not ret:
        break
    
    gaus = cv2.GaussianBlur(image, (7,7), 0)
    dst = cv2.Canny(gaus, 35, 80, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 200, None, 120, 25)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    
    cv2.imshow("Detected Lines (in red) - Probabilistic Hough Transform", cdst)
    cv2.waitKey(1)

cap.release()
"""

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