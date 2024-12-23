import os
from ultralytics import YOLO
import cv2

path_to_model = ""
path_to_images = "C:/Users/willi/OneDrive/Desktop/HITNET/data/datasets/games1_40/YOLODataset_seg/images/val"
# Use custom model to run first round of analysis for labelme.
model = YOLO(path_to_model)


for image_file in os.listdir(path_to_images):
    # Open image
    img = cv2.imread(f"{path_to_images}z/{image_file}")
    # Run annotator on image
    results = model(img)
    # Convert annotation masks to polygons
    print(results)
    # Save polygons

# Open polygons with images in labelme