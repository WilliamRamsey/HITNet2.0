import os
from ultralytics import YOLO
import cv2
# strictly for visualization
from ultralytics.utils.plotting import Annotator, colors

path_to_model = "C:/Users/willi/OneDrive/Desktop/HITNET/runs/segment/train/weights/best.pt"
path_to_images = "C:/Users/willi/OneDrive/Desktop/HITNET/data/datasets/games1_40/YOLODataset_seg/images/val"
# Use custom model to run first round of analysis for labelme.
model = YOLO(path_to_model)


for image_file in os.listdir(path_to_images):
    # Open image
    img = cv2.imread(f"{path_to_images}/{image_file}")
    # Run model on image
    results = model(img)
    # for result in results:
        # print(result.show())
    # Display annotation mask
    annotator = Annotator(img, line_width=1)
    # Convert annotation masks to polygons

    # Save polygons
    break

# Open polygons with images in labelme