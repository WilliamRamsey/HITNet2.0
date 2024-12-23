import os
from ultralytics import YOLO
import cv2
from ultralytics.data.annotator import auto_annotate
import json

path_to_model = "C:/Users/willi/OneDrive/Desktop/HITNET/runs/segment/train/weights/best.pt"
path_to_images = "C:/Users/willi/OneDrive/Desktop/HITNET/data_tools/test"
# Use custom model to run first round of analysis for labelme.
auto_annotate(  
    data="data/datasets/games1_40/YOLODataset_seg/images/val",
    det_model=path_to_model,
    sam_model="mobile_sam.pt",
    output_dir=f"{path_to_images}/annotations",
)

# convert yolo to labelme json format
for file in os.listdir(f"{path_to_images}/annotations"):
    # read dimensions based on origional image
    img = cv2.imread(f"{path_to_images}/{file[:-4]}.png")
    h, w = img.shape[:2]

    # generate json template at image level
    output_dict = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": f"..\\{path_to_images}\\{file[:-4]}.png",
        "imageHeight": h,
        "imageWidth": w
    }

    # read yolo formatted data
    with open(f"{path_to_images}/annotations/{file}") as yolo_annotation:
        # iterate through lines in yolo annotation
        for yolo_helmet in yolo_annotation.readlines():
            yolo_helmet = yolo_helmet.split(" ")
            # grab first value as class ID
            print(yolo_helmet)
            class_id = yolo_helmet[0]
            # grab the x and y values as their own extended lists
            yolo_x_list = yolo_helmet[1::2]
            yolo_y_list = yolo_helmet[2::2]
            points = []
            # iterate through values, scale up, and append to list
            for yolo_x, yolo_y in zip(yolo_x_list, yolo_y_list):
                point = [float(yolo_x)*w, float(yolo_y)*h]
                points.append(point)
            # ADJUTST "Helmet" to class ID for production
            output_dict["shapes"].append({
                "label": "Helmet",
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None
            })

        yolo_annotation.close()
    with open(f"{path_to_images}/{file[:-4]}.json", "w") as output_json:
        json.dump(output_dict, output_json, indent=2)
    output_json.close()
    # save dict as json

# Format of labelme json files
"""
{
  "version": "5.5.0",
  "flags": {},
  "shapes": [
    {
      "label": "CLASS ID",
      "points": [
        [
          X VALUE,
          Y VALUE
        ],
        [
          X VALUE,
          Y VALUE
        ]
      ],
      "group_id": null,
      "description": "",
      "shape_type": "polygon",
      "flags": {},
      "mask": null
    },
  ],
  "imagePath": "..\\images\\1.jpg",
  "imageHeight": 720,
  "imageWidth": 1280
}
"""

# Code that converts labelme points to yolo
"""
label_id = self._label_id_map[shape['label']]

if self._to_seg:
    retval = [label_id]
    for i in shape['points']:
        i[0] = round(float(i[0]) / img_w, 6)
        i[1] = round(float(i[1]) / img_h, 6)
        retval.extend(i)
    # save retval
# Open polygons with images in labelme
"""
