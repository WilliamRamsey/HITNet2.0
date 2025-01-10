import os
import base64
import json
import cv2
from ultralytics.data.annotator import auto_annotate

path_to_model = "C:/Users/willi/OneDrive/Desktop/HITNET/runs/segment/train6/weights/best.pt"
path_to_images = "C:/Users/willi/OneDrive/Desktop/HITNET/data/datasets/wrapped3"

# Use custom model to run first round of analysis for labelme.
auto_annotate(  
    data=path_to_images,
    det_model=path_to_model,
    sam_model="C:/Users/willi/OneDrive/Desktop/HITNET/models/mobile_sam.pt",
    output_dir=f"{path_to_images}/annotations",
)


# convert yolo to labelme json format
for file in os.listdir(f"{path_to_images}/annotations"):
  # read dimensions based on origional image and add image byte data
  img = cv2.imread(f"{path_to_images}/{file[:-4]}.jpg")
  h, w = img.shape[:2]
  with open(f"{path_to_images}/{file[:-4]}.jpg", "rb") as img_file:
    img_binary = img_file.read()
  encoded_binary = base64.b64encode(img_binary).decode("utf-8")

  # generate json template at image level
  output_dict = {
    "version": "5.5.0",
    "flags": {},
    "shapes": [],
    "imagePath": f"..\\{file[:-4].replace("/", "\\")}.jpg",
    "imageData": encoded_binary,
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
  
  
  # save dict as json
  with open(f"{path_to_images}/{file[:-4]}.json", "w") as output_json:
    json.dump(output_dict, output_json, indent=2)

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
