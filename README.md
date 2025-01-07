# HITNET Auto-Segmentation Helmet Detection

## Generating Datasets

Imperfect annotations working directory: <code>data/datasets/wrapped2</code>
Perfect autosegmented working directory: <code>data/datasets/wrapped2</code>
Data current segmentation model was trained on: <code>data/datasets/games1_40</code>

### Instructions for Dataset Creation

Select images using <code>data_tools/tools.py</code> -> <code>select_images()</code>
Automatically generate segmentations using <code>data_tools/automatic_annotator.py</code>
Use <code>labelme</code> to verify annotations
Convert annotations to YOLO format using <code>python data_tools/labelme2yolo.py --json_dir {path to annotations} --seg</code>

## Training and Running Models

<code>train.py</code>
<code>run.py</code>

## Product Plan

| Preseed MVP | First Public Release | Second Public Release | Full Research Tool |
| ----------- | -------------------- | --------------------- | ------------------ |
| Select video off computer |  |  |  |
| Select desired helmet annotation |  |  |  |
| Skips until segmentation is lost |  |  |  |
| Counts number of collisions |  |  |  |

### Preseed MVP

Bulletproof segmentaiton model:
[] Implement larger YOLOV11 model
[] Train on 1000 images from different years - outsource to matthew/friends/dad/india [https://www.freelancer.com/u/schoudhary1553]

Collision detection

UI
[] Emulate python package archatecture
[] Select desired helmet
[] Skip until helmet is lost
[] Skip until new helmets enter
[] Display total collisions

## Long term plan for real distances

- run current segmentation model and save annotations
- detect field lines that make a rectangle from birds eye view
- use CV2 to calculate transform that makes this rectangle, well a rectangle
- apply transform to both annotations and images
- find distance between hashes and use as constant to relate pixles to meters
- plug data into physics engine
