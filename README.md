# HITNET Auto-Segmentation Instructions

HITNET is split up into three micro-services. The ModelFactory, PhysicsEngine, and UploadPortal.

## ModelFactory

A tool for streamlined image selection, annotation, and model training. The actual usage of these models is housed within the PhysicsEngine.
LABELME format is used for annotation creation and storage.
A final YOLO format conversion is performed for model training.

### Creating New Datasets

~~~ Python
# Create new dataset folder or reinstante old one.
data = Dataset(path) # Dataset.py

# Select images for annotation
data.select_images_from_video(video_path, some way to determine image spacing)
# or merge in an old dataset.
data.add_data(path_to_other_dataset)

# Use best model to automatically annotate
data.generate_auto_annotations(model_path)
# This saves the annotations in the desired labelme format, but creates a yolo annotations folder.
~~~

Next, use the LABELME software to fix all the mistakes made by the auto annotator.
This is where we move into the Compiled-Data folder if following the current format.

~~~ Bash
labelme
~~~

Once annotations are fixed, manually delete the YOLO annotations folder and move the dataset to Human-Verified by creating a new Dataset instance and merging if following the current schema.

Finally, create a the YOLO annotations for the dataset.

~~~ python
data.format_dataset_as_yolo()
~~~

or

~~~ bash
python ModelFactory/labelme2yolo.py --json_dir {path to annotations} --seg
~~~

### Training and Running Models

To fine tune an existing YOLO model see segmentation_train.py
A library of YOLO models is available in HITNET/models, as well as the sam model used for auto annotations.

The output of these models is saved in HITNET/runs/segment/train{id}/weights

### Structure of Current Datasets

HITNET/data/datasets/Helmets

- Auto-Segmented: Used for raw imports and auto annotations.
- Compiled-Data: In the process of being corrected by humans
- Human-Verified: Fixed by humans with LABELME

HITNET/data/datasets/Field-Lines

## Physics Engine

## Upload Portal

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

### Long term plan for real distances

- run current segmentation model and save annotations
- detect field lines that make a rectangle from birds eye view
- use CV2 to calculate transform that makes this rectangle, well a rectangle
- apply transform to both annotations and images
- find distance between hashes and use as constant to relate pixles to meters
- plug data into physics engine
