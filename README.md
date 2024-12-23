# HITNET Auto-Segmentation Helmet Detection

## Dataset Addition Outline

- working directory is data/datasets/wrapped2
- partially completed datasets are in /games1_10 and /games1_25
- continue generating new annotations by running following code

## Speedy annotations tool

- 

## Long term plan for real distances

- run current segmentation model and save annotations
- detect field lines that make a rectangle from birds eye view
- use CV2 to calculate transform that makes this rectangle, well a rectangle
- apply transform to both annotations and images
- find distance between hashes and use as constant to relate pixles to meters
- plug data into physics engine

<code>
labelme
python data_tools/labelme2yolo.py --json_dir {path to annotations} --seg
</code>
