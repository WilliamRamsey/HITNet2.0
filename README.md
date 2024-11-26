# HITNET Auto-Segmentation Helmet Detection

## Dataset Addition Outline

- working directory is data/datasets/wrapped2
- partially completed datasets are in /games1_10 and /games1_25
- continue generating new annotations by running following code
    labelme
    python data_tools/labelme2yolo.py --json_dir {path to annotations} --seg
