from ultralytics import YOLO

model = YOLO("models/yolov8n-seg.pt")
results = model.train(data="C:/Users/willi/OneDrive/Desktop/HITNET/data/datasets/games1_25/YOLODataset_seg/dataset.yaml", epochs=100)
