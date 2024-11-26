from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
results = model.train(data="C:/Users/willi/OneDrive/Desktop/HITNET/data/datasets/games1_10/YOLODataset/dataset.yaml", epochs=100)
