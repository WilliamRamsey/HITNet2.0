from ultralytics import YOLO

# Switch to yolov11n-seg.pt
model = YOLO("models/yolov8n-seg.pt")
results = model.train(data="C:/Users/willi/OneDrive/Desktop/HITNET/data/datasets/games1_40/YOLODataset_seg/dataset.yaml", epochs=100)
