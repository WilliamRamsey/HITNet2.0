from ultralytics import YOLO

# Switch to yolov11n-seg.pt
model = YOLO("models/yolo11m-seg.pt")
results = model.train(data="C:/Users/willi/OneDrive/Desktop/HITNET/data/datasets/frames75/YOLODataset_seg/dataset.yaml", epochs=100)
