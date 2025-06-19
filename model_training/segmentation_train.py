from ultralytics import YOLO


model = YOLO("models/yolo11m-seg.pt")
results = model.train(data="C:/Users/willi/OneDrive/Desktop/HITNET/data/datasets/Compiled-Data/YOLODataset_seg/dataset.yaml", epochs=100, imgsz=[720, 1280], rect=True)
