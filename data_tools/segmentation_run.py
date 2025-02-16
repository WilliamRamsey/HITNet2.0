import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Runs a given yolo segmentation model on a video
def run_model(model_path, video_path):
    model = YOLO(model_path)  # Loads segmentation model
    cap = cv2.VideoCapture(video_path)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter("data/output/obj.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    while True:
        # Read the image from video
        ret, im0 = cap.read()
        # Stop if image is empty
        if not ret:
            break
        
        # Initialize annotation drawer
        annotator = Annotator(im0, line_width=1)
        # Actually run the model
        results = model.track(im0, persist=True)

        # Make sure output is not empty
        if results[0].boxes.id is not None and results[0].masks is not None:
            # List of numpy arrays [x, y] arrays of each segmentation point
            masks = results[0].masks.xy 

            # List of coorisponding ids for each helmet
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # 
            for mask, track_id in zip(masks, track_ids):
                annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), label=str(track_id))
                # annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), label=None)

        out.write(im0)
        cv2.imshow("instance-segmentation-object-tracking", im0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()

# C:\Users\willi\OneDrive\Desktop\HITNET\yolov8n-seg.pt
# C:/Users/willi/OneDrive/Desktop/HITNET/runs/segment/train3/weights/best.pt
run_model("C:/Users/willi/OneDrive/Desktop/HITNET/runs/segment/train6/weights/best.pt",
          "C:/Users/willi/OneDrive/Desktop/HITNET DATA/2024 Playoff/3.mp4")
