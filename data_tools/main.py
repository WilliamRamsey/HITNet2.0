import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

class Helmet:
    def __init__(self, IDs = [], masks = [], collisions = []):
        self.IDs = IDs # List of IDs that helmet mask is appears in image
        self.masks = masks # List of masks -> Should we transform or not?
        self.collisions = collisions # List of indexes where helmet experienced a collision

# Runs a given yolo segmentation model on a video
def run_model(model_path, video_path, display_output=True):
    # Loads segmentation model
    model = YOLO(model_path)
    # Loads video
    cap = cv2.VideoCapture(video_path)
    # Initializes video parameters
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)) 
    
    # Begins output window object
    if display_output:
        out = cv2.VideoWriter("data/output/obj.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    trackedHelmet = Helmet()
    mask_id = None
    while True:
        # Read the image from video
        ret, im0 = cap.read()
        # Stop if image is empty
        if not ret:
            break
        
        # Initialize annotation drawer
        if display_output:
            annotator = Annotator(im0, line_width=1)
        
        # Actually run the model
        results = model.track(im0, persist=True)

        # Make sure output is not empty
        if results[0].boxes.id is not None and results[0].masks is not None:
            # List of numpy arrays [x, y] arrays of each segmentation point
            masks = results[0].masks.xy 

            # List of coorisponding ids for each helmet
            track_ids = results[0].boxes.id.int().cpu().tolist()
            

            if display_output:
                for mask, track_id in zip(masks, track_ids):
                    annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), label=str(track_id))
                    # annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), label=None)

        if display_output:
            out.write(im0)
            cv2.imshow("instance-segmentation-object-tracking", im0)
        
        # Check if user has assigned a track ID
        # 
        if (mask_id is None or not (mask_id in track_ids)) and (results[0].boxes.id is not None and results[0].masks is not None):
            mask_id = input("Enter the mask ID>")
            if mask_id == "":
                mask_id = None
            else:
                mask_id = int(mask_id)
            trackedHelmet.IDs.append(mask_id)

        if mask_id is None:
            trackedHelmet.masks.append(None)
        else:
            trackedHelmet.masks.append(results[0].masks.xy)
        print(trackedHelmet.masks)

        if display_output & cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    if display_output:
        out.release()
    cap.release()
    cv2.destroyAllWindows()

# C:\Users\willi\OneDrive\Desktop\HITNET\yolov8n-seg.pt
# C:/Users/willi/OneDrive/Desktop/HITNET/runs/segment/train3/weights/best.pt
run_model("C:/Users/willi/OneDrive/Desktop/HITNET/runs/segment/train6/weights/best.pt",
          "C:/Users/willi/OneDrive/Desktop/HITNET DATA/2024 Playoff/3.mp4", display_output=True)
