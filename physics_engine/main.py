import cv2
import math
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# [{id:str, accelerations:[int] collicions:[int (indicies)]}]

# [(x pos, y pos, radius)]
# track id = index + 1
def track_helmets(model_path, video_path, display_output=True) -> list[list[tuple]]:
    # Loads segmentation model
    model = YOLO(model_path)
    # Loads video
    cap = cv2.VideoCapture(video_path)
    # Initializes video parameters
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)) 
    
    # Begins output window object
    if display_output:
        out = cv2.VideoWriter("data/output/obj.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    paths = []
    while True:
        # Read the image from video
        ret, im0 = cap.read()
        # Stop if image is empty
        if not ret:
            break
        
        # Initialize annotation drawer
        if display_output:
            annotator = Annotator(im0, line_width=1)
        
        results = model.track(im0, persist=True) # Actually run the model

        if results[0].boxes.id is not None and results[0].masks is not None: # Make sure output is not empty
            masks = results[0].masks.xy # List of numpy arrays [x, y] arrays of each segmentation point
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            # find middle and radius
            for mask, track_id in zip(masks, track_ids):
                # finding middle point
                x_sum = 0
                y_sum = 0
                for point in mask:
                    x_sum += point[0]
                    y_sum += point[1]
                x_average = x_sum / len(mask)
                y_average = y_sum / len(mask)

                radius_sum = 0
                for point in mask:
                    radius_sum += math.sqrt((x_average - point[0])**2 + (y_average - point[1])**2)
                radius = radius_sum / len(mask)
                print(track_id, type(track_id))

                if len(paths) < track_id:
                    paths.append([(x_average, y_average, radius)])
                else:
                    paths[track_id - 1].append((x_average, y_average, radius))
                if display_output: # add the output to the image
                    annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), label=str(track_id))
                    # annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), label=None)
                    
        if display_output:
            out.write(im0)
            cv2.imshow("instance-segmentation-object-tracking", im0)
        
        if display_output & cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    if display_output:
        out.release()

    cap.release()
    cv2.destroyAllWindows()
    return paths


# [(x pos, y pos, radius)]
# track id = index + 1
def track_helmets(model_path, video_path, display_output=True) -> list[list[tuple]]:
    # Loads segmentation model
    model = YOLO(model_path)
    # Loads video
    cap = cv2.VideoCapture(video_path)
    # Initializes video parameters
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)) 
    
    # Begins output window object
    if display_output:
        out = cv2.VideoWriter("data/output/obj.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    paths = []
    while True:
        # Read the image from video
        ret, im0 = cap.read()
        # Stop if image is empty
        if not ret:
            break
        
        # Initialize annotation drawer
        if display_output:
            annotator = Annotator(im0, line_width=1)
        
        results = model.track(im0, persist=True) # Actually run the model

        if results[0].boxes.id is not None and results[0].masks is not None: # Make sure output is not empty
            masks = results[0].masks.xy # List of numpy arrays [x, y] arrays of each segmentation point
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            # find middle and radius
            for mask, track_id in zip(masks, track_ids):
                # finding middle point
                x_sum = 0
                y_sum = 0
                for point in mask:
                    x_sum += point[0]
                    y_sum += point[1]
                x_average = x_sum / len(mask)
                y_average = y_sum / len(mask)

                radius_sum = 0
                for point in mask:
                    radius_sum += math.sqrt((x_average - point[0])**2 + (y_average - point[1])**2)
                radius = radius_sum / len(mask)
                print(track_id, type(track_id))

                if len(paths) < track_id:
                    paths.append([(x_average, y_average, radius)])
                else:
                    paths[track_id - 1].append((x_average, y_average, radius))
                if display_output: # add the output to the image
                    annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), label=str(track_id))
                    # annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), label=None)
                    
        if display_output:
            out.write(im0)
            cv2.imshow("instance-segmentation-object-tracking", im0)
        
        if display_output & cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    if display_output:
        out.release()

    cap.release()
    cv2.destroyAllWindows()
    return paths

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
        

        if display_output & cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    if display_output:
        out.release()

    cap.release()
    cv2.destroyAllWindows()


# C:\Users\willi\OneDrive\Desktop\HITNET\yolov8n-seg.pt
# C:/Users/willi/OneDrive/Desktop/HITNET/runs/segment/train3/weights/best.pt
track_helmets("C:/Users/willi/OneDrive/Desktop/HITNET/runs/segment/train6/weights/best.pt",
              "C:/Users/willi/OneDrive/Desktop/HITNET DATA/2024 Playoff/1.mp4", display_output=True)
