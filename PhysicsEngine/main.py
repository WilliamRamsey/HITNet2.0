import cv2
import math
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

class Track:
    def __init__(self, id, initial_position, time_step):
        self.id = id
        self.positions = initial_position
        self.time_step = time_step
        self.actual_helmet_radius = 0.124 # TODO not abstracted for all tracks.
    
    def update_position(self, new_position):
        self.positions = np.append(self.positions, new_position, axis=0)

    def calculate_velocities(self):
        """
        Returns (x, y) rows of numpy array coorisponding to velocities.
        Each velocity is measured in meters per second based on the helmet radius and fps.
        """
        velocities = np.empty((len(self.positions) - 1, 2))
        if len(self.positions[:, 0]) > 1:
            for i in range(len(self.positions[:,0]) - 1):
                dx = (self.actual_helmet_radius / self.positions[i, 2]) * (self.positions[i + 1, 0] - self.positions[i, 0])
                dy = (self.actual_helmet_radius / self.positions[i, 2]) * (self.positions[i + 1, 1] - self.positions[i, 1])
                velocities[i] = [dx, dy]
        return velocities

    def calculate_last_velocity(self):
        """
        Get the last velocity of a helmet
        """
        if len(self.positions[:, 0]) > 1:
            dx = (self.actual_helmet_radius / self.positions[-1, 2]) * (self.positions[-1, 0] - self.positions[-2, 0])
            dy = (self.actual_helmet_radius / self.positions[-1, 2]) * (self.positions[-1, 1] - self.positions[-2, 1])
            return (dx, dy)
        else:
            return (0, 0)
        
    def calculate_accelerations(self):
        """
        Acceleration length is two less than the length of the positions list.
        Velocity length is one less than the length of the positions list.
        """
        velocities = self.calculate_velocities()
        for i in range(len(velocities)):
            pass

# Physics struct -> {id:{positions: np.array(), time_step: float}}
# Numpy array format:
# px x, px y, radius, (eventually collisions)
def track_objects(model_path, video_path, display_output=True):
    """
    Runs segmentation model and returns a numpy array of object tracts and sizes.
    """
    # Loads segmentation model
    model = YOLO(model_path)
    # Loads video
    cap = cv2.VideoCapture(video_path)
    # Initializes video parameters
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)) 
    # Begins output window object
    if display_output:
        out = cv2.VideoWriter("data/output/obj.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    tracks = {}
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

                # finding radius
                radius_sum = 0
                for point in mask:
                    radius_sum += math.sqrt((x_average - point[0])**2 + (y_average - point[1])**2)
                radius = radius_sum / len(mask)
                
                new_position = np.array([x_average, y_average, radius], ndmin=2)
                # if track id already exists
                if str(track_id) in tracks.keys():
                    tracks[str(track_id)].update_position(new_position)
                else:
                    tracks[str(track_id)] = Track(track_id, new_position, 1/fps)

                # print(tracks[str(track_id)].calculate_velocities())
                
                if display_output: # add the output to the 
                    path = tracks[str(track_id)]
                    pts = path.positions[:, :2].astype(np.int32).reshape(-1, 1, 2)
                    vx_color = 2250 * abs(tracks[str(track_id)].calculate_last_velocity()[0])
                    vy_color = 2250 * abs(tracks[str(track_id)].calculate_last_velocity()[1])
                    print(vy_color)
                    print(vx_color)
                    cv2.polylines(im0, [pts], False, color=(vx_color, vy_color, 0))
                    annotator.seg_bbox(mask=mask, mask_color=(vx_color, vy_color, 0), label=str(track_id))
                    # annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), label=str(track_id))
                    # annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), label=None)
                
        if display_output:
            out.write(im0)
            cv2.imshow("instance-segmentation-object-tracking", im0)
        
        if display_output and (cv2.waitKey(1) & 0xFF == ord("q")):
            break
        
    if display_output:
        out.release()

    cap.release()
    cv2.destroyAllWindows()
    return tracks

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

track_objects("C:/Users/willi/OneDrive/Desktop/HITNET/runs/segment/train6/weights/best.pt",
              "C:/Users/willi/OneDrive/Desktop/HITNET DATA/2024 Playoff/1.mp4", display_output=True)
