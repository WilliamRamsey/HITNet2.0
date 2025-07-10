import cv2
import math
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

class Trajectory:
    def __init__(self):
        self.pixels = []
    
class HelmetTrack:
    """
    ## Represents the path taken by a helmet.

    Will eventually abstract to all tracked objects, but we use the helmet's raduis
    to scale pixels to meters. When we transition to using the distance for field lines
    to scale pixels to meters, we will use this scalar value for all objects instead.

    The class really is a wrapper for the kinematics list.
    Do we want to save the origional masks?
    [(frame_number, x_pos, y_pos, radius, x_velo, y_velo, x_accel, y_accel)]
    """
    @staticmethod
    def find_mask_center(mask) -> tuple[float, float, float]:
        """
        ## Finds the center and raduis of a mask in pixels.

        ### Args:
            **mask** (ArrayType@array): [(x, y), (x, y)]
        
        ### Returns:
            average_x, average_y, radius (tuple[float, float, float]):
        """
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

        return (x_average, y_average, radius)

    def __init__(self, id, time_step):
        self.id = id
        self.time_step = time_step
        self.masks = [] # []
        self.kinematics = [] # [(frame_number, x_pos, y_pos, radius, x_velo, y_velo, x_accel, y_accel)]
        self.collisions = [] # [frame_number, frame_number... ]
        self.actual_helmet_radius = 0.124

    def add_mask(self, frame_number: int, mask) -> None:
        self.masks.append(mask)
        self.kinematics_from_mask(frame_number, mask)

    def kinematics_from_mask(self, frame_number: int, mask) -> None:
        """
        ## Appends all kinematic information based on a mask.

        Calculates the average pixel positions.
        Calculates the average radius at each point.
        Calculates velocity in meters/s
        Calculates acceleration in meters/s^2
        """
        x, y, r = self.find_mask_center(mask)

        if len(self.kinematics) == 0:
            next_sample = (frame_number, x, y, r, None, None, None, None)

        elif len(self.kinematics) == 1:
            x_velo, y_velo = self.calculate_new_velocity(x, y, r)
            next_sample = (frame_number, x, y, r, x_velo, y_velo, None, None)

        else:
            x_velo, y_velo = self.calculate_new_velocity(x, y, r)
            x_accel, y_accel = self.calculate_new_acceleration(x_velo, y_velo)
            next_sample = (frame_number, x, y, r, x_velo, y_velo, x_accel, y_accel)

        self.kinematics.append(next_sample)
        
    def calculate_new_velocity(self, x, y, r) -> tuple[float, float]:
        """
        ## Returns the X and Y velocities from the last entry into kinematics and the passed position.
        Does not append a new sample.
        Performs conversion from pixels to meters.
        """
        # Pixel scalar represents the meters per pixel by comparing the actual helmet radius to
        # the average of the last two computed helmet radi.
        pixel_scalar = self.actual_helmet_radius / (r + self.kinematics[-1][3]) / 2
        x_velo = pixel_scalar * (x - self.kinematics[-1][1]) / self.time_step
        y_velo = pixel_scalar * (y - self.kinematics[-1][2]) / self.time_step
        return x_velo, y_velo

    def calculate_new_acceleration(self, x_velo, y_velo) -> tuple[float, float]:
        """
        ## Returns the X and Y acceleration from the last entry into kinematics and the passed position.
        Does not append a new sample.
        Assumes velocity previous sample is not none.
        """
        x_accel = (x_velo - self.kinematics[-1][4]) / self.time_step
        y_accel = (y_velo - self.kinematics[-1][5]) / self.time_step
        return x_accel, y_accel

class Run:
    """
    ## Represents a segmentation run on a video.

    Used for organization and retrival 
    """
    def __init__(self, model_path, video_path, output_path):
        self.model_path = model_path
        self.video_path = video_path
        self.output_path = output_path
        self.tracks = {}

    def track_objects(self):
        pass

    def run_model(self):
        model = YOLO(self.model_path) # loads segmentation model
        cap = cv2.VideoCapture(self.video_path) # loads video capture
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        
        frame_number = 1 # yes, I know it's one indexed. Deal with it.
        while True:
            ret, image = cap.read() # reads the next image from the video stream.

            if not ret:
                break # stops on empty frame

            results = model.track(image, persist = True) # run the model on the image.

            if results[0].boxes.id is not None and results[0].masks is not None: # Make sure output is not empty
                masks = results[0].masks.xy # List of numpy arrays [x, y] arrays of each segmentation point
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for mask, track_id in zip(masks, track_ids):
                    track_id = str(track_id)

                    # generate a reference to the track
                    if str(track_id) in self.tracks.keys():
                        track = self.tracks[track_id]
                    else:
                        track = HelmetTrack(track_id, time_step = 1/fps)

                    track.add_mask(frame_number, mask)
                    
            frame_number += 1

    def display_video(self):
        pass

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
