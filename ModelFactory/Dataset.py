from LabelMe2YOLO import Labelme2YOLO
import cv2
import os
import base64
import json
from ultralytics.data.annotator import auto_annotate

class Dataset:
    """
    ## Dataset class that manages the annotation development process for a dataset.
    
    Datasets follow this format:

    Dataset-Name/
    - YOLODataset_seg/ {folder containing YOLO annotations}
    - 1.jpg
    - 1.json {labelme annotations for 1.jpg}
    ... {patter for images continues in ascending order}
    
    ### Attributes:
        **path** (str): The absolute path to the storage location of data.
    """
    def __init__(self, path):
        """
        ## Creates a new Dataset at the specified path.

        If there is already data within the path it does not write over.
        
        ### Args:
            **path** (str): Absolute path to the new dataset.

        ### Returns:
            None
        """
        self.path = path
        os.makedirs(path, exist_ok=True)

    def select_images_from_video(self, video_path, num_frames=None, time_interval=None, frame_interval=None):
        """
        ## Adds selected images from video to directory without annotating.

        Finds image with largest index and adds images in ascending order after it.
        Does not check content of images and can add duplicates.
        Can mix annotated and unannotated images.

        ### Args:
            **video_path** (str): path to the video images will be selected from.
            **num_frames** (int): saves number of frames evenly spaced.
            or
            **time_interval** (int): saves n * time interval frames.
            or
            **frame_interval** (int): saves every nth frame.
        
        ### Returns:
            None
        """
        capture = cv2.VideoCapture(video_path)

        # Determine the frequency of frame selection
        if num_frames != None:
            vid_len = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = vid_len // num_frames

        elif time_interval != None:
            fps = int(capture.get(cv2.CAP_PROP_FPS))
            frame_interval = (time_interval * fps) // 1

        elif frame_interval == None:
            frame_interval = 60
        
        image_index = self.get_highest_image_index() + 1
        frame_remainder = 0
        ret, frame = capture.read()

        while ret:
            if frame_remainder % frame_interval == 0:
                cv2.imwrite(f"{self.path}/{image_index}.jpg", frame)
                image_index += 1
            frame_remainder += 1
            ret, frame = capture.read()
        
        capture.release()
        cv2.destroyAllWindows()

    def add_data(self, new_data_path):
        """
        ## Appends this dataset with the images and LABELME annotations in new_data_path.

        Assumes the images and annotations in this dataset are correctly titled 1+n.
        Assumes the YOLO annotations folder of the new dataset has been deleted.
        Basically a merge method.

        ### Args:
            new_data_path (str): path to the new data to add.
        """
        # Get file with largest numeric name in current files. BOLDLY ASSUMES ALL FILE NAMES CAN BE CAST AS INTS
        highest_image_index = self.get_highest_image_index()
        for filename in os.listdir(new_data_path):
            highest_image_index = highest_image_index + 1
            extension = filename.split(".")[-1]
            source_path = new_data_path + "/" + filename
            destination_path = self.path + "/" + str(highest_image_index) + "." + extension
            os.rename(source_path, destination_path)

    # Generates labelme annotations in directory
    # Writes over old saved annotations
    # Does not produce json file for images without annotation
    # Leaves behind labelme annotation dir
    def generate_auto_annotations(self, model_path):

        auto_annotate(data=self.path,
                      det_model=model_path,
                      sam_model="C:/Users/willi/OneDrive/Desktop/HITNET/models/mobile_sam.pt",
                      output_dir=f"{self.path}/annotations"
        )
        
        # convert yolo to labelme json format
        for file in os.listdir(f"{self.path}/annotations"):
            self.format_annotation_as_labelme(file)
            # Delete old annotations file
            os.remove(f"{self.path}/annotations/{file}")
        # delete annotations folder
        os.rmdir(f"{self.path}/annotations")
        
    # Assumes annotations are stored in /annotations
    def format_annotation_as_labelme(self, file):
        # read dimensions based on origional image and add image byte data
        img = cv2.imread(f"{self.path}/{file[:-4]}.jpg")
        h, w = img.shape[:2]
        with open(f"{self.path}/{file[:-4]}.jpg", "rb") as img_file:
            img_binary = img_file.read()
        encoded_binary = base64.b64encode(img_binary).decode("utf-8")

        # generate json template at image level
        output_dict = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": f"..\\{file[:-4].replace("/", "\\")}.jpg",
        "imageData": encoded_binary,
        "imageHeight": h,
        "imageWidth": w
        }

        # read yolo formatted data
        with open(f"{self.path}/annotations/{file}") as yolo_annotation:
        # iterate through lines in yolo annotation
            for yolo_helmet in yolo_annotation.readlines():
                yolo_helmet = yolo_helmet.split(" ")
                # grab first value as class ID
                # print(yolo_helmet)
                class_id = yolo_helmet[0]
                # grab the x and y values as their own extended lists
                yolo_x_list = yolo_helmet[1::2]
                yolo_y_list = yolo_helmet[2::2]
                points = []
                # iterate through values, scale up, and append to list
                for yolo_x, yolo_y in zip(yolo_x_list, yolo_y_list):
                    point = [float(yolo_x)*w, float(yolo_y)*h]
                    points.append(point)
                
                # ADJUTST "Helmet" to class ID for production
                output_dict["shapes"].append({
                "label": "Helmet",
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None
                })
            
            
            # save dict as json
            with open(f"{self.path}/{file[:-4]}.json", "w") as output_json:
                json.dump(output_dict, output_json, indent=2)

    # Helper function used to convert saved YOLO annotations to labelme
    def format_annotation_as_yolo(self, annotation_path):
        pass

    # Generates final YOLO dataset in folder for training
    def format_dataset_as_yolo(self):
        yolo_dataset = Labelme2YOLO(self.path, True)
    
    # Helper function to get current largest image
    def get_highest_image_index(self):
        # Get file with largest numeric name in current files. BOLDLY ASSUMES ALL FILE NAMES CAN BE CAST AS INTS
        current_files = [int(filename.split(".")[0]) for filename in os.listdir(self.path)]
        current_files.sort()
        if current_files != []:
            return current_files[-1]
        else:
            return 0

autoSegmented = Dataset("C:/Users/willi/OneDrive/Desktop/HITNET/data/datasets/Auto-Segmented")
humanVerified = Dataset("C:/Users/willi/OneDrive/Desktop/HITNET/data/datasets/Human-Verified")
# humanVerified.select_images_from_video("C:/Users/willi/OneDrive/Desktop/HITNET DATA/2024 Week 15/1.mp4", num_frames=17)
# autoSegmented.add_data("C:/Users/willi/OneDrive/Desktop/HITNET/data/datasets/Human-Verified")
autoSegmented.generate_auto_annotations("C:/Users/willi/OneDrive/Desktop/HITNET/runs/segment/train6/weights/best.pt")

# myData.select_images_from_video("C:/Users/willi/OneDrive/Desktop/HITNET DATA/2024 Week 15/1.mp4", num_frames=25)
# myData.add_data("C:/Users/willi/OneDrive/Desktop/HITNET/data/datasets/Human-Verified")
# autoSegmented.generate_auto_annotations("C:/Users/willi/OneDrive/Desktop/HITNET/runs/segment/train6/weights/best.pt")
