import cv2
import math

def select_images(video_path, out_path, num_frames=None, frame_interval=None):
    capture = cv2.VideoCapture(video_path)
    vid_len = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    
    if num_frames:
        frame_interval = math.floor(vid_len / num_frames)
    if frame_interval is None:
        raise Exception("select_images()\n^^^ Please specify the number of frames or interval of frame slelection")
    

    i_in = 0
    i_out = 0
    ret, frame = capture.read()

    while ret:
        if (i_in % frame_interval) == 0:
            i_out += 1
            cv2.imwrite(f"{out_path}/{i_out}.jpg", frame)
        i_in += 1
        ret, frame = capture.read()

    capture.release()
    cv2.destroyAllWindows()
    

select_images("C:/Users/willi/OneDrive/Desktop/HITNET DATA/2024 Week 15/1.mp4", 
              "C:/Users/willi/OneDrive/Desktop/HITNET/data/datasets/Auto-Segmented",
              num_frames=30)
