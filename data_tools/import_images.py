from tools import *

data_path = "C:/Users/willi/OneDrive\Desktop/HITNET DATA/"
work_dir = "C:/Users/willi/OneDrive/Desktop/HITNET/"

for i in range(1,11):
    select_images(f"{data_path}{i}.mp4", f"{work_dir}data/raw/images/{i}.jpg", num_frames=10)
