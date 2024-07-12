from tools import *
import os

data_path = "C:/Users/willi/OneDrive\Desktop/HITNET DATA/"
work_dir = "C:/Users/willi/OneDrive/Desktop/HITNET/"

for i in range(2,11):
    save_dir = f"{work_dir}data/raw/images/{i}"
    os.makedirs(save_dir)
    select_images(f"{data_path}{i}.mp4", f"{save_dir}/", num_frames=10)
    print(i)
