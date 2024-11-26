import os

dir = "C:/Users/willi/OneDrive/Desktop/HITNET/data/raw/images"

i = 0
for folder in os.listdir(dir):
    for file in os.listdir(f"{dir}/{folder}"):
        i += 1
        os.rename(f"{dir}/{folder}/{file}", f"C:/Users/willi/OneDrive/Desktop/HITNET/data/raw/wrapped/{i}.jpg")
