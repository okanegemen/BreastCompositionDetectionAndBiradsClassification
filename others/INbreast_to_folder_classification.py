import os
import pandas as pd
import shutil

def birads_to_int(birads):
    if birads.isdigit():
        return int(birads)
    else:
        return 4

root = "/home/alican/Documents/Datasets/INBreast/storage/"
path = "/home/alican/Documents/Datasets/INBreast/storage/images"
csv = "/home/alican/Documents/Datasets/INBreast/INbreast.csv"

images = os.listdir(path)
csv = pd.read_csv(csv,delimiter=";").drop(["Patient ID","Patient age","Laterality","View","Acquisition date","ACR"],axis=1)


for image_name in images:
    birads = csv[csv["File Name"]==int(image_name.split("_")[0])]["Bi-Rads"].item()
    
    shutil.copyfile(os.path.join(path,image_name), f"/home/alican/Documents/AnkAI/Test/Dataset/{birads_to_int(birads)}/{image_name}")


print("finish")