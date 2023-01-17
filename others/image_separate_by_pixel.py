from torchvision import transforms as T
import torch
import torch.nn as nn
from PIL import Image
import cv2 as cv
import os
import shutil
import numpy as np
import pydicom
import time
import pydicom.data
from torchvision.utils import save_image
import gc
from PIL import Image,ImageOps,ImageFilter
import shutil

gc.collect()
path = "/home/alican/Documents/Datasets/teknofest"
f = "TEKNOFEST_MG_EGITIM_1"
transform = T.ToTensor()
count = 0
folders = [folder for folder in os.listdir(os.path.join(path,f)) if folder.split(".")[-1]!="xlsx"]
for folder in folders:
    files = os.listdir(os.path.join(path,f,folder))[0]
    if len(files)==0:
        print(folder,files)
    dicom_path = os.path.join(folder,files)

    name = pydicom.data.data_manager.get_files(os.path.join(path,f),dicom_path )[0]
    ds = pydicom.dcmread(name)
    ds = np.round((ds.pixel_array/4095)*255)
    if ds.mean()<150:
        count += 1

print(count)
    # print(files,ds.mean())
    # image = Image.fromarray(ds.astype(np.uint8))
    # image.show()
    # time.sleep(0.5)
    # while True:
    #     try:
    #         value = transform(image).mean()
    #         if value < 0.105:
    #             res = "Y"
    #         elif value >0.155:
    #             res = "N"
    #         else:
    #             image.show()
    #             print(value)
    #             res = str(input())
            

    #         if res.capitalize() == "Y":
    #             shutil.move(os.path.join(path,folder),os.path.join(path,"Temiz",folder))
    #         elif res.capitalize() == "N":
    #             shutil.move(os.path.join(path,folder),os.path.join(path,"Kirli",folder))
    #         else:
    #             raise Exception("Incorrect input")
    #     except:
    #         continue
        # break