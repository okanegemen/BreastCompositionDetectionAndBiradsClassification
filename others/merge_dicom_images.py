import pydicom
from PIL import Image,ImageOps
import os
import numpy as np
import time
from tqdm import tqdm
import torchvision.transforms as T
import torch

MAIN_DIR = "/home/alican/Documents/"
DATASET_DIR = os.path.join(MAIN_DIR,"Datasets/")
TEKNOFEST = os.path.join(DATASET_DIR,"TEKNOFEST_MG_EGITIM_1")

def dicom_open(path):
    name = pydicom.data.data_manager.get_files(TEKNOFEST,path)[0]
    
    ds = pydicom.dcmread(name)
    img = ds.pixel_array
    img = np.array(img).astype(np.float64)/4095.
    return img

if __name__ == "__main__":

    dcm_names = ["LMLO","LCC","RMLO","RCC"]
    patients = [i for i in os.listdir(TEKNOFEST) if len(i.split("."))<2]

    for dcm in dcm_names:
        info = {
            "mixedUp":Image.new('L', size=(500, 400)),
            "mean":[],
            "max":[],
            "min":[],
            "count":0
            }
        false_images = []
        count = 0
        for i,folder in enumerate(tqdm(patients)):
            dicom = dicom_open(os.path.join(TEKNOFEST,folder,dcm+".dcm"))*255.
            image = Image.fromarray(dicom)
            image = image.convert('L')
            image = image.resize([500,400],Image.Resampling.LANCZOS)
            array = np.array(image)
            if array[:,-60:].mean() > 240.:    
                weight = 1/(i+1+count)
                array = ImageOps.equalize(Image.blend(info["mixedUp"],image, alpha=weight))
                img = np.array(array)[:,-60:].mean()
                if img >240.:
                    info["mixedUp"] = array
                    info["mean"].append(dicom.mean()) 
                    info["max"].append(dicom.max()) 
                    info["min"].append(dicom.min()) 
                    info["count"] = 1/weight
                else:
                    count -= 1   
                    false_images.append(folder) 

                if i%100==0 and len(info["max"])>0:
                    ImageOps.equalize(info["mixedUp"]).show()
                    time.sleep(1)
        print(1/weight)
        print(false_images)
        