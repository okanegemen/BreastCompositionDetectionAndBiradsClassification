import matplotlib.pyplot as plt
import pydicom
import pydicom.data
from XML_utils import XML_files
from torch import nn
import torch
import numpy as np
import pandas as pd
import cv2
from torchvision import datasets
import os
import random
import math
from PIL import Image
import torchvision.transforms.functional as TF



class DICOM_Dataset(datasets.VisionDataset):
    def __init__(self,dataset: pd.DataFrame,imgs_dir:str,threshold=2):
        super().__init__()
        self.dataset = dataset
        self.threshold = threshold
        self.imgs_name = {img.split("/")[-1].split("_")[0]:img for img in os.listdir(imgs_dir)}


    def loadImg(self,filename):
        img_path = self.imgs_name[filename]
        image = Image.open(img_path)
        x = TF.to_tensor(image)
        return x

    def __getitem__(self, index: int):
        data = self.dataset.iloc[index,:]
        dicti = data.to_dict()
        image = self.loadImg(self.imgs_name[dicti["File Name"]])

        laterality = torch.tensor(self.laterality_to_int(dicti["Laterality"]))
        view = torch.tensor(self.view_to_int(dicti["View"]))
        acr = torch.tensor(dicti["ACR"] if isinstance(dicti["ACR"],int) else 0)
        bi_rads = torch.tensor(self.bi_rads_to_int(dicti["Bi-Rads"]))

        target = {
            "Laterality": laterality,
            "View": view,
            "ACR": acr,
            "Bi-Rads":bi_rads 
        }

        return  image,target

    @staticmethod
    def bi_rads_to_int(a):
        if isinstance(a,int):
            return a
        else:
            return 4

    @staticmethod
    def view_to_int(a:str):
        if a == "MLO":
            return 0
        elif a == "CC":
            return 1

    @staticmethod
    def laterality_to_int(a:str):
        if a == "L":
            return 0
        elif a == "R":
            return 1

    def __str__(self):
        return str(self.dataset)

if __name__=="__main__":
    path = "/home/alican/Documents/AnkAI/"
    