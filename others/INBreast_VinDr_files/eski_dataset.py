import matplotlib.pyplot as plt
import pydicom
import pydicom.data
from others.INBreast_VinDr_files.XML_utils import XML_files
from torch import nn
import torch
import numpy as np
import pandas as pd
import cv2
from torchvision import datasets
import os
import random
import math

class DICOM_Dataset(datasets.VisionDataset):
    def __init__(self,dataset: pd.DataFrame,dicom_folder_name="AllDICOMs",xml_dicom_folder = "AllXML",threshold=2):
        super().__init__(root)
        self.dicom_folder_name = dicom_folder_name
        self.xml_folder_name = xml_dicom_folder
        self.dataset = dataset
        self.threshold = threshold

    def __getitem__(self, index: int):
        data = self.dataset.iloc[index,:]
        dicti = data.to_dict()
        self.dicom_name = dicti["dicom_names"]

        segmentation = []
        if isinstance(dicti["segmentations"],list):    
            for seg in dicti["segmentations"]:
                if len(seg)>self.threshold:
                    segmentation.append(seg)

        dicom = torch.tensor(self.dicom_open(dicti["dicom_names"])/4095)
        print([len(a) for a in segmentation])
        print(self.dicom_name)
        print(dicti["xml_names"])
        mask ,mask_all= self.convert_points_to_boolmask(segmentation,dicom.shape)
        mask = torch.tensor(mask)

        laterality = torch.tensor(self.laterality_to_int(dicti["Laterality"]))
        view = torch.tensor(self.view_to_int(dicti["View"]))
        acr = torch.tensor(dicti["ACR"] if isinstance(dicti["ACR"],int) else 0)
        bi_rads = torch.tensor(self.bi_rads_to_int(dicti["Bi-Rads"]))

        target = {
            "mask":mask,
            "mask_all":mask_all,
            "Laterality": laterality,
            "View": view,
            "ACR": acr,
            "Bi-Rads":bi_rads 
        }

        return  dicom,target

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

    def convert_points_to_boolmask(self,points, img_shape):
        if len(points)>0:
            mask = np.zeros((len(points),*img_shape), dtype=np.uint8)
        else:
            mask = np.zeros((1,*img_shape), dtype=np.uint8)

        mask_all = np.zeros((1,*img_shape), dtype=np.uint8)

        if len(points) >0:
            for i in range(len(points)):
                cv2.fillPoly(mask[i], pts=[np.array(points[i])], color =(1,0,0))
            mask_all = torch.clamp(torch.sum(torch.tensor(mask),0),0,1)
            print(mask_all.shape)
            return mask ,mask_all
        else: 
            return mask ,mask_all

    def dicom_open(self,filename):
        # enter DICOM image name for pattern
        # result is a list of 1 element
        try:
            dicom_path = os.path.join(self.dicom_folder_name,filename)
            name = pydicom.data.data_manager.get_files(root, dicom_path)[0]
            
            ds = pydicom.dcmread(name)
            array = ds.pixel_array # normal
            return array

        except:
            raise Exception(f"{filename} is not a dicom file.")

    def __str__(self):
        return str(self.dataset)

def display_dicom(dicom,target):
    mask = target["mask"]
    bi_rads = target["Bi-Rads"]
    mask_all = target["mask_all"]
    print(mask.shape,mask_all.shape)

    count,width,height = mask.shape
    if count == 0:
        count = 1
    print(count+2)
    fig = plt.figure(figsize=(15,15))
    rows= int(input("row:"))
    cols = int(input("col:"))

    
    fig.add_subplot(rows,cols, 1)
    plt.imshow(dicom, cmap=plt.cm.bone)  # set the color map to bone
    plt.title("dicom")

    if len(mask)>0:
        for i in range(count):
            fig.add_subplot(rows,cols,i+2)
            plt.imshow(mask[i],cmap=plt.cm.bone)
            plt.title(f"mask {i+1}, Bi-Rads {bi_rads}")

    fig.add_subplot(rows,cols, i+3)
    plt.imshow(mask_all.squeeze(0), cmap=plt.cm.bone)
    plt.title("mask_all")

    plt.show()

if __name__=="__main__":
    root = "/home/alican/Documents/AnkAI/yoloV5/INbreast Release 1.0"
    train, test = XML_files(root).return_datasets()
    train = DICOM_Dataset(train)
    dicom,target = train[75]
    display_dicom(dicom,target)


