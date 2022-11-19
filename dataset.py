import matplotlib.pyplot as plt
import pydicom
import pydicom.data
from torch import nn
import torch
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from torchvision import datasets
import os
import pandas as pd

class DICOM_Dataset(datasets.VisionDataset):
    def __init__(self, root,xml_folder_name="AllXML",dicom_folder_name="AllDICOMs", transform=None, target_transform=None, transforms=None):
        super().__init__(root, transform=None, target_transform=None, transforms=None)
        self.root = root
        self.xml_folder_name = xml_folder_name
        self.dicom_folder_name = dicom_folder_name

        self.xls = self.open_xls()

        dicom_names,filename,xml_names,CC_or_ML = [],[],[],[]
        for name in os.listdir(os.path.join(root,dicom_folder_name)):
            dicom_names.append(name)
            filename.append(int(float(name.split("_")[0])))
            xml_names.append(name.split("_")[0]+".xml" )
        self.dicom_info = pd.DataFrame(list(zip(dicom_names,filename,xml_names)),columns=["dicom_names","File Name","xml_names"])
        self.merged = self.merge_xls_and_info()

    def merge_xls_and_info(self):
        merged = pd.merge(left=self.dicom_info,right=self.xls,left_on="File Name",right_on="File Name")
        # merged.to_csv('raw_data.csv', index=False)
        return merged

    def dicom_open(self,filename):
        # enter DICOM image name for pattern
        # result is a list of 1 element
        try:
            pass_dicom = os.path.join(self.dicom_folder_name,filename)
            name = pydicom.data.data_manager.get_files(root, pass_dicom)[0]
            
            ds = pydicom.dcmread(name)
            array = ds.pixel_array # normal
            return array

        except:
            print(filename)

    def open_xls(self,xls_filename="INbreast.xls",row_end = 410):
        xls = pd.ExcelFile(os.path.join(self.root,xls_filename))
        sheetX = xls.parse(0).iloc[:row_end,2:]
        sheetX["File Name"].apply(lambda x:pd.to_numeric(x))
        return sheetX

    def return_df(self):
        return self.merge_xls_and_info()

if __name__=="__main__":
    root = r"/home/alican/Documents/yoloV5/INbreast Release 1.0"

    df = DICOM_Dataset(root).return_df()
    print(df.head(10))
    # print(len(df[df["View"]=="MLO"]))