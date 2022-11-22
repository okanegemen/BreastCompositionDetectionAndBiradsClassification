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
from XML_utils import XML_files

class DICOM_Dataset(datasets.VisionDataset):
    def __init__(self, root,xml_folder_name="AllXML",dicom_folder_name="AllDICOMs", transform=None, target_transform=None, transforms=None):
        super().__init__(root, transform=None, target_transform=None, transforms=None)
        self.root = root
        self.xml_folder_name = xml_folder_name
        self.dicom_folder_name = dicom_folder_name
        self.xml_df = XML_files(root).return_segmentations()


        self.xls = self.open_xls()

        dicom_names,filename,xml_names = [],[],[]
        for name in os.listdir(os.path.join(root,dicom_folder_name)):
            if name != ".DS_Store":
                dicom_names.append(name)
                filename.append(int(float(name.split("_")[0])))
                xml_names.append(name.split("_")[0]+".xml" )

            
        self.dicom_info = pd.DataFrame(list(zip(dicom_names,filename,xml_names)),columns=["dicom_names","File Name","xml_names"])
        self.merged = self.merge_dfs()
        self.eliminate_columns_of_df()
        
    
    def __getitem__(self, index: int):
        pass
    
    def eliminate_columns_of_df(self):
        eliminated_columns_names = ["Other Notes","Other Annotations","Acquisition date","Pectoral Muscle Annotation","Asymmetry","Distortion","Micros","Mass ","Findings Notes (in Portuguese)","Lesion Annotation Status"]
        self.merged = self.merged.drop(eliminated_columns_names, axis=1)

    def fill_na_in_df(self):
        fill_na_columns = []
        # self.merged['DataFrame Column'] = self.merged['DataFrame Column'].fillna(0)

    def display_dicom(self,dir,dicom_name,xml_name=None,sub_folder=["AllDICOMs","AllXML"]):
        dicom_path = os.path.join(sub_folder[0],dicom_name)
        data = self.dicom_open(dir,dicom_path)

        if xml_name!=None:
            ann = os.path.join(dir,sub_folder[1], xml_name)
            ann = self.open_annotation(ann)
            mask = self.convert_points_to_boolmask(ann,data.shape)

            fig = plt.figure(figsize=(10,10))
            rows=1
            cols = 2

            fig.add_subplot(rows,cols, 1)
            plt.imshow(data, cmap=plt.cm.bone)  # set the color map to bone
            plt.title("dicom")

            fig.add_subplot(rows,cols,2)
            plt.imshow(mask,cmap=plt.cm.bone)
            plt.title("mask")

            plt.show()
        
        else:
            plt.imshow(data,cmap=plt.cm.bone)
            plt.title("dicom")
            plt.show()

    def convert_points_to_boolmask(self,points, img_shape):
        mask = np.zeros(img_shape, dtype=np.uint8)
        if points!= None:
            cv2.fillPoly(mask, pts=[points], color =(1,0,0))
        else:
            pass
        return mask != 0

    def merge_dfs(self):
        merged_2 = pd.merge(left=self.dicom_info,right=self.xls,left_on="File Name",right_on="File Name",how="outer")
        merged_3 = pd.merge(left=self.xml_df,right=merged_2,left_on="File Name",right_on="File Name",how="outer")
        # merged.to_csv('raw_data.csv', index=False)
        return merged_3

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
            print(filename)

    def open_xls(self,xls_filename="INbreast.xls",row_end = 410):
        xls = pd.ExcelFile(os.path.join(self.root,xls_filename))
        sheetX = xls.parse(0).iloc[:row_end,2:]
        sheetX["File Name"].apply(lambda x:pd.to_numeric(x))
        return sheetX

    def return_df(self):
        return self.merge_xls_and_info()

if __name__=="__main__":
    root = "/home/alican/Documents/AnkAI/yoloV5/INbreast Release 1.0"
    df = DICOM_Dataset(root).merged
    
    print(df)
