from torch import nn
import xml.etree.ElementTree as ET
import os
from collections import defaultdict
import pandas as pd
import random

def default_dict():
    return None

class XML_files():
    def __init__(self,root,train_split=0.8,xml_folder_name="AllXML",dicom_folder_name="AllDICOMs"):
        self.root = root
        self.xml_folder_name = xml_folder_name
        self.dicom_folder_name = dicom_folder_name
        self.train_split = train_split

        self.xls = self.open_xls(self.root,xls_filename="INbreast.xls",row_end=410)
        self.xmls, filename = self.get_xml_files(self.root,self.xml_folder_name)
        segs = self.get_xmls_points(self.root,self.xml_folder_name)
        self.xml_segs = pd.DataFrame(list(zip(filename,segs.values())),columns=["File Name","segmentations"])

        self.dicom_info = self.list_to_df()

        self.merged = self.merge_dfs()
        self.df = self.eliminate_columns_of_df(self.merged)


    def return_datasets(self):
        self.idxs = [*range(len(self.df))]
        random.shuffle(self.idxs)
        self.split = int(self.train_split*len(self.idxs))
        self.train_idxs = self.idxs[:self.split]
        self.test_idxs = self.idxs[self.split:]
        train = self.df[self.df.index.isin(self.train_idxs)].sample(frac = 1).reset_index(drop=True)
        test = self.df[self.df.index.isin(self.test_idxs)].sample(frac = 1).reset_index(drop=True)
        return train,test

    def list_to_df(self):
        dicom_names,filename,xml_names = [],[],[]
        for name in os.listdir(os.path.join(self.root,self.dicom_folder_name)):
            if name != ".DS_Store":
                dicom_names.append(name)
                filename.append(int(float(name.split("_")[0])))
                xml_names.append(name.split("_")[0]+".xml" )

        dicom_info = pd.DataFrame(list(zip(dicom_names,filename,xml_names)),columns=["dicom_names","File Name","xml_names"])
        return dicom_info
    
    def get_xmls_points(self,root,xml_folder_name):
        points = defaultdict(default_dict)
        for xml in self.xmls:
            path = os.path.join(root,xml_folder_name,xml)
            xml_file = self.open_annotation(path)
            
            if xml_file == None:
                raise Exception(f"{xml} is not an xml file")

            seg_points = self.get_points(xml_file)
            points[xml.split(".")[0]] = seg_points
        return points

    # def map_segs(self):
    #     self.xml_segs["segmentations"] = self.xml_segs["segmentations"].map(lambda x: [a for a in x.values()])

    @staticmethod
    def get_xml_files(root,xml_folder_name):
        xmls_name = os.listdir(os.path.join(root,xml_folder_name))
        filename = [int(xml_name.split(".")[0]) for xml_name in xmls_name]
        return xmls_name,filename

    @staticmethod
    def open_annotation(path):
        try:
            with open(path, "r") as f:
                f.read()

            tree = ET.parse(path)
            root = tree.getroot()
            return root
        except:
            return None

    @staticmethod
    def open_xls(root,xls_filename,row_end):
        xls = pd.ExcelFile(os.path.join(root,xls_filename))
        sheetX = xls.parse(0).iloc[:row_end,2:]
        sheetX["File Name"].apply(lambda x:pd.to_numeric(x))
        return sheetX
    
    @staticmethod
    def eliminate_columns_of_df(df):
        eliminated_columns_names = ["Other Notes","Other Annotations","Acquisition date","Pectoral Muscle Annotation","Asymmetry","Distortion","Micros","Mass ","Findings Notes (in Portuguese)","Lesion Annotation Status"]
        df = df.drop(eliminated_columns_names, axis=1)
        return df

    def merge_dfs(self):
        merged_2 = pd.merge(left=self.dicom_info,right=self.xls,left_on="File Name",right_on="File Name",how="outer")
        merged_3 = pd.merge(left=self.xml_segs,right=merged_2,left_on="File Name",right_on="File Name",how="outer")
        return merged_3

    @staticmethod
    def get_points(xml):

        anns = xml[0][1][0][5].iter()
        segs = []
        a = -2
        for i,x in enumerate(anns):
            if x.text == "Point_px":
                a = i
            if a+1 == i:
                segs.append([a.text for a in x.findall("string")])
            
        segs = [[[int(float(value)) for value in tuples.strip("()").split(", ")] for tuples in part_cord] for part_cord in segs]
        return segs

if __name__ == "__main__":
    root = "/home/alican/Documents/AnkAI/yoloV5/INbreast Release 1.0"
    train,test = XML_files(root).return_datasets()

    print(train)
    print(test)