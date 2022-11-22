from torch import nn
import xml.etree.ElementTree as ET
import os
from collections import defaultdict
import pandas as pd

def default_dict():
    return None

class XML_files():
    def __init__(self,root,sub_folder = "AllXML"):
        self.root = root
        self.sub_folder = sub_folder
        self.xmls = self.get_xml_files()
        self.segs = self.get_xmls_points()
        self.xml_segs = pd.DataFrame(list(zip(self.filename,self.segs.values())),columns=["File Name","segmentations"])
        # self.map_segs()

    def return_segmentations(self):
        return self.xml_segs

    def get_xmls_points(self):
        points = defaultdict(default_dict)
        for xml in self.xmls:
            path = self.get_xml_file_path(xml)
            xml_file = self.open_annotation(path)
            seg_points = self.get_points(xml_file)
            points[xml.split(".")[0]] = seg_points
        return points

    def map_segs(self):
        self.xml_segs["segmentations"] = self.xml_segs["segmentations"].map(lambda x: [a for a in x.values()])

    def get_xml_files(self):
        xmls_name = os.listdir(os.path.join(self.root,self.sub_folder))
        self.filename = [int(xml_name.split(".")[0]) for xml_name in xmls_name]
        return xmls_name


    def get_xml_file_path(self,xml_name):
        return os.path.join(self.root,self.sub_folder,xml_name)

    def open_annotation(self,path):
        try:
            with open(path, "r") as f:
                f.read()

            tree = ET.parse(path)
            root = tree.getroot()
            return root
        except:
            return None

    def get_points(self,xml):

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
    anns = XML_files(root).xml_segs

    print(anns)