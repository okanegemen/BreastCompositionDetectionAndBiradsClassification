import matplotlib.pyplot as plt
import pydicom
import pydicom.data
from torch import nn
import torch
import xml.etree.ElementTree as ET
import numpy as np
import cv2

def open_annotation(annotations):
    with open(annotations, "r") as f:
        f.read()

    tree = ET.parse(annotations)
    root = tree.getroot()
    anns = [ann for ann in root[0][1][0][5][0]]
    coordinates_str = [[cord.text for cord in anns[i+1].findall("string")] for i,ann in enumerate(anns) if ann.text == "Point_px"]
    coordinates_str = [[[int(float(value)) for value in tuples.strip("()").split(", ")] for tuples in part_cord] for part_cord in coordinates_str]
    return np.array(coordinates_str)

def convert_points_to_boolmask(points, img_shape):
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, pts=[points], color =(1,0,0))
    return mask != 0

def dicom_open(base,pass_dicom):
    # enter DICOM image name for pattern
    # result is a list of 1 element
    filename = pydicom.data.data_manager.get_files(base, pass_dicom)[0]
    
    ds = pydicom.dcmread(filename)
    return ds.pixel_array # normal



if __name__ == "__main__":
    # Full path of the DICOM file is passed in base
    base = "/home/alican/Documents/AnkAI/yoloV5/sample"

    pass_dicom = "24065530_d8205a09c8173f44_MG_L_ML_ANON.dcm"  # file name is 1-12.dcm
    data = dicom_open(base,pass_dicom)

    annotations = base+"/24065530.xml"
    annotations = open_annotation(annotations)
    mask = convert_points_to_boolmask(annotations,data.shape)
    
    print(np.shape(mask),np.unique(mask))
    print(np.shape(data),np.max(data),np.min(data))

    fig = plt.figure(figsize=(10,10))
    rows=1
    cols = 2

    fig.add_subplot(rows,cols, 1)
    plt.imshow(data, cmap=plt.cm.bone)  # set the color map to bone
    plt.title("image")

    fig.add_subplot(rows,cols,2)
    plt.imshow(mask,cmap=plt.cm.bone)
    plt.title("mask")

    plt.show()
