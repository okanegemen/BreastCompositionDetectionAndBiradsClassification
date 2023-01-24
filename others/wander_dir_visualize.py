import os
import torchvision.transforms as T
from visualize_one_patient import four_image_show
import time
import gc
import numpy as np
from PIL import Image
import config
import pydicom
import torch 
import scipy.ndimage as ndi
from process.fiximage import fit_image
import pandas as pd
import cv2

gc.collect()
dcm_names = ["LCC","LMLO","RCC","RMLO"]

def dicom_paths_func():
    folder_names =  [folder for folder in os.listdir(config.TEKNOFEST) if len(folder.split("."))<2]
    return folder_names

def eliminate_unused_dicoms(dicom_folders:dict,dataset:pd.DataFrame):
    # dataset = dataset[~dataset["HASTANO"].isin(hastano_from_txt())]
    dataset = dataset[dataset["HASTANO"].isin(dicom_folders)]
    return dataset

def dicom_open(path):
    dicom_img = pydicom.dcmread(path)
    numpy_pixels = dicom_img.pixel_array
    print(numpy_pixels.shape)
    return dicom_img,numpy_pixels

def get_concat_h(im1, im2):
    dst = Image.new('L', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('L', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def four_image_show(hastano,w = config.INPUT_IMAGE_HEIGHT,h = config.INPUT_IMAGE_WIDTH):
    dcm_names = ["LMLO","LCC","RMLO","RCC"]
    images = []

    for dcm in dcm_names:
        dicom,image = dicom_open(os.path.join(config.TEKNOFEST,str(hastano),dcm+".dcm"))
        image = cv2.resize(image, (400, 500))
        image = fit_image(image)
        img = torch.from_numpy(image).float()
        img = T.ToPILImage()(img/255.)
        images.append(img)

    for img in images:
        img.show()
    # a = get_concat_v(images[0],images[1])
    # b = get_concat_v(images[2],images[3])

    # c = get_concat_h(b,a)
    # c.show()

def hastano_from_txt(txt_path = os.path.join(config.MAIN_DIR,"yoloV5","others","kirli_resimler.txt")):
    with open(txt_path) as text_file:
        lines = text_file.readlines()
    dcm_folders = [int(line.split("\t")[0]) for line in lines]
    return dcm_folders

def hastano_from_dir():
    folders = [int(f) for f in os.listdir(config.TEKNOFEST) if len(f.split("."))<2]
    return folders
    

if __name__ == "__main__":

    patients = hastano_from_dir()#list(set([int(i) for i in os.listdir(TEKNOFEST) if len(i.split("."))<2]))
    images = []
    k = 23
    for i,folder in enumerate(patients[k:]):
        four_image_show(folder)
        # print(k+i,folder)
        # a = input()

        # if a == 'n':
        #     with open(os.path.join(DATASET_DIR,"images.txt"), "a") as text_file:
        #         text_file.write(str(folder)+"\n")



