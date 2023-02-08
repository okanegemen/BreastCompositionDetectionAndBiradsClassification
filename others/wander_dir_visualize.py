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
from fiximage import fit_image
import pandas as pd
import cv2
import cv2
import shutil
import imutils
from tqdm import tqdm

gc.collect()
dcm_names = ["LCC","LMLO","RCC","RMLO"]

def dicom_paths_func():
    folder_names =  [folder for folder in os.listdir(config.TEKNOFEST) if len(folder.split("."))<2]
    return folder_names

def eliminate_unused_dicoms(dicom_folders:dict,dataset:pd.DataFrame):
    # dataset = dataset[~dataset["HASTANO"].isin(hastano_from_txt())]
    dataset = dataset[dataset["HASTANO"].isin(dicom_folders)]
    return dataset

# def dicom_open(path):
#     dicom_img = pydicom.dcmread(path)
#     numpy_pixels = dicom_img.pixel_array
#     print(numpy_pixels.shape)
#     return dicom_img,numpy_pixels

def dicom_open(hastano,dcm):
    path = os.path.join(config.TEKNOFEST,hastano,dcm)
    dicom_img = pydicom.dcmread(path)
    numpy_pixels = dicom_img.pixel_array
    image = numpy_pixels-numpy_pixels.min()
    image = image/image.max()
    if image.mean()>0.5:
        image = 1-image
    return image

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
    dcm_folders = [int(line.strip("\n").split("\t")[0]) for line in lines]
    return dcm_folders

def hastano_from_dir():
    folders = [int(f) for f in os.listdir(config.TEKNOFEST) if len(f.split("."))<2]
    return folders
    

if __name__ == "__main__":
    path = "/home/alican/Documents/Datasets/TeknofestPNG"
    to_path = "/home/alican/Documents/Datasets/TeknofestExtractedPNG"
    patients = list(set([i for i in os.listdir(path) if len(i.split("."))<2]))
    for patient in tqdm(patients):
        patient_path = os.path.join(path,patient)
        # os.makedirs(patient_path)
        mammos = os.listdir(patient_path)
        for mammo in mammos:
            # image_np = dicom_open(patient,mammo)
            # image_np = imutils.resize(image_np,height=800)
            # cv2.imwrite(patient_path+"/"+mammo.split(".")[0]+".png", image_np*255)
            shutil.copyfile(os.path.join(patient_path,mammo),os.path.join(to_path,patient+"_"+mammo))