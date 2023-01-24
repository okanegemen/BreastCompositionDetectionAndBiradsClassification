import os
import cv2
import numpy as np
import pandas as pd
import torch
import time
from tqdm.notebook import tqdm
import cv2
from torchvision.ops import masks_to_boxes

DEBUG = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    BATCH_SIZE = 16
else:
    BATCH_SIZE = 2

     
    
class CFG:
    resize_dim = 1024
    aspect_ratio = True
    img_size = [1024, 512]


from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import re
import pydicom

MAIN_DIR = "/home/alican/Documents/"
DATASET_DIR = os.path.join(MAIN_DIR,"Datasets/")
TEKNOFEST = os.path.join(DATASET_DIR,"TEKNOFEST_MG_EGITIM_1")

def true_norm(img):
    norm = (img - img.min())
    norm = (norm / norm.max()) * 255
    return norm.astype(np.uint8)

def mask_external_contour(pixels):
    contours, _ = cv2.findContours(pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(pixels.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    return cv2.bitwise_and(pixels, mask)

def fit_image(dicom,img, size=1024):
    # Some images have narrow exterior "frames" that complicate selection of the main data. Cutting off the frame
    img = true_norm(img)*255
    img = cv2.resize(img, (size, size)).astype(np.uint8)
    img = img[5:-5, 5:-5]
    img = img*(img>15)
    
    img = cv2.equalizeHist(img)
    if img.mean()>0.5:
        img = 1 - img

    mask = mask_external_contour(img).astype(np.uint8)
    
    x, y = np.nonzero(mask)
    xl,xr = x.min(),x.max()
    yl,yr = y.min(),y.max()
    img=img[xl:xr+1, yl:yr+1]

    return img