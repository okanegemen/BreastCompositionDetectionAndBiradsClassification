if __name__ == "__main__":    
    from XLS_utils import XLS 
    from utils import get_class_weights,get_sampler
    import config
    
else:
    from .XLS_utils import XLS 
    from .utils import get_class_weights,get_sampler
    import DataLoaders.config as config

import torch
import pandas as pd
from torchvision import datasets
import os
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
import pydicom
import scipy.ndimage as ndi
import random
import cv2
import time
def rand_prob():
    return 0.5*random.random()

def get_transforms(train=True):
    if train:
        transform = T.Compose([
                            T.ToPILImage(),
                            # T.RandomHorizontalFlip(0.5),
                            # T.RandomRotation(7*random.random()),
                            T.Resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)),
                            T.GaussianBlur(5),
                            T.ToTensor(),
                        ])
    else:
        transform = T.Compose([
                            T.ToPILImage(),
                            T.Resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)),
                            T.GaussianBlur(5),
                            T.ToTensor(),
                        ])
    return transform


class Dataset(datasets.VisionDataset):
    def __init__(self,dataset: pd.DataFrame,train_transform=True):
        super().__init__(self,dataset)
        self.train_transform = train_transform
        if self.train_transform:
            print("Train data is preparing...")
        else:
            print("Test data is preparing...")

        self.dcm_names = ["LCC","LMLO","RCC","RMLO"]

        self.dataset = dataset
        self.dataset_name = config.TEKNOFEST
        self.transform = get_transforms(train_transform)

        self.dicom_paths = self.dicom_paths_func()
        self.dataset = self.eliminate_unused_dicoms(self.dicom_paths,self.dataset) # eliminates rows in dataframe of dataset which are not in the image directory, deleted or moved
        categories = self.dataset["BIRADS KATEGORİSİ"].to_list()

        self.ids = [x for x in list(categories)]

        class_weights = get_class_weights(self.ids)
        self.sampler = get_sampler(self.ids,class_weights)

    def __getitem__(self, index: int):
        data = self.dataset.iloc[index,:]
        dicti = data.to_dict()

        images = self.loadImg(dicti["HASTANO"])

        birads = torch.tensor(dicti["BIRADS KATEGORİSİ"],dtype=torch.int64)
        acr = torch.tensor(dicti["MEME KOMPOZİSYONU"])
        # kadran_r = torch.tensor(dicti["KADRAN BİLGİSİ (SAĞ)"])
        # kadran_l = torch.tensor(dicti["KADRAN BİLGİSİ (SOL)"])

        for name,image in images.items():
            image = torch.from_numpy(image).float().unsqueeze(0)
            images[name] = self.transform(image)
            # print(images[name].max())
            # T.ToPILImage()(images[name]).show()
            # time.sleep(1)
        images = {key:image for key,image in images.items()}

        image = torch.stack([image.squeeze() for image in images.values()])
        target = {
            "birads":birads,
            "acr":acr
            # "kadran_r":kadran_r,
            # "kadran_l":kadran_l,
            # "names":images.keys()
        }
        return  image,birads
    

    def loadImg(self,hastano):
        images = {}
        for dcm in self.dcm_names:
            image = self.dicom_open(hastano,dcm)
            images[dcm] = image

        return images

    def dicom_paths_func(self):
        folder_names =  [folder for folder in os.listdir(config.TEKNOFEST) if len(folder.split("."))<2]
        return folder_names

    def eliminate_unused_dicoms(self,dicom_folders:dict,dataset:pd.DataFrame):
        if config.ELIMINATE_CORRUPTED_PATIENTS and self.train_transform:
            dataset = dataset[~dataset["HASTANO"].isin(hastano_from_txt())]

        dataset = dataset[dataset["HASTANO"].isin(dicom_folders)]
        return dataset

    def dicom_open(self,hastano,dcm):
        path = os.path.join(config.TEKNOFEST,hastano,dcm+".dcm")
        dicom_img = pydicom.dcmread(path)
        numpy_pixels = dicom_img.pixel_array
        img = np.array(numpy_pixels,dtype="float32")
        return img/np.max(img)

    @classmethod
    def kadran_to_bool(cls, kadranlar:list, choices = ["ÜST DIŞ","ÜST İÇ","ALT İÇ","ALT DIŞ", "MERKEZ"]):
        binary_list = [0,0,0,0,0]
        if len(kadranlar)>0:
            for kadran in kadranlar:
                binary_list[choices.index(kadran)] += 1
        return binary_list
    
    @classmethod
    def kompozisyon_to_bool(cls, kompozisyon:str, choices = ["A","B","C","D"]):
        binary_list = [0,0,0,0]
        binary_list[choices.index(kompozisyon)] += 1
        return binary_list
    
    @classmethod
    def birads_to_bool(cls, birads:str, choices = ["BI-RADS0","BI-RADS1-2","BI-RADS4-5"]):
        binary_list = [0,0,0]
        binary_list[choices.index(birads)] += 1
        return binary_list

    def __len__(self) -> int:
        return len(self.dataset)

    def __str__(self):
        return str(self.dataset)


def hastano_from_txt(txt_path = os.path.join(config.MAIN_DIR,"yoloV5","others","kirli_resimler.txt")):
    with open(txt_path) as text_file:
        lines = text_file.readlines()
    dcm_folders = [line.split("\t")[0].strip() for line in lines]
    return dcm_folders

if __name__=="__main__":
    train, test= XLS().return_datasets()

    train = Dataset(train,True)
    test = Dataset(test,False)

    print(train[3])