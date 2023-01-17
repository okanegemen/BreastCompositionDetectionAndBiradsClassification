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
import time

def get_transforms(train=True):
    if train:
        transform = T.Compose([
                            # T.RandomHorizontalFlip(0.5),
                            # T.RandomRotation(7*random.random()),
                            T.Resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)),
                            T.ToTensor(),
                        ])
    else:
        transform = T.Compose([
                            T.Resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)),
                            T.ToTensor(),
                        ])
    return transform

# image = T.ToPILImage()(img)
# image = image.resize((config.INPUT_IMAGE_WIDTH,config.INPUT_IMAGE_HEIGHT)) # width,height
# image = TF.to_tensor(image).float()


class Dataset(datasets.VisionDataset):
    def __init__(self,dataset: pd.DataFrame,train_transform=True):
        super().__init__(self,dataset)
        if train_transform:
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
        kadran_r = torch.tensor(dicti["KADRAN BİLGİSİ (SAĞ)"])
        kadran_l = torch.tensor(dicti["KADRAN BİLGİSİ (SOL)"])

        for name,image in images.items():
            images[name] = self.transform(image)
            max_value = images[name].max()
            # print(images[name].unique())
            # T.ToPILImage()(images[name]*255).show()
            # time.sleep(1)
        images = {key:image/max_value for key,image in images.items()}
           
        target = {
            "birads":birads,
            "acr":acr,
            "kadran_r":kadran_r,
            "kadran_l":kadran_l,
            "names":images.keys()
        }
        return  images,target
    

    def loadImg(self,hastano):
        images = {}
        for dcm in self.dcm_names:
            image = self.dicom_open(os.path.join(hastano,dcm+".dcm"))

            image = Image.fromarray(image)

            if config.NUM_CHANNELS == 1:
                image = ImageOps.grayscale(image)

            elif config.NUM_CHANNELS == 3:
                image = image.convert('RGB')

            if config.EQUALIZE:
                image = ImageOps.equalize(image)
            
            if config.AUTO_CONTRAST:
                image = ImageOps.autocontrast(image)
            
            images[dcm] = image

        return images

    def dicom_paths_func(self):
        folder_names =  [folder for folder in os.listdir(config.TEKNOFEST) if len(folder.split("."))<2]
        return folder_names

    def eliminate_unused_dicoms(self,dicom_folders:dict,dataset:pd.DataFrame):
        dataset = dataset[dataset["HASTANO"].isin(dicom_folders)]
        return dataset

    def dicom_open(self,path):
        # enter DICOM image name for pattern
        # result is a list of 1 element
        name = pydicom.data.data_manager.get_files(config.TEKNOFEST,path)[0]
        
        ds = pydicom.dcmread(name)
        img = ds.pixel_array
        img = np.array(img).astype(np.float64)
        return img

    @staticmethod
    def bi_rads_to_int(a):
        if isinstance(a,int):
            return a

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

if __name__=="__main__":
    train, test= XLS().return_datasets()

    train = Dataset(train,True)
    test = Dataset(test,False)

    print(train[0])