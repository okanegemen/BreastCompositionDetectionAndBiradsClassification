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

def get_transforms(train=True):
    if train:
        transform = T.Compose([
                            T.RandomHorizontalFlip(0.5),
                            T.RandomRotation(7*random.random()),
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
    def __init__(self,dataset: pd.DataFrame,imgs_dir:str,train_transform=True):
        super().__init__(imgs_dir)
        self.dataset = dataset
        self.imgs_dir = imgs_dir
        self.dataset_name = config.DATASET_NAME
        self.transform = get_transforms(train_transform)

        if self.dataset_name == "INBreast":
            self.imgs_name = self.INBreast()
        elif self.dataset_name == "VinDr":
            self.imgs_name = self.VinDr()
            self.dataset = self.eliminate_unused_dicoms_VinDr(self.imgs_name,self.dataset) # eliminates rows in dataframe of dataset which are not in the image directory, deleted or moved

        categories = self.dataset["Bi-Rads"].to_list()
        self.ids = [x-1 for x in list(categories)]
        # category_ids = list(categories)
        class_weights = get_class_weights(self.ids)
        self.sampler = get_sampler(self.ids,class_weights)

    def __getitem__(self, index: int):
        data = self.dataset.iloc[index,:]
        dicti = data.to_dict()

        self.view = dicti["View"]+"_"+dicti["Laterality"]
        image = self.loadImg(self.imgs_name[dicti["File Name"]],self.view)

        bi_rads = torch.tensor(dicti["Bi-Rads"]-1,dtype=torch.int64) # -1 to convert classes to 0,1,2,3,4,5 instead of 1,2,3,4,5,6

        image = self.transform(image)
        img = T.ToPILImage()(image)
        return  image,bi_rads

    def loadImg(self,filename,view):
        if self.dataset_name == "INBreast":
            array = self.dicom_open(filename=filename)
            image = Image.fromarray(array)
        
        elif self.dataset_name == "VinDr":
            image = Image.open(os.path.join(self.imgs_dir,filename))

        image = ImageOps.grayscale(image)

        if config.EQUALIZE:
            image = ImageOps.equalize(image)
        
        if config.AUTO_CONTRAST:
            image = ImageOps.autocontrast(image)
        
        if config.MINIMIZE_IMAGE:
            transform = T.Compose([
                        T.PILToTensor()
                        ])
                
            img = transform(image)
            if self.dataset_name == "VinDr":
                img = img/255
            _,H,W = img.size()

            ignore = config.IGNORE_SIDE_PIXELS
            temp = img[:,ignore:-ignore,ignore:-ignore]

            _,centerH,centerW = ndi.center_of_mass(temp.detach().cpu().numpy())
            centerH, centerW = int(centerH)+ignore,int(centerW)+ignore
            distance_to_sideR = W - centerW

            if view == "MLO_L":
                img = img[:,centerH-int(H*0.25):centerH+int(H*0.4),:centerW+int(W*0.3)]
                _,Hx,Wx = img.size()
                transform = T.Compose([
                    T.Pad((0,0,int(Hx/1.75)-Wx,0)),
                    T.ToPILImage(),])

                img = transform(img)
            elif view == "MLO_R":
                img = img[:,centerH-int(H*0.25):centerH+int(H*0.4),centerW-distance_to_sideR -int(W*0.08):]
                _,Hx,Wx = img.size()
                transform = T.Compose([
                    T.Pad((int(Hx/1.75)-Wx,0,0,0)),
                    T.ToPILImage(),])
                img = transform(img)
            elif view == "CC_L":
                img = img[:,centerH-int(H*0.3):centerH+int(H*0.3),:centerW+int(W*0.3)]
                _,Hx,Wx = img.size()
                transform = T.Compose([
                    T.Pad((0,0,int(Hx/1.75)-Wx,0)),
                    T.ToPILImage(),])

                img = transform(img)
            elif view == "CC_R":
                img = img[:,centerH-int(H*0.3):centerH+int(H*0.3),centerW-distance_to_sideR-int(W*0.08):]
                _,Hx,Wx = img.size()
                transform = T.Compose([ 
                    T.Pad((int(Hx/1.75)-Wx,0,0,0)),
                    T.ToPILImage(),])

                img = transform(img)
            else:
                raise Exception(f"{view} is not an available option for View!")

            # img = img[:,centerH-int(H*0.3):centerW+int(H*0.5),:]

        return img

    def VinDr(self):
        dicom_paths = {}
        folder_names =  os.listdir(self.imgs_dir)
        for folder in folder_names:
            dicom_names = os.listdir(os.path.join(self.imgs_dir,folder))
            for dicom_name in dicom_names:
                dicom_paths[os.path.join(folder,dicom_name.split(".")[0])] = os.path.join(folder,dicom_name)
        return dicom_paths

    def eliminate_unused_dicoms_VinDr(self,dicom_paths:dict,dataset:pd.DataFrame):
        dataset = dataset[dataset["File Name"].isin(list(dicom_paths.keys()))]
        return dataset


    def INBreast(self):
        return {img.split("/")[-1].split("_")[0]:img for img in os.listdir(self.imgs_dir)}

    def dicom_open(self,filename):
        # enter DICOM image name for pattern
        # result is a list of 1 element
        try:
            name = pydicom.data.data_manager.get_files(self.imgs_dir, filename)[0]
            
            ds = pydicom.dcmread(name)
            array = (ds.pixel_array/4095) # normal
            return array

        except:
            print(f"{filename} is not a dicom file.")
            permission = input("Do you want to delete the file? Y/N\n")
            if permission == "Y":
                os.remove(os.path.join(self.imgs_dir,filename))

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
        
    def __len__(self) -> int:
        return len(self.dataset)

    def __str__(self):
        return str(self.dataset)

if __name__=="__main__":
    dataset_name = "VinDr"
    train, test ,imgs_dir, img_type= XLS(dataset_name).get_all_info()

    train = Dataset(train,imgs_dir,dataset_name)
    test = Dataset(test,imgs_dir,dataset_name)

    print(next(iter(train)))