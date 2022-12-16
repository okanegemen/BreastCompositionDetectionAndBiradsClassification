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

class Dataset(datasets.VisionDataset):
    def __init__(self,dataset: pd.DataFrame,imgs_dir:str):
        super().__init__(imgs_dir)
        self.dataset = dataset
        self.imgs_dir = imgs_dir
        self.dataset_name = config.DATASET_NAME

        if self.dataset_name == "INBreast":
            self.imgs_name = self.INBreast()
        elif self.dataset_name == "VinDr":
            self.imgs_name = self.VinDr()



        categories = self.dataset["Bi-Rads"].to_list()
        self.ids = [x-1 for x in list(categories)]
        # category_ids = list(categories)
        class_weights = get_class_weights(self.ids)
        self.sampler = get_sampler(self.ids,class_weights)

    def loadImg(self,filename):
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
            a = transform(image)

            temp = torch.nonzero(a)
            temp = [torch.min(temp[:,1]),torch.max(temp[:,1]),torch.min(temp[:,2]),torch.max(temp[:,2])]
            a = a[:,temp[0]:temp[1],temp[2]:temp[3]]
            image = T.ToPILImage()(a)

        image = image.resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)) # convert grayscale
        image = TF.to_tensor(image).float()

        return image

    def VinDr(self):
        dicom_paths = {}
        folder_names =  os.listdir(self.imgs_dir)
        for folder in folder_names:
            dicom_names = os.listdir(os.path.join(self.imgs_dir,folder))
            for dicom_name in dicom_names:
                dicom_paths[os.path.join(folder,dicom_name.split(".")[0])] = os.path.join(folder,dicom_name)
        return dicom_paths


    def INBreast(self):
        return {img.split("/")[-1].split("_")[0]:img for img in os.listdir(self.imgs_dir)}

    def dicom_open(self,filename):
        # enter DICOM image name for pattern
        # result is a list of 1 element
        try:
            name = pydicom.data.data_manager.get_files(self.imgs_dir, filename)[0]
            
            ds = pydicom.dcmread(name)
            array = (ds.pixel_array/4095)*255 # normal
            return array

        except:
            print(f"{filename} is not a dicom file.")
            permission = input("Do you want to delete the file? Y/N\n")
            if permission == "Y":
                os.remove(os.path.join(self.imgs_dir,filename))

    def __getitem__(self, index: int):
        data = self.dataset.iloc[index,:]
        dicti = data.to_dict()
        image = self.loadImg(self.imgs_name[dicti["File Name"]])

        # laterality = torch.tensor(self.laterality_to_int(dicti["Laterality"]))
        # view = torch.tensor(self.view_to_int(dicti["View"]))
        # acr = torch.tensor(dicti["ACR"] if isinstance(dicti["ACR"],int) else 0)
        bi_rads = torch.tensor(dicti["Bi-Rads"]-1,dtype=torch.int64) # -1 to make classes 0,1,2,3,4,5 instead of 1,2,3,4,5,6
        # bi_rads = torch.nn.functional.one_hot(bi_rads-1, num_classes=config.NUM_CLASSES)
        # target = {
        #     "Laterality": laterality,
        #     "View": view,
        #     "ACR": acr,
        #     "Bi-Rads":bi_rads 
        # }

        return  image,bi_rads

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