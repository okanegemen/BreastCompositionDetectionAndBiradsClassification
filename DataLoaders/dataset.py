import torch
import pandas as pd
from torchvision import datasets
import os
from PIL import Image
import torchvision.transforms.functional as TF
from XLS_utils import XLS 


class Dataset(datasets.VisionDataset):
    def __init__(self,dataset: pd.DataFrame,imgs_dir:str):
        super().__init__(imgs_dir)
        self.dataset = dataset
        self.imgs_dir = imgs_dir
        self.imgs_name = {img.split("/")[-1].split("_")[0]:img for img in os.listdir(imgs_dir)}

    def loadImg(self,filename):
        image = Image.open(os.path.join(self.imgs_dir,filename)).resize((512,512))
        image = TF.to_tensor(image).float()
        return image

    def __getitem__(self, index: int):
        data = self.dataset.iloc[index,:]
        dicti = data.to_dict()
        image = self.loadImg(self.imgs_name[dicti["File Name"]])

        laterality = torch.tensor(self.laterality_to_int(dicti["Laterality"]))
        view = torch.tensor(self.view_to_int(dicti["View"]))
        acr = torch.tensor(dicti["ACR"] if isinstance(dicti["ACR"],int) else 0)
        bi_rads = torch.tensor(self.bi_rads_to_int(dicti["Bi-Rads"]))

        target = {
            "Laterality": laterality,
            "View": view,
            "ACR": acr,
            "Bi-Rads":bi_rads 
        }

        return  image,target

    @staticmethod
    def bi_rads_to_int(a):
        if isinstance(a,int):
            return a
        else:
            return 4

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

    def __str__(self):
        return str(self.dataset)

if __name__=="__main__":
    path = "/home/alican/Documents/AnkAI/yoloV5/INbreast Release 1.0"
    train,test = XLS(path).return_datasets()

    imgs_dir = "/home/alican/Documents/AnkAI/yoloV5/INbreast Release 1.0/images"

    train = Dataset(train,imgs_dir)
    test = Dataset(test,imgs_dir)

    print(next(iter(train)))