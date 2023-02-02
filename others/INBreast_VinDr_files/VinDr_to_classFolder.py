import DataLoaders.config as config
from DataLoaders.XLS_utils import XLS
import pandas as pd
import os
import shutil

class Dataset():
    def __init__(self,dataset: pd.DataFrame,imgs_dir:str,train_transform=True):
        if train_transform:
            print("Train data is preparing...")
        else:
            print("Test data is preparing...")

        self.dataset = dataset
        self.imgs_dir = imgs_dir
        self.dataset_name = config.DATASET_NAME

        if self.dataset_name == "VinDr":
            self.imgs_name = self.VinDr()
            self.dataset = self.eliminate_unused_dicoms_VinDr(self.imgs_name,self.dataset) # eliminates rows in dataframe of dataset which are not in the image directory, deleted or moved

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

if __name__=="__main__":

    root = "/home/alican/Documents/Datasets/VinDr-mammo/Dicom_images/"

    train, test ,imgs_dir= XLS().get_all_info()

    train = Dataset(train,imgs_dir,True).dataset
    test = Dataset(test,imgs_dir,False).dataset

    for data in train.values:
        image_name = data[-1]
        birads = data[3]
        new_name = data[1]+"_"+data[0]+"_"+str(birads)+"_"+image_name.split("/")[1]+".png"

        shutil.copyfile(os.path.join(root,image_name+".png"), f"/home/alican/Documents/AnkAI/Test/Dataset/{int(birads)}/{new_name}")

