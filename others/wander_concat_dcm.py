import os
import pydicom
import numpy as np
import pandas as pd
import torchvision.transforms as T
import torch
import time
from XLS_utils import XLS 
import config

dcm_names = ["LCC","LMLO","RCC","RMLO"]

def hastano_from_txt(txt_path = os.path.join(config.MAIN_DIR,"yoloV5","others","kirli_resimler.txt")):
    with open(txt_path) as text_file:
        lines = text_file.readlines()
    dcm_folders = [line.split("\t")[0].strip() for line in lines]
    return dcm_folders

class Dataset():
    def __init__(self,dataset: pd.DataFrame,train_transform=True):
        self.train_transform = train_transform
        if self.train_transform:
            print("Train data is preparing...")
        else:
            print("Test data is preparing...")

        self.dcm_names = ["LCC","LMLO","RCC","RMLO"]

        self.dataset = dataset
        self.dataset_name = config.TEKNOFEST
        self.transform = T.Compose([
                            T.ToPILImage(),
                            T.Resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)),
                            T.ToTensor(),
                        ])

        self.dicom_paths = self.dicom_paths_func()
        self.dataset = self.eliminate_unused_dicoms(self.dicom_paths,self.dataset) # eliminates rows in dataframe of dataset which are not in the image directory, deleted or moved
        categories = self.dataset["BIRADS KATEGORİSİ"].to_list()
        self.ids = [x for x in list(categories)]

    def __getitem__(self, index: int):
        data = self.dataset.iloc[index,:]
        dicti = data.to_dict()

        images = self.loadImg(dicti["HASTANO"])

        birads = torch.tensor(dicti["BIRADS KATEGORİSİ"],dtype=torch.int64)

        for name,image in images.items():
            image = torch.from_numpy(image).float().unsqueeze(0)

            images[name] = self.transform(image)

            # images[name][images[name]>0.9] = 0.
            # images[name][images[name]<0.1] = 0.

            # print(images[name].max())
            # T.ToPILImage()(images[name]).show()
            # time.sleep(1)
        images = {key:image for key,image in images.items()}

        image = torch.stack([image.squeeze() for image in images.values()])

        return  image
    
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
        # dataset = dataset[~dataset["HASTANO"].isin(hastano_from_txt())]
        dataset = dataset[dataset["HASTANO"].isin(dicom_folders)]
        return dataset

    def dicom_open(self,hastano,dcm):
        path = os.path.join(config.TEKNOFEST,hastano,dcm+".dcm")
        dicom_img = pydicom.dcmread(path)
        numpy_pixels = dicom_img.pixel_array
        img = np.array(numpy_pixels,dtype="float32")
        return img/np.max(img)

    def __len__(self) -> int:
        return len(self.dataset)

    def __str__(self):
        return str(self.dataset)


if __name__ == "__main__":
    train, test= XLS().return_datasets()

    train = Dataset(train,True)

    mean = torch.zeros(4)
    mean_d = torch.zeros(4)
    mean_l = torch.zeros(4)

    std = torch.zeros(4)
    std_d = torch.zeros(4)
    std_l = torch.zeros(4)

    count_d = 0
    count_l = 0

    #["LCC","LMLO","RCC","RMLO"]
    for i in range(len(train)):
        m = train[i].mean(dim=(1,2))
        s = train[i].std(dim=(1,2))
        mean += m
        std += s
        if ((m<0.5).sum()<=2):
            mean_d += m
            std_d += s
            count_d += 1
        else:
            mean_l += m
            std_l += s
            count_l += 1
    

    print(mean,std)
    print(mean_d,std_d)
    print(mean_l,std_l)

    # tensor([2832.8076, 2752.6228, 2834.8845, 2748.5737]) tensor([457.7245, 574.3817, 459.0716, 582.6831])
    # tensor([2793.7437, 2703.4209, 2796.0762, 2699.2837]) tensor([396.3914, 506.3709, 397.8712, 514.8105])
    # tensor([39.0623, 49.2026, 38.8069, 49.2884]) tensor([61.3330, 68.0109, 61.2002, 67.8724])

    print(mean/len(train),std/len(train))
    print(mean_d/count_d,std_d/count_d)
    print(mean_l/count_l,std_l/count_l)

    # [0.8579, 0.8336, 0.8585, 0.8324], [0.1386, 0.1739, 0.1390, 0.1765]
    # [0.9254, 0.8955, 0.9262, 0.8941], [0.1313, 0.1677, 0.1318, 0.1705]
    # [0.1380, 0.1739, 0.1371, 0.1742], [0.2167, 0.2403, 0.2163, 0.2398]
