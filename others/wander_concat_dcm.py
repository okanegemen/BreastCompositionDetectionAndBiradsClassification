import os
import pydicom
import numpy as np
import pandas as pd
import torchvision.transforms as T
import torch
import time
from XLS_utils import XLS 
import config
from DataLoaders.dataset import Dataset

dcm_names = ["LCC","LMLO","RCC","RMLO"]

def hastano_from_txt(txt_path = os.path.join(config.MAIN_DIR,"yoloV5","others","kirli_resimler.txt")):
    with open(txt_path) as text_file:
        lines = text_file.readlines()
    dcm_folders = [line.split("\t")[0].strip() for line in lines]
    return dcm_folders

if __name__ == "__main__":
    train, test= XLS().return_datasets()

    train = Dataset(train,True)

    mean = torch.zeros(4).to(config.DEVICE)
    mean_d = torch.zeros(4).to(config.DEVICE)
    mean_l = torch.zeros(4).to(config.DEVICE)

    std = torch.zeros(4).to(config.DEVICE)
    std_d = torch.zeros(4).to(config.DEVICE)
    std_l = torch.zeros(4).to(config.DEVICE)

    count_d = 0
    count_l = 0

    #["LCC","LMLO","RCC","RMLO"]
    for i in range(len(train)):
        m = train[i][0].mean(dim=(1,2)).to(config.DEVICE)
        s = train[i][0].std(dim=(1,2)).to(config.DEVICE)
        mean += m
        std += s
        if ((m>0.7).sum()<=2):
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


# tensor([527.1708, 498.0023, 528.9872, 499.4696], device='cuda:0') tensor([892.8487, 884.1351, 893.8290, 884.8680], device='cuda:0')
# tensor([527.1708, 498.0023, 528.9872, 499.4696], device='cuda:0') tensor([892.8487, 884.1351, 893.8290, 884.8680], device='cuda:0')
# tensor([0., 0., 0., 0.], device='cuda:0') tensor([0., 0., 0., 0.], device='cuda:0')
# tensor([0.1704, 0.1610, 0.1710, 0.1615], device='cuda:0') tensor([0.2887, 0.2859, 0.2890, 0.2861], device='cuda:0')
# tensor([0.1704, 0.1610, 0.1710, 0.1615], device='cuda:0') tensor([0.2887, 0.2859, 0.2890, 0.2861], device='cuda:0')
# tensor([nan, nan, nan, nan], device='cuda:0') tensor([nan, nan, nan, nan], device='cuda:0')

# tensor([609.6924, 510.1406, 606.7292, 503.0339], device='cuda:0') tensor([934.7719, 871.1933, 934.4750, 863.7477], device='cuda:0')
# tensor([609.6924, 510.1406, 606.7292, 503.0339], device='cuda:0') tensor([934.7719, 871.1933, 934.4750, 863.7477], device='cuda:0')
# tensor([0., 0., 0., 0.], device='cuda:0') tensor([0., 0., 0., 0.], device='cuda:0')
# tensor([0.1846, 0.1545, 0.1837, 0.1523], device='cuda:0') tensor([0.2831, 0.2638, 0.2830, 0.2616], device='cuda:0')
# tensor([0.1846, 0.1545, 0.1837, 0.1523], device='cuda:0') tensor([0.2831, 0.2638, 0.2830, 0.2616], device='cuda:0')
# tensor([nan, nan, nan, nan], device='cuda:0') tensor([nan, nan, nan, nan], device='cuda:0')

    print(mean/len(train),std/len(train))
    print(mean_d/count_d,std_d/count_d)
    print(mean_l/count_l,std_l/count_l)

    # # [0.8579, 0.8336, 0.8585, 0.8324], [0.1386, 0.1739, 0.1390, 0.1765]
    # # [0.9254, 0.8955, 0.9262, 0.8941], [0.1313, 0.1677, 0.1318, 0.1705]
    # # [0.1380, 0.1739, 0.1371, 0.1742], [0.2167, 0.2403, 0.2163, 0.2398]




# class Dataset():
#     def __init__(self,dataset: pd.DataFrame,train_transform=True):
#         self.train_transform = train_transform
#         if self.train_transform:
#             print("Train data is preparing...")
#         else:
#             print("Test data is preparing...")

#         self.dcm_names = ["LCC","LMLO","RCC","RMLO"]

#         self.dataset = dataset
#         self.dataset_name = config.TEKNOFEST
#         self.transform = T.Compose([
#                             T.ToPILImage(),
#                             T.Resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)),
#                             T.ToTensor(),
#                         ])

#         self.dicom_paths = self.dicom_paths_func()
#         self.dataset = self.eliminate_unused_dicoms(self.dicom_paths,self.dataset) # eliminates rows in dataframe of dataset which are not in the image directory, deleted or moved
#         categories = self.dataset["BIRADS KATEGORİSİ"].to_list()
#         self.ids = [x for x in list(categories)]

#     def __getitem__(self, index: int):
#         data = self.dataset.iloc[index,:]
#         dicti = data.to_dict()

#         images = self.loadImg(dicti["HASTANO"])

#         birads = torch.tensor(dicti["BIRADS KATEGORİSİ"],dtype=torch.int64)

#         for name,image in images.items():
#             image = torch.from_numpy(image).float().unsqueeze(0)

#             images[name] = self.transform(image)

#             # images[name][images[name]>0.9] = 0.
#             # images[name][images[name]<0.1] = 0.

#             # print(images[name].max())
#             # T.ToPILImage()(images[name]).show()
#             # time.sleep(1)
#         images = {key:image for key,image in images.items()}

#         image = torch.stack([image.squeeze() for image in images.values()])

#         return  image

#     def loadImg(self,hastano):
#         images = {}
#         for dcm in self.dcm_names:
#             dicom,image = self.dicom_open(hastano,dcm)
#             image = fiximage.fit_image(dicom,image)
#             images[dcm] = image

#         return images

#     def dicom_paths_func(self):
#         folder_names =  [folder for folder in os.listdir(config.TEKNOFEST) if len(folder.split("."))<2]
#         return folder_names

#     def eliminate_unused_dicoms(self,dicom_folders:dict,dataset:pd.DataFrame):
#         # dataset = dataset[~dataset["HASTANO"].isin(hastano_from_txt())]
#         dataset = dataset[dataset["HASTANO"].isin(dicom_folders)]
#         return dataset

#     def dicom_open(self,hastano,dcm):
#         path = os.path.join(config.TEKNOFEST,hastano,dcm+".dcm")
#         dicom_img = pydicom.dcmread(path)
#         numpy_pixels = dicom_img.pixel_array
#         return dicom_img,numpy_pixels

#     def __len__(self) -> int:
#         return len(self.dataset)

#     def __str__(self):
#         return str(self.dataset)
