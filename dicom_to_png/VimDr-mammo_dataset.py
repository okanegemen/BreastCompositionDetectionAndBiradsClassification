import pydicom
import pydicom.data
import torch
import numpy as np
import os
from torchvision.utils import save_image
import gc
from PIL import Image
import shutil

gc.collect()

class dicoms_to_png():
    def __init__(self,dicom_folders_path,png_folder = "/home/alican/Documents/Dicom_images"):
        self.dicom_folder_name = dicom_folders_path
        self.dicoms_folders = os.listdir(self.dicom_folder_name)
        self.dir = png_folder

    def dicom_open(self):
        # enter DICOM image name for pattern
        # result is a list of 1 element
        main_path = self.dicom_folder_name
        folders = self.dicoms_folders
        counter = 0
        for folder in folders:
            if len(folder.split(".")) == 1:
                dicoms = os.listdir(os.path.join(main_path,folder))

                for dicom in dicoms:
                    if dicom.split(".")[1] == "dicom":
                        counter +=1
                        dicom_path = os.path.join(folder,dicom)
                        name = pydicom.data.data_manager.get_files(main_path, dicom_path)[0]
                        
                        ds = pydicom.dcmread(name)
                        
                        ds = np.round((ds.pixel_array/4095)*255)
                        image = Image.fromarray(ds.astype(np.uint8))
                        dir_exist = os.path.join(self.dir,folder)

                        if not(os.path.isdir(dir_exist)):
                            os.mkdir(dir_exist)
                        image.save(os.path.join(self.dir,folder,dicom.split(".")[0]+".png"))
                        print(counter)
                    else:
                        original = os.path.join(main_path,folder,dicom)
                        target = os.path.join(self.dir,folder,dicom)
                        if not(os.path.isdir(os.path.join(self.dir,folder))):
                            os.mkdir(os.path.join(self.dir,folder))
                        shutil.copyfile(original, target)
            else:
                original = os.path.join(main_path,folder)
                target = os.path.join(self.dir,folder)
                shutil.copyfile(original, target)
                

if __name__=="__main__":
    root = "/home/alican/Documents/physionet.org/files/vindr-mammo/1.0.0/images"

    imgs = dicoms_to_png(root).dicom_open()

    print("finish")
