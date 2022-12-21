from PIL import Image
from torchvision import transforms as T
import os
import shutil

path = "/home/alican/Documents/Datasets/VinDr-mammo"
f = "Dicom_images"

transform = T.ToTensor()

folders = os.listdir(os.path.join(path,f))
for folder in folders:
    files = os.listdir(os.path.join(path,f,folder))
    image = Image.open(os.path.join(path,f,folder,files[0]))
    while True:
        try:
            value = transform(image).mean()
            if value < 0.105:
                res = "Y"
            elif value >0.155:
                res = "N"
            else:
                image.show()
                print(value)
                res = str(input())
            

            if res.capitalize() == "Y":
                shutil.move(os.path.join(path,f,folder),os.path.join(path,"Temiz",folder))
            elif res.capitalize() == "N":
                shutil.move(os.path.join(path,f,folder),os.path.join(path,"Kirli",folder))
            else:
                raise Exception("Incorrect input")
        except:
            continue
        break