from DataLoaders.XLS_utils import XLS
import os
from PIL import Image
import shutil
from torchvision import transforms as T
import sys

train = XLS().df

path = "/home/alican/Documents/Datasets/VinDr-mammo"
f = "Kirli"

folders = os.listdir(os.path.join(path,f))
for folder in folders:
    files = os.listdir(os.path.join(path,f,folder))
    File_Name = folder+"/"+files[0].split(".")[0]
    bi_rads = train[train["File Name"]==File_Name]["Bi-Rads"].values[0]
    if not os.path.exists(os.path.join(path,f"{bi_rads}")):
        os.mkdir(os.path.join(path,f"{bi_rads}"))
    shutil.move(os.path.join(path,f,folder),os.path.join(path,f,f"{bi_rads}",folder))
    

# transform = T.ToTensor()

# folders = os.listdir(os.path.join(path,f))
# for folder in folders:
#     files = os.listdir(os.path.join(path,f,folder))
#     image = Image.open(os.path.join(path,f,folder,files[0]))
#     while True:
#         try:
#             value = transform(image).mean()
#             if value < 0.105:
#                 res = "Y"
#             elif value >0.155:
#                 res = "N"
#             else:
#                 image.show()
#                 print(value)
#                 res = str(input())
            

#             if res.capitalize() == "Y":
#                 shutil.move(os.path.join(path,f,folder),os.path.join(path,"Temiz",folder))
#             elif res.capitalize() == "N":
#                 shutil.move(os.path.join(path,f,folder),os.path.join(path,"Kirli",folder))
#             else:
#                 raise Exception("Incorrect input")
#         except:
#             continue
#         break