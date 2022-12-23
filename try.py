
from PIL import Image
import torchvision
from torchvision.models import efficientnet_b5;
from torchvision import transforms as T
import os
import shutil

import torch
import torch.nn as nn
from PIL import Image
import cv2 as cv
import numpy as np
import pydicom as dicom 

path = "/Users/okanegemen/Desktop/yoloV5/INbreast Release 1.0/AllDICOMs/20586986_6c613a14b80a8591_MG_L_ML_ANON.dcm"

dicom_img = dicom.dcmread(path)

numpy_pixels = dicom_img.pixel_array
img = np.resize(numpy_pixels,(600,600))
img = np.array(img,dtype="float32")


tensor = torch.from_numpy(img)
tensor = tensor.float()
tensor = torch.reshape(tensor,[1, 1, 600, 600])



class StochasticDepth(nn.Module):
    """StochasticDepth
    paper: https://link.springer.com/chapter/10.1007/978-3-319-46493-0_39
    :arg
        - prob: Probability of dying
        - mode: "row" or "all". "row" means that each row survives with different probability
    """
    def __init__(self, prob, mode):
        super(StochasticDepth, self).__init__()
        self.prob = prob
        self.survival = 1.0 - prob
        self.mode = mode

    def forward(self, x):
        # if self.prob == 0.0 or not self.training:
        #     return x
        # else:
            shape = [x.size(0)] + [1] * (x.ndim - 1) if self.mode == 'row' else [1]
            x = x * torch.empty(shape).bernoulli_(self.survival).div_(self.survival).to(x.device)
            print(x)
            return x

out = StochasticDepth(0.19487179487179487, 'row')

tensor = torch.rand(10)
out = out(tensor)

print(out.size())

model = efficientnet_b5()
max = 0.19636363636363638
min= 0.18545454545454548
tensor = torch.rand(4)*(max-min)+min
tensor = tensor.sort()
print(tensor)


binomial = torch.distributions.binomial.Binomial(probs=0.01)
a=0.19272727272727275
b = 0.1890909090909091
c = a/b


print(c/a)
print(c)
print(a-b)

# path = "/home/alican/Documents/Datasets/VinDr-mammo"
# f = "Dicom_images"

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

import time
import random
from qqdm import qqdm, format_str

tw = qqdm(range(10), desc=format_str('bold', 'Description'))

a = {2:1,3:2,4:4}
print({3:1,
        **a})

import time
import random
from qqdm import qqdm, format_str

tw = qqdm(range(10), desc=format_str('bold', 'Description'))

a = {2:1,3:2,4:4}
print({3:1,
        **a})

