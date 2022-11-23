from Pytorch_model.unet.unet_model import UNet
import torch
from torch.autograd import Variable
import pydicom as dicom 
import matplotlib.pyplot as plt
from ConnectedSegnet.connectedSegnet_model import *
from ConnectedSegnet.connectedSegnet_elements import *
from PIL import Image
import cv2 as cv
import numpy as np



path = "/Users/okanegemen/yoloV5/INbreast Release 1.0/AllDICOMs/20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm"

dicom_img = dicom.dcmread(path)

numpy_pixels = dicom_img.pixel_array
img = np.resize(numpy_pixels,(600,600))
img = np.array(img,dtype="float32")



tensor = torch.from_numpy(img)
tensor = tensor.float()
tensor = torch.reshape(tensor,[1,1,600,600])
#tensor = torch.view_as_real(tensor)
model = ConSegnetsModel(1)

output = model(tensor)

numpy_img = output.cpu().detach().numpy()

numpy_img = np.resize(numpy_img,(596,596))


image = Image.fromarray(numpy_img,'L')
image.show()

