import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np 
from Pytorch_model.unet import unet_model,unet_parts
import pydicom
import pydicom.data as data
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

tensor_image = torch.rand(1,1,1000,1000)
tensor_image.to(mps_device)
a = unet_model.UNet(1,2)
x = a(tensor_image)
