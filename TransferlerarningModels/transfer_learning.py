import torch

import torch.nn as nn
import torchvision

import  torchvision.models as models
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


import warnings

warnings.filterwarnings("ignore")




class efficientNet_v2L(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(efficientNet_v2L,self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.model = models.efficientnet_v2_l(weights = models.EfficientNet_V2_L_Weights,pretrained = False)
        self.First= nn.Sequential(nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),

                                        nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.SiLU(inplace=True)
                                )
        
        
        self.classifier = nn.Sequential(nn.Dropout(p=0.4,inplace=True),

                                            nn.Linear(in_features=1280,out_features=num_classes))


        self.avg = nn.AdaptiveAvgPool2d(1)
        

        self.modelBody = self.model.features[1:]


        print(self.modelBody[0])

        
    def forward(self,inputs):

        out = self.First(inputs)
        out = self.modelBody(out)
        out = self.avg(out)
        out = out.view(out.size(0),-1)
        out = self.classifier(out)

        
        return out 






class ResNet101(nn.Module):
    def __init__(self,in_channels,num_classes) :

        super(ResNet101,self).__init__()

        model = models.resnet101(weights = models.ResNet101_Weights,pretrained = False)

        self.first_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=False))
        self.body = nn.Sequential(model.layer1,
                            model.layer2,
                            model.layer3,
                            model.layer4)

        self.avg = model.avgpool

        self.fc = nn.Linear(in_features=2048,out_features=num_classes,bias=True)



    def forward(self,inputs):

        out = self.first_layer(inputs)
        out = self.body(out)
        out = self.avg(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)


        return out











