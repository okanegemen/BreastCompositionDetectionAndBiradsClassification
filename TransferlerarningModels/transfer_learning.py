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


model = models.resnet50()
liste = [module for module in model.modules()]

class Resnet50(nn.Module):
    def __init__(self,in_channels):
        super(Resnet50,self).__init__()

        model = models.resnet50()

        module_list = [module for module in model.children()]



        first_layer = nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=7,stride = 2,padding = 3,bias=False)

        body = module_list[1:-1]
        self.featureExtracture = nn.Sequential(first_layer,*body)
        self.birads = nn.Linear(in_features=2048,out_features=3)
        self.composition = nn.Linear(in_features=2048,out_features=4)
        self.kadran = nn.Linear(in_features=2048,out_features=10)

        


    def forward(self,input):

        out = self.featureExtracture(input)

        out = out.view(out.size(0),-1)
        birads = self.birads(out)
        composition = self.composition(out)
        kadran = self.kadran(out)



        return {"birads":birads , "acr":composition ,"kadran":kadran} 


resnet = Resnet50(1)   

out = resnet(tensor)
print(torch.argmax(out["birads"]))

class FeaturesImg(nn.Module):
    def __init__(self ,inplanes):
        super(FeaturesImg,self).__init__()
        self.inplanes = inplanes


        self.conv1 = nn.Conv2d(in_channels=inplanes,out_channels=16,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=5,padding = 2,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)


    def forward(self,inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)


        return out



class ConcatModel(nn.Module):

    def __init__(self,inplanes,model):
        super(ConcatModel,self).__init__()

        self.inplanes = inplanes
        

        

        self.img1 = FeaturesImg(inplanes)
        self.img2 = FeaturesImg(inplanes)
        self.img3 = FeaturesImg(inplanes)
        self.img4 = FeaturesImg(inplanes)

        firstBlock,firstBody,body  = self._modifyFirstLayertakeBody(model=model)
        if  firstBody !=0 :
            self.featureExtrator = nn.Sequential(*firstBlock,*firstBody,*body)
        else:self.featureExtrator = nn.Sequential(*firstBlock,*body)

        buffer,linear = self._changeLastlayer(model)
        if linear !=0 :
            self.birads = nn.Sequential(linear,nn.Linear(buffer[0].in_features,3))
            self.composition = nn.Sequential(linear,nn.Linear(buffer[0].in_features,4))
            self.kadran = nn.Sequential(linear,nn.Linear(buffer[0].in_features,10))
        else:
            self.birads = nn.Linear(buffer[0].in_features,3)
            self.composition = nn.Linear(buffer[0].in_features,4)
            self.kadran = nn.Linear(buffer[0].in_features,10)

            





        
        
        
    def _modifyFirstLayertakeBody(self,model):
        module_list = [modules for modules in model.children()]

        temp = module_list[0]
        body = module_list[1:-1]
        last = module_list[-1]
        buffer = []

        firstBlock = []

        count = 0

        while True:
            buffer.append(temp)
            try:
                childOrParent = next(iter(buffer[-1].children()))[0]
                temp = childOrParent

                
                count +=1
            except:
                
                if count==1:
                    firstBlock.append(buffer[0][0])
                    break
                    
                firstBlock.append(temp)
                break 


        oneOrTwoDim = 0

        length = 0

        try:
            length=len(firstBlock[0])

            if length!=0:
                oneOrTwoDim = 2
        except TypeError:
            
            oneOrTwoDim = 1





        lookFor = module_list[0]
        try:

            if len(lookFor)!=0:
                
                firstBody=lookFor[1:]
        except:
            firstBody = 0



        try:
            if oneOrTwoDim == 1:




                firstBlock[0] = nn.Conv2d(256,
                                            firstBlock[0].out_channels,
                                            firstBlock[0].kernel_size,
                                            firstBlock[0].stride,
                                            firstBlock[0].padding,
                                            firstBlock[0].dilation,
                                            firstBlock[0].groups,
                                            bias=firstBlock[0].bias)

            else: 
                firstBlock[0][0] = nn.Conv2d(256,
                                                firstBlock[0][0].out_channels,
                                                firstBlock[0][0].kernel_size,
                                                firstBlock[0][0].stride,
                                                firstBlock[0][0].padding,
                                                firstBlock[0][0].dilation,
                                                firstBlock[0][0].groups,
                                                bias=firstBlock[0][0].bias)
        except:
            firstBlock[0].conv = nn.Conv2d(256,
                                            firstBlock[0].conv.out_channels,
                                            firstBlock[0].conv.kernel_size,
                                            firstBlock[0].conv.stride,
                                            firstBlock[0].conv.padding,
                                            firstBlock[0].conv.dilation,
                                            firstBlock[0].conv.groups,
                                            firstBlock[0].conv.bias
                                            )


                    

        return firstBlock,firstBody,body


        
    def _changeLastlayer(self,model):
        buffer2 = []
        module_list2 = [modules for modules in model.children()]
        temp2 = module_list2[-1]





        while True:
            
            buffer2.append(temp2)
            try:
                childOrParent = next(iter(buffer2[-1].children()))[0]
                temp2 = childOrParent
                

            except:
                linear = temp2
            break
        new_buffer = []

        
        try:
            
            if len(linear)!=1:
                new_buffer.append(linear[1])
                linear = linear[0]
                
        except:
            new_buffer.append(linear)
            linear = 0


        return new_buffer,linear
            


        
        
          
    def forward(self,input1,input2,input3,input4):
        out1 = self.img1(input1)

        out2 = self.img2(input2)

        out3 = self.img3(input3)

        out4 = self.img4(input4)

        concat1 = torch.cat((out1,out2),1)
        
        concat2 = torch.cat((out3,out4),1)
        final = torch.cat((concat1,concat2),1)
        final = torch.reshape(final,[1,256,600,600])
        print(final.size())
        features = self.featureExtrator(final)

        features = features.view(features.size(0),-1)


        birads = self.birads(features)
        composition = self.composition(features)
        kadran = self.kadran(features)


        return {"birads":birads , "acr":composition ,"kadran":kadran}

















        

