import torch

import torch.nn as nn
import torchvision

import  torchvision.models as models
from PIL import Image
import cv2 as cv
import numpy as np
import pydicom as dicom 



import warnings

warnings.filterwarnings("ignore")



class efficientNet_v2L(nn.Module):
    def __init__(self,in_channels,num_classes,pretrained=False):
        super(efficientNet_v2L,self).__init__()

        model = models.efficientnet_v2_l(pretrained = pretrained)

        modules = [module for module in model.children()]

        modules_first = modules[0]
        


        self.first_block_will_using = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                                    nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                                                    nn.SiLU(inplace=True)
        )

        body_first = modules_first[1:]

        body_second = modules[1:-1]
        self.body = nn.Sequential(*body_first,*body_second)

      
    def forward(self,inputs):
        out = self.first_block_will_using(inputs)

        out = self.body(out)
        
        out = out.view(out.size(0),-1)
        out = self.classifier(out)

        return {"birads":out} 





    


class efficientNetv2s(nn.Module):
    def __init__(self,in_channels=4, weight : bool = False):
        super(efficientNetv2s,self).__init__()

        model = models.efficientnet_v2_s(weights = models.EfficientNet_V2_S_Weights.DEFAULT)


        modules = [module for module in model.children()]
        modules_first = modules[0]
        self.first_block_will_using = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels= 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        
                                                    nn.BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                                                    nn.SiLU(inplace=True)
        )
        
        body_first = modules_first[1:]

        body_second = modules[1:-1]
        self.body = nn.Sequential(*body_first,*body_second)

    def forward(self,inputs):
        out = self.first_block_will_using(inputs)

        out = self.body(out)
        return out
        
         









# model = models.efficientnet_v2_s()

# modules = [module for module in model.children()]
# first_block = modules[0][0]
# print(first_block)
# print(modules[0])
class Resnet18(nn.Module):
    def __init__(self,in_channels=4,num_classes=3) :
        super(Resnet18,self).__init__()

        model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)

        modules = [module for module in model.children()]

        
        first_block = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        body = modules[1:-1]
        self.features = nn.Sequential(first_block,*body)
        self.classifier = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self,input):

        
        out = self.features(input)
        out = out.view(out.size(0),-1)
        out = self.classifier(out)

        return out




class Resnet34(nn.Module):

    def __init__(self,in_channels=4,num_classes=3):

        super(Resnet34,self).__init__()

        model = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)

        modules = [module for module in model.children()]


        self.first = nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False)
        body = modules[1:-1]

        self.body = nn.Sequential(*body)


        self.last = nn.Linear(in_features=512,out_features=num_classes)

    def forward(self,input1):


        out= self.first(input1)
        out = self.body(out)

        out = out.view(out.size(0),-1)
        out = self.last(out)

        return out






class ResNet101(nn.Module):
    def __init__(self,in_channels=4,num_classes=3) :

        super(ResNet101,self).__init__()

        model = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)

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



class Resnet50(nn.Module):
    def __init__(self,pretrained=True,in_channels=4):
        super(Resnet50,self).__init__()

        if pretrained:
            model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        else:
            model = models.resnet50()
    
        module_list = [module for module in model.children()]



        first_layer = nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=7,stride = 2,padding = 3,bias=False)

        body = module_list[1:-1]
        self.featureExtracture = nn.Sequential(first_layer,*body)
        
        self.dropout = nn.Dropout(p=0.4,inplace=True)

        self.b1 = nn.Linear(in_features=2048,out_features=512)
        self.b2 = nn.Linear(in_features=512,out_features=128)
        self.b3 = nn.Linear(in_features=128,out_features=3)

        self.c1 = nn.Linear(in_features=2048,out_features=512)
        self.c2 = nn.Linear(in_features=512,out_features=128)
        self.c3 = nn.Linear(in_features=128,out_features=4)
        
        self.k1 = nn.Linear(in_features=2048,out_features=512)
        self.k2 = nn.Linear(in_features=512,out_features=128)
        self.k3 = nn.Linear(in_features=128,out_features=10)

    def forward(self,input):

        out = self.featureExtracture(input)

        out = out.view(out.size(0),-1)
        b_out = self.b1(out)
        b_out = self.dropout(b_out)
        b_out = self.b2(b_out)
        b_out = self.dropout(b_out)
        b_out = self.b3(b_out)

        # composition = self.composition(out)
        # kadran = self.kadran(out)



        return b_out




class FeaturesImg(nn.Module):
    def __init__(self ,inplanes):
        super(FeaturesImg,self).__init__()
        self.inplanes = inplanes

        self.conv1 = nn.Conv2d(in_channels=inplanes,out_channels=16,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding = 1,bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding = 2,bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,padding = 2,bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,padding = 2,bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,padding = 2,bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU(inplace=True)

    def forward(self,inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)
        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu6(out)
        return out

class SEBlock(nn.Module):
  """Squeeze-and-excitation block"""
  def __init__(self, n_in, r=24):
    super().__init__()

    self.squeeze = nn.AdaptiveAvgPool2d(1)
    self.excitation = nn.Sequential(nn.Conv2d(n_in, n_in//r, kernel_size=1),
                                    nn.SiLU(),
                                    nn.Conv2d(n_in//r, n_in, kernel_size=1),
                                    nn.Sigmoid())
  
  def forward(self, x):
    y = self.squeeze(x)
    y = self.excitation(y)
    return x * y

class ConcatModel(nn.Module):

    def __init__(self,model,inplanes=3):
        super(ConcatModel,self).__init__()
        self.inplanes = inplanes
        self.img1 = FeaturesImg(inplanes)
        self.img2 = FeaturesImg(inplanes)
        self.img3 = FeaturesImg(inplanes)
        self.img4 = FeaturesImg(inplanes)
        self.ch_drop = nn.Dropout2d()

        firstBlock,firstBody,body  = self._modifyFirstLayertakeBody(model=model,in_channels = 256)

        if  firstBody !=0 :
            self.featureExtrator = nn.Sequential(*firstBlock,*firstBody,*body)
        else:self.featureExtrator = nn.Sequential(*firstBlock,*body)

        buffer,linear = self._changeLastlayer(model)
        if linear !=0 :
            self.birads = torch.nn.Sequential(
                                torch.nn.Linear(buffer[0].in_features,256),
                                torch.nn.Dropout(),
                                torch.nn.Linear(256,64),
                                torch.nn.Dropout(),
                                torch.nn.Linear(64,3)
                            )
            # self.composition = nn.Sequential(linear,nn.Linear(buffer[0].in_features,4))
            # self.kadran = nn.Sequential(linear,nn.Linear(buffer[0].in_features,10))
        else:
            self.birads = torch.nn.Sequential(
                                torch.nn.Linear(buffer[0].in_features,256),
                                torch.nn.Dropout(),
                                torch.nn.Linear(256,64),
                                torch.nn.Dropout(),
                                torch.nn.Linear(64,3)
                            )
            # self.composition = nn.Linear(buffer[0].in_features,4)
            # self.kadran = nn.Linear(buffer[0].in_features,10)

    def forward(self,inputs):

        input1 = inputs[:,0,:,:,:].squeeze(1)
        input2 = inputs[:,1,:,:,:].squeeze(1)
        input3 = inputs[:,2,:,:,:].squeeze(1)
        input4 = inputs[:,3,:,:,:].squeeze(1)
        
        input1 = self.img1(input1)
        input2 = self.img2(input2)
        input3 = self.img3(input3)
        input4 = self.img4(input4)

        concat1 = torch.cat((input1,input2),1)
        concat2 = torch.cat((input3,input4),1)
        final = torch.cat((concat1,concat2),1)

        final = self.ch_drop(final)

        features = self.featureExtrator(final)

        features = features.view(features.size(0),-1)

        birads = self.birads(features)
        # composition = self.composition(features)
        # kadran = self.kadran(features)


        return birads#{"birads":birads , "acr":composition ,"kadran":kadran}
        
    def _modifyFirstLayertakeBody(self,model,in_channels=256):
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
                    print(buffer)
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


                if firstBlock[0].bias is not None:


                    firstBlock[0] = nn.Conv2d(in_channels = in_channels,
                                                out_channels=firstBlock[0].out_channels,
                                                kernel_size = firstBlock[0].kernel_size,
                                                stride = firstBlock[0].stride,
                                                padding =firstBlock[0].padding,
                                                dilation=firstBlock[0].dilation,
                                                groups =firstBlock[0].groups,
                                                bias=True)
                else:
                    firstBlock[0] = nn.Conv2d(in_channels = in_channels,
                                                out_channels=firstBlock[0].out_channels,
                                                kernel_size = firstBlock[0].kernel_size,
                                                stride = firstBlock[0].stride,
                                                padding =firstBlock[0].padding,
                                                dilation=firstBlock[0].dilation,
                                                groups =firstBlock[0].groups,
                                                bias=False)

            elif oneOrTwoDim == 2:


                if firstBlock[0][0].bias is not None:

                    firstBlock[0][0] = nn.Conv2d(in_channels = in_channels,
                                                    out_channels=firstBlock[0][0].out_channels,
                                                    kernel_size=firstBlock[0][0].kernel_size,
                                                    stride = firstBlock[0][0].stride,
                                                    padding = firstBlock[0][0].padding,
                                                    dilation = firstBlock[0][0].dilation,
                                                    groups = firstBlock[0][0].groups,
                                                    bias=True)
                else:
                    firstBlock[0][0] = nn.Conv2d(in_channels = in_channels,
                                                    out_channels=firstBlock[0][0].out_channels,
                                                    kernel_size=firstBlock[0][0].kernel_size,
                                                    stride = firstBlock[0][0].stride,
                                                    padding = firstBlock[0][0].padding,
                                                    dilation = firstBlock[0][0].dilation,
                                                    groups = firstBlock[0][0].groups,
                                                    bias=False)

        except:

            if firstBlock[0].conv.bias is not None:


                firstBlock[0].conv = nn.Conv2d(in_channels,
                                                firstBlock[0].conv.out_channels,
                                                firstBlock[0].conv.kernel_size,
                                                firstBlock[0].conv.stride,
                                                firstBlock[0].conv.padding,
                                                firstBlock[0].conv.dilation,
                                                firstBlock[0].conv.groups,
                                                bias = True
                                                )
            else:
                firstBlock[0].conv = nn.Conv2d(in_channels,
                                            firstBlock[0].conv.out_channels,
                                            firstBlock[0].conv.kernel_size,
                                            firstBlock[0].conv.stride,
                                            firstBlock[0].conv.padding,
                                            firstBlock[0].conv.dilation,
                                            firstBlock[0].conv.groups,
                                            bias=False
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




class AlexnetCat(nn.Module):

    def __init__(self,in_channels=1,num_classes=[3,4,10]):
        super(AlexnetCat,self).__init__()

        self.img1 = FeaturesImg(in_channels)
        self.img2 = FeaturesImg(in_channels)
        self.img3 = FeaturesImg(in_channels)
        self.img4 = FeaturesImg(in_channels)

        self.se1 = SEBlock(n_in=256)


        self.model = efficientNet_v2L(in_channels=256)

        self.dropout = nn.Dropout(p=0.4,inplace=True)
        self.fc1 = nn.Linear(1280,512)

        self.fc2 = nn.Linear(512,256)

        self.fc3 = nn.Linear(256,128)

        self.fc4 = nn.Linear(128,num_classes[0])

    def forward(self,inputs:dict):

        input1 = inputs[:,0,:,:].unsqueeze(1)
        input2 = inputs[:,1,:,:].unsqueeze(1)
        input3 = inputs[:,2,:,:].unsqueeze(1)
        input4 = inputs[:,3,:,:].unsqueeze(1)

        out1 = self.img1(input1)
        out2 = self.img2(input2)
        out3 = self.img3(input3)
        out4 = self.img4(input4)

        cat1 = torch.cat((out1,out2),dim=1)

        cat2 = torch.cat((out3,out4),dim=1)

        cat_last = torch.cat((cat1,cat2),dim=1)
        
        out = self.se1(cat_last)

        out = self.model(out)

        out = out.view(out.size(0),-1)

        out = self.dropout(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.fc4(out)
        return out


class AlexnetCat2(nn.Module):

    def __init__(self,in_channels=1,num_classes=[3,4,10]):
        super(AlexnetCat2,self).__init__()

        self.img1 = FeaturesImg(in_channels)
        self.img2 = FeaturesImg(in_channels)
        self.img3 = FeaturesImg(in_channels)
        self.img4 = FeaturesImg(in_channels)

        self.se1 = SEBlock(n_in=256)


        self.model =   efficientNetv2s(in_channels=256)

        self.dropout = nn.Dropout(p=0.4,inplace=True)
        self.fc1 = nn.Linear(1280,512)

        self.fc2 = nn.Linear(512,256)

        self.fc3 = nn.Linear(256,128)

        self.fc4 = nn.Linear(128,num_classes[0])

    def forward(self,inputs:dict):

        input1 = inputs[:,0,:,:].unsqueeze(1)
        input2 = inputs[:,1,:,:].unsqueeze(1)
        input3 = inputs[:,2,:,:].unsqueeze(1)
        input4 = inputs[:,3,:,:].unsqueeze(1)
        
        out1 = self.img1(input1)
        out2 = self.img2(input2)

        out3 = self.img3(input3)

        out4 = self.img4(input4)

        cat1 = torch.cat((out1,out2),dim=1)

        cat2 = torch.cat((out3,out4),dim=1)

        cat_last = torch.cat((cat1,cat2),dim=1)
        
        out = self.se1(cat_last)

        out = self.model(out)

        out = out.view(out.size(0),-1)

        out = self.dropout(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.fc4(out)
        return out



if __name__ == "__main__":


    import os 

    model = AlexnetCat2(1)



    with open("efficientNET.txt","w") as f:

        for module in model.children():

            f.write(str(module)+"\n")

        f.close()











# if __name__ == "__main__":
    # path = "/Users/okanegemen/Desktop/yoloV5/INbreast Release 1.0/AllDICOMs/20586986_6c613a14b80a8591_MG_L_ML_ANON.dcm"

    # dicom_img = dicom.dcmread(path)

    # numpy_pixels = dicom_img.pixel_array
    # img = np.resize(numpy_pixels,(600,600))
    # img = np.array(img,dtype="float32")


    # tensor = torch.from_numpy(img)
    # tensor = tensor.float()
    # tensor = torch.reshape(tensor,[1, 1, 600, 600])

    # model = models.resnet101()
