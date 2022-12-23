import torch
import torch.nn as nn 
from networkParts import *
from linear_block import *








class MBConv1(MBConvN):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1):
        super(MBConv1,self).__init__(inplanes, outplanes, expansion=1, kernel_size=kernel_size, stride=stride)


class MBConv6(MBConvN):
    def __init__(self, inplanes, outplanes,kernel_size=3, stride=1):
        super(MBConv6,self).__init__(inplanes, outplanes, expansion=6, kernel_size=kernel_size, stride=stride)





class EfficientModel(nn.Module):
    def __init__(self,class_num,in_channels,w_factor=1,d_factor=1.):
        super(EfficientModel,self).__init__()



        base_widths = [(32, 16), (16, 24), (24, 40),
               (40, 80), (80, 112), (112, 192),
               (192, 320), (320, 1280)]
        base_depths = [1, 2, 2, 3, 3, 4, 1]


        scaled_widths = [(scale_width(w[0], w_factor), scale_width(w[1], w_factor)) 
                 for w in base_widths]
        scaled_depths = [math.ceil(d_factor*d) for d in base_depths]


        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]

        strides = [1, 2, 2, 2, 1, 2, 1]
        self.stem = BasicBlock(in_channels=in_channels,out_channels=scaled_widths[0][0],kernel_size=3,stride=2,padding=1)

        blockes = []

        for i in range(7):

            layer = MBConv1 if i == 0 else MBConv6
            block=self._make_layer(layer,in_channels=scaled_widths[i][0],out_channels=scaled_widths[i][1],
                             depth_layer=scaled_depths[i],kernel_size=kernel_sizes[i],stride=strides[i])
            blockes.append(block)


            
        
        self.mbconvblocks = nn.Sequential(*blockes)


        self.pre_head = BasicBlock(in_channels=scaled_widths[-1][0],out_channels=scaled_widths[-1][1],kernel_size=1)

        self.avg = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.5)

        self.linear = FullyConnected(scaled_widths[-1][1],1024,n_classes=class_num)

        

        

        

            


    def _make_layer(self,block,in_channels,out_channels,depth_layer,kernel_size,stride):


        layers = []
        layers.append(block(in_channels,out_channels,kernel_size,stride))
        for _ in range(depth_layer-1):
            layers.append(block(out_channels,out_channels,kernel_size,stride))
        return nn.Sequential(*layers)

    def forward(self,input):
        out = self.stem(input)
        out = self.mbconvblocks(out)
        out = self.pre_head(out)

        out = self.avg(out)
        out = self.dropout(out)

        out = out.view(out.size(0),-1)


        out = self.linear(out)

        return out

#YOU CAN TRY WİTH INPUT WHICH IS GIVEN IMAGE

# if __name__ == "__main__ ":

w_factor = [1, 1 , 1.1 , 1.2 , 1.4 , 1.6 , 1.8 , 2]
num_classes = 3
d_factor = [1 , 1.1 , 1.2 , 1.4 , 1.8 , 2.2 , 2.6 , 3.1]


models = []


for i in range(8):

    model = EfficientModel(num_classes,1,w_factor=w_factor[i],d_factor=d_factor[i])
    models.append(model)


from PIL import Image
import cv2 as cv
import numpy as np
import pydicom as dicom 

path = "/Users/okanegemen/yoloV5/INbreast Release 1.0/AllDICOMs/20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm"

dicom_img = dicom.dcmread(path)

numpy_pixels = dicom_img.pixel_array
img = np.resize(numpy_pixels,(600,600))
img = np.array(img,dtype="float32")



tensor = torch.from_numpy(img)
tensor = tensor.float()
tensor = torch.reshape(tensor,[1, 1, 600, 600])


# model_num = int(input("PLEASE ENTER NUMBER FROM 0 TO 7 TO TRY EFFICIENT NET BLOCKES : "))

# model = models[model_num]

model.eval()

out = model(tensor)

print("FOR EFFICIENTB{} PREDİCTİON VALUES IS : {} ".format(model_num,out))

print(model.parameters)





# print(out)



            

            
            














