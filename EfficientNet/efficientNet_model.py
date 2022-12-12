import torch
import torch.nn as nn 
from networkParts import *




class MBConv1(MBConvN):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, Rs=0.25):
        super().__init__(inplanes, outplanes, expansion=1, kernel_size=kernel_size, stride=stride, Rs=Rs)


class MBConv6(MBConvN):
    def __init__(self, inplanes, outplanes,kernel_size=3, stride=1, Rs=0.25):
        super().__init__(inplanes, outplanes, expansion=6, kernel_size=kernel_size, stride=stride, Rs=Rs)





class EfficientModel(nn.Module):
    def __init__(self,sample_num,w_factor,d_factor):
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
        stem = BasicBlock(in_channels=1,out_channels=scale_width[0][0],kernel_size=3,stride=2,padding=1)

    def _make_layer(self,block,in_channels,out_channels,depth_layer,kernel_size,stride,rs):


        layers = []

        for _ in range(depth_layer):
            layers.append(block(in_channels,out_channels,kernel_size,stride,rs))


        return nn.Sequential(*layers)

         
          










