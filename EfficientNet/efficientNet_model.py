import torch
import torch.nn as nn 
from networkParts import *



class MBConv1(MBConvN):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, Rs=0.25):
        super().__init__(inplanes, outplanes, expansion=1, kernel_size=kernel_size, stride=stride, Rs=Rs)


class MBConv6(MBConvN):
    def __init__(self, inplanes, outplanes,kernel_size=3, stride=1, Rs=0.25):
        super().__init__(inplanes, outplanes, expansion=6, kernel_size=kernel_size, stride=stride, Rs=Rs)



a = MBConv6(32,24)


