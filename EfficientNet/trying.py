import torch
from torch import nn
import math
import configToMac as config

class ConvBnAct(nn.Module):
  """Layer grouping a convolution, batchnorm, and activation function"""
  def __init__(self, n_in, n_out, kernel_size=3, 
               stride=1, padding=0, groups=1, bias=False,
               bn=True, act=True):
    super().__init__()
    
    self.conv = nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                          stride=stride, padding=padding,
                          groups=groups, bias=bias)
    self.bn = nn.BatchNorm2d(n_out) if bn else nn.Identity()
    self.act = nn.SiLU() if act else nn.Identity()
  
  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.act(x)
    return x


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


class MBConvN(nn.Module):
  """MBConv with an expansion factor of N, plus squeeze-and-excitation"""
  def __init__(self, n_in, n_out, expansion_factor,
               kernel_size=3, stride=1, r=24, p=0):
    super().__init__()

    padding = (kernel_size-1)//2
    expanded = expansion_factor*n_in
    self.skip_connection = (n_in == n_out) and (stride == 1)

    self.expand_pw = nn.Identity() if (expansion_factor == 1) else ConvBnAct(n_in, expanded, kernel_size=1)
    self.depthwise = ConvBnAct(expanded, expanded, kernel_size=kernel_size, 
                               stride=stride, padding=padding, groups=expanded)
    self.se = SEBlock(expanded, r=r)
    self.reduce_pw = ConvBnAct(expanded, n_out, kernel_size=1,
                               act=False)
    self.dropsample = DropSample(p)
  
  def forward(self, x):
    residual = x

    x = self.expand_pw(x)
    x = self.depthwise(x)
    x = self.se(x)
    x = self.reduce_pw(x)

    if self.skip_connection:
      x = self.dropsample(x)
      x = x + residual

    return x



class DropSample(nn.Module):
  """Drops each sample in x with probability p during training"""
  def __init__(self, p=0):
    super().__init__()

    self.p = p
    print(self.p)
  
  def forward(self, x):
    if (not self.p) or (not self.training):
      return x
    
    batch_size = len(x)
    random_tensor = torch.FloatTensor(batch_size, 1, 1, 1).uniform_()
    bit_mask = self.p<random_tensor.to(config.DEVICE)

    x = x.div(1-self.p)
    x = x * bit_mask
    return x


class MBConv1(MBConvN):
  def __init__(self, n_in, n_out, kernel_size=3,
               stride=1, r=24, p=0):
    super().__init__(n_in, n_out, expansion_factor=1,
                     kernel_size=kernel_size, stride=stride,
                     r=r, p=p)
    
 
class MBConv6(MBConvN):
  def __init__(self, n_in, n_out, kernel_size=3,
               stride=1, r=24, p=0):
    super().__init__(n_in, n_out, expansion_factor=6,
                     kernel_size=kernel_size, stride=stride,
                     r=r, p=p)
def scale_width(w, w_factor):
  """Scales width given a scale factor"""
  w *= w_factor
  new_w = (int(w+4) // 8) * 8
  new_w = max(8, new_w)
  if new_w < 0.9*w:
     new_w += 8
  return int(new_w)


def create_stage(n_in, n_out, num_layers, layer_type, 
                 kernel_size=3, stride=1, r=24, p=0):
  """Creates a Sequential consisting of [num_layers] layer_type"""
  layers = [layer_type(n_in, n_out, kernel_size=kernel_size,
                       stride=stride, r=r, p=p)]
  layers += [layer_type(n_out, n_out, kernel_size=kernel_size,
                        r=r, p=p) for _ in range(num_layers-1)]
  layers = nn.Sequential(*layers)
  return layers


class EfficientNet(nn.Module):
  """Generic EfficientNet that takes in the width and depth scale factors and scales accordingly"""
  def __init__(self, w_factor=1, d_factor=1,
               out_sz=1000):
    super().__init__()

    base_widths = [(32, 16), (16, 24), (24, 40),
                   (40, 80), (80, 112), (112, 192),
                   (192, 320), (320, 1280)]
    base_depths = [1, 2, 2, 3, 3, 4, 1]

    scaled_widths = [(scale_width(w[0], w_factor), scale_width(w[1], w_factor)) 
                     for w in base_widths]
    scaled_depths = [math.ceil(d_factor*d) for d in base_depths]
    
    kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
    strides = [1, 2, 2, 2, 1, 2, 1]
    ps = [0, 0.029, 0.057, 0.086, 0.114, 0.143, 0.171]
    
    self.stem = ConvBnAct(1, scaled_widths[0][0], stride=2, padding=1)
    
    stages = []
    for i in range(7):
      layer_type = MBConv1 if (i == 0) else MBConv6
      r = 4 if (i == 0) else 24
      stage = create_stage(*scaled_widths[i], scaled_depths[i],
                           layer_type, kernel_size=kernel_sizes[i], 
                           stride=strides[i], r=r, p=ps[i])
      stages.append(stage)
    self.stages = nn.Sequential(*stages)

    self.pre_head = ConvBnAct(*scaled_widths[-1], kernel_size=1)

    self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                              nn.Flatten(),
                              nn.Linear(scaled_widths[-1][1], out_sz))

  def feature_extractor(self, x):
    x = self.stem(x)
    x = self.stages(x)
    x = self.pre_head(x)
    return x

  def forward(self, x):
    x = self.feature_extractor(x)
    x = self.head(x)
    return x


class EfficientNetB0(EfficientNet):
  def __init__(self, out_sz=1000):
    w_factor = 1
    d_factor = 1
    super().__init__(w_factor, d_factor, out_sz)
    
    
class EfficientNetB1(EfficientNet):
  def __init__(self, out_sz=1000):
    w_factor = 1
    d_factor = 1.1
    super().__init__(w_factor, d_factor, out_sz)
   
  
class EfficientNetB2(EfficientNet):
  def __init__(self, out_sz=1000):
    w_factor = 1.1
    d_factor = 1.2
    super().__init__(w_factor, d_factor, out_sz)
    
    
class EfficientNetB3(EfficientNet):
  def __init__(self, out_sz=1000):
    w_factor = 1.2
    d_factor = 1.4
    super().__init__(w_factor, d_factor, out_sz)
    
    
class EfficientNetB4(EfficientNet):
  def __init__(self, out_sz=1000):
    w_factor = 1.4
    d_factor = 1.8
    super().__init__(w_factor, d_factor, out_sz)
    
    
class EfficientNetB5(EfficientNet):
  def __init__(self, out_sz=1000):
    w_factor = 1.6
    d_factor = 2.2
    super().__init__(w_factor, d_factor, out_sz)
    
    
class EfficientNetB6(EfficientNet):
  def __init__(self, out_sz=1000):
    w_factor = 1.8
    d_factor = 2.6
    super().__init__(w_factor, d_factor, out_sz)
    
    
class EfficientNetB7(EfficientNet):
  def __init__(self, out_sz=6):
    w_factor = 2
    d_factor = 3.1
    super().__init__(w_factor, d_factor, out_sz)



from PIL import Image
import cv2 as cv
import numpy as np
import pydicom as dicom 

path = "/Users/okanegemen/Desktop/yoloV5/INbreast Release 1.0/AllDICOMs/20586986_6c613a14b80a8591_MG_L_ML_ANON.dcm"

dicom_img = dicom.dcmread(path)

numpy_pixels = dicom_img.pixel_array
img = np.resize(numpy_pixels,(600,600))
img = np.array(img,dtype="float32")


model = EfficientNetB7()
tensor = torch.from_numpy(img)
tensor = tensor.float()
tensor = torch.reshape(tensor,[1, 1, 600, 600])


# model_num = int(input("PLEASE ENTER NUMBER FROM 0 TO 7 TO TRY EFFICIENT NET BLOCKES : "))

# model = models[model_num]

model.eval()

out = model(tensor)
# print(model.parameters)






