import torch
import torch.nn as nn 
import math



def conv1x1(in_planes,out_planes,stride=1):
    
    return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride,bias = False)


class DownSampling(nn.Module):

    def __init__(self, inplanes,planes,stride):
        super(DownSampling, self).__init__()

        self.conv1x1 = conv1x1(inplanes,planes,stride = stride)

        self.bn = nn.BatchNorm2d(planes)
    def forward(self,x):

        out = self.conv1x1(x)
        return self.bn(out)

    





class BasicBlock(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=0,bias=False,activation=True,groups=1):
        super(BasicBlock,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups = groups,
                    bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)

        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    
    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)

        return out




class SENet(nn.Module):

    def __init__(self,inplanes,expansion=1):
        super(SENet,self).__init__()
        Rs = 0.25/expansion
        self.squezee = nn.AdaptiveAvgPool2d(1)

        self.excitation = nn.Sequential(nn.Conv2d(in_channels=inplanes,out_channels=int(inplanes*Rs),kernel_size=1),
                                        nn.SiLU(inplace=True),
                                        nn.Conv2d(in_channels=int(inplanes*Rs),out_channels=inplanes,kernel_size=1),
                                        nn.Sigmoid()
        )



    def forward(self,input):
        out = self.squezee(input)
        out = self.excitation(out)

        return input*out

class MBConvN(nn.Module):

    def __init__(self,inplanes,outplanes,expansion=1,kernel_size=3,stride=1):
        super(MBConvN,self).__init__()

        padding = int((kernel_size-1)/2)
        expanded = int(expansion*inplanes)

        self.expanded_block = None if (expansion==1) else BasicBlock(inplanes,expanded,kernel_size=1)

        self.depthwise =  BasicBlock(expanded,expanded,kernel_size=kernel_size,stride=stride,padding=padding,groups=expanded)

        self.SENet = SENet(inplanes=expanded,expansion=expansion)

        self.reduce_pw = BasicBlock(in_channels=expanded,out_channels=outplanes,kernel_size=1,activation=False)
        
        self.dropsample = None if (stride==1) and (inplanes==outplanes) else DownSampling(inplanes=inplanes,planes=outplanes,stride=stride)
        

    def forward(self, x):

        identity = x
        if self.expanded_block is not None:
            x = self.expanded_block(x)
        out = self.depthwise(x)
        out = self.SENet(out)
        out = self.reduce_pw(out)
        if self.dropsample is not None:
            identity = self.dropsample(identity)
            return (out+identity)
        
        return out



def scale_width(w, w_factor):
  w *= w_factor
  new_w = (int(w+4) // 8) * 8
  new_w = max(8, new_w)
  if new_w < 0.9*w:
     new_w += 8
  return int(new_w)























        
        
# downsample = None
# if stride !=1 or self.inplanes != planes*block.expansion:

#     downsample = nn.Sequential(
#         conv1x1(self.inplanes,planes*block.expansion,stride=stride),
#         nn.BatchNorm2d(planes*block.expansion))

# layers = []

# layers.append(block(self.inplanes,planes,stride = stride,downsample=downsample))
# self.inplanes = planes*block.expansion












