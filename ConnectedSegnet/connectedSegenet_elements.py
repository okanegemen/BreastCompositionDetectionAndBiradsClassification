import torch
import torch.nn as nn
import torch.nn.functional as F




#For blocks which created by double convolutional block

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels,encoder=False):
        super(DoubleConv).__init__()
        if encoder:
            self.conv=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,3,padding=1,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels,out_channels,3,padding=1,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv=nn.Sequential(
                nn.ConvTranspose2d(in_channels,out_channels,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            

        


    def forward(self,x):
        return self.conv(x)

        
        

#For blocks which created by Tripple convolutional block if encoder true it will create encoder block and conv layer 
# else if encoder false create convTranspose layer
class TripleConv(nn.Module):
    def __init__(self,in_channels,out_channels,encoder=False):
        super(TripleConv).__init__()
        if encoder:

            self.tripple_conv=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        else:
            self.tripple_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels,out_channels,kernel_size=3,padding=1,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(out_channels,out_channels,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(out_channels,out_channels,kernel_size=3,padding=1,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        

    def forward(self,x):
        return self.tripple_conv(x)


#created conv layer which have 1 kernel_size
def conv1x1(in_channels,out_channels,padding=None,stride=None):

    conv1 =nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,padding = padding),
        nn.ReLU(inplace=True)
        )
    return conv1 
#dilation layer
def dilationConv(in_channels,out_channels,):
    dilation = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding = 1,dilation=3)
    return dilation
    
    
#maxpooling with indices for downsampling,indices for upsampling
def maxpooling(conv_output,kernel_size=2,stride=2,return_indices=True):
    return F.max_pool2d(conv_output,kernel_size,stride=stride,return_indices=return_indices)
#unmaxpooling to create upsampling layer
def unmaxpooling(conv_output,maxpool_indices,dim,stride=2,kernel_size=2):
    return F.max_unpool2d(input=conv_output,indices=maxpool_indices,stride=stride,kernel_size=kernel_size,output_size=dim)

#concate 2 output
def cat(x1,x2):
    new_output=torch.cat([x1,x2],dim=1)
    return new_output