import torch.nn as nn
import torch

array = torch.tensor([[1,2,3],[4,5,6]])

net = nn.Conv2d(in_channels=2,out_channels=2,padding =1,dilation=3,bias=False)

output = net(array)

print(output)