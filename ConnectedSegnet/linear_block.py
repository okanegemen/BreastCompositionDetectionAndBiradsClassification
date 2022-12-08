import torch
import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(self,in_features,out_features,n_classes):
        super(FullyConnected,self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features=in_features,out_features=out_features)
        self.fc2 = nn.Linear(in_features=out_features,out_features=n_classes)


    def forward(self,x):
        out = self.fc1(x)
        out = self.fc2(out)

        return out



def make_layer(conv_block,fc_block):
     layer_list = []
     layer_list.append(conv_block)
     layer_list.append(fc_block)
     return nn.Sequential(*layer_list)

from connectedSegnet_model import ConSegnetsModel
from connectedSegnet_elements import *

fc = FullyConnected(64*20*20,4096,5)


model = ConSegnetsModel(1,5)

model = make_layer(model,fc)

print(model.parameters)

        

        

        
    