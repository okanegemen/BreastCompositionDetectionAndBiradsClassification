
import torch
import torch.nn as nn





def conv3x3(in_planes,out_planes,stride=1):
    
    return nn.Conv2d(in_planes,out_planes,kernel_size = 3 ,stride = stride,padding=1,bias = False)
    
    
def conv1x1(in_planes,out_planes,stride=1):
    
    return nn.Conv2d(in_planes,out_planes,kernel_size = 1, stride = stride,bias = False)


class BasicBlock(nn.Module):
    
    expansion = 1
    identity = 0
    
    def __init__(self,in_planes,planes,stride = 1 ,downsample = None):
        
        super(BasicBlock,self).__init__()
        
        self.conv1 = conv3x3(in_planes,planes,stride)
        
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.relu = nn.ReLU(inplace= True)
        
        self.drop = nn.Dropout(0.6)
        
        self.conv2 = conv3x3(planes,planes)
        
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride 
        
        
    def forward(self,x):
        
        
        identity = x
           
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
           
        out = self.conv2(out)
           
        out = self.bn2(out)
        out = self.drop(out)
           
           
        if self.downsample is not None:
            identity = self.downsample(x)
         
        out = out+identity
        out = self.relu(out)
        
        return out
       
           
           
        

    
class ResNet50(nn.Module):
    
    def __init__(self,block,layers,num_classes = 3):
        
        super(ResNet50,self).__init__()
        
        self.inplanes=64
        self.conv1 = nn.Conv2d(1,64,kernel_size=7,stride = 2,padding = 3)
        
        self.bn1 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU(inplace = True)
        
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride = 2,padding = 1)
        
        self.layer1 = self._make_layer(block,64,layers[0],stride=1)
        
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)
        
        self.layer3 = self._make_layer(block,256,layers[2],stride=2)
        
        self.layer4 = self._make_layer(block,512,layers[3],stride=2)
        
        
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc1  = nn.Linear(512*block.expansion,num_classes)
        
        self.softmax = nn.Softmax()

        
        for mod in self.modules():
            
            if isinstance(mod,nn.Conv2d):
                
                nn.init.kaiming_normal_(mod.weight,mode = "fan_out", nonlinearity = "relu")
            elif isinstance(mod,nn.BatchNorm2d):
                nn.init.constant_(mod.weight,1)
                nn.init.constant_(mod.bias,0)
               
        
    def _make_layer(self,block,planes,blocks,stride = 1):
        
        
        downsample = None
        if stride !=1 or self.inplanes != planes*block.expansion:
        
            downsample = nn.Sequential(
                conv1x1(self.inplanes,planes*block.expansion,stride=stride),
                nn.BatchNorm2d(planes*block.expansion))
        
        layers = []
        
        layers.append(block(self.inplanes,planes,stride = stride,downsample=downsample))
        self.inplanes = planes*block.expansion
        
        for _ in range(1,blocks):
            
            layers.append(block(self.inplanes,planes))
            
        return nn.Sequential(*layers)
    
    def forward(self,x):
        
    
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avg(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        return out
        
        