from Pytorch_model.unet.unet_model import UNet
import torch
from torch.autograd import Variable

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

rand = torch.rand(1,1,500,500).to(dev)
rand = Variable(rand)

net = UNet(1,1,False).to(dev)
