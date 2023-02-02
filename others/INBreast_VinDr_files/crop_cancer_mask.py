import os
from PIL import Image
import torch
from torchvision import transforms

path = "/home/alican/Documents/Datasets/INBreast/storage/cancer_areas"

transform_pil = transforms.Compose([transforms.ToTensor()])
transform_torch = transforms.ToPILImage()

masks_all = os.listdir(path)
masks_all = {mask:Image.open(os.path.join(path,mask)) for mask in masks_all}

for name, image in masks_all.items():
    img = transform_pil(image)
    temp = torch.nonzero(img)
    temp = [torch.min(temp[:,1]),torch.max(temp[:,1]),torch.min(temp[:,2]),torch.max(temp[:,2])]
    a = img[:,temp[0]:temp[1],temp[2]:temp[3]] 
    a = transform_torch(a)
    print(name, torch.stack(temp).detach().cpu(),temp[1]-temp[0],temp[3]-temp[2])

# CC_L.PNG tensor([ 144, 2902,    0, 1532]) tensor(2758) tensor(1532)
# MLO_L.PNG tensor([ 495, 2911,    0, 1757]) tensor(2416) tensor(1757)
# MLO_R.PNG tensor([ 511, 2981,  916, 2559]) tensor(2470) tensor(1643)
# CC_R.PNG tensor([ 330, 2516, 1165, 2559]) tensor(2186) tensor(1394)