import os
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms

def get_mask_info(filename):
    id, _, _, laterality, view, _ = filename.split("_")
    return laterality, view

def open_mask(path):
    img = Image.open(path).resize((2560,3328))
    img = ImageOps.grayscale(img)
    return img

def merge_masks(list_mask,transform):
    path = "/home/alican/Documents/Datasets/INBreast/storage/All_masks"
    mask_all = torch.zeros((1,3328,2560))
    for mask in list_mask:
        img = open_mask(os.path.join(path,mask))
        img = transform(img)
        mask_all = mask_all + img
    return mask_all

def classify(masks):

    mask_dict = {"MLO_R":[], "MLO_L":[], "CC_R":[], "CC_L":[]}

    for mask in masks:
        laterality, view = get_mask_info(filename = mask)
        if laterality == "R" and view == "ML":
            mask_dict["MLO_R"].append(mask)

        elif laterality == "R" and view == "CC":
            mask_dict["CC_R"].append(mask)

        elif laterality == "L" and view == "ML":
            mask_dict["MLO_L"].append(mask)

        elif laterality == "L" and view == "CC":
            mask_dict["CC_L"].append(mask)

    return mask_dict

path = "/home/alican/Documents/Datasets/INBreast/storage/All_masks"

masks = os.listdir(path)

mask_dict = classify(masks)
mask_all_dict = {"MLO_R":0, "MLO_L":0, "CC_R":0, "CC_L":0}
transform = transforms.Compose([transforms.ToTensor()])
for key, list_mask in mask_dict.items():
    mask_all_dict[key] = merge_masks(list_mask,transform)

transform = transforms.ToPILImage()

for key, mask_all in mask_all_dict.items():
    img = transform(mask_all)
    img.show()