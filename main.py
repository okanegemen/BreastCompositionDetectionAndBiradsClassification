from DataLoaders.dataset import Dataset
from DataLoaders.XLS_utils import XLS
# from Pytorch_model.unet import UNet as load_model
from AllModels.TransferlerarningModels.transfer_learning import ConcatModel as load_model
from torchvision.models import resnet34 as upload_model,ResNet34_Weights as upload_weight
# from AllModels.unet_model import UNet as upload_model

import DataLoaders.config as config
import math
import sys
import os
from torch.nn import CrossEntropyLoss as Loss
from torch.optim import Adam,RMSprop,NAdam
from torch.utils.data import DataLoader,SubsetRandomSampler
from sklearn.model_selection import KFold
from torchvision import transforms as T
import matplotlib.pyplot as plt
import torch
import random
import time
from qqdm import qqdm, format_str
from DataLoaders.scores import scores
import numpy 

from engine import testing,training
import json
import imp
import shutil


def collate_fn(batch):
    return tuple(zip(*batch))

def get_resnet18():
    model = load_model()
    in_channels = model.fc.in_features
    model.fc = torch.nn.Linear(in_channels,3)
    for id,(name,param) in enumerate(model.named_parameters()):
        print(name,param.requires_grad)

    return model.to(config.DEVICE)

def get_model():
    if config.LOAD_NEW_MODEL:
        # kwargs = dict({"num_classes":config.NUM_CLASSES})
        model = load_model(upload_model(weights=upload_weight.DEFAULT)) # upload_model(weights=upload_weight.DEFAULT)

        # model.conv1.in_channels = config.NUM_CHANNELS
        # model.fc.out_features = config.NUM_CLASSES

        print("Random Weighted ModelS loaded.")
        # print(model)

    else:
        try:
            model = load_model()
            print("############# Previous weights loaded. ###################")
            model.load_state_dict(torch.load(config.MODEL_PATH))
        except:
            model = torch.load(config.MODEL_PATH) 
        # print(model.classifier)
        # model.classifier = torch.nn.Linear(1024,config.NUM_CLASSES)

    if config.FREEZE_LAYER>0:
        ct = 0
        for param in model.parameters():
            ct += 1
        print("Number of layers:",ct)

        lt=config.FREEZE_LAYER*ct
        cntr=0

        for param in model.parameters():
            cntr+=1
            if cntr < 44:
                param.requires_grad = True

            elif cntr < lt:
                param.requires_grad = False
            elif cntr == ct:
                param.requires_grad = True

    # if config.SKIP_FREEZE[1]>0:
    #     cntr=0
    #     ct -= config.SKIP_FREEZE[2]
    #     freeze = config.SKIP_FREEZE[0]
    #     skip = config.SKIP_FREEZE[1]
    #     freeze_b = lt

    #     for name,param in model.named_parameters():
    #         cntr+=1
    #         if cntr>freeze_b+freeze:
    #             freeze_b += skip
    #         if cntr != 0 and cntr != ct-1 and cntr>lt:
    #             if freeze_b<cntr<=freeze_b+freeze:
    #                 param.requires_grad = False
    #                 print(name)

    for id,(name,param) in enumerate(model.named_parameters()):
        print(name,param.requires_grad)
    return model.to(config.DEVICE)

def get_others(model):

    lossFunc = Loss()
    # opt = RMSprop(model.parameters(),lr=config.INIT_LR)
    opt = Adam(model.parameters(), lr=config.INIT_LR,weight_decay=1e-4)#,weight_decay=1e-6
    print("LossFunc:",lossFunc)
    print("Optimizer:",opt)

    return lossFunc,opt


def save_model_and_metrics(model,fold_metrics):
    print("\nSaving Model...")
    name = ""+model.__class__.__name__+"_"+config.DATE_FOLDER
    print(name)
    os.makedirs(os.path.join(config.SAVE_FOLDER,name))
    torch.save(model, os.path.join(config.SAVE_FOLDER,name,model.__class__.__name__+".pth"))

    jso = json.dumps(fold_metrics)
    with open(os.path.join(config.SAVE_FOLDER,name,model.__class__.__name__+".json"),"w") as f:
        f.write(jso)

    srcs = ["DataLoaders/XLS_utils.py","DataLoaders/fiximage.py","DataLoaders/dataset.py","DataLoaders/config.py","main.py","engine.py","AllModels/TransferlerarningModels/transfer_learning.py"]
    for src in srcs:
        dst = os.path.join(config.BASE_OUTPUT,"results_models",name,src.split("/")[-1].split(".")[0]+".txt")
        shutil.copyfile(os.path.join(config.BASE_OUTPUT,src),dst)


def get_dataloaders(train_valDS,train_sampler,val_sampler):
    trainLoader = DataLoader(train_valDS,sampler=train_sampler, shuffle=False, batch_size=config.BATCH_SIZE, num_workers=0,collate_fn=collate_fn)
    valLoader = DataLoader(train_valDS,sampler=val_sampler, shuffle=False, batch_size=config.BATCH_SIZE, num_workers=0,collate_fn=collate_fn)

    return trainLoader, valLoader
    
def base():
    test_acc = []
    model = get_model() #load_model().to(config.DEVICE)
    total_time_start = time.time()
    metrics = {"training":[],"test":[]}
    lossFunc, opt= get_others(model)

    data = XLS()
    train_val,test = data.return_datasets()
    print(f"[INFO] found {len(train_val)} examples in the train and val set...")
    print(f"[INFO] found {len(test)} examples in the test set...")   

    train_val_indexs,train_val = data.train_val_k_fold(train_val)

    test = Dataset(test,False)
    testLoader = DataLoader(test,config.BATCH_SIZE,shuffle=False,num_workers=4,collate_fn=collate_fn)
    
    for loop in range(config.TEKRAR):
        imp.reload(config)
        print(f'TEKRAR {loop}')
        print('--------------------------------')
        for fold,(train_idxs,val_idxs) in enumerate(zip(train_val_indexs["train"],train_val_indexs["val"])):
            print(f'FOLD {fold}')
            print('--------------------------------')

            train = Dataset(train_val.iloc[train_idxs],True)
            val = Dataset(train_val.iloc[val_idxs],False,True)
            print(f"[INFO] found {len(train)} examples in the train set...")
            print(f"[INFO] found {len(val)} examples in the val set...")

            trainLoader = DataLoader(train,sampler=train.sampler, batch_size=config.BATCH_SIZE, num_workers=4,collate_fn=collate_fn)
            valLoader = DataLoader(val, shuffle=False, batch_size=config.BATCH_SIZE, num_workers=4,collate_fn=collate_fn)

            training_metrics = training(model,trainLoader,lossFunc,opt,valLoader,fold)
            metrics["training"].append(training_metrics)

        test_metrics = testing(model,lossFunc,testLoader)
        metrics["test"].append(test_metrics)
        print(f"--------Passed time: {int(time.time()-total_time_start)/60}-------")
        



    # image,_ = train_valDS[0]
    # for id,img in enumerate(image):
    #     # print(img.min(),img.max())
    #     T.ToPILImage()(img).show()
    #     time.sleep(1)
    # input()
        if config.CV_K_FOLDS < 2:
            trainLoader = DataLoader(train_val,config.BATCH_SIZE,sampler=train_val.sampler,num_workers=4,collate_fn=collate_fn)
            training_metrics = training(model,trainLoader,lossFunc,opt)
            metrics["training"].append(training_metrics)

            test_metrics = testing(model,lossFunc,testLoader)
            metrics["test"].append(test_metrics)

        test_acc.append(metrics["test"][-1]["acc"])

    total_time = int(time.time()-total_time_start)/60
    print(f"---------- Training_time:{total_time} minute ----------")
    save_model_and_metrics(model,metrics)
    
    print(test_acc)
    print(sum(test_acc)/len(test_acc))
if __name__ == "__main__":
    base()