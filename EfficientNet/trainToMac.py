import pandas as pd
import numpy as np 
import torch 
import torch.nn as nn 
import torch.backends as backends
import torch.nn.functional as F

from torch.utils.data import DataLoader,Dataset

import matplotlib.pyplot as plt
import os
from torch.optim import Adam
from torchvision import transforms












from DataLoaders.dataset import Dataset
from DataLoaders.XLS_utils import XLS
from trying import EfficientNet as load_model
# from ConnectedSegnet.connectedSegnet_model import ConSegnetsModel as load_model
import configToMac as config
import math
import sys
import os
from torch.nn import CrossEntropyLoss as Loss
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import time
from tqdm import tqdm

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model():
    if config.LOAD_NEW_MODEL:
        model = load_model(w_factor=2,d_factor=3.1,out_sz=config.NUM_CLASSES).to(config.DEVICE)
        print("Random Weighted Model loaded.")
        return model
    else:
        model = load_model(config.NUM_CHANNELS,config.NUM_CLASSES).to(config.DEVICE)
        model.load_state_dict(torch.load(config.MODEL_PATH))
        print("############# Previous weights loaded. ###################")
        return model

def get_dataset():
    train,test,imgs_dir = XLS().get_all_info()

    train = Dataset(train,imgs_dir)
    test = Dataset(test,imgs_dir)

    return train, test

def get_dataloaders(trainDS,testDS):
    trainLoader = DataLoader(trainDS,sampler=trainDS.sampler, shuffle=False,
        batch_size=config.BATCH_SIZE, num_workers=0,collate_fn=collate_fn)
    testLoader = DataLoader(testDS,sampler=testDS.sampler, shuffle=False,
        batch_size=config.BATCH_SIZE, num_workers=0,collate_fn=collate_fn)

    return trainLoader, testLoader

def get_others(model):

    lossFunc = Loss()
    opt = RMSprop(model.parameters(), lr=config.INIT_LR)

    return lossFunc,opt

def plot(H):
    train_epochs = [*range(len(H["train_loss"]))]
    val_epochs = [epoch for epoch in train_epochs if epoch%config.VALIDATE_PER_EPOCH==0]
    print(train_epochs,val_epochs)
    print(H)

    plt.plot(train_epochs,H["train_acc"])
    plt.plot(val_epochs,H["val_acc"])
    plt.savefig(config.PLOT_ACC_PATH)
    plt.clf()

    plt.plot(train_epochs,H["train_loss"])
    plt.plot(val_epochs,H["val_loss"])
    plt.savefig(config.PLOT_LOSS_PATH)





def training(model, trainLoader, lossFunc, optimizer, valLoader, H):
    count = 0
    # loop over epochs
    model.train()
    print("[INFO] training the network...")
    for epoch in range(config.NUM_EPOCHS):
        # set the model in training mode
        

        lr_scheduler = None
        # if epoch == 0:
        #     warmup_factor = 1.0 / 1000
        #     warmup_iters = min(1000, len(trainLoader) - 1)

        #     lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         optimizer,mode="min",
        #     )        

        # loop over the training set

        train_loss = 0
        train_acc = 0
        train_count = 0
        for idx_t,traindata in enumerate(pbar:=tqdm(trainLoader,ncols=100)):
            images,targets = traindata



            # send the input to the device
            # images = torch.stack([image.to(config.DEVICE) for image in images])
            # targets = torch.stack([v.to(config.DEVICE) for v in targets]).float()
            images, targets = torch.stack(images).to(config.DEVICE), torch.stack(targets).view(-1).to(config.DEVICE)            
            optimizer.zero_grad()
            outputs = model(images)
            print(outputs)
            
            loss_train = lossFunc(outputs,targets)
            train_count += 1
            train_loss += loss_train.item()
            temp_loss = train_loss / train_count
            train_acc += 1*(torch.argmax(outputs,dim=1)==targets).sum().item()
            temp_acc = train_acc/(train_count*config.BATCH_SIZE)

            
            
            loss_train.backward()
            

            # torch.nn.utils.clip_grad_norm_(model.parameters(),25)
            if not math.isfinite(loss_train):
                print(f"Loss is {loss_train}, stopping training")
                print(temp_acc)
                sys.exit(1)
            optimizer.step()
        
            

            pbar.set_description(f"Epoch:[{epoch+1}] lr: {optimizer.param_groups[0]['lr']:.7f}  train_loss: {temp_loss:.4f}  train_acc:{temp_acc:.4f}")
            
            if lr_scheduler is not None:
                lr_scheduler.step()
        
        H["train_acc"].append(temp_acc)
        H["train_loss"].append(temp_loss)
        n_threads = torch.get_num_threads()
        
        if epoch % config.VALIDATE_PER_EPOCH == 0 and (epoch != 0 or config.VALIDATE_PER_EPOCH == 1):
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set

            val_acc = 0
            val_loss = 0
            val_counter = 0
            # switch off autograd
            
            with torch.no_grad():
                for idx_v, valData in enumerate(pbar:=tqdm(valLoader,ncols=80)):
                    images, targets = valData

                    # send the input to the device
                    images, targets = torch.stack(images).to(config.DEVICE), torch.stack(targets).view(-1).to(config.DEVICE)
                    outputs = model(images)
                    loss_val = lossFunc(outputs,targets)

                    val_counter += 1
                    val_loss += loss_val.item()
                    temp_loss = val_loss/val_counter

                    val_acc += 1*(torch.argmax(outputs)==targets).sum().item()
                    temp_acc = val_acc/val_counter

                    pbar.set_description(f"Val: val_loss: {temp_loss:.4f}  val_acc:{temp_acc:.4f}")

                H["val_acc"].append(temp_acc)
                H["val_loss"].append(temp_loss)

        if (epoch % config.SAVE_MODEL_PER_EPOCH == 0 and (epoch != 0 or config.VALIDATE_PER_EPOCH == 1)) or epoch == config.NUM_EPOCHS-1:
            print("Saving Model State Dict...")
            torch.save(model.state_dict(), config.MODEL_PATH)

        text_file = open(config.HISTORY_PATH,"a" if os.path.exists(config.HISTORY_PATH) else "x")
        text_file.write(" ".join([str(values[-1]) for key,values in H.items()])+"\n")
        text_file.close()

        # accumulate predictions from all images
        torch.set_num_threads(n_threads)

    return H

def base():

    trainDS, valDS = get_dataset()

    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(valDS)} examples in the val set...")

    trainLoader, valLoader = get_dataloaders(trainDS, valDS)

    model = get_model()

    lossFunc, opt= get_others(model)

    H = {"train_loss": [], "val_loss": [],"train_acc":[],"val_acc":[]}


    H = training(model,trainLoader,lossFunc,opt,valLoader,H)

    plot(H)

if __name__ == "__main__":
    base()