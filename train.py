from DataLoaders.dataset import Dataset
from TransferlerarningModels.transfer_learning import Resnet50 as load_model
from DataLoaders.XLS_utils import XLS
# from Pytorch_model.unet import UNet as load_model
# from ConnectedSegnet.connectedSegnet_model import ConSegnetsModel as load_model
import DataLoaders.config as config
import math
import sys
import os
from torch.nn import CrossEntropyLoss as Loss
from torch.optim import Adam,RMSprop,NAdam
from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import random
import time
from qqdm import qqdm, format_str
from DataLoaders.scores import scores
import numpy 
from sklearn.model_selection import KFold
from test import testing
import json

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model():
    if config.LOAD_NEW_MODEL:
        # kwargs = dict({"num_classes":config.NUM_CLASSES})
        model = load_model(4)
        
        # model.conv1.in_channels = config.NUM_CHANNELS
        # model.fc.out_features = config.NUM_CLASSES

        print("Random Weighted ModelS loaded.")
        # print(model)

        return model.to(config.DEVICE)
    else:
        model = load_model(4)
        print("############# Previous weights loaded. ###################")
        model.load_state_dict(torch.load(config.MODEL_PATH))
        
        # print(model.classifier)
        # model.classifier = torch.nn.Linear(1024,config.NUM_CLASSES)

        # for name,param in model.named_parameters():
        #     print(name,param.requires_grad)

        return model.to(config.DEVICE)

def get_dataset():
    train,test = XLS().return_datasets()

    train = Dataset(train,True)
    test = Dataset(test,False)

    return train, test

def get_others(model):

    lossFunc = Loss()
    # opt = RMSprop(model.parameters(),lr=config.INIT_LR)
    opt = Adam(model.parameters(), lr=config.INIT_LR,weight_decay=1e-6)
    print("LossFunc:",lossFunc)
    print("Optimizer:",opt)

    return lossFunc,opt

def plot(H):
    train_epochs = [*range(len(H["train_loss"]))]
    val_epochs = [epoch for epoch in train_epochs if epoch%config.VALIDATE_PER_EPOCH==0]

    plt.plot(train_epochs,H["train_acc"])
    plt.plot(val_epochs,H["val_acc"])
    plt.savefig(config.PLOT_ACC_PATH)
    plt.clf()

    plt.plot(train_epochs,H["train_loss"])
    plt.plot(val_epochs,H["val_loss"])
    plt.savefig(config.PLOT_LOSS_PATH)


def training(model, trainLoader, lossFunc, optimizer, valLoader,fold):
    metrics = {"train":[],"val":[]}

    # loop over epochs
    print("[INFO] training the network...")
    for epoch in range(config.NUM_EPOCHS):
        scores_train = scores()
        # set the model in training mode
        model.train()

        lr_scheduler = None
        # if epoch == 0:
        #     warmup_factor = 1.0 / 1000
        #     warmup_iters = min(1000, len(trainLoader) - 1)

        #     lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        #     )
        

        # loop over the training set

        train_loss = []
        for idx_t,traindata in enumerate(tw :=qqdm(trainLoader, desc=format_str('bold', 'Description'))):
            images,targets = traindata
            # send the input to the device
            images, targets = torch.stack(images).to(config.DEVICE), torch.stack(targets).view(-1).to(config.DEVICE)

            with torch.cuda.amp.autocast():
                outputs = model(images)["birads"]
                loss_train = lossFunc(outputs,targets)
            
            optimizer.zero_grad()

            train_loss.append(loss_train.item())
            temp_loss = sum(train_loss[-20:]) / min([len(train_loss),20])

            if not math.isfinite(loss_train):
                print(f"Loss is {loss_train}, stopping training")
                sys.exit(1)

            loss_train.backward()
            optimizer.step()

            scores_train.update(outputs,targets)
            tw.set_infos({"Fold":fold,
                            "Epoch":f"{epoch}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.7f}",
                            "loss":"%.4f"%temp_loss,
                            **scores_train.metrics()})

            if lr_scheduler is not None:
                lr_scheduler.step()
        
        n_threads = torch.get_num_threads()

        metrics["train"].append(scores_train.metric)
        
        if epoch % config.VALIDATE_PER_EPOCH == 0 and (epoch != 0 or config.VALIDATE_PER_EPOCH == 1):
            print(f'Validation')
            print('--------------------------------')
            scores_val = scores()
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            val_loss = []

            # switch off autograd
            with torch.no_grad():
                for idx_v, valData in enumerate(tw :=qqdm(valLoader, desc=format_str('bold', 'Description'))):
                    images, targets = valData

                    # send the input to the device
                    images, targets = torch.stack(images).to(config.DEVICE), torch.stack(targets).view(-1).to(config.DEVICE)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    outputs = model(images)["birads"]
                    loss_val = lossFunc(outputs,targets)

                    val_loss.append(loss_val.item())
                    temp_loss = sum(val_loss[-20:]) / min([len(val_loss),20])

                    scores_val.update(outputs,targets)
                    tw.set_infos({"loss":"%.4f"%temp_loss,
                                **scores_val.metrics()})

        if (epoch % config.SAVE_MODEL_PER_EPOCH == 0 and (epoch != 0 or config.VALIDATE_PER_EPOCH == 1)) or epoch == config.NUM_EPOCHS-1:
            print("/nSaving Model State Dict...")
            torch.save(model.state_dict(), config.MODEL_PATH.strip(".pth")+"_fold"+str(fold)+"_epoch"+str(epoch)+".pth")

        # accumulate predictions from all images
        torch.set_num_threads(n_threads)
    return metrics

def save_model_and_metrics(model,fold_metrics):
    print("/nSaving Model...")
    name = "model_"+model.__class__.__name__+".pth"
    print(name)
    if name not in os.listdir(config.BASE_OUTPUT):
        torch.save(model.state_dict(), os.path.join(config.BASE_OUTPUT,name))
    
    jso = json.dumps(fold_metrics)
    f = open(f"output/metrics_{model.__class__.__name__}.json","a")
    f.write(jso)
    f.close()
    

def get_dataloaders(train_valDS,train_sampler,val_sampler):
    trainLoader = DataLoader(train_valDS,sampler=train_sampler, shuffle=False, batch_size=config.BATCH_SIZE, num_workers=0,collate_fn=collate_fn)
    valLoader = DataLoader(train_valDS,sampler=val_sampler, shuffle=False, batch_size=config.BATCH_SIZE, num_workers=0,collate_fn=collate_fn)

    return trainLoader, valLoader
    
def base():
    train_valDS, testDS = get_dataset()
    model = get_model()
    lossFunc, opt= get_others(model)

    print(f"[INFO] found {len(train_valDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")
    
    total_time_start = time.time()

    folds_metrics = {"training":[],"test":[]}
    kfold = KFold(n_splits=config.CV_K_FOLDS, shuffle=True)

    testLoader = DataLoader(testDS,config.BATCH_SIZE,shuffle=False,sampler=testDS.sampler,num_workers=0,collate_fn=collate_fn)


    for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_valDS)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(valid_ids)
        trainLoader, valLoader = get_dataloaders(train_valDS,train_sampler, val_sampler)

        training_metrics = training(model,trainLoader,lossFunc,opt,valLoader,fold)
        folds_metrics["training"].append(training_metrics)

        test_metrics = testing(model,lossFunc,testLoader)
        folds_metrics["test"].append(test_metrics)

    total_time = int(time.time()-total_time_start)/60
    print(f"---------- Training_time:{total_time} minute ----------")
    save_model_and_metrics(model,folds_metrics)
    
if __name__ == "__main__":
    base()