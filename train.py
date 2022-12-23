from DataLoaders.dataset import Dataset
from DataLoaders.XLS_utils import XLS
from Pytorch_model.unet import UNet as load_model
# from ConnectedSegnet.connectedSegnet_model import ConSegnetsModel as load_model
import DataLoaders.config as config
import math
import sys
import os
from torch.nn import CrossEntropyLoss as Loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import random
import time
from qqdm import qqdm, format_str
from DataLoaders.scores import scores

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model():
    if config.LOAD_NEW_MODEL:
        model = load_model(config.NUM_CHANNELS,config.NUM_CLASSES).to(config.DEVICE)
        print("Random Weighted Model loaded.")
        return model
    else:
        model = load_model(config.NUM_CHANNELS,config.NUM_CLASSES).to(config.DEVICE)
        model.load_state_dict(torch.load(os.path.join(config.LOAD_MODEL_DIR,"model.pth")))
        print("############# Previous weights loaded. ###################")
        return model

def get_dataset():
    train,test,imgs_dir = XLS().get_all_info()

    train = Dataset(train,imgs_dir,True)
    test = Dataset(test,imgs_dir,False)

    return train, test

def get_dataloaders(trainDS,testDS):
    trainLoader = DataLoader(trainDS,sampler=trainDS.sampler, shuffle=False,
        batch_size=config.BATCH_SIZE, num_workers=0,collate_fn=collate_fn)
    testLoader = DataLoader(testDS,sampler=testDS.sampler, shuffle=False,
        batch_size=config.BATCH_SIZE, num_workers=0,collate_fn=collate_fn)

    return trainLoader, testLoader

def get_others(model):

    lossFunc = Loss()
    opt = Adam(model.parameters(), lr=config.INIT_LR)

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


def training(model, trainLoader, lossFunc, optimizer, valLoader):
    scores_train = scores()

    # loop over epochs
    print("[INFO] training the network...")
    for epoch in range(config.NUM_EPOCHS):
        # set the model in training mode
        model.train()

        lr_scheduler = None
        # if epoch == 0:
        #     warmup_factor = 1.0 / 1000
        #     warmup_iters = min(1000, len(trainLoader) - 1)

        #     lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         optimizer,mode="min",
        #     )        

        # loop over the training set

        train_loss = []
        for idx_t,traindata in enumerate(tw :=qqdm(trainLoader, desc=format_str('bold', 'Description'))):
            images,targets = traindata
            # send the input to the device
            # images = torch.stack([image.to(config.DEVICE) for image in images])
            # targets = torch.stack([v.to(config.DEVICE) for v in targets]).float()
            images, targets = torch.stack(images).to(config.DEVICE), torch.stack(targets).view(-1).to(config.DEVICE)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss_train = lossFunc(outputs,targets)
            

            train_loss.append(loss_train.item())
            temp_loss = sum(train_loss[-20:]) / min([len(train_loss),20])

            if not math.isfinite(loss_train):
                print(f"Loss is {loss_train}, stopping training")
                sys.exit(1)

            loss_train.backward()
            optimizer.step()

            scores_train.update(outputs,targets)
            tw.set_infos({"Epoch":f"{epoch}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.5f}",
                            "loss":"%.4f"%temp_loss,
                            **scores_train.metrics()})

            if lr_scheduler is not None:
                lr_scheduler.step()
        
        n_threads = torch.get_num_threads()
        
        if epoch % config.VALIDATE_PER_EPOCH == 0 and (epoch != 0 or config.VALIDATE_PER_EPOCH == 1):
            scores_test = scores()
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
                    outputs = model(images)
                    loss_val = lossFunc(outputs,targets)

                    val_loss.append(loss_val.item())
                    temp_loss = sum(val_loss[-20:]) / min([len(val_loss),20])


                    scores_test.update(outputs,targets)
                    tw.set_infos({"loss":"%.4f"%temp_loss,
                                **scores_test.metrics()})

        if (epoch % config.SAVE_MODEL_PER_EPOCH == 0 and (epoch != 0 or config.VALIDATE_PER_EPOCH == 1)) or epoch == config.NUM_EPOCHS-1:
            print("Saving Model State Dict...")
            torch.save(model.state_dict(), config.MODEL_PATH)

        # accumulate predictions from all images
        torch.set_num_threads(n_threads)

def base():

    trainDS, valDS = get_dataset()

    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(valDS)} examples in the val set...")

    trainLoader, valLoader = get_dataloaders(trainDS, valDS)

    model = get_model()

    lossFunc, opt= get_others(model)

    training(model,trainLoader,lossFunc,opt,valLoader)

if __name__ == "__main__":
    base()