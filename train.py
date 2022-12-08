from DataLoaders.dataset import Dataset
from DataLoaders.XLS_utils import XLS
from ConnectedSegnet.connectedSegnet_model import ConSegnetsModel
import config
import math
import sys
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import time
import torchvision



def get_model(load_last = False):
    if load_last == False:
        return ConSegnetsModel(3,6).to(config.DEVICE)
    else:
        return torch.load(config.MODEL_PATH) # load previous weights

def get_dataset():
    path = "/home/alican/Documents/AnkAI/yoloV5/INbreast Release 1.0"
    train,test = XLS(path).return_datasets()

    imgs_dir = "/home/alican/Documents/AnkAI/yoloV5/INbreast Release 1.0/images"

    train = Dataset(train,imgs_dir)
    test = Dataset(test,imgs_dir)

    return train, test

def get_dataloaders(trainDS,testDS):
    trainLoader = DataLoader(trainDS, shuffle=True,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=2)
    testLoader = DataLoader(testDS, shuffle=False,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=1)

    return trainLoader, testLoader

def get_others(unet, trainDS, testDS):

    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=config.INIT_LR)

    trainSteps = len(trainDS) // config.BATCH_SIZE
    testSteps = len(testDS) // config.BATCH_SIZE

    return lossFunc,opt,trainSteps,testSteps

def func(x):
    return x

def training(model, trainLoader, lossFunc, optimizer, valLoader, trainSteps, valSteps, H, scaler=None):

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
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

        train_loss = 0
        train_acc = 0.
        train_count = 0
        len_trainLoader = len(trainLoader)
        for idx_t,traindata in enumerate(trainLoader):
            images,targets = traindata
            # send the input to the device
            images = torch.stack([image.to(config.DEVICE) for image in images])
            targets = torch.stack([v.to(config.DEVICE) for v in targets]).float()

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(images)
                loss_train = lossFunc(outputs,targets)

            
            train_count += 1

            train_loss += torch.mean(loss_train)
            temp_loss = train_loss / train_count

            train_acc += 1*config.BATCH_SIZE if torch.argmax(outputs)==torch.argmax(targets) else 0
            temp_acc = train_acc/train_count

            if not math.isfinite(loss_train):
                print(f"Loss is {loss_train}, stopping training")
                print(temp_acc)
                sys.exit(1)


            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss_train).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_train.backward()
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
        
            if idx_t % config.PRINT_FREQ == 0 and idx_t != 0:
                print(f"Epoch: [{epoch}]  [{idx_t}/{len_trainLoader}]  lr: {optimizer.param_groups[0]['lr']}  train_loss: {temp_loss:.4f}  train_acc:{temp_acc:.4f}")

        print(f"Epoch: [{epoch}]  [{idx_t}/{len_trainLoader}]  lr: {optimizer.param_groups[0]['lr']}  train_loss: {temp_loss:.4f}  train_acc:{temp_acc:.4f}")
        # switch off autograd
        with torch.no_grad():

            n_threads = torch.get_num_threads()
            # set the model in evaluation mode
            cpu_device = torch.device("cpu")
            model.eval()
            # loop over the validation set

            val_acc = 0.
            val_loss = 0
            val_counter = 0
            len_valLoader = len(valLoader)
            for idx_v, valData in enumerate(valLoader):
                imgs, targets = valData
                # send the input to the device
                images = torch.stack([img.to(config.DEVICE) for img in imgs])
                targets = torch.stack([target.to(config.DEVICE) for target in targets]).float()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                model_time = time.time()
                outputs = model(images)
                val_loss = lossFunc(outputs,targets)
                outputs = torch.stack([v.to(cpu_device) for v in outputs]).squeeze(0)
                model_time = time.time() - model_time

                targets = targets.squeeze(0).float().to(cpu_device)
                loss_val = lossFunc(outputs,targets)

                val_counter += 1
                val_loss += torch.mean(loss_val)
                temp_loss = val_loss/val_counter

                val_acc += 1 if torch.argmax(outputs)==torch.argmax(targets) else 0
                temp_acc = val_acc/val_counter

                evaluator_time = time.time()
                evaluator_time = time.time() - evaluator_time
                if idx_v % config.PRINT_FREQ == 0 and idx_v != 0:
                    print(f"Val: [{idx_v}/{len_valLoader}]  val_loss: {temp_loss:.4f}  val_acc:{temp_acc:.4f}")

            print(f"Val: [{idx_v}/{len_valLoader}]  val_loss: {temp_loss:.4f}  val_acc:{temp_acc:.4f}")
        # accumulate predictions from all images
        torch.set_num_threads(n_threads)
    return H


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def plot(H):
    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(config.PLOT_PATH)
    # serialize the model to disk

def base():

    trainDS, testDS = get_dataset()

    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    trainLoader, valLoader = get_dataloaders(trainDS, testDS)

    model = get_model()

    lossFunc, opt, trainSteps, valSteps = get_others(model, trainDS, testDS)

    H = {"train_loss": [], "test_loss": []}


    H = training(model,trainLoader,lossFunc,opt,valLoader,trainSteps,valSteps,H)

    plot(H)

    torch.save(model, config.MODEL_PATH)


if __name__ == "__main__":
    base()