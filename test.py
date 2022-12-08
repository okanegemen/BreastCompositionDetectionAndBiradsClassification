from DataLoaders.dataset import Dataset
from DataLoaders.XLS_utils import XLS
from ConnectedSegnet.connectedSegnet_model import ConSegnetsModel
import config
import math
import sys
from utils import MetricLogger,SmoothedValue, reduce_dict
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import time
import torchvision
from tqdm import tqdm


def get_model():
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

def training(model, lossFunc, optimizer, testLoader, testSteps):
    # switch off autograd
    with torch.no_grad():

        n_threads = torch.get_num_threads()
        # set the model in evaluation mode
        cpu_device = torch.device("cpu")
        model.eval()
        # loop over the validation set

        test_acc = 0.
        test_counter = 0
        len_testLoader = len(testLoader)
        for idx,testdata in enumerate(testLoader):
            imgs, targets = testdata
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
            loss = lossFunc(outputs,targets)

            test_counter += 1
            test_acc += 1 if torch.argmax(outputs)==torch.argmax(targets) else 0
            temp_acc = test_acc/test_counter

            evaluator_time = time.time()
            evaluator_time = time.time() - evaluator_time
            if idx % config.PRINT_FREQ == 0  and idx != 0:
                print(f"Test:  [{idx}/{len_testLoader}]  test_loss: {loss}  test_acc: {temp_acc}  evaluator_time: {evaluator_time}")

    print(f"Test:  [{idx}/{len_testLoader}]  test_loss: {loss}  test_acc: {temp_acc}  evaluator_time: {evaluator_time}")

    # accumulate predictions from all images
    torch.set_num_threads(n_threads)

    # Val:  [81/82]  eta: 0:00:00  model_time: 0.0408 (0.0406)  val_loss: 0.3412 (0.3879)  val_acc: 0.4697 (0.5080)  evaluator_time: 0.0000 (0.0000)  eta: 0  batch_num: 80  batch_len: 82  time: 0.1450  data: 0.1036  max mem: 1817