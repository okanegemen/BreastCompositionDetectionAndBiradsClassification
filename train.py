from dataset import SegmentationDataset
from model import UNet
import config
import math
import sys
from vision.coco_eval import CocoEvaluator
from vision.coco_utils import get_coco_api_from_dataset
from vision.utils import MetricLogger,SmoothedValue, reduce_dict
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import torchvision
def get_model(load_last = True):
    if load_last == False:
        return UNet().to(config.DEVICE)
    else:
        return torch.load(config.MODEL_PATH) # load previous weights

def get_dataset():
    imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
    maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

    split = train_test_split(imagePaths, maskPaths,
        test_size=config.TEST_SPLIT, random_state=42)

    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    print("[INFO] saving testing image paths...")
    f = open(config.TEST_PATHS, "w")
    f.write("\n".join(testImages))
    f.close()

    transform = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((config.INPUT_IMAGE_HEIGHT,
            config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()])

    trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
        transforms=transform)
    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
        transforms=transform)

    return trainDS, testDS

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

def concat_to_list(images,targets):
    imgs = [t for t in images]
    keys = targets.keys()
    targ = [{"mask":mask} for mask in targets["masks"]]

    return imgs,targ

def func(x):
    return x

def training(model, trainLoader, lossFunc, optimizer, testLoader, trainSteps, testSteps, H, scaler=None):

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for epoch in range(config.NUM_EPOCHS):
        # set the model in training mode
        model.train()

        metric_logger_train = MetricLogger(delimiter="  ")
        metric_logger_train.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.10f}"))
        header = f"Epoch: [{epoch}]"

        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(trainLoader) - 1)

            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )        
        # loop over the training set
        for images,targets in metric_logger_train.log_every(trainLoader,10,header):
            # send the input to the device
            images = torch.stack([image.to(config.DEVICE) for image in images])
            targets = torch.stack([v.to(config.DEVICE) for v in targets]).squeeze(0)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(images)
                loss_dict = lossFunc(outputs,targets)
                losses = loss_dict

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = {"masks":reduce_dict(loss_dict)}

            losses_reduced = losses

            loss_value = losses

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict_reduced)
                sys.exit(1)


            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger_train.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger_train.update(lr=optimizer.param_groups[0]["lr"])

            params = metric_logger_train.__dict__()
            
        # switch off autograd
        with torch.no_grad():

            n_threads = torch.get_num_threads()
            # set the model in evaluation mode
            cpu_device = torch.device("cpu")
            model.eval()
            # loop over the validation set
            metric_logger_test = MetricLogger(delimiter="  ")
            header = "Test:"   

            for imgs, targets in metric_logger_test.log_every(testLoader, 10, header):
                # send the input to the device
                images = torch.stack([img.to(config.DEVICE) for img in imgs])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                model_time = time.time()
                outputs = model(images)
                outputs = torch.stack([v.to(cpu_device) for v in outputs]).squeeze(0)
                model_time = time.time() - model_time

                losses = lossFunc(outputs,targets)

                evaluator_time = time.time()
                evaluator_time = time.time() - evaluator_time
                metric_logger_test.update(model_time=model_time, evaluator_time=evaluator_time)

        # gather the stats from all processes
        metric_logger_test.synchronize_between_processes()
        print("Averaged stats:", metric_logger_test)

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

    trainLoader, testLoader = get_dataloaders(trainDS, testDS)

    unet = get_model()

    lossFunc, opt, trainSteps, testSteps = get_others(unet, trainDS, testDS)

    H = {"train_loss": [], "test_loss": []}


    H = training(unet,trainLoader,lossFunc,opt,testLoader,trainSteps,testSteps,H)

    plot(H)

    torch.save(unet, config.MODEL_PATH)


if __name__ == "__main__":
    base()