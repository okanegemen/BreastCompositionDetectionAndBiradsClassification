import torch
import os

DATASET_NAME = "VinDr"
DATASET_NAMES = ["INBreast","VinDr"] # available datasets

DATASET_PATH = os.path.join("dataset","train")

IMAGE_DATASET_PATH = os.path.join(DATASET_PATH,"images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH,"masks")
HISTORY_PATH =  "/home/alican/Documents/AnkAI/yoloV5/output/history.txt"

MINIMIZE_IMAGE = True
ONLY_CC = False

CONVERT_BI_RADS = False # Eğer True ise 3 -> 2, 4 ->3, 5->3 olur. birads 6 silinir.

EQUALIZE = False # true enables histogram equalization
AUTO_CONTRAST = False # true enables contrast func

TRAIN_SPLIT = 0.80
TEST_SPLIT = 0.12
VAL_SPLIT = 0.8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

PIN_MEMORY = True if DEVICE == "cuda" else False

NUM_CHANNELS = 1
NUM_CLASSES = 6 if CONVERT_BI_RADS != True else 3
NUM_LEVELS = 1

INIT_LR = 0.00001
NUM_EPOCHS = 18
BATCH_SIZE = 5

SAVE_MODEL_PER_EPOCH = 1
VALIDATE_PER_EPOCH = 1

INPUT_IMAGE_WIDTH = 200 # yatay
INPUT_IMAGE_HEIGHT = 400 # dikey

THRESHOLD = 0.5
PRINT_FREQ = None

BASE_OUTPUT = "output"

LOAD_NEW_MODEL = False
MODEL_PATH = os.path.join(BASE_OUTPUT,"unet_tgs_salt.pth")
PLOT_ACC_PATH = os.path.sep.join([BASE_OUTPUT,"plot_acc.png"])
PLOT_LOSS_PATH = os.path.sep.join([BASE_OUTPUT,"plot_loss.png"])
PLOT_TEST = os.path.sep.join([BASE_OUTPUT,"plot_test.png"])

CM_COLUMNS = ['1','2','3','4','5','6'] if CONVERT_BI_RADS != True else ['1','2','3']

TEST_PATHS = os.path.sep.join([BASE_OUTPUT,"test_paths.txt"])