import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

MAIN_DIR = "/home/alican/Documents/"
DATASET_DIR = os.path.join(MAIN_DIR,"Datasets/")
VINDR_DIR = os.path.join(DATASET_DIR,"VinDr-mammo")
DATASET_NAME = "VinDr" # available datasets VinDr

DATASET_PATH = os.path.join("dataset","train")

IMAGE_DATASET_PATH = os.path.join(DATASET_PATH,"images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH,"masks")
HISTORY_PATH =  "output/history.txt"

MINIMIZE_IMAGE = True
IGNORE_SIDE_PIXELS = 100 

CONVERT_BI_RADS = True # Eğer True ise df['Bi-Rads'] = df['Bi-Rads'].replace([0,1,2,4, 5], [0,1,1,2,2]). Bi-Rads 3 ve 6 silinir

EQUALIZE = False # true enables histogram equalization
AUTO_CONTRAST = False # true enables contrast func

TRAIN_SPLIT = 0.80
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1

PIN_MEMORY = True if DEVICE == "cuda" else False

NUM_CHANNELS = 3
NUM_CLASSES = 3
NUM_LEVELS = 1

CV_K_FOLDS = 5
INIT_LR = 0.0001
NUM_EPOCHS = 1
BATCH_SIZE = 32

SAVE_MODEL_PER_EPOCH = 3
VALIDATE_PER_EPOCH = 3

INPUT_IMAGE_WIDTH = 120 # yatay
INPUT_IMAGE_HEIGHT = 210 # dikey

THRESHOLD = 0.5
PRINT_FREQ = None

BASE_OUTPUT = os.path.join(MAIN_DIR,"yoloV5/output")

LOAD_NEW_MODEL = False

MODEL_PATH = os.path.join(BASE_OUTPUT,"model_DenseNet.pth")
PLOT_ACC_PATH = os.path.sep.join([BASE_OUTPUT,"plot_acc.png"])
PLOT_LOSS_PATH = os.path.sep.join([BASE_OUTPUT,"plot_loss.png"])
PLOT_TEST = os.path.sep.join([BASE_OUTPUT,"plot_test.png"])

CM_COLUMNS = ['0','1-2','4-5']

TEST_PATHS = os.path.sep.join([BASE_OUTPUT,"test_paths.txt"])