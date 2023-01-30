import torch,torchvision
import os
import datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(torch.__version__)


MAIN_DIR = "/home/alican/Documents/"
DATASET_DIR = os.path.join(MAIN_DIR,"Datasets/")
TEKNOFEST = os.path.join(DATASET_DIR,"TEKNOFEST_MG_EGITIM_1")
BASE_OUTPUT = os.path.join(MAIN_DIR,"yoloV5")

DATE = datetime.datetime.now().astimezone().timetuple()
DATE_FOLDER = str(DATE[1])+"_"+str(DATE[2])+"_"+str(DATE[3])+"_"+str(DATE[4])
MID_FOLDER = os.path.join(BASE_OUTPUT,"output")
SAVE_FOLDER = os.path.join(BASE_OUTPUT,"results_models")

CROP_DATA = 0.
TEST_SPLIT = 0.17
K_FOLD = True
CV_K_FOLDS = 4
INIT_LR = 0.0001
NUM_EPOCHS = 8
BATCH_SIZE = 16

ELIMINATE_CORRUPTED_PATIENTS = True # only for train

MODEL_INPUT_CONCATED = True
FREEZE_LAYER = 0.6  # baştan 
SKIP_FREEZE = [1,1,3] # 1 dondur 1 atla son 3 katmana karışma

NUM_CHANNELS = 4
NUM_CLASSES = 3
NUM_LEVELS = 1

CLAHE_CLIP = 2
FOCAL_LOSS = True
L1regularization = False
L2regularization = False

SAVE_MODEL_PER_EPOCH = 3
VALIDATE_PER_EPOCH = 3

INPUT_IMAGE_WIDTH = 100 # yatay
INPUT_IMAGE_HEIGHT = 100 # dikeyweights = models.EfficientNet_V2_L_Weights,pretrained = False
CROP_RATIO = 0.9
PAD_PIXELS = 7
NORMALIZE = False
PRINT_FREQ = None


LOAD_NEW_MODEL = True

MODEL_PATH = os.path.join(BASE_OUTPUT,"results_models/AlexnetCat2_1_29_17_15/AlexnetCat2.pth")
PLOT_ACC_PATH = os.path.sep.join([BASE_OUTPUT,"output/plot_acc.png"])
PLOT_LOSS_PATH = os.path.sep.join([BASE_OUTPUT,"output/plot_loss.png"])
PLOT_TEST = os.path.sep.join([BASE_OUTPUT,"output/plot_test.png"])