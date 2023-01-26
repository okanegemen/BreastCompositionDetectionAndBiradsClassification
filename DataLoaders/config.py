import torch,torchvision
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(torch.__version__)

MAIN_DIR = "/home/robotik/Documents/"
DATASET_DIR = os.path.join(MAIN_DIR,"Datasets/")
TEKNOFEST = os.path.join(DATASET_DIR,"TEKNOFEST_MG_EGITIM_1")

CROP_DATA = 0.
TEST_SPLIT = 0.17
CV_K_FOLDS = 5

ELIMINATE_CORRUPTED_PATIENTS = False

MODEL_INPUT_CONCATED = True

NUM_CHANNELS = 4
NUM_CLASSES = 3
NUM_LEVELS = 1


INIT_LR = 0.0001
NUM_EPOCHS = 10
BATCH_SIZE = 32

L1regularization = False
L2regularization = False

SAVE_MODEL_PER_EPOCH = 10
VALIDATE_PER_EPOCH = 4

INPUT_IMAGE_WIDTH = 40 # yatay
INPUT_IMAGE_HEIGHT = 40 # dikey
CROP_RATIO = 0.9
PAD_PIXELS = 7
NORMALIZE = True

PRINT_FREQ = None

BASE_OUTPUT = os.path.join(MAIN_DIR,"yoloV5/output")

LOAD_NEW_MODEL = True

MODEL_PATH = os.path.join(BASE_OUTPUT,"model_Resnet34.pth019.pth")
PLOT_ACC_PATH = os.path.sep.join([BASE_OUTPUT,"plot_acc.png"])
PLOT_LOSS_PATH = os.path.sep.join([BASE_OUTPUT,"plot_loss.png"])
PLOT_TEST = os.path.sep.join([BASE_OUTPUT,"plot_test.png"])