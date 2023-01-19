import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(torch.version.cuda)

MAIN_DIR = "/home/robotik/Documents/"
DATASET_DIR = os.path.join(MAIN_DIR,"Datasets/")
TEKNOFEST = os.path.join(DATASET_DIR,"TEKNOFEST_MG_EGITIM_1")

CROP_DATA = 0.
TEST_SPLIT = 0.17
CV_K_FOLDS = 5

MODEL_INPUT_CONCATED = True

NUM_CHANNELS = 4
NUM_CLASSES = 3
NUM_LEVELS = 1


INIT_LR = 0.0001
NUM_EPOCHS = 8
BATCH_SIZE = 32

L1regularization = False
L2regularization = False

SAVE_MODEL_PER_EPOCH = 10
VALIDATE_PER_EPOCH = 4

INPUT_IMAGE_WIDTH = 160 # yatay
INPUT_IMAGE_HEIGHT = 200 # dikey

PRINT_FREQ = None

BASE_OUTPUT = os.path.join(MAIN_DIR,"yoloV5/output")

LOAD_NEW_MODEL = True

MODEL_PATH = os.path.join(BASE_OUTPUT,"model_ResNet.pth")
PLOT_ACC_PATH = os.path.sep.join([BASE_OUTPUT,"plot_acc.png"])
PLOT_LOSS_PATH = os.path.sep.join([BASE_OUTPUT,"plot_loss.png"])
PLOT_TEST = os.path.sep.join([BASE_OUTPUT,"plot_test.png"])