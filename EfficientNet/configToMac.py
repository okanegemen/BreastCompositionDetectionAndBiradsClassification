import torch
import os

DATASET_PATH = os.path.join("dataset","train")

IMAGE_DATASET_PATH = os.path.join(DATASET_PATH,"images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH,"masks")

TRAIN_SPLIT = 0.70
TEST_SPLIT = 0.20
VAL_SPLIT = 0.10

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PIN_MEMORY = True if DEVICE == "mps" else False

NUM_CHANNELS = 1
NUM_CLASSES = 2
NUM_LEVELS = 1

INIT_LR = 0.0001
NUM_EPOCHS = 10
BATCH_SIZE = 1

INPUT_IMAGE_WIDTH = 700
INPUT_IMAGE_HEIGHT = 700

THRESHOLD = 0.5
PRINT_FREQ = 20

BASE_OUTPUT = "output"

MODEL_PATH = os.path.join(BASE_OUTPUT,"unet_tgs_salt.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT,"plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT,"test_paths.txt"])
