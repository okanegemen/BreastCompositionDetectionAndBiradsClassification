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
MID_FOLDER = os.path.join(BASE_OUTPUT,"output") # model eğitilirken kaydedilen modeller buraya gider
SAVE_FOLDER = os.path.join(BASE_OUTPUT,"results_models") # eğitim bitince model ve dosyalar buraya kaydedilir

CROP_DATA = 0.                              # 0-1 arasında değer girilir. girilen değerin % si kadar veriyi yok sayar
TEST_SPLIT = 0.17                           # test oranı

FREEZE_LAYER = 0.                           # baştan % kaç layer ın freeze edeceğini belirtir.

NUM_CHANNELS = 1                            # dicomların kaç channel olarak yükleneceğini belirler
NUM_CLASSES = 3

TEKRAR = 3                                  # aynı eğitimi aynı modeli devam ettirerek kaç defa eğitileceği
CV_K_FOLDS = 4                              # 2 den küçük sayı girilirse sadece train ve test yapılır
NUM_EPOCHS = 3                              # her fold da olacak epoch sayısı
INIT_LR = 0.0001                            # learning rate
BATCH_SIZE = 64 

CAT_MODEL = True

SAVE_MODEL_PER_FOLD = 2                     # CV için her kaç foldda save edeceğini belirtir
VALIDATE_PER_EPOCH = 1                      # val kaç epochta bir olacağını belirtir

ELIMINATE_CORRUPTED_PATIENTS = True         # kirli verileri sadece train verisinden çıkartır

CLAHE_CLIP = 2                              # clahe fonksiyonunun etki miktarını belirler
FOCAL_LOSS = True                           # cross entropy üzerine focal loss kullanır

L1regularization = False
L2regularization = False

INPUT_IMAGE_WIDTH = 80                      # yatay
INPUT_IMAGE_HEIGHT = 80                     # dikey

CROP_RATIO = 0.9                            # Resmin kırpılma oranına etki eder

NORMALIZE = True                            # Normalize kullanır

LOAD_NEW_MODEL = True

MODEL_PATH = os.path.join(BASE_OUTPUT,"results_models/","AlexnetCat2_1_31_12_29/AlexnetCat2.pth")