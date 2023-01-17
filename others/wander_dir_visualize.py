import os
import torchvision.transforms as T
from others.visualize_one_patient import four_image_show
import time
import gc

gc.collect()

MAIN_DIR = "/home/alican/Documents/"
DATASET_DIR = os.path.join(MAIN_DIR,"Datasets/")
TEKNOFEST = os.path.join(DATASET_DIR,"TEKNOFEST_MG_EGITIM_1")

if __name__ == "__main__":

    patients = list(set([int(i) for i in os.listdir(TEKNOFEST) if len(i.split("."))<2]))
    images = []
    k = 3924
    for i,folder in enumerate(patients[k:]):
        four_image_show(folder)
        print(k+i,folder)
        a = input()
        if a == 'n':
            with open(os.path.join(DATASET_DIR,"images.txt"), "a") as text_file:
                text_file.write(str(folder)+"\n")
