import os
import torchvision.transforms as T
from visualize_one_patient import four_image_show
import time
import gc

gc.collect()

MAIN_DIR = "/home/alican/Documents/"
DATASET_DIR = os.path.join(MAIN_DIR,"Datasets/")
TEKNOFEST = os.path.join(DATASET_DIR,"TEKNOFEST_MG_EGITIM_1")

def hastano_from_txt(txt_path = os.path.join(MAIN_DIR,"yoloV5","others","kirli_resimler.txt")):
    with open(txt_path) as text_file:
        lines = text_file.readlines()
    dcm_folders = [int(line.split("\t")[0]) for line in lines]
    return dcm_folders


if __name__ == "__main__":

    patients = hastano_from_txt()#list(set([int(i) for i in os.listdir(TEKNOFEST) if len(i.split("."))<2]))
    images = []
    k = 0
    for i,folder in enumerate(patients[k:]):
        four_image_show(folder)
        print(k+i,folder)
        a = input()

        # if a == 'n':
        #     with open(os.path.join(DATASET_DIR,"images.txt"), "a") as text_file:
        #         text_file.write(str(folder)+"\n")
