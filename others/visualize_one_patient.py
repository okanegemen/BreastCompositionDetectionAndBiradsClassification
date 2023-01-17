import pydicom
from PIL import Image,ImageOps
import os
import numpy as np
from skimage import exposure
import time

MAIN_DIR = "/home/alican/Documents/"
DATASET_DIR = os.path.join(MAIN_DIR,"Datasets/")
TEKNOFEST = os.path.join(DATASET_DIR,"TEKNOFEST_MG_EGITIM_1")

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

def dicom_open(path):
    name = pydicom.data.data_manager.get_files(TEKNOFEST,path)[0]
    
    ds = pydicom.dcmread(name)
    img = ds.pixel_array
    img = np.array(img).astype(np.float64)
    print(img.max())
    return (img-img.min())/img.max()

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def four_image_show(hastano,w = 120,h = 150):
    dcm_names = ["LMLO","LCC","RMLO","RCC"]
    images = []

    for dcm in dcm_names:
        image = dicom_open(os.path.join(str(hastano),dcm+".dcm"))
        image = Image.fromarray(image*255)
        image = image.convert('L')
        image = image.resize([w,h],Image.Resampling.LANCZOS)
        images.append(image)
    a = get_concat_v(images[0],images[1])
    b = get_concat_v(images[2],images[3])

    c = get_concat_h(b,a)
    c.show()

def hastano_from_txt(txt_path = os.path.join(MAIN_DIR,"yoloV5","others","kirli_resimler.txt")):
    with open(txt_path) as text_file:
        lines = text_file.readlines()
    dcm_folders = [int(line.split("\t")[0]) for line in lines]
    return dcm_folders

def hist_eq(image):
    hist, bins = exposure.histogram(image, nbins=256, normalize=False)
    # append any remaining 0 values to the histogram
    hist = np.hstack((hist, np.zeros((255 - bins[-1])))) 
    cdf = 255*(hist/hist.sum()).cumsum()
    equalized = cdf[image].astype(np.uint8)

    return equalized

def four_concat(dcm_folders, dcm_names = ["LMLO","LCC","RMLO","RCC"]):
    for hastano in dcm_folders:
        images = []
        for dcm in dcm_names:
            image = dicom_open(os.path.join(str(hastano),dcm+".dcm"))
            image = Image.fromarray(image*255)
            image = image.convert('RGB')
            images.append(image)
        a = get_concat_v(images[0],images[1])
        b = get_concat_v(images[2],images[3])

        c = get_concat_h(b,a)
        c.show()
        time.sleep(1)
if __name__ == "__main__":
    four_image_show(845285991,w = 1000,h=1250)