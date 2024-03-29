import pydicom
from PIL import Image,ImageOps
import os
import numpy as np
from skimage import exposure
import config
import torchvision.transforms as T
import time
import torch
import imutils
import fiximage
import cv2

MAIN_DIR = "/home/alican/Documents/"
DATASET_DIR = os.path.join(MAIN_DIR,"Datasets/")
TEKNOFEST = os.path.join(DATASET_DIR,"TEKNOFEST_MG_EGITIM_1")

transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)),
                ])

def norm():
    Norm = T.Normalize([0.1525, 0.1502, 0.1543, 0.1522],[0.2215, 0.2315, 0.2231, 0.2336])
    return torch.nn.Sequential(Norm)

def norm_image(norm_imgs):
    image = torch.stack(norm_imgs).squeeze()
    image = norm()(image).unsqueeze(1)
    images = []
    for img in image:    
        image = transform(img.float())
        images.append(image)
    a = get_concat_v(images[0],images[1])
    b = get_concat_v(images[2],images[3])
    c = get_concat_h(b,a)
    return c

def tensor_concat(image):
    images = []
    for img in image:    
        image = transform(img)
        images.append(image)
    a = get_concat_v(images[0],images[1])
    b = get_concat_v(images[2],images[3])
    c = get_concat_h(b,a)
    return c

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
    path = os.path.join(config.TEKNOFEST,path)
    dicom_img = pydicom.dcmread(path)
    numpy_pixels = dicom_img.pixel_array
    img = np.array(numpy_pixels,dtype="float32")
    return img/np.max(img)

def dicom_open_norm(hastano,dcm):
    path = os.path.join(config.TEKNOFEST,str(hastano),dcm+".dcm")
    dicom_img = pydicom.dcmread(path)
    numpy_pixels = dicom_img.pixel_array
    numpy_pixels = imutils.resize(numpy_pixels,height=config.INPUT_IMAGE_HEIGHT*3)
    return numpy_pixels

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

def four_image_show_norm(hastano,w = config.INPUT_IMAGE_HEIGHT,h = config.INPUT_IMAGE_WIDTH):
    dcm_names = ["LMLO","LCC","RMLO","RCC"]
    norm_imgs = []
    images = []

    start = time.time()
    for dcm in dcm_names:
        image = dicom_open_norm(os.path.join(str(hastano)),dcm)

        norm_img = fiximage.fit_image(image)
        norm_img = imutils.resize(norm_img,height = config.INPUT_IMAGE_HEIGHT)
        h,w = norm_img.shape

        if list(dcm)[0] == "R":
            try:
                norm_img = np.pad(norm_img, ((0, 0), (h-w,0)), 'constant')
            except:
                norm_img = norm_img[:,w-h:] # image = image[:,w-h:]
        else:
            try:
                norm_img = np.pad(norm_img, ((0, 0), (0,h-w)), 'constant')
            except:
                norm_img = norm_img[:,:h]
        img = torch.from_numpy(norm_img).unsqueeze(0).float()/ 255.
        images.append(img)
        norm_img =transform(img)
        norm_imgs.append(norm_img)
    print(time.time()-start)
    a = get_concat_v(norm_imgs[0],norm_imgs[1])
    b = get_concat_v(norm_imgs[2],norm_imgs[3])
    c = get_concat_h(b,a)
    return c,images

def four_image_show(hastano,w = config.INPUT_IMAGE_HEIGHT,h = config.INPUT_IMAGE_WIDTH):
    dcm_names = ["LMLO","LCC","RMLO","RCC"]
    images = []
    
    for dcm in dcm_names:
        image = dicom_open(os.path.join(str(hastano),dcm+".dcm"))
        image = transform(torch.from_numpy(image).float().unsqueeze(0))
        images.append(image)
    a = get_concat_v(images[0],images[1])
    b = get_concat_v(images[2],images[3])
    c = get_concat_h(b,a)

    return c

def hastano_from_txt(txt_path = os.path.join(MAIN_DIR,"yoloV5","others","kalan_kirli.txt")):
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
            image.show()
            images.append(image)
            
        a = get_concat_v(images[0],images[1])
        b = get_concat_v(images[2],images[3])

        c = get_concat_h(b,a)
        c.show()
        time.sleep(1)
if __name__ == "__main__":
    hastanos = os.listdir(TEKNOFEST)
    # hastanos = hastano_from_txt()
    k = 0
    for hastano in hastanos[k:]:
        x = four_image_show(hastano)
        y,norm_imgs = four_image_show_norm(hastano)
        images = norm_image(norm_imgs)
        z = get_concat_h(x,y)
        t = get_concat_h(z,images)

        t.show()
        print(k,hastano)
        input()
        k += 1
