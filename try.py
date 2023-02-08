import cv2
import numpy as np
import time
import cv2
import imutils
import scipy.ndimage as ndi
# import config

# if __name__ == "__main__":
#     import config
# else:
#     import DataLoaders.config as config
    
def true_norm(img):
    norm = (img - img.min())
    norm = (norm / norm.max())*255
    return norm.astype(np.uint8)

def mask_external_contour(pixels):
    contours, _ = cv2.findContours(pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(pixels.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    mask = imutils.resize(mask,height=int(pixels.shape[0]/config.INPUT_IMAGE_HEIGHT*15))
    mask = imutils.resize(mask,height=pixels.shape[0])
    mask = np.pad(mask, ((0, 0), (pixels.shape[1]-mask.shape[1],0)), 'constant')
    mask = np.array(mask>0)
    pixels = pixels*mask
    x, y = np.nonzero(mask)
    xl,xr = x.min(),x.max()
    yl,yr = y.min(),y.max()
    pixels=pixels[xl:xr+1, yl:yr+1]
    return  pixels

def fit_image(X):

    h,w = X.shape
    ratio = (4*(1-config.CROP_RATIO)/10)
    X = X[int(config.INPUT_IMAGE_HEIGHT*ratio):int(h-config.INPUT_IMAGE_WIDTH*ratio), int(config.INPUT_IMAGE_HEIGHT*ratio):int(w-config.INPUT_IMAGE_HEIGHT*ratio)]
    h_c,w = X.shape
    X = true_norm(X)
    if X.mean()>120.:
        X = 255 - X

    X = X*(X>23) #57
    X = mask_external_contour(X).astype(np.uint8)

    length = 15
    L = (imutils.resize(X,height=length)>15).astype(np.uint8)
    cv2.imshow("img",L)
    cv2.waitKey(0)
    L = np.pad(L, ((0,length+2), (0,0)), 'constant')

    l_h,l_w = L.shape
    ratio_T = h_c/l_h
    u_v = 0
    l_v = length

    u_i = 0
    l_i = length
    for i,l in enumerate(L):
        if  u_v>=sum(l) and i<length/2:
            u_v = sum(l)
            u_i = i
        elif l_v>=sum(l) and i>length/2:
            l_v = sum(l)
            l_i = i

    cv2.imshow("img",L)
    cv2.waitKey(0)
    L = np.transpose(L)

    cv2.imshow("img",L)
    cv2.waitKey(0)
    length_T = L.shape[1]
    le_v = length_T
    ri_v = length_T

    le_i = 0
    ri_i = length_T
    print(ri_i)
    for i,l in enumerate(L):
        if  le_v>=sum(l) and i<length_T/2:
            le_v = sum(l)
            le_i = i
        elif ri_v>=sum(l) and i>length_T/2:
            ri_v = sum(l)
            ri_i = i
    X = X[int((u_i+1)*ratio):int((l_i)*ratio),int((ri_i+1)*ratio_T):int((le_i)*ratio_T)]
    X = imutils.resize(X,height = config.INPUT_IMAGE_HEIGHT)


    # clahe = cv2.createCLAHE(clipLimit = 2*config.CLAHE_CLIP)
    # X = clahe.apply(X)

    # X = cv2.equalizeHist(X)
    # clahe = cv2.createCLAHE(clipLimit = config.CLAHE_CLIP)
    # X = clahe.apply(X)
    return X

if __name__ == "__main__":
    # start = time.time()
    # fit_image()
    # print(time.time()-start)
    img = cv2.imread("/home/alican/Documents/Datasets/TeknofestPNG/822670340/LCC.png")
    cv2.imshow("img", img*255)
    cv2.waitKey(0)