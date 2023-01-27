import cv2
import numpy as np
import time
import cv2
import config
import imutils

def true_norm(img):
    norm = (img - img.min())
    norm = (norm / norm.max())*255
    return norm.astype(np.uint8)

def mask_external_contour(pixels):
    contours, _ = cv2.findContours(pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(pixels.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    mask = imutils.resize(mask,height=int(pixels.shape[0]/10))
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
    w,h = X.shape
    ratio = (4*(1-config.CROP_RATIO)/10)
    X = X[int(config.INPUT_IMAGE_HEIGHT*ratio):int(w-config.INPUT_IMAGE_WIDTH*ratio), int(config.INPUT_IMAGE_HEIGHT*ratio):int(h-config.INPUT_IMAGE_HEIGHT*ratio)]
    w,h = X.shape
    X = true_norm(X)
    if X.mean()>120.:
        X = 255 - X
    # thresh_x = imutils.resize(X.copy(),height=int(X.shape[0]/5))
    # thresh_x = imutils.resize(thresh_x,height=X.shape[0])
    # thresh_x = np.pad(thresh_x, ((0, 0), (X.shape[1]-thresh_x.shape[1],0)), 'constant')
    X = X*(X>57)
    X = mask_external_contour(X).astype(np.uint8)
    X = cv2.equalizeHist(X)
    return X

if __name__ == "__main__":
    start = time.time()
    fit_image()
    print(time.time()-start)