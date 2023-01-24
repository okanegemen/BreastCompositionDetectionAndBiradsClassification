import cv2
import numpy as np
import time
import cv2

def true_norm(img):
    norm = (img - img.min())
    norm = (norm / norm.max())*255
    return norm.astype(np.uint8)

def mask_external_contour(pixels):
    contours, _ = cv2.findContours(pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(pixels.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    x, y = np.nonzero(mask)
    xl,xr = x.min(),x.max()
    yl,yr = y.min(),y.max()
    pixels=pixels[xl:xr+1, yl:yr+1]
    return  pixels

def fit_image(X):
    X = true_norm(X)
    if X.mean()>120.:
        X = 255 - X
    X = X*(X>20)
    X = mask_external_contour(X).astype(np.uint8)
    X = cv2.equalizeHist(X)
    
    return X

if __name__ == "__main__":
    start = time.time()
    fit_image()
    print(time.time()-start)