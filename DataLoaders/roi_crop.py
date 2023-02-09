import cv2
import numpy as np
import time
import cv2
import imutils
import scipy.ndimage as ndi
# import config

if __name__ == "__main__":
    import config
else:
    import DataLoaders.config as config
    
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
    pixels=pixels[xl:xr, yl:yr]
    return  pixels

def fit_image(X):
    # start = time.time()

    h,w = X.shape
    X = X[int(config.INPUT_IMAGE_HEIGHT*0.15):,:]

    X = true_norm(X)
    if X.mean()>120.:
        X = 255 - X

    # clahe = cv2.createCLAHE(clipLimit = 2,tileGridSize=(3,3))
    # X = clahe.apply(X)
    X = (X/255)**3
    X = true_norm(X)

    X = mask_external_contour(X).astype(np.uint8)
    X = np.sqrt(X)
    X = true_norm(X)

    clahe = cv2.createCLAHE(clipLimit = 10,tileGridSize=(50,50))
    X = clahe.apply(X)

    # X = cv2.Laplacian(X,cv2.CV_64F, ksize=5)

    # gaussian blur
    X = cv2.medianBlur(X,3)
    # X = cv2.GaussianBlur(X,(5,5),0)
    X = cv2.fastNlMeansDenoising(X, None, 8, 8, 8) 

    X = (X/255)**2
    X = true_norm(X)
    # kernel_sharpening = np.array([[-1,-1,-1],
    #                           [-1, 9,-1],
    #                           [-1,-1,-1]])

    # X = cv2.filter2D(X, -1, kernel_sharpening)

    # X = cv2.calcHist(X,[0], None, [256], [0,256])
    # clahe = cv2.createCLAHE(clipLimit = 20)
    # X = clahe.apply(X)

    # X = mask_external_contour(X).astype(np.uint8)
    h,w = X.shape

    if h/w>2:
        X = cv2.resize(X, dsize=(w,int(w*2)), interpolation=cv2.INTER_CUBIC)

    # length = 15
    # L = (imutils.resize(X,height=length)>0).astype(np.uint8)
    # ratio = int(h_c/length)

    # u_i = 0
    # l_i = length
    # for i in range(int(length/2)):
    #     if (length/4)>sum(L[i]):
    #         u_i =i
    #     if (length/4)>sum(L[length-1-i]):
    #         l_i = length-1-i

    # a = int((u_i)*ratio)
    # b = int((l_i)*ratio)
    # if a<0:
    #     a = 0
    # if b>h_c:
    #     b = h_c

    # X = X[a:b,:]
    clahe = cv2.createCLAHE(clipLimit = 2)
    X = clahe.apply(X)
    X = true_norm(X)
    # print(time.time()-start)
    
    # cv2.imshow("img",X)
    # cv2.waitKey(0)
    return X

if __name__ == "__main__":
    start = time.time()
    fit_image()
    print(time.time()-start)