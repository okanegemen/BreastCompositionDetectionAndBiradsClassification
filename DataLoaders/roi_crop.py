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

    h,w = X.shape
    ratio = (4*(1-config.CROP_RATIO)/10)
    X = X[int(config.INPUT_IMAGE_HEIGHT*0.95):,:]
    h_c,w_c = X.shape
    X = true_norm(X)
    if X.mean()>120.:
        X = 255 - X

    clahe = cv2.createCLAHE(clipLimit = 2)
    X = clahe.apply(X)

    X = X*(X>23) #57
    clahe = cv2.createCLAHE(clipLimit = 20)
    X = clahe.apply(X)

    X = X*(X>23) #57
    X = mask_external_contour(X).astype(np.uint8)
    X = cv2.equalizeHist(X)

    clahe = cv2.createCLAHE(clipLimit = 20)
    X = clahe.apply(X)

    h,w = X.shape
    if h/w>2:
        X = cv2.resize(X, dsize=(w,int(w*2)), interpolation=cv2.INTER_CUBIC)
    h,w = X.shape

    X = X*(X>23) #57
    length = 15
    L = (imutils.resize(X,height=length)>20).astype(np.uint8)
    ratio = int(h_c/length)

    u_i = 0
    l_i = length
    for i in range(int(length/4)):
        if 5>sum(L[i]):
            u_i =i
        if 5>sum(L[length-1-i]):
            l_i = length-1-i

    # L = np.transpose(L)
    # length_T = L.shape[0]
    # ratio_T = int(w_c/length_T +1)

    # le_i = 0
    # ri_i = length_T
    # for i in range(int(length_T/2)):
    #     if sum(L[i])!=0 and le_i==0:
    #         le_i =i
    #     if sum(L[length_T-1-i])!=0 and ri_i>length_T-1:
    #         ri_i = length_T-1-i
    a = int((u_i-2)*ratio)
    b = int((l_i+1)*ratio)
    if a<0:
        a = 0
    if b>h_c:
        b = h_c

    X = X[a:b,:] #int((le_i-2)*ratio_T):int((ri_i+1)*ratio_T)
    # X = imutils.resize(X,height = config.INPUT_IMAGE_HEIGHT)
    # h,w = X.shape
    # if h/w>2.7:
    #     X = X[int(h-w*2.7):,:]
    # L = (imutils.resize(X,height=length)>0).astype(np.uint8)

    # h,w = X.shape
    # ratio = h/L.shape[0]
    # del_up_pix = 0
    # a = 0.00
    # l_h,l_w = L.shape
    # while True:
    #     g_h,g_w = ndi.center_of_mass(L)
    #     # g_h = int(g_h+0.5)
    #     if len(L)*0.48<g_h:
    #         break
    #     else:
    #         L = L[1:,:]
    #         del_up_pix += 1
    
    # try:
    #     a =  int((del_up_pix-1)*ratio)
    #     if a<0:
    #         raise 
    # except:
    #     a = 0

    # try:
    #     b =  int((del_up_pix+g_h*2+1)*ratio)
    #     if b>h:
    #         raise 
    # except:
    #     b=int(l_w*ratio)

    # try:
    #     c =  int(g_w+w*0.4-w/2)
    #     if c<0:
    #         raise 
    # except:
    #     c = 0

    # try:
    #     d =  int(g_w+w*0.4+w/2)
    #     if d>w:
    #         raise 
    # except:
    #     d=w
    # t = X[a:b,:]
    # if t.shape[0]*t.shape[1]<200:
    #     pass
    # else:
    #     X = t

    # cv2.imshow("img",X)
    # cv2.waitKey(0)

    # X = cv2.equalizeHist(X)
    # clahe = cv2.createCLAHE(clipLimit = config.CLAHE_CLIP)
    # X = clahe.apply(X)
    return X

if __name__ == "__main__":
    start = time.time()
    fit_image()
    print(time.time()-start)