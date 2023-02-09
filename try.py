import cv2
import numpy as np
import cv2
from matplotlib import pyplot as plt
def draw_image_histogram(image, channels, color='k'):
    hist = cv2.calcHist([image], channels, None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

def show_grayscale_histogram(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    draw_image_histogram(grayscale_image, [0])
    plt.show()

img = cv2.imread('/home/alican/Documents/yoloV5/normalized_breast.PNG',0)
show_grayscale_histogram(img)
# print(X.shape)
# cv2.imshow("img",X)
# cv2.waitKey(0)