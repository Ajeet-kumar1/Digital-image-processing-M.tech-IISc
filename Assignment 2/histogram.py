#import cv2
#import matplotlib.pyplot as plt
import numpy as np

# Find the histogram of an image
def histogram(img):
    row, col = img.shape
    # Declare a array of zeros of lenth 256(Pixel values)
    y = np.zeros(256, dtype = int)
    x = np.arange(0,256)
    for i in range(0,row):
        for j in range(0,col):
            y[img[i,j]] = y[img[i,j]]+ 1
    return x, y, row, col, img



#if __name__=='__main__':
    ##img = cv2.imread('ECE.png')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #x,y,_,_, _ = histogram(img)
    #plt.bar(x,y)
    #plt.show()