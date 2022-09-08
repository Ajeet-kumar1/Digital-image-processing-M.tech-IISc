# import all libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt


######################Histogram function###################

def histogram_function(img):
   row, col = img.shape
   y = [0 for i in range(256)]
   for i in range(0,row):
      for j in range(0,col):
         y[img[i,j]] = y[img[i,j]]+ 1
   x = [k for k in range(256)]
   return x, y
 

if __name__== '__main__':
    img1 = cv2.imread('GulmoharMarg.png')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    x,y = histogram_function(img1)
    plt.subplot(3,2,1)
    plt.bar(x,y)
    plt.xlabel("Gray scale value k)")
    plt.title("Histogram by manual code")
   
    
    
    img2 = cv2.imread('GulmoharMargDark.png')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    x,y = histogram_function(img2)
    plt.subplot(3,2,3)
    plt.bar(x,y)
    plt.xlabel("Gray scale value k)")
    plt.ylabel("Number of pixel having gray scale H(k)")
    
  

    img3 = cv2.imread('GulmoharMargBright.png')
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    x,y = histogram_function(img3)
    plt.subplot(3,2,5)
    plt.bar(x,y)
    plt.xlabel("Gray scale value k)")
    

    ######################################Library code for Histogram #############################
    plt.subplot(3,2,2)
    hist = cv2.calcHist([img1], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title("Histogram by built-in function")

    plt.subplot(3,2,4)
    hist = cv2.calcHist([img2], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.ylabel("Number of pixel having gray scale H(k)")

    plt.subplot(3,2,6)
    hist = cv2.calcHist([img3], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.xlabel("Gray scale value k)")
    plt.show()
    