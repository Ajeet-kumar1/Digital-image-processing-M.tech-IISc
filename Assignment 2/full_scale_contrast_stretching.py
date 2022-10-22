#Import packages and histogram function
import numpy as np
import matplotlib.pyplot as plt
import cv2
from histogram import histogram

# You may get the condition where B-A(Defined below in line 20) could be zero. Division with zero give error, to avaoid that use below line.
np.seterr(divide='ignore', invalid='ignore')

# Define streching function
def streching(img):
   #Get the histogram and find minimum and max pixel value in image
   x, y, row, col, img  = histogram(img)
   A = np.amin(img)
   B = np.amax(img)

   # Now apply the formula of streching in each pixel
   for i in range(row):
     for j in range(col):
      img[i,j] = 255/(B-A) * (img[i,j] -A)

   # Again evaluate histogram of streched image
   x1,y1,_,_,_ = histogram(img)
   return x, y, img, x1, y1
 





##################################### Driver code with command ###############################################
if __name__== '__main__':
   # Read the ECE.png image and convert into gray scale
    img_ECE = cv2.imread('images\ECE.png')
    img11=img_ECE
    img_ECE_gray = cv2.cvtColor(img_ECE, cv2.COLOR_BGR2GRAY)

    # Read the IIScMain.png and convert into gray scale
    img_IISc = cv2.imread('images\IIScMain.png')
    img_IISc_gray = cv2.cvtColor(img_IISc, cv2.COLOR_BGR2GRAY)
    
    # Find histogram and do the streching of image and plot original and streched histogram
    x,y,J, x1,y1 = streching(img_ECE_gray)
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.bar(x,y)
    plt.title('Original bar graph of ECE.png')
    plt.subplot(2,2,2)
    plt.bar(x1,y1)
    plt.title('Streched of ECE.png')
   
   # Also show the original and streched picture
    plt.figure(2)
    plt.subplot(2,2,1)
    plt.imshow(img11,cmap='gray',vmin=0,vmax=255)
    plt.title('original of ECE.png')
    plt.subplot(2,2,2)
    plt.imshow(J,cmap='gray',vmin=0,vmax=255)
    plt.title('Stretched of ECE.png')
    
    # Now do the streching on IIScMain image also and plot both histograms
    x,y,J, x1,y1 = streching(img_IISc_gray)
    plt.figure(1)
    plt.subplot(2,2,3)
    plt.bar(x,y)
    plt.title('Original of IIScMain')
    plt.subplot(2,2,4)
    plt.bar(x1,y1)
    plt.title('Stretched of IIScMain')
   
   # Display the original and streched image
    plt.figure(2)
    plt.subplot(2,2,3)
    plt.imshow(img_IISc_gray,cmap='gray',vmin=0,vmax=255)
    plt.title('Original of IIScMain')
    plt.subplot(2,2,4)
    plt.imshow(J,cmap='gray',vmin=0,vmax=255)
    plt.title('Stretched of IIScMain')
    plt.show()
    




