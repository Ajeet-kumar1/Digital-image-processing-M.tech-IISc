import numpy as np
import cv2
import matplotlib.pyplot as plt
from histogram_computation import histogram_function



#################################### Mathematical function for otsu's binarization ####################################

def otsus_binarization(image_gray):
    within = []
    between = []
    total_variance = []
    d =0
    row, col = np.shape(image_gray)
    # Histogram evaluation using first problem's function
    x, histg = histogram_function(image_gray)
    for i in range(255):
        x, y = np.split(histg, [i])
        x1 = np.sum(x)/(image_gray.shape[0]*image_gray.shape[1])
        y1 = np.sum(y) /(image_gray.shape[0]* image_gray.shape[1])
        x2 = np.sum([j*t for j,t in enumerate(x)])/np.sum(x)
        x2 = np.nan_to_num(x2)
        y2 = np.sum([(j+d)*t for j,t in enumerate(y)])/np.sum(y)
        x3 = np.sum([(j-x2)**2*t for j,t in enumerate(x)])/np.sum(x)
        x3 = np.nan_to_num(x3)
        y3 = np.sum([(j-y2+d)**2*t for j,t in enumerate(y)])/np.sum(y)
        sigma2_w = x1*x3 + y1*y3
        d = d+1
        within.append(sigma2_w)
        between.append(x1*y1*(x2-y2)*(x2-y2))
        total_variance.append(sigma2_w+x1*y1*(x2-y2)*(x2-y2))

    thresold_intra = np.argmin(within)
    thresold_inter= np.argmax(between)
    for i in range(row):
        for j in range(col):
            if image_gray[i,j]>=thresold_intra:
                image_gray[i,j]=1
            else:
                image_gray[i,j]=0
    return image_gray, thresold_intra, thresold_inter, within, between, total_variance

########################################################### Driver code ############################################
if __name__ =='__main__':
    image = cv2.imread('coins.png')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image_gray, thresold_intra,thresold_inter, within, between, total_variance = otsus_binarization(image_gray)
    binary_img = image_gray
    print('Minimun value of intra class variance',thresold_intra)
    print('Maximum value of inter class variance', thresold_inter)
    t = [i for i in range(0, 255)]
    plt.subplot(2,2,1)
    plt.plot(t, within)
    plt.title('Intra class variance')
    plt.subplot(2,2,2)
    plt.plot(t, between)
    plt.title('Inter class variance')
    plt.subplot(2,2,3)
    plt.plot(t, total_variance)
    plt.title('Total variance(For verification)')
    plt.figure()
    plt.imshow(binary_img)
    plt.show()


