
from otsus_binarization import otsus_binarization
import numpy as np
import matplotlib.pyplot as plt
import cv2

def shape_count_function(img):
    img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img, thresold_intra,thresold_inter, within, between, total_variance = otsus_binarization(img)
    row, col = np.shape(img)
    R = np.zeros([row, col])
    k = 1
    for i in range(0,row):
        for j in range(0, col):
            if img[i,j] ==0:
                pass
            elif img[i,j]==1 and img[i, j-1]==0 and img[i-1,j]==0:
                R[i,j]= k
                k = k+1
            elif img[i,j]==1 and img[i-1,j]==1 and img[i, j-1]==0:
                R[i,j]=R[i-1,j]
            elif img[i,j]==1 and img[i-1,j]==0 and img[i, j-1]==1:
                R[i,j]=R[i,j-1]
            elif img[i,j]==1 and img[i-1,j]==1 and img[i, j-1]==1:
                if R[i, j-1] != R[i-1, j]:
                    R[R == max(R[i, j-1], R[i-1, j])] = min(R[i, j-1], R[i-1, j])
                    k = k-1
                R[i,j]=R[i-1,j]
            
    number_of_shapes=k-1          
    counter = [0 for i in range(k)]
    for i in range(row):
        for j in range(col):
            counter[int(R[i,j])] = counter[int(R[i,j])] + 1 
    circle_count = 0
    number_of_shapes1 = 0
    for k in counter:
        if k>50:
            number_of_shapes1 = number_of_shapes1+ 1
        if k>500 and k<550:
            circle_count = circle_count+1
    number_of_shapes1 = number_of_shapes1 -1
    return number_of_shapes, circle_count, counter, number_of_shapes1
    
if __name__ == '__main__':
    shape_image = cv2.imread("Shapes.png")
    number_of_shapes, number_of_circles, counter, number_of_shapes1 = shape_count_function(shape_image)
    print('Number of shapes:',number_of_shapes1, '\nNumber of circles:',number_of_circles)
 



