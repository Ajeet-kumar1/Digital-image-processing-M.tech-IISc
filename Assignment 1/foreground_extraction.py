from otsus_binarization import otsus_binarization
import numpy as np
import cv2
import matplotlib.pyplot as plt





##########################################Function which take two images and supeimpose them #######################################
def image_superimpose(background_img, text_binary, text_image):
    for i in range(512):
        for j in range(512):
            if text_binary[i,j] == 1:

                background_img[i,j, 0]= text_image[i,j,0]
                background_img[i,j, 1]= text_image[i,j,1]
                background_img[i,j, 2]= text_image[i,j,2]
            else:
                pass
    superimposed = background_img
 
    return superimposed

############### Driver code #####################################

if __name__ =='__main__':
    text_image = cv2.imread('IIScText.png')
    image1_gray = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)
    text_binary, thresold_intra,thresold_inter, within, between, total_variance = otsus_binarization(image1_gray)
    
 
    main_build = cv2.imread('IIScMainBuilding.png')
    output2 = image_superimpose(main_build, text_binary, text_image)
    cv2.imshow('Super imposed',output2)
    cv2.waitKey(0)
    plt.show()




 