# Import required library and module
import cv2
import matplotlib.pyplot as plt
import numpy as np
from histogram_computation import histogram_function
from otsus_binarization import otsus_binarization
from foreground_extraction import image_superimpose
from connected_components import shape_count_function


a = int(input(" To run the question 1 Enter '1'.\n To run the question 2 Enter '2'.\n To run the question 3 Enter '3'.\n To run the question 4 Enter '4'. \n"))
if a==1:    ################################# Question 1 driver code ###################
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

  ####################### Driver code for Question 2 #############################################################################
  #    
elif a==2:
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

############################################# Driver code for question 3##################################################################
elif a ==3:
   text_image = cv2.imread('IIScText.png')
   image1_gray = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)
   text_binary, thresold_intra,thresold_inter, within, between, total_variance = otsus_binarization(image1_gray)
    
 
   main_build = cv2.imread('IIScMainBuilding.png')
   output2 = image_superimpose(main_build, text_binary, text_image)
   cv2.imshow('Super imposed',output2)
   cv2.waitKey(0)
   plt.show()

######################################### Driver code for question 4 ######################################################################

elif a ==4:
   shape_image = cv2.imread("Shapes.png")
   number_of_shapes_total, number_of_circles, counter, number_of_shapes_visible = shape_count_function(shape_image)
   print('Number of shapes visible:',number_of_shapes_visible, '\nNumber of circles:',number_of_circles)

else:
   print("Invalid input! Please select correct input")