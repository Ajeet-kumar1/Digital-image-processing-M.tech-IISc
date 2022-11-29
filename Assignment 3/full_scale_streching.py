import numpy as np


# You may get the condition where B-A(Defined below in line 20) could be zero. Division with zero give error, to avaoid that use below line.
np.seterr(divide='ignore', invalid='ignore')

# Define streching function
def streching(img):
   #Get the histogram and find minimum and max pixel value in image
   row, col = img.shape
   A = np.amin(img)
   B = np.amax(img)

   # Now apply the formula of streching in each pixel
   for i in range(row):
     for j in range(col):
      img[i,j] = 255/(B-A) * (img[i,j] - A)

   return img