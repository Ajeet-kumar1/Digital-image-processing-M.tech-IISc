import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread('images\Flowers.png')
img1  = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


# Read the original image and know its type
f = 3

# Obtain the size of the original image
[m, n] = img1.shape
print('Image Shape:', m, n)

# Up sampling

# Create matrix of zeros to store the upsampled image
img3 = np.zeros((m, n))
# new size
for i in range(0, m-1, f):
	for j in range(0, n-1, f):
		img3[i, j] = img2[i//f][j//f]

# Nearest neighbour interpolation-Replication
# Replicating rows

for i in range(1, m-(f-1), f): 
	for j in range(0, n-(f-1)):
		img3[i:i+(f-1), j] = img3[i-1, j]

# Replicating columns
for i in range(0, m-1):
	for j in range(1, n-1, f):
		img3[i, j:j+(f-1)] = img3[i, j-1]