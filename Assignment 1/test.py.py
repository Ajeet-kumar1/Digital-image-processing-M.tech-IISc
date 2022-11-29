import cv2
import numpy as np
from matplotlib import pyplot as plt
d =0
x = [1, 2,3,4]
y = [1,2,3,4]
ajeet = [j*t for j,t in enumerate(x)]
x2 = np.sum([j*t for j,t in enumerate(x)])/np.sum(x)
x2 = np.nan_to_num(x2)
y2 = np.sum([(j+d)*t for j,t in enumerate(y)])/np.sum(y)
x3 = np.sum([(j-x2)**2*t for j,t in enumerate(x)])/np.sum(x)
x3 = np.nan_to_num(x3)
y3 = np.sum([(j-y2+d)**2*t for j,t in enumerate(y)])/np.sum(y)
print(ajeet)
print(x2)
print(y2)
print(x3)
print(y3)