import numpy as np
import matplotlib.pyplot as plt
#import cv2

def dummy(img2):
	f = 3
	m,n = img2.shape
	img3 = np.zeros((m, n))
	I = img2
	for i in range(0, m):
		for j in range(0, n):
			t = np.floor(i/f)
			r = np.floor(j/f)
			t1 = round(t)
			r1 = round(r)
			A = [[1,0,0,0],[1,1,0,0],[1, 0, 1,0],[1,1,1,1]]
			Y = [I[t1,r1], I[t1+1, r1], I[t1,r1+1], I[t1+1,r1+1]]
			res = np.linalg.inv(A).dot(Y)
			img3[i, j] = round(res[0] + res[1]*(t - t1) + res[2]*(r -r1) +res[3]*(r-r1)*(t -t1))
	return img3
#img = cv2.imread('images\Flowers.png')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img3 = dummy(img)
#plt.imshow(img3, cmap='gray')
#plt.show()