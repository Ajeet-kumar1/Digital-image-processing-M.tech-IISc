import numpy as np
import cv2
import matplotlib.pyplot as plt
from histogram_equalization import equalizer

# CLAHE without overlap function
def clahe_no_overlap(gray_img):
    r = 64
    c = 64
    for i in range(0,gray_img.shape[0] , r):           # Iterate all 64x64 tiles
        for j in range(0,gray_img.shape[1] , c):
            window = gray_img[i:i+r,j:j+c]             # Get the values inside the tiles and equalize 
            window = equalizer(window)                 # Do equalization and clipping
            diff = np.abs(np.max(window)- 220)
            avg_diff = diff/256
            window = np.add(window, avg_diff)
            window = np.clip(window, diff, 250)
            gray_img[i:i+r,j:j+c] = window
    return gray_img

def clahe_with_overlap(gray_img):
    r = 64
    c = 64
    for i in range(0,gray_img.shape[0] , 48):           # 25% overlap
        for j in range(0,gray_img.shape[0] , 48):
            window = gray_img[i:i+r,j:j+c]
            window = equalizer(window)
            diff = np.abs(np.max(window)- 220)
            avg_diff = diff/256
            window = np.add(window, avg_diff)
            window = np.clip(window, diff, 250)
            gray_img[i:i+r,j:j+c] = window

    return gray_img



if __name__=='__main__':
    img_lion = cv2.imread('images\lion.png')
    gray_lion = cv2.cvtColor(img_lion, cv2.COLOR_BGR2GRAY)
    lion_ahe = clahe_no_overlap(gray_lion)
    img_stone = cv2.imread('images\StoneFace.png')
    gray_stone = cv2.cvtColor(img_stone, cv2.COLOR_BGR2GRAY)
    stone_ahe = clahe_no_overlap(gray_stone)
    
    plt.figure(1)
    plt.imshow(lion_ahe, cmap='gray', vmin=0, vmax=255)
    plt.title('Lion contrast adapative equalization')
        
    plt.figure(2)
    plt.imshow(stone_ahe, cmap='gray', vmin=0, vmax=255)
    plt.title('StoneFace contrast adapative equalization')

    stone_clahe = clahe_with_overlap(gray_stone)
    lion_clahe = clahe_with_overlap(gray_lion)
    
    plt.figure(3)
    plt.imshow(lion_clahe, cmap='gray', vmin=0, vmax=255)
    plt.title('Lion contrast limiting adapative equalization with overlap')
        
    plt.figure(4)
    plt.imshow(stone_clahe, cmap='gray', vmin=0, vmax=255)
    plt.title('StoneFace contrast limit apative equalization with overlap')


    plt.show()
   