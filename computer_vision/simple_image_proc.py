import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

if not os.path.exists('output'):
    os.makedirs('output')

# read image from file and convert to RGB
img_bgr = cv2.imread('./input/Waymo-1.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

# greyscale image and save
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
cv2.imwrite('./output/waymo_grayscale.png', img_gray)

# edge detection
img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
cv2.imwrite('./output/waymo_sobel_vert.png', img_sobel_x)
cv2.imwrite('./output/waymo_sobel_horz.png', img_sobel_y)

# gaussian blur
img_blur_1 = cv2.GaussianBlur(img_gray, (9, 9), 1)
img_blur_5 = cv2.GaussianBlur(img_gray, (9, 9), 5)
img_blur_10 = cv2.GaussianBlur(img_gray, (9, 9), 10)
cv2.imwrite('./output/waymo_blur_sigma_1.png', img_blur_1)
cv2.imwrite('./output/waymo_blur_sigma_5.png', img_blur_5)
cv2.imwrite('./output/waymo_blur_sigma_10.png', img_blur_10)

# dilation and erosion
img_binary = img_gray.copy()
cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY, img_binary)
kernel = np.ones((5, 5), np.uint8)
img_erosion = cv2.erode(img_binary, kernel, iterations=1)
img_dilation = cv2.dilate(img_binary, kernel, iterations=1)
cv2.imwrite('./output/waymo_binary.png', img_binary)
cv2.imwrite('./output/waymo_eroded.png', img_erosion)
cv2.imwrite('./output/waymo_dilated.png', img_dilation)

