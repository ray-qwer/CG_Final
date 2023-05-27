#  Beginning with the contents of the detected bounding box (a), 
#  we convert to grayscale and apply adaptive thresholding (b), 
#  perform morphological closing (c) 
#  and dilating (d), 
#  flood filling (e), 
#  and retain only the largest polygon (f ). 
# -------------------------------------------------------------------------------------------

import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt


# input
ip = "dragon_cat.jpg"
# ip = 'bear.jpg'
img = cv2.imread(ip)
m,n=img.shape[:2]

# smooth the input img if the background is complex (if necessary)

# BGR to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# gaussian blur to denoise
img = cv2.medianBlur(img, 5)

# adaptive binary thresholding
ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# img_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
img_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imwrite('./thresh_result.jpg', img_th)

# closing 
kernel = np.ones((2,2),dtype=np.uint8)
closing = cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, kernel)

cv2.imwrite('./closing_result.jpg', closing)

# # dilating
# dilating = cv2.dilate(closing, None)

# cv2.imwrite('./dilating_result.jpg', dilating)


#--------retain the largest polygan--------------------
# pre-process for finding contours (considered image boundary)
closing = 255 - closing

# contours
contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# find max contour
areas = [cv2.contourArea(cnt) for cnt in contours]
max_idx = np.argmax(areas)
# print(max_idx)
# print(hierarchy)

# mask
mask = cv2.drawContours(closing, contours, max_idx, 255, thickness=cv2.FILLED)
# mask = 255 - mask

cv2.imwrite('./mask_result.jpg', mask)


# flood filling
# mask = np.zeros([m+2,n+2],np.uint8)
# ret, mask = cv2.threshold(closing, 1, 255, cv2.THRESH_BINARY)
# cv2.floodFill(mask, mask, seedPoint=(round(m/2),round(n/2)), newVal=255)


# cv2.imwrite('./floodfill_result.jpg', mask)
