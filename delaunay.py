import numpy as np
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
import random
from segmentation_mask import SegmentationMask
import cv2
from tqdm import tqdm
from skimage.draw import polygon2mask, polygon
from skimage.feature import corner_harris, corner_peaks, corner_shi_tomasi

# try numba for speeding up maybe?

ip = "drawing_data/dragon_cat.jpg"
ip = "drawing_data/bear.jpg"

# create a mask
segmentationMask = SegmentationMask(image_name=ip, isShowResult=False)
mask = segmentationMask.get_segmentation_mask()
img = cv2.imread(ip)

# get contour from mask
img_shape = img.shape
tmp_mask_width_choosing = np.arange(img_shape[1]-1)
tmp_mask_height_choosing = np.arange(img_shape[0]-1)
mask_down_shift = np.vstack((mask[0],mask[tmp_mask_height_choosing]))
mask_up_shift = np.vstack((mask[tmp_mask_height_choosing+1],mask[-1]))
mask_left_shift = np.hstack((np.expand_dims(mask[:,0], axis=1), mask[:,tmp_mask_width_choosing]))
mask_right_shift = np.hstack((mask[:,tmp_mask_width_choosing+1],np.expand_dims(mask[:,-1], axis=1)))
mask_border = (mask ^ mask_down_shift) |      \
              (mask ^ mask_up_shift)   |     \
              (mask ^ mask_right_shift)|      \
              (mask ^ mask_left_shift) &       \
              mask

# gray scale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# corners
corners = corner_peaks(corner_harris(gray*mask), min_distance=20, threshold_rel=0.00005)
for corner in corners:
    y, x = corner
    cv2.circle(img, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

# bwlabel
# cv2.connectedComponent

# Display the image with corners
cv2.imshow('Image with Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# get edge in the image
blurred = cv2.GaussianBlur(gray, (5,5), 1)

edges = cv2.Canny(blurred, threshold1 = 30, threshold2 = 50)    # add edge points
edges = edges * mask    # only the edges at edge block will preserve

# plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.subplot(122), plt.imshow(edges, cmap="gray")
# plt.show()
edges[np.where(edges > 0)] = True

indice_candidate = edges | mask_border

indice = np.array(np.where(indice_candidate == True))
border_dots = random.sample(list(range(indice.shape[1])), indice.shape[1]//100)  # hyper parameter
border_dots = (indice.T)[border_dots]

print(border_dots.shape)
print(corners.shape)
dots = np.concatenate((border_dots, corners), axis =0)


tri_in_mask = []
tri_in_mask_color = []

tri = Delaunay(dots)

# print(tri_mask.shape)
# To check the triangle is in the mask: by testing the centroid.
# if the centroid is outside the mask, then ignore it, else append it.
tri_color = np.zeros(img.shape, dtype=np.uint8)

tri_count = 1
for triangle in tqdm(tri.simplices):
    tri_vertices = dots[triangle]

    # calculate centroid of triangle
    centroid =(np.sum(tri_vertices, axis=0)//3)
    centroid_4_dots = np.array([[centroid[0], centroid[0], centroid[0]+1, centroid[0]+1], \
                                [centroid[1], centroid[1]+1, centroid[1], centroid[1]+1]])
    
    if np.any(mask[centroid_4_dots[0], centroid_4_dots[1]] == False):
        continue
    else: 
        tri_in_mask.append(triangle)
        tri_vertices = np.array([tri_vertices[0:, 1], tri_vertices[0:,0]], dtype=np.int32).T
        i, j = polygon(tri_vertices[:,1], tri_vertices[:,0], tri_color.shape)
        color = np.mean(img[i,j], axis = 0)
        tri_color[i, j] = color
        tri_in_mask_color.append(color)
        

        

tri.simplices = tri_in_mask

# tri_image = np.zeros(img.shape)
# for i in tqdm(range(1, tri_count+1)):
#     tri_area = np.where(tri_mask == i)
#     if tri_area[0].shape == (0,):
#         continue
#     tri_color = np.mean(img[tri_area], axis=0, dtype=np.uint8)
    
#     tri_image[tri_area,:] = tri_color

plt.imshow(tri_color)
# plt.triplot(dots[:, 1], dots[:, 0], tri.simplices)
# plt.plot(border_dots[:, 0], border_dots[:, 1], 'o')
plt.show()


