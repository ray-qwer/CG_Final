import numpy as np
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
import random

image = np.zeros((400, 400), dtype=bool)

# create a mask
mask = np.zeros((400, 400),dtype=bool)
mask[100:200, 100:200] = True
mask[100:150, 200:250] = True
mask[100:300, 250:300] = True

tmp_mask_choosing = np.arange(400-1)
mask_down_shift = np.vstack((mask[0],mask[tmp_mask_choosing]))
mask_up_shift = np.vstack((mask[tmp_mask_choosing+1],mask[-1]))
mask_left_shift = np.hstack((np.expand_dims(mask[:,0], axis=1), mask[:,tmp_mask_choosing]))
mask_right_shift = np.hstack((mask[:,tmp_mask_choosing+1],np.expand_dims(mask[:,-1], axis=1)))

mask_border = (mask ^ mask_down_shift) |      \
              (mask ^ mask_right_shift)
mask = mask | mask_border

indice = np.array(np.where(mask_border == True))
border_dots = random.sample(list(range(indice.shape[1])), indice.shape[1]//2)
border_dots = (indice.T)[border_dots]

tri_in_mask = []

tri = Delaunay(border_dots)

# To check the triangle is in the mask: by testing the centroid.
# if the centroid is outside the mask, then ignore it, else append it.
for triangle in tri.simplices:
    tri_vertices = border_dots[triangle]

    # calculate centroid of triangle
    centroid =(np.sum(tri_vertices, axis=0)//3)
    centroid_4_dots = np.array([[centroid[0], centroid[0], centroid[0]+1, centroid[0]+1], \
                                [centroid[1], centroid[1]+1, centroid[1], centroid[1]+1]])
    
    if np.any(mask[centroid_4_dots[0], centroid_4_dots[1]] == False):
        continue
    else: tri_in_mask.append(triangle)

tri.simplices = tri_in_mask

plt.triplot(border_dots[:, 0], border_dots[:, 1], tri.simplices)
plt.plot(border_dots[:, 0], border_dots[:, 1], 'o')
plt.show()
# plt.imshow(mask_border)
# plt.show()