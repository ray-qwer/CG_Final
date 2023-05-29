import numpy as np
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
import random
from segmentation_mask import SegmentationMask
import cv2
from tqdm import tqdm
from skimage.draw import polygon
from skimage.feature import corner_harris, corner_peaks

class DelaunayTriangles:
    def __init__(self, img_path, isShowResult=True):
        self.img_ori = cv2.imread(img_path)
        segmentationMask = SegmentationMask(image_name = img_path, isShowResult=False)
        self.SegMask = segmentationMask.get_segmentation_mask()
        self._mask_border = self._get_contour_from_mask()   # return matrix as large as h*w, dtype=bool. True if the pixel is border
        self._corner_pnts = self._get_corner_pnts()         # return (n, 2), n is the detected corners 

    def _get_contour_from_mask(self):
        h, w = self.img_ori.shape[0], self.img_ori.shape[1]
        tmp_mask_width_choosing = np.arange(w-1)
        tmp_mask_height_choosing = np.arange(h-1)
        mask_down_shift = np.vstack((self.SegMask[0],self.SegMask[tmp_mask_height_choosing]))
        mask_up_shift = np.vstack((self.SegMask[tmp_mask_height_choosing+1],self.SegMask[-1]))
        mask_left_shift = np.hstack((np.expand_dims(self.SegMask[:,0], axis=1), self.SegMask[:,tmp_mask_width_choosing]))
        mask_right_shift = np.hstack((self.SegMask[:,tmp_mask_width_choosing+1],np.expand_dims(self.SegMask[:,-1], axis=1)))
        return  (self.SegMask ^ mask_down_shift) |      \
                (self.SegMask ^ mask_up_shift)   |     \
                (self.SegMask ^ mask_right_shift)|      \
                (self.SegMask ^ mask_left_shift) &       \
                self.SegMask
        

    def _get_corner_pnts(self, min_dist=20, th_rel=0.00005):         # adaptive choosing: min_distance, threshold_rel
        # gray scale the image
        gray = cv2.cvtColor(self.img_ori, cv2.COLOR_RGB2GRAY)
        return corner_peaks(corner_harris(gray*self.SegMask), min_distance=min_dist, threshold_rel=th_rel) 
    
    