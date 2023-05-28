import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure

class SegmentationMask():
    def __init__(self, image_name, isBlur=True, isShowResult=False):
        self.img_ori = cv2.imread(image_name)
        self.img = cv2.cvtColor(self.img_ori, cv2.COLOR_BGR2GRAY)
        if isBlur:
            self.img = cv2.medianBlur(self.img, 5)
        self.isShowResult = isShowResult

    def _adaptive_thresh(self, img, blockSize=11, tolerance=10):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize, tolerance)

    def _closing(self, img, kernelSize=13):
        kernel = np.ones((kernelSize, kernelSize),dtype=np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    def _dilation(self, img, kernelSize=11, iterations=1):
        kernel = np.ones((kernelSize, kernelSize), np.uint8)
        return cv2.dilate(img, kernel, iterations=iterations)
    
    def _flood_filling(self, img):
        binary_mask = np.where(img == 255, 1, 0) # turn uint8 into binary first
        return ndimage.binary_fill_holes(binary_mask).astype(np.uint8)
    
    def _retain_largest_polygon(self, mask):        
        region = measure.label(mask)
        total_regions = np.max(region)

        largest_region_area = 0
        largest_region_label = 0
        for i in range(1, total_regions+1):
            area = np.sum(np.where(region == i, 1, 0))
            if area > largest_region_area:
                largest_region_area = area
                largest_region_label = i

        return np.where(region == largest_region_label, 1, 0)
    
    def show_result(self, ):
        mask3d = np.repeat(self.mask[:, :, np.newaxis], 3, axis=2)
        result = mask3d * self.img_ori
        plt.imshow(result)
        plt.show()

    
    def get_segmentation_mask(self, ):
        '''
        main function here
        '''
        self.img = self._adaptive_thresh(self.img)
        self.img = self._closing(self.img)
        self.img = self._dilation(self.img)
        self.mask = self._flood_filling(self.img)
        self.mask = self._retain_largest_polygon(self.mask)
        if self.isShowResult:
            self.show_result()
        return self.mask


if __name__ == "__main__":
    # dry run here:
    ip = "drawing_data/dragon_cat.jpg"
    # ip = 'drawing_data/bear.jpg'

    segmentationMask = SegmentationMask(image_name=ip, isShowResult=True)
    result = segmentationMask.get_segmentation_mask()
