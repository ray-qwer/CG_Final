import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure
from PIL import Image, ImageTk, ImageDraw
import io


class SegmentationMask():
    def __init__(self, image_name, image=None, isBlur=True, isShowResult=False):
        if np.array(image).any() == None:
            self.img_ori = cv2.imread(image_name)
        else:
            self.img_ori = image
        
        """
        To ensure our drawing figure can move without exceeding the canvas boundaries,
        we need to do padding to the canvas.
        The size of padding can be controlled by 'hori_pad_size' and 'veri_pad_size'.
        e.g. hori_pad_size = 0.2 means making the canvas become 1+(2*0.2) = 1.4 original width
        """
        hori_pad_size = 0.2 
        veri_pad_size = 0.15
        H,W,C = self.img_ori.shape
        background_color = self.img_ori[10,10,:]
        img_padding = np.ones((int((1+2*veri_pad_size)*H), int((1+2*hori_pad_size)*W), 3), dtype=np.uint8) * background_color
        img_padding[int(veri_pad_size*H):int((1+veri_pad_size)*H), int(hori_pad_size*W):int((1+hori_pad_size)*W), :] = self.img_ori
        self.img_ori = img_padding

        # rescale the input image: (H,W) -> (768, W')
        H,W,C = self.img_ori.shape
        scale = 768 / H
        H_prime, W_prime = (int(H*scale), int(W*scale))
        self.img_ori = cv2.resize(self.img_ori, ((W_prime, H_prime)), interpolation=cv2.INTER_AREA)
 
        self.img = cv2.cvtColor(self.img_ori, cv2.COLOR_BGR2GRAY)
        if isBlur:
            self.img = cv2.medianBlur(self.img, 5)
        self.isShowResult = isShowResult

    def _adaptive_thresh(self, img, blockSize=7, tolerance=6):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize, tolerance)

    def disk_mask(self, kernelSize) :
        center = kernelSize // 2
        y,x = np.ogrid[-center:kernelSize-center, -center:kernelSize-center]
        mask = x*x + y*y <= center*center
        array = np.zeros((kernelSize, kernelSize))
        array[mask] = 1
        return array

    def _closing(self, img, kernelSize=7):
        kernel = self.disk_mask(kernelSize).astype(np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    def _dilation(self, img, kernelSize=5, iterations=1):
        kernel = self.disk_mask(kernelSize).astype(np.uint8)
        return cv2.dilate(img, kernel, iterations=iterations)
    
    def _erosion(self, img, kernelSize=5, iterations=1):
        kernel = self.disk_mask(kernelSize).astype(np.uint8)
        return cv2.erode(img, kernel, iterations=iterations)
    
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
        mask3d = np.repeat(self.mask3[:, :, np.newaxis], 3, axis=2)
        result = mask3d * self.img_ori
        plt.subplot(2,4,1), plt.imshow(self.img1, cmap="gray"), plt.title("adaptive threshold")
        plt.subplot(2,4,2), plt.imshow(self.img2, cmap="gray"), plt.title("closing")
        plt.subplot(2,4,3), plt.imshow(self.img3, cmap="gray"), plt.title("dilation")
        plt.subplot(2,4,4), plt.imshow(self.mask, cmap="gray"), plt.title("flood filling")
        plt.subplot(2,4,5), plt.imshow(self.mask1, cmap="gray"), plt.title("dilation")
        plt.subplot(2,4,6), plt.imshow(self.mask2, cmap="gray"), plt.title("erosion")
        plt.subplot(2,4,7), plt.imshow(self.mask3, cmap="gray"), plt.title("retain largest polygon")
        plt.subplot(2,4,8), plt.imshow(result, cmap="gray"), plt.title("result")
        plt.show()
    
    def get_segmentation_mask(self, D1_kernel=7, D2_kernel=5, D1_iter=3, D2_iter=2, blockSize=7, tolerance=6, showInterResult=False):
        '''
        main function here
        '''
        self.img1 = self._adaptive_thresh(self.img, blockSize=blockSize, tolerance=tolerance)
        self.img2 = self._closing(self.img1)
        self.img3 = self._dilation(self.img2, kernelSize=D1_kernel, iterations=D1_iter)
        self.mask = self._flood_filling(self.img3)
        self.mask1 = self._dilation(self.mask, kernelSize=D2_kernel, iterations=D2_iter)
        self.mask2 = self._erosion(self.mask1, kernelSize=int((D1_kernel+D2_kernel)/2), iterations=D1_iter+D2_iter-1)
        self.mask3 = self._retain_largest_polygon(self.mask2)
        if self.isShowResult:
            self.show_result()
        return self.mask3


if __name__ == "__main__":
    # dry run here:
    # ip = "drawing_data/dragon_cat.jpg"
    # ip = 'drawing_data/bear.jpg'
    # ip = "drawing_data/maoli.jpg"
    # ip = "drawing_data/maoli_lattice.jpg"
    # ip = "drawing_data/maoli_stripes.jpg"
    # ip = "drawing_data/shit.jpg"
    # ip = "drawing_data/ghost.jpg"
    ip = "drawing_data/pig.jpg"


    segmentationMask = SegmentationMask(image_name=ip, isShowResult=True)
    para = {"D1_kernel":3, "D1_iter":2, "D2_kernel":3, "D2_iter":1, "blockSize":25, "tolerance":2}
    result = segmentationMask.get_segmentation_mask(**para)
