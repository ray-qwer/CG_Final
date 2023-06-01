import numpy as np
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
from segmentation_mask import SegmentationMask
import cv2
from tqdm import tqdm
from skimage.draw import polygon
from skimage.feature import corner_harris, corner_peaks
from skimage.measure import approximate_polygon

"""
    every point is keypoint...
"""

class BFTriangle:
    def __init__(self, img_path, seg_mask, skeleton_path="", isShowResult=True, strip=1):
        """
        Arg:
            strip: the interval size between each triangle vertex (or we should say keypoint)
        """
        self.img = cv2.imread(img_path)
        # resize the image
        H, W, _ = self.img.shape
        scale = 512/H
        self.img = cv2.resize(self.img,(int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
        self.img_path = img_path
        
        # H and W
        self.H = 512
        self.W = int(W*scale)

        # mask
        self.seg_mask = seg_mask

        self.isShowResult = isShowResult

        # skeleton pnts
        if skeleton_path != "":
            self.skeleton_pts = np.load(skeleton_path)
        else:
            self.skeleton_pts = np.zeros((0,2))
        
        # delaunay ttriangles
        self.get_delaunay_triangles(strip)

    def get_delaunay_triangles(self, strip=1):
        # meshgrid
        H_strip = np.arange(0, self.H, strip)
        W_strip = np.arange(0, self.W, strip)
        xv, yv = np.meshgrid(W_strip, H_strip)
        index_mask_tmp = self.seg_mask.copy()
        index_mask_tmp[self.skeleton_pts[:,1], self.skeleton_pts[:,0]] = 0  # in order that the keypoints and skeleton points wont be duplicated
        index_mask = (index_mask_tmp[yv, xv]).astype(bool)
        xv_keypoint = xv[index_mask]
        yv_keypoint = yv[index_mask]

        self._keypnts = np.array([xv_keypoint, yv_keypoint]).T
        self._keypnts = np.concatenate((self._keypnts, self.skeleton_pts), axis=0)

        self.tri = Delaunay(self._keypnts)
        tri_in_mask = []
        self.tri_color = []
        for triangle in tqdm(self.tri.simplices):
            if self._tri_in_mask(triangle):
                tri_in_mask.append(triangle)
                tri_vertices = self._keypnts[triangle][:,[1,0]]
                i, j = polygon(tri_vertices[:,0], tri_vertices[:,1], self.img.shape)
                color = np.mean(self.img[i, j], axis=0)
                self.tri_color.append(color.astype(np.uint8))
        self.tri.simplices = np.array(tri_in_mask)
        if self.isShowResult:
            self.show_result()
        return self.tri

    def _tri_in_mask(self, triangle):
        """
            to check if the triangle is in the mask.
            if it is in the mask, the centroid will be in the mask too, else not
        """
        tri_vertices = self._keypnts[triangle][:,[1,0]]
        centroid = (np.sum(tri_vertices, axis=0)//3) 
        centroid_4_dots = np.array([[centroid[0], centroid[0], centroid[0]+1, centroid[0]+1], \
                                [centroid[1], centroid[1]+1, centroid[1], centroid[1]+1]]).astype(int)
        return np.all(self.seg_mask[centroid_4_dots[0], centroid_4_dots[1]] == True)
    
    def show_result(self, show_dots=False):
        tri_color = np.zeros(self.img.shape, dtype=np.uint8)
        for idx, triangle in (enumerate(self.tri.simplices)):
            tri_vertices = self._keypnts[triangle][:,[1,0]]
            i, j = polygon(tri_vertices[:,0], tri_vertices[:,1], self.img.shape)
            tri_color[i, j] = self.tri_color[idx]
        plt.imshow(tri_color)
        if show_dots:
            plt.scatter(self._keypnts[:,0], self._keypnts[:,1],c="r",s=1)
        plt.show()

    def vertex_to_simplex(self):
        """
            return a [H,W] size of numpy array, where each entry (pixel) indicates that 
            which triangle index does this entry (pixel) belongs to.
            returning -1 means that it does not belong to any triangle.
        """
        self._vertex_to_simplex = np.ones((self.seg_mask.shape), dtype=np.int32)*(-1)
        for idx, triangle in tqdm(enumerate(self.tri.simplices)):
            tri_vertices = self._keypnts[triangle][:,[1,0]]
            i, j = polygon(tri_vertices[:,0], tri_vertices[:,1], self.seg_mask.shape)
            self._vertex_to_simplex[i,j] = idx
        return self._vertex_to_simplex

    def __getitem__(self, val):
        return {
            "triangle": self._keypnts[self.tri.simplices[val]],
            "color": self.tri_color[val]
        }

    def get_skeleton_pnts(self):
        return self.skeleton_pts
    
    def get_keypnts(self):
        return self._keypnts
        


if __name__ == "__main__":
    name = "dragon_cat"
    img_path = f"drawing_data/{name}.jpg"
    sk_path = f"drawing_data/{name}_skeleton.npy"
    seg = SegmentationMask(image_name=img_path, isShowResult=False)
    seg_mask = seg.get_segmentation_mask()
    tri = BFTriangle(img_path=img_path, seg_mask=seg_mask, skeleton_path=sk_path, strip=2)
    print(tri.vertex_to_simplex().shape)