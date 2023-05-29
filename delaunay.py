import numpy as np
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
import random
from segmentation_mask import SegmentationMask
import cv2
from tqdm import tqdm
from skimage.draw import polygon
from skimage.feature import corner_harris, corner_peaks
from skimage.measure import approximate_polygon


"""
steps:
    1. outline keypoint: get poly contour by skimage.measure.approximate_polygon(contours, tolerance)
        tolerance is a hyperparameters, contour can be obtained by _get_contour_from_mask
    2. inline keypoint: edge and corner point
        before we get the inline keypoint, do erosion first to avoid duplicate keypoint with outline keypoint
    3. sampling the inline keypoint
    4. delaunay triangles
    5. drawing
"""

class DelaunayTriangles:
    def __init__(self, img_path, skeleton_path="", isShowResult=True, ):
        self.img_ori = cv2.imread(img_path)
        self.img_path = img_path
        segmentationMask = SegmentationMask(image_name = img_path, isShowResult=False)
        self.SegMask = segmentationMask.get_segmentation_mask()
        self.isShowResult = isShowResult
        if skeleton_path != "":
            self.skeleton_pts = np.load(skeleton_path)
        else:
            self.skeleton_pts = np.zeros((0,2))
    def _get_polygon_contour_from_mask(self, tol=0.02):
        """
            outline keypoint, if tolerance up, the keypoint will be less, vise versa.
            parameters: 
                1. tolerance: the tolerance of polygon approximate
        """
        contours, _ =  cv2.findContours(self.SegMask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _contour_tmp = approximate_polygon(contours[0].squeeze(), tolerance=tol)
        return np.array([_contour_tmp[:,1],_contour_tmp[:,0]],dtype=np.int32).T
    
    def _get_corner_pnts(self, min_dist=20, th_rel=0.00005):         # adaptive choosing: min_distance, threshold_rel
        # gray scale the image
        """
            innter keypoints
            steps: 1. do erosion first.
        """
        gray = cv2.cvtColor(self.img_ori, cv2.COLOR_RGB2GRAY)
        return corner_peaks(corner_harris(gray*self._erosed_mask), min_distance=min_dist, threshold_rel=th_rel) 
    
    def _get_edge_pnts(self, cannyth1=30, cannyth2=50):
        """
            Warning: This edge pnts dont contain corner
        """
        gray = cv2.cvtColor(self.img_ori, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 1)
        edges = cv2.Canny(blurred, threshold1 = cannyth1, threshold2 = cannyth2)    # add edge points
        edges = edges * self._erosed_mask    # only the edges at edge block will preserve
        edges[self._corner_pnts] = 0
        return np.array(np.where(edges>0)).T

    def _get_erosed_mask(self, kernelSize=(5,5), iterations=3):
        """
            used for edge and corner finding. To avoid duplicated keypoints 
        """
        kernel = np.ones(kernelSize, dtype= np.uint8)
        erosed_mask = cv2.erode(self.SegMask.astype(np.uint8), kernel, iterations=iterations)
        return np.where(erosed_mask > 0, True, False)
    
    def _get_keypnts(self, sampling=5):
        """
            outline: dont move
            inline: 
                corner: dont move
                edges: sampling
        """
        edge_pnts_sampling = self._edge_pnts[np.random.choice(self._edge_pnts.shape[0], size=self._edge_pnts.shape[0]//sampling, replace=False)]
        return np.concatenate((edge_pnts_sampling, self._corner_pnts, self._polygon_border, self.skeleton_pts), axis=0)
    
    def _tri_in_mask(self, triangle):
        """
            to check if the triangle is in the mask.
            if it is in the mask, the centroid will be in the mask too, else not
        """
        tri_vertices = self._keypnts[triangle]
        centroid = (np.sum(tri_vertices, axis=0)//3) 
        centroid_4_dots = np.array([[centroid[0], centroid[0], centroid[0]+1, centroid[0]+1], \
                                [centroid[1], centroid[1]+1, centroid[1], centroid[1]+1]]).astype(int)
        return np.all(self.SegMask[centroid_4_dots[0], centroid_4_dots[1]] == True)
        
    def show_result(self,):
        tri_color = np.zeros(self.img_ori.shape, dtype=np.uint8)
        for idx, triangle in (enumerate(self.tri.simplices)):
            tri_vertices = self._keypnts[triangle]
            tri_vertices = np.array([tri_vertices[0:, 1], tri_vertices[0:,0]], dtype=np.int32).T
            i, j = polygon(tri_vertices[:,1], tri_vertices[:,0], self.img_ori.shape)
            tri_color[i, j] = self.tri_color[idx]
        plt.imshow(tri_color)
        plt.scatter(self._keypnts[:,1], self._keypnts[:,0],c="r",s=0.1)
        plt.show()

    def get_delaunay_triangles(self):
        """
            main function
        """
        self._polygon_border = self._get_polygon_contour_from_mask(10)   # return matrix as large as h*w, dtype=bool. True if the pixel is border
        self._erosed_mask = self._get_erosed_mask(kernelSize=5, iterations=3)
        self._corner_pnts = self._get_corner_pnts()         # return (n, 2), n is the detected corners 
        self._edge_pnts = self._get_edge_pnts()
        self._keypnts = self._get_keypnts()

        self.tri = Delaunay(self._keypnts)
        tri_in_mask = []
        self.tri_color = []

        for triangle in tqdm(self.tri.simplices):
            if self._tri_in_mask(triangle):
                tri_in_mask.append(triangle)
                tri_vertices = self._keypnts[triangle]

                tri_vertices = np.array([tri_vertices[0:, 1], tri_vertices[0:,0]], dtype=np.int32).T
                i, j = polygon(tri_vertices[:,1], tri_vertices[:,0], self.img_ori.shape)
                color = np.mean(self.img_ori[i,j], axis = 0)
                self.tri_color.append(color.astype(np.uint8))
        self.tri.simplices = tri_in_mask
        if self.isShowResult:
            self.show_result()
        return self.tri
    
    def __getitem__(self, val):
        return {
            "triangle": self._keypnts[self.tri.simplices[val]],
            "color": self.tri_color[val]
        }

if __name__ == "__main__":
    # img_path = "drawing_data/dragon_cat.jpg"
    name = "bear"
    img_path = f"drawing_data/{name}.jpg"
    sk_path = f"drawing_data/{name}_skeleton.npy"
    delaunay = DelaunayTriangles(img_path,)
    delaunay.get_delaunay_triangles()
