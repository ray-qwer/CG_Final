import numpy as np
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
from segmentation_mask import SegmentationMask
import cv2
from tqdm import tqdm
from skimage.draw import polygon
from skimage.feature import corner_harris, corner_peaks
from skimage.measure import approximate_polygon
from collections import deque

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

        """
        To ensure our drawing figure can move without exceeding the canvas boundaries,
        we need to do padding to the canvas.
        The size of padding can be controlled by 'hori_pad_size' and 'veri_pad_size'.
        e.g. hori_pad_size = 0.2 means making the canvas become 1+(2*0.2) = 1.4 original width
        """
        hori_pad_size = 0.2
        veri_pad_size = 0.15
        H,W,C = self.img.shape
        background_color = self.img[10,10,:]
        img_padding = np.ones((int((1+2*veri_pad_size)*H), int((1+2*hori_pad_size)*W), 3), dtype=np.uint8) * background_color
        img_padding[int(veri_pad_size*H):int((1+veri_pad_size)*H), int(hori_pad_size*W):int((1+hori_pad_size)*W), :] = self.img
        self.img = img_padding

        # rescale the input image: (H,W) -> (768, W')
        H,W,C = self.img.shape
        scale = 768 / H
        H_prime, W_prime = (int(H*scale), int(W*scale))
        self.img = cv2.resize(self.img, ((W_prime, H_prime)), interpolation=cv2.INTER_AREA)

        self.img_path = img_path
        
        # H and W
        H, W, _ = self.img.shape
        self.H = H
        self.W = W

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
        self._tri_label_with_joints()

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
    
    def show_result(self, show_dots=False, returnResult=False):
        """ show result with hierarchy """
        tri_color = np.zeros(self.img.shape, dtype=np.uint8)
        for idx, triangle in (enumerate(self.tri.simplices)):
            tri_vertices = self._keypnts[triangle][:,[1,0]]
            i, j = polygon(tri_vertices[:,0], tri_vertices[:,1], self.img.shape)
            tri_color[i, j] = self.tri_color[idx]
        if returnResult:
            return tri_color
        else:
            plt.imshow(tri_color)
            if show_dots:
                plt.scatter(self._keypnts[:,0], self._keypnts[:,1],c="r",s=1)
            plt.show()
    def show_result_H(self, hierarchy=None, returnResult=True):
        if hierarchy is None:
            if self.skeleton_pts.shape[0] == 15:
                hierarchy = np.array([0, 2, 9, 8, 11, 13, 10, 12, 14, 1, 4, 6, 3, 5, 7])
            elif self.skeleton_pts.shape[0] == 17:
                hierarchy = np.array([0, 15, 16, 2, 9, 8, 11, 13, 10, 12, 14, 1, 4, 6, 3, 5, 7])
            elif self.skeleton_pts.shape[0] == 19:
                hierarchy = np.array([0, 15, 16, 17, 18, 2, 9, 8, 11, 13, 10, 12, 14, 1, 4, 6, 3, 5, 7])
        assert len(hierarchy) == self.skeleton_pts.shape[0]
        tri_color = np.zeros_like(self.img, dtype=np.uint8)
        for i in hierarchy:
            shading = np.where(self.tri_shading == i)[0]
            for s in shading:
                triangle = self.tri.simplices[s]
                tri_vertices = self._keypnts[triangle][:, [1,0]]
                i, j = polygon(tri_vertices[:,0], tri_vertices[:,1], self.img.shape)
                tri_color[i, j] = self.tri_color[s]
        if returnResult:
            return tri_color
        else:
            plt.imshow(tri_color)
            plt.show()

    def _tri_label_with_joints(self, showResult=False):
        """ bfs from joints """
        self.vertex_to_simplex()    # to get all triangles' label
        skeleton_tri = self._vertex_to_simplex[self.skeleton_pts[:,1], self.skeleton_pts[:,0]]
        self.tri_shading = np.ones(len(self.tri.simplices), dtype=np.uint8)*(-1)

        queue = deque()
        visited = set()
        
        h, w = self.seg_mask.shape
        print(h, w)
        for idx, start_joint in enumerate(skeleton_tri):
            queue.append(start_joint)
            visited.add(start_joint)
            self.tri_shading[start_joint] = idx
        
        while queue:
            node = queue.popleft()

            tri_neighbors = self.find_neighbors_triangle(node)

            for neighbor in tri_neighbors:
                if neighbor not in visited and neighbor >=0 and neighbor < len(self.tri.simplices):
                    queue.append(neighbor)
                    visited.add(neighbor)
                    self.tri_shading[neighbor] = self.tri_shading[node]
        
        residual = np.where(self.tri_shading == -1)[0]
        for i in residual:
            # i is the simplices index
            tri_neighbors = self.find_neighbors_triangle(i)

            for neighbor in tri_neighbors:
                if neighbor >=0 and neighbor < len(self.tri.simplices):
                    self.tri_shading[i] = self.tri_shading[neighbor]
                    continue

        """ test shading """
        if showResult:
            tri_color = np.zeros(self.img.shape, dtype=np.uint8)
            for idx in range(self.skeleton_pts.shape[0]):
                shadings = self.tri.simplices[np.where(self.tri_shading == idx)]
                color = (np.random.random(3)*256).astype(np.uint8)
                for triangle in shadings:
                    tri_vertices = self._keypnts[triangle][:,[1,0]]
                    i, j = polygon(tri_vertices[:,0], tri_vertices[:,1], self.img.shape)
                    tri_color[i, j] = color

            plt.imshow(tri_color)
            plt.show()

    def find_neighbors_triangle(self, node):
        """ this is an alternative function of delaunay.neighbors, because we have trimmed the triangles out of mask """
        h, w, _ = self.img.shape
        triangle = self.tri.simplices[node]
        tri_vertices = self._keypnts[triangle][:, [1,0]]
        tri_v_up   = tri_vertices - np.array([1,0])
        tri_v_down = tri_vertices + np.array([1,0])
        tri_v_left = tri_vertices - np.array([0,1])
        tri_v_right= tri_vertices + np.array([0,1])
        tri_directions = np.vstack((tri_v_up, tri_v_down, tri_v_right, tri_v_left))
        tri_directions_y = tri_directions[:, 0]
        tri_directions_y[tri_directions_y >= h] = h-1
        tri_directions_y[tri_directions_y < 0] = 0
        tri_directions_x = tri_directions[:, 1]
        tri_directions_x[tri_directions_x >= w] = w-1
        tri_directions_x[tri_directions_x < 0] = 0
        tri_neighbors = self._vertex_to_simplex[tri_directions_y, tri_directions_x]
        tri_neighbors = set(tuple(tri_neighbors))
        return tri_neighbors
    
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
    name = "bear"
    img_path = f"drawing_data/{name}.jpg"
    sk_path = f"drawing_data/{name}_skeleton.npy"
    seg = SegmentationMask(image_name=img_path, isShowResult=False)
    para = {"D1_kernel":11, "D1_iter":2, "D2_kernel":7, "D2_iter":1, "blockSize":49, "tolerance":2}
    seg_mask = seg.get_segmentation_mask(**para)
    tri = BFTriangle(img_path=img_path, seg_mask=seg_mask, skeleton_path=sk_path, strip=2)
    print(tri.vertex_to_simplex().shape)
    tri._tri_label_with_joints()
