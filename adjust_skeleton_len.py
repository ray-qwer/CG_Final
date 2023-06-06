import numpy as np
import math
import matplotlib.pyplot as plt

class AdjustSkeletonLength:
	def __init__(self, drawing_sk_array):
		'''
		input: 
			drawing_sk_array: the [15,2] size numpy array of the (x,y) skeleton coordinate,
							  corresponding to the hand drawing figure.
		'''
		self.draw_sk = drawing_sk_array
		self.draw_sk_length = self.get_sk_length(self.draw_sk)

	def get_length(self, pt1, pt2):
		'''
		L2 distance between 2 point
		'''
		return np.sqrt(np.sum(np.square(pt1 - pt2)))
	
	def get_sk_length(self, sk):
		'''
		obtain the length of each skeleton edge.
		'''
		assert (sk.shape[0] >= 15 and sk.shape[0] <= 19), "skeleton points must be >= 15 and <= 19"
		sk_lengths = np.zeros((sk.shape[0],))
		origin = ((sk[2] + sk[9]) / 2).astype(np.int32)
		sk_lengths[0] = self.get_length(sk[0], sk[2])
		sk_lengths[1] = self.get_length(sk[1], sk[2])
		sk_lengths[2] = self.get_length(sk[2], sk[3])
		sk_lengths[3] = self.get_length(sk[1], sk[4])
		sk_lengths[4] = self.get_length(sk[4], sk[6])
		sk_lengths[5] = self.get_length(sk[3], sk[5])
		sk_lengths[6] = self.get_length(sk[5], sk[7])
		sk_lengths[7] = self.get_length(sk[2], origin)
		sk_lengths[8] = self.get_length(sk[9], origin)
		sk_lengths[9] = self.get_length(sk[8], sk[9])
		sk_lengths[10] = self.get_length(sk[9], sk[10])
		sk_lengths[11] = self.get_length(sk[8], sk[11])
		sk_lengths[12] = self.get_length(sk[11], sk[13])
		sk_lengths[13] = self.get_length(sk[10], sk[12])
		sk_lengths[14] = self.get_length(sk[12], sk[14])
		if sk.shape[0] > 15:
			sk_lengths[15] = self.get_length(sk[0], sk[15])
			sk_lengths[16] = self.get_length(sk[0], sk[16])
		if sk.shape[0] > 17:
			sk_lengths[17] = self.get_length(sk[15], sk[17])
			sk_lengths[18] = self.get_length(sk[16], sk[18])
		return sk_lengths
	
	def get_theta(self, pt1, pt2):
		'''
		the theta is return in radians
		'''
		return math.atan((pt2[1]-pt1[1])/ (pt2[0]-pt1[0]))
	
	def adjust(self, pt1, pt2, draw_sk_length):
		'''
		input:
			pt1: the "origin" of the skeleton
			pt2: the other side of the skeleton, that required adjustment
		output:
			argument1: the adjusted pt2's (x,y) coordinate
			argument2: the offset required to apply on pt2's neighbor
		'''
		theta = self.get_theta(pt1, pt2)
		x_prime = draw_sk_length * math.cos(theta)
		if pt1[0] > pt2[0]:
			x_prime = -x_prime
		y_prime = draw_sk_length * abs(math.sin(theta))
		if pt1[1] > pt2[1]:
			y_prime = -y_prime
		adjusted_offset = np.array([x_prime, y_prime])
		return pt1 + adjusted_offset, (pt1 + adjusted_offset) - pt2

	def adjust_ear(self, pt1, pt2, draw_sk_length, LorR):	# LorR: left or right side, True for left
		motion_theta = self.get_theta(pt1, pt2)
		if LorR:
			draw_theta = self.get_theta(self.draw_sk[16], self.draw_sk[18])
			theta = motion_theta - math.pi + draw_theta + 25/180*math.pi
		else:
			draw_theta = self.get_theta(self.draw_sk[15], self.draw_sk[17])
			theta = motion_theta + draw_theta - 25/180*math.pi
		
		x_prime = draw_sk_length * math.cos(theta)
		y_prime = draw_sk_length * math.sin(theta)
		adjusted_offset = np.array([x_prime, y_prime])
		return pt1 + adjusted_offset, (pt1 + adjusted_offset) - pt2

	def adjust_all(self, target_motion_vec):
		'''
		adjust the entire target motion vector's skeleton length
		input:
			target_motion_vec: a [T,sk_pts,2] size numpy array, where T is the total frame number.
		'''
		origins = ((target_motion_vec[:, 2] + target_motion_vec[:, 9]) / 2).astype(np.int32)
		
		for frameIdx, frame_sk in enumerate(target_motion_vec):
			frame_sk = target_motion_vec[frameIdx, :, :]
			origin = origins[frameIdx]
			# adjust the length of skeleton points
			# then, adjust its neighbor (or we should say, its affected skeleton points) to ensure the rigidity
			frame_sk[2], offset_head = self.adjust(origin, frame_sk[2], self.draw_sk_length[7])
			frame_sk[[0,1,3,4,5,6,7]] += offset_head
			frame_sk[0], offset_nose = self.adjust(frame_sk[2], frame_sk[0], self.draw_sk_length[0])

			if frame_sk.shape[0] > 15:
				frame_sk[[15,16]] += offset_head + offset_nose
				frame_sk[15], offset_l_eye = self.adjust(frame_sk[0], frame_sk[15], self.draw_sk_length[15])
				frame_sk[16], offset_r_eye = self.adjust(frame_sk[0], frame_sk[16], self.draw_sk_length[16])
			if frame_sk.shape[0] > 17:
				frame_sk[[17,18]] += offset_head + offset_nose
				frame_sk[[17]] += offset_l_eye
				frame_sk[[18]] += offset_r_eye
				frame_sk[17], offset = self.adjust_ear(frame_sk[15], frame_sk[17], self.draw_sk_length[17], False)
				frame_sk[18], offset = self.adjust_ear(frame_sk[16], frame_sk[18], self.draw_sk_length[18], True)
				# frame_sk[17], offset = self.adjust(frame_sk[15], frame_sk[17], self.draw_sk_length[17])
				# frame_sk[18], offset = self.adjust(frame_sk[16], frame_sk[18], self.draw_sk_length[18])
	
			frame_sk[1], offset = self.adjust(frame_sk[2], frame_sk[1], self.draw_sk_length[1])
			frame_sk[[4,6]] += offset
			frame_sk[3], offset = self.adjust(frame_sk[2], frame_sk[3], self.draw_sk_length[2])
			frame_sk[[5,7]] += offset
			frame_sk[4], offset = self.adjust(frame_sk[1], frame_sk[4], self.draw_sk_length[3])
			frame_sk[6] += offset
			frame_sk[6], offset = self.adjust(frame_sk[4], frame_sk[6], self.draw_sk_length[4])
			frame_sk[5], offset = self.adjust(frame_sk[3], frame_sk[5], self.draw_sk_length[5])
			frame_sk[7] += offset
			frame_sk[7], offset = self.adjust(frame_sk[5], frame_sk[7], self.draw_sk_length[6])
			frame_sk[9], offset = self.adjust(origin, frame_sk[9], self.draw_sk_length[8])
			frame_sk[[8,10,11,12,13,14]] += offset
			frame_sk[8], offset = self.adjust(frame_sk[9], frame_sk[8], self.draw_sk_length[9])
			frame_sk[[11,13]] += offset
			frame_sk[10], offset = self.adjust(frame_sk[9], frame_sk[10], self.draw_sk_length[10])
			frame_sk[[12,14]] += offset
			frame_sk[11], offset = self.adjust(frame_sk[8], frame_sk[11], self.draw_sk_length[11])
			frame_sk[13] += offset
			frame_sk[13], offset = self.adjust(frame_sk[11], frame_sk[13], self.draw_sk_length[12])
			frame_sk[12], offset = self.adjust(frame_sk[10], frame_sk[12], self.draw_sk_length[13])
			frame_sk[14] += offset
			frame_sk[14], offset = self.adjust(frame_sk[12], frame_sk[14], self.draw_sk_length[14])

		return target_motion_vec


if __name__ == "__main__":
	from config import dragon_cat, bear, maoli, shit, pig
	model = pig
	img_path = model["img_path"]
	sk_path = model["skeleton_path"]
	segmask_config = model["segmask_config"]

	from bfTriangles import BFTriangle
	from segmentation_mask import SegmentationMask
	seg = SegmentationMask(image_name=img_path, isShowResult=False)
	seg_mask = seg.get_segmentation_mask(**segmask_config)
	triangle = BFTriangle(img_path=img_path, seg_mask=seg_mask, skeleton_path=sk_path, strip=4)
	
	drawing_skeleton_pts = triangle.skeleton_pts

	from target_motion import TargetMotion
	targetMotion = TargetMotion(isDraw=False)
	video_name = "target_motion_data/11.mp4"
	target_motion_vec = targetMotion.get_motion_vec(video_name, sk_pts=19)
	origin = ((target_motion_vec[:, 2, :] + target_motion_vec[:, 9, :]) / 2).astype(np.int32)
	origin = np.expand_dims(origin, axis=1)
	target_motion_vec_normalized = (target_motion_vec - origin).astype(np.int32)
	
	#####################
	#	example usage	#
	#####################
	adjustment = AdjustSkeletonLength(drawing_skeleton_pts)
	target_motion_vec = adjustment.adjust_all(target_motion_vec)

	origin = ((target_motion_vec[:, 2, :] + target_motion_vec[:, 9, :])/2).astype(np.int32)
	origin = np.expand_dims(origin, axis=1)
	target_motion_vec_normalized = (target_motion_vec - origin).astype(np.int32)

	from label_skeleton import LabelingGUI
	import tkinter as tk
	root = tk.Tk()
	gui = LabelingGUI(root, img_path)

	skeleton_pts = triangle.skeleton_pts
	vertices = triangle._keypnts
	triangles = triangle.tri.simplices
	from arap import ARAP
	print("initializing ARAP...")
	arap = ARAP(pins_xy=skeleton_pts, vertices=vertices, triangles=triangles)

	skeleton_pts_origin = ((drawing_skeleton_pts[2,:] + drawing_skeleton_pts[9,:])/2).astype(np.int32)
	for frame in range(30):
		new_pins_xy = target_motion_vec_normalized[frame] + skeleton_pts_origin
		new_vertices = arap.solve(new_pins_xy)
		gui.check_skeletal(new_labeled_points=new_pins_xy, isSwapXY=False)
		triangle._keypnts = new_vertices
		triangle.show_result()