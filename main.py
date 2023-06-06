from bfTriangles import BFTriangle
from segmentation_mask import SegmentationMask
from target_motion import TargetMotion
from label_skeleton import LabelingGUI
from arap import ARAP
from adjust_skeleton_len import AdjustSkeletonLength
from config import dragon_cat, bear, maoli, shit, stickman

from argparse import ArgumentParser
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
import os 

def choose_drawing(name):
	if name == "dragon_cat":
		return dragon_cat
	if name == "bear":
		return bear
	if name == "maoli":
		return maoli
	if name == "shit":
		return shit
	if name == "stickman":
		return stickman


if __name__ == "__main__":
	parser = ArgumentParser()
	'''add argument here'''
	parser.add_argument("--drawing", type=str, default="stickman", choices=["dragon_cat", 
																			"bear", 
																			"maoli", 
																			"shit", 
																			"stickman"]) 
	parser.add_argument("--motion", type=str, default=8) # see line 68
	parser.add_argument("--strip", type=int, default=4) # see line 59
	parser.add_argument("--output", type=str, default="output/test_output.mp4") # see line 112
	args = parser.parse_args()
	out_name = args.output.split("/")
	if len(out_name)>1:
		os.makedirs(out_name[0], exist_ok=True)

	############################
	##	choose drawing figure ##
	############################
	model = choose_drawing(args.drawing)
	img_path = model["img_path"]
	sk_path = model["skeleton_path"]
	segmask_config = model["segmask_config"]

	###############################
	##	obtain segmentation mask ##
	###############################
	seg = SegmentationMask(image_name=img_path, isShowResult=False)
	seg_mask = seg.get_segmentation_mask(**segmask_config)

	###################################
	##	calculate delaunay triangles ##
	###################################
	triangle = BFTriangle(img_path=img_path, seg_mask=seg_mask, skeleton_path=sk_path, strip=args.strip, isShowResult=False)
	drawing_skeleton_pts = triangle.skeleton_pts
	vertices = triangle._keypnts
	triangles = triangle.tri.simplices

	##################################
	##	obtain target motion vector ##
	##################################
	targetMotion = TargetMotion(isDraw=False)
	video_name = f"target_motion_data/{args.motion}.mp4"
	target_motion_vec = targetMotion.get_motion_vec(video_name)

	##################################
	##	adjust skeleton length of 	##
	## 	target motion vector		##
	##################################
	adjustment = AdjustSkeletonLength(drawing_skeleton_pts)
	target_motion_vec = adjustment.adjust_all(target_motion_vec)

	##########################################
	##	normalize target motion vector		##
	##	according to figure's origin/center	##
	##########################################
	origin = ((target_motion_vec[:, 2, :] + target_motion_vec[:, 9, :]) / 2).astype(np.int32)
	origin = np.expand_dims(origin, axis=1)
	target_motion_vec_normalized = (target_motion_vec - origin).astype(np.int32)

	############################
	##	As-Rigid-As-Possible  ##
	## 	Shape Manipilation	  ##
	############################
	"""
	the motion to be applied on the drawing figure:
	[(normalized) & (skeleton length adjusted) target motion vector] + [origin of drawing figure]
	"""
	print("initializing ARAP...")
	arap = ARAP(pins_xy=drawing_skeleton_pts, vertices=vertices, triangles=triangles)
	skeleton_pts_origin = ((drawing_skeleton_pts[2,:] + drawing_skeleton_pts[9,:])/2).astype(np.int32)
	output_video = []
	for i in range(target_motion_vec_normalized.shape[0]):
		print(f"applying ARAP on each frame... (frame {i+1}/{target_motion_vec.shape[0]})", end="\r")
		new_pins_xy = target_motion_vec_normalized[i] + skeleton_pts_origin
		new_vertices = arap.solve(new_pins_xy)
		triangle._keypnts = new_vertices
		frame_result = triangle.show_result_H(returnResult=True)
		output_video.append(frame_result)

	#################################
	##	output the frames as video ##
	#################################
	frame_H, frame_W, _ = output_video[0].shape
	fps = 30
	print(f"saving frames to video")
	out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_W, frame_H))
	for frame in output_video:
		out.write(frame.astype(np.uint8))
	out.release()

	