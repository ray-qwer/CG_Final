import cv2
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import os
from scipy import ndimage

from segmentation_mask import SegmentationMask
from config import *

if __name__ == "__main__":
	parser = ArgumentParser()
	'''add argument here'''
	parser.add_argument("--pos", type=str, default="right")
	parser.add_argument("--vid1", type=str, default="test_output.mp4")
	parser.add_argument("--vid2", type=str, required=True) 
	parser.add_argument("--output", type=str, default="output/test_background_output.mp4")
	args = parser.parse_args()
	out_name = args.output.split("/")
	if len(out_name)>1:
		os.makedirs(out_name[0], exist_ok=True)

	######################################################
	##	read motion retargetted drawing figure's video 	##
	######################################################
	print("reading drawing video...")
	cap = cv2.VideoCapture(f"output/{args.vid1}")
	drawing_vid = []
	while cap.isOpened():
		isValid, frame = cap.read()
		if not isValid:
			break
		drawing_vid.append(frame)
	frame = drawing_vid[0]
	H2,W2,_ = frame.shape

	##############################
	##	read background video 	##
	##############################
	print("reading background video...")
	cap = cv2.VideoCapture(args.vid2)
	background_vid = []
	while cap.isOpened():
		isValid, frame = cap.read()
		if not isValid:
			break
		background_vid.append(frame)
	frame = background_vid[0]
	H1,W1,_ = frame.shape
	
	segmentationMask = SegmentationMask(image_name="", image=drawing_vid[0], isShowResult=False)
	kernel = segmentationMask.disk_mask(5).astype(np.uint8)
	doErode = True

	if args.pos != "right":
		W1 = W2

	for i in range(len(background_vid)):
		frame = background_vid[i]
		canvas = frame[H1-H2:H1, W1-W2:W1, :]
		segMask = cv2.cvtColor(drawing_vid[i], cv2.COLOR_BGR2GRAY)	
		segMask = np.where(segMask > 0, 0, 1).astype(np.uint8)
		segMask = ndimage.median_filter(segMask, size=15)
		canvas[:,:,0] *= segMask
		canvas[:,:,1] *= segMask
		canvas[:,:,2] *= segMask
		canvas += drawing_vid[i]
		frame[H1-H2:H1, W1-W2:W1, :] = canvas


	#################################
	##	output the frames as video ##
	#################################
	frame_H, frame_W, _ = background_vid[0].shape
	fps = 30
	print(f"saving frames to video")
	out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_W, frame_H))
	for frame in background_vid:
		out.write(frame.astype(np.uint8))
	out.release()