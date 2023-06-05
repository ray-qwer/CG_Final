import cv2
import matplotlib.pyplot as plt
import numpy as np

from segmentation_mask import SegmentationMask
from config import stickman

if __name__ == "__main__":
	print("reading drawing video...")
	video_name = "output/stickman.mp4"
	cap = cv2.VideoCapture(video_name)
	drawing_vid = []
	while cap.isOpened():
		isValid, frame = cap.read()
		if not isValid:
			break
		drawing_vid.append(frame)

	print("reading background video...")
	video_name = "target_motion_data/8.mp4"
	cap = cv2.VideoCapture(video_name)
	background_vid = []
	while cap.isOpened():
		isValid, frame = cap.read()
		if not isValid:
			break
		background_vid.append(frame)

	frame = background_vid[0]
	H1,W1,_ = frame.shape
	frame = drawing_vid[0]
	H2,W2,_ = frame.shape

	para = stickman["segmask_config"]
	segmentationMask = SegmentationMask(image_name="", image=drawing_vid[0], isShowResult=False)
	kernel = segmentationMask.disk_mask(7).astype(np.uint8)
	doErode = True

	for i in range(len(background_vid)):
		frame = background_vid[i]
		canvas = frame[H1-H2:H1, W1-W2:W1, :]
		segmentationMask = SegmentationMask(image_name="", image=drawing_vid[i], isShowResult=False)
		segMask = segmentationMask.get_segmentation_mask(**para).astype(np.uint8)
		if doErode:
			segMask = cv2.erode(segMask, kernel, iterations=1)
		segMask = np.logical_not(segMask)
		segMask = np.repeat(segMask[:, :, np.newaxis], 3, axis=2)
		canvas *= segMask
		canvas += drawing_vid[i]
		frame[H1-H2:H1, W1-W2:W1, :] = canvas

	#################################
	##	output the frames as video ##
	#################################
	frame_H, frame_W, _ = background_vid[0].shape
	fps = 30
	print(f"saving frames to video")
	out = cv2.VideoWriter("stickman_background.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_W, frame_H))
	for frame in background_vid:
		out.write(frame.astype(np.uint8))
	out.release()