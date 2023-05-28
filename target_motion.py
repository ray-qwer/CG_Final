import cv2
import mediapipe as mp
import numpy as np


class TargetMotion():
	def __init__(self, isDraw=False):
		'''
		:isDraw: boolean to specify whether to draw the output, default=False.
		'''
		self.isDraw = isDraw
		if self.isDraw:
			self.mp_drawing = mp.solutions.drawing_utils
		self.mp_pose = mp.solutions.pose
		self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

	def get_motion_vec(self, video_name):
		'''
		---input:
		:video_name: string to specify the location of video file
		---output:
		:motion_vec: a [T, 15, 2] size numpy array. (T: total frame; 15 part of body; 2: x and y coordinates)
		'''
		# create capture object
		cap = cv2.VideoCapture(video_name)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		motion_vec = np.zeros((total_frames, 15, 2), dtype=np.float32)

		t = -1
		while cap.isOpened():
			# read frame from capture object
			t += 1
			_, frame_ = cap.read()

			try:
				W, H, _ = frame_.shape
				frame = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)

				results = self.pose.process(frame)
				landmarks = results.pose_landmarks.landmark

				if self.isDraw:
					# draw detected skeleton on the frame
					self.mp_drawing.draw_landmarks(frame_, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
					cv2.imshow('Output', frame_)

			except: # no more frames to be read
				break
				
			if self.isDraw and (cv2.waitKey(1) == ord('q')): 
				#----------------------------#
				# 		Press Q to Exit		 #
				#----------------------------#
				break

			shoulder_mid_x = (landmarks[11].x + landmarks[12].x) / 2
			shoulder_mid_y = (landmarks[11].y + landmarks[12].y) / 2
			hip_mid_x = (landmarks[23].x + landmarks[24].x) / 2
			hip_mid_y = (landmarks[23].y + landmarks[24].y) / 2

			''' 
			2. left eye
			5. right eye
			7. left ear
			8. right ear
			9. left mouth
			10. right mouth
			currently, 2, 5, 7, 8, 9, 10 is not used.
			'''
			motion_vec[t] = np.array([[landmarks[0].x,  landmarks[0].y ], # nose
									[landmarks[11].x, landmarks[11].y], # left shoulder
									[shoulder_mid_x,  shoulder_mid_y ], # mid shoulder / neck
									[landmarks[12].x, landmarks[12].y], # right shoulder
									[landmarks[13].x, landmarks[13].y], # left elbow
									[landmarks[14].x, landmarks[14].y], # right elbow
									[landmarks[15].x, landmarks[15].y], # left wrist
									[landmarks[16].x, landmarks[16].y], # right wrist
									[landmarks[23].x, landmarks[23].y], # left hip
									[hip_mid_x, 		hip_mid_y	   ], # mid hip
									[landmarks[24].x, landmarks[24].y], # right hip
									[landmarks[25].x, landmarks[25].y], # left knee
									[landmarks[26].x, landmarks[26].y], # right knee
									[landmarks[27].x, landmarks[27].y], # left ankle
									[landmarks[28].x, landmarks[28].y], # right ankle
									])
			motion_vec[t,:, 0] *= W
			motion_vec[t,:, 1] *= H

		return motion_vec


if __name__ == "__main__":
	'''
	test dry run here
	'''
	targetMotion = TargetMotion(isDraw=True)
	video_name = "target_motion_data/3.mp4"
	target_motion_vec = targetMotion.get_motion_vec(video_name)