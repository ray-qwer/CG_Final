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

	def get_motion_vec(self, video_name, sk_pts=19):
		'''
		---input:
		:video_name: string to specify the location of video file
		---output:
		:motion_vec: a [T, sk_pts, 2] size numpy array. (T: total frame; (sk_pts) part of body; 2: x and y coordinates)
		'''
		assert (sk_pts >= 15 and sk_pts <= 19), "skeleton points must be 15 <= sk_pts <= 19"
		# create capture object
		cap = cv2.VideoCapture(video_name)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		motion_vec = np.zeros((total_frames, sk_pts, 2), dtype=np.float32)

		t = -1
		while cap.isOpened():
			# read frame from capture object
			t += 1
			_, frame_ = cap.read()

			try:
				H, W, _ = frame_.shape
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

			vec = np.array([[landmarks[0].x,  landmarks[0].y ], # nose
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
			
			extra_pts = np.array([[landmarks[2].x, landmarks[2].y], # left eye
								  [landmarks[5].x, landmarks[5].y], # right eye
								  [landmarks[7].x, landmarks[7].y], # left ear
								  [landmarks[8].x, landmarks[8].y], # right ear
								  ])
			
			if vec.shape[0] < sk_pts:
				motion_vec[t] = np.concatenate((vec, extra_pts[:sk_pts-vec.shape[0]]), axis=0)
			else:
				motion_vec[t] = vec

			motion_vec[t,:, 0] *= W
			motion_vec[t,:, 1] *= H

		return motion_vec


if __name__ == "__main__":
	'''
	test dry run here
	'''
	targetMotion = TargetMotion(isDraw=True)
	video_name = "target_motion_data/14.mp4"
	target_motion_vec = targetMotion.get_motion_vec(video_name, sk_pts=19)

	origin = ((target_motion_vec[:, 2, :] + target_motion_vec[:, 9, :])/2).astype(np.int32)
	origin = np.expand_dims(origin, axis=1)
	target_motion_vec_normalized = (target_motion_vec - origin).astype(np.int32)

	