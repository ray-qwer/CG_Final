python .\main.py --drawing pig2 --motion 12 --strip 4 --output output/pig2_12_ear_adjust_17pts.mp4 --sk_pts 17
python .\main.py --drawing pig2 --motion 12 --strip 4 --output output/pig2_12_ear_adjust_19pts.mp4 --sk_pts 19
python .\mp4_2_gif.py --video output/pig2_12_ear_adjust_17pts.mp4 --output output/pig2_12_ear_adjust_17pts.gif
python .\mp4_2_gif.py --video output/pig2_12_ear_adjust_19pts.mp4 --output output/pig2_12_ear_adjust_19pts.gif