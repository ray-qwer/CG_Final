from argparse import ArgumentParser
from moviepy.editor import VideoFileClip
import os

parser = ArgumentParser()
'''add argument here'''
parser.add_argument("--vedio", type=str)
parser.add_argument("--output", type=str, default="")
args = parser.parse_args()
if os.path.exists(args.vedio):

    videoClip = VideoFileClip(args.vedio)
    if args.output != "":
        videoClip.write_gif(args.output, fps=10,program='imageio')
    else:
        out_name = args.vedio.split("/")
        filename = out_name[-1].rsplit(".",1)
        videoClip.write_gif(filename[0]+".gif", fps=10,program='imageio')
else:
    print("No such vedio:", args.vedio)