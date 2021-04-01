from utilities.runpipeline_video_latest_frame import runPipelineOnVideo
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# Worked initially with hough transform. This presents using hough transform on a video and getting left and right lines. Left lines are black where as right lanes are yellow.
# This is only for hough transform demo purposes and is not a final solution.
# Hough transform was dropped later because it only deals with straight lines.
runPipelineOnVideo(dir_path + "/../input/input_video_lab.mp4")
runPipelineOnVideo(None)
