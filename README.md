# LaneDetection
Lane Detection code developed for my Masters Thesis.

/input directory contains files of bigger size which delay the cloning of this repository, but in-turn provides exact source used for tuning the program so users can get started quickly.

/LaneDetection contains python code used to perform lane detection on pre-recorded videos containing simulated lanes in laboratory.

Three different kinds of approaches are used for lane detection: sliding window, clustering and augmented sliding window approaches.


Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg


# Getting Started - Setup
```shell
pip3 install matplotlib
pip3 install numpy
pip3 install opencv-python
pip3 install sklearn
pip3 install pyrealsense2 -f https://github.com/cansik/pyrealsense2-macosx/releases
python3 augmented_slw_lane_detecting_with_splitting_and_merge_lanes_031302.py
```
