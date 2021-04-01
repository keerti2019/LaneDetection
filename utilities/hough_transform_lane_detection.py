from utilities.imageHelper import showImage
from utilities.imageHelper import displayAllHoughLines
from utilities.imageHelper import display_all_hough_lines_based_on_category
from utilities.imageHelper import draw_lines
from utilities.image_debugger import ImageDebugger
from utilities.pre_processing_pipeline import pipeline
from utilities.video_writer import VideoWriter
import math
import cv2
import numpy as np

import matplotlib.pyplot as plt


class HoughTransformLaneDetection:
    def __init__(self, show_every_frame=False, save_every_frame=False,
                 build_input_video=False, build_output_video=False, frames_per_sec=0, width=0, height=0):
        self.centerOfCar = width / 2
        print("Center of car:" + str(self.centerOfCar))
        self.frame_count = 0
        self.show_every_frame = show_every_frame
        self.save_every_frame = save_every_frame

        self.build_output_video = build_output_video
        self.build_input_video = build_input_video
        self.frames_per_sec = frames_per_sec
        self.width = width
        self.height = height
        self.output_VideoWriter = None
        self.input_VideoWriter = None
        self.image_debugger = ImageDebugger(show_every_frame)

        if build_output_video:
            self.output_VideoWriter = VideoWriter('output_video.mp4', frames_per_sec, width, height)
        if build_input_video:
            self.input_VideoWriter = VideoWriter('input_video.mp4', frames_per_sec, width, height)

    def processFrame(self, image):
        self.frame_count = self.frame_count + 1
        print("Frame count " + str(self.frame_count))

        if self.build_input_video:
            self.input_VideoWriter.add_image(image)

        cropped_image, wit = pipeline(image, self.image_debugger, cropped_image=True)

        lines = cv2.HoughLinesP(
            cropped_image,
            rho=8,
            theta=np.pi / 60,
            threshold=100,
            lines=np.array([]),
            minLineLength=15,
            maxLineGap=10
        )
        # print(len(lines))
        # print(lines)
        # qq = lines[0]
        # print(qq[0, 2])
        # print("Total hough lines: ", len(lines))
        if self.show_every_frame:
            showImage("Image With All HOUGH Lines", displayAllHoughLines(lines, image))

        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []
        leftCategory = []
        rightCategory = []
        ommittedCategory = []
        error = 0
        delta = -1
        #centeroflanes = -1
        centerofimage = 320

        if lines is None:
            print("Hough Lines is None")
            error = 1

        else:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x2 == x1:
                        ommittedCategory.append([x1, y1, x2, y2])
                        # self.writeToFrame(image, "OM X1=X2: ", (x2, y2), (255, 255, 255))
                        continue
                    slope = (y2 - y1) / (x2 - x1)
                    if math.fabs(slope) < 0.1 or math.fabs(slope) > 15:
                        ommittedCategory.append([x1, y1, x2, y2])
                        # self.writeToFrame(image, "OM SLOPE: ", (x2, y2), (255, 255, 255))
                        continue
                    if slope <= 0:
                        # if x1 > self.centerOfCar:
                        #     ommittedCategory.append([x1, y1, x2, y2])
                        #     # self.writeToFrame(image, "OM LonR: ", (x2, y2), (255, 255, 255))
                        #     continue
                        left_line_x.extend([x1, x2])
                        left_line_y.extend([y1, y2])
                        leftCategory.append([x1, y1, x2, y2])
                    else:
                        # if x1 < self.centerOfCar:
                        #     ommittedCategory.append([x1, y1, x2, y2])
                        #     # self.writeToFrame(image, "OM RonL: ", (x2, y2), (255, 255, 255))
                        #     continue
                        right_line_x.extend([x1, x2])
                        right_line_y.extend([y1, y2])
                        rightCategory.append([x1, y1, x2, y2])

        if self.show_every_frame:
            display_all_hough_lines_based_on_category(leftCategory, rightCategory, ommittedCategory, image)

        min_y = int(image.shape[0] * (1 / 4))
        max_y = int(image.shape[0])
        finalLines = []
        leftX = 0
        rightX = 0

        plt.plot(left_line_x, left_line_y, 'o', color='black')
        plt.plot(right_line_x, right_line_y, 'o', color='yellow')
        plt.imshow(image)
        plt.show()

        if len(left_line_y) != 0 and len(left_line_x) != 0:
            poly_left = np.poly1d(np.polyfit(
                left_line_y,
                left_line_x,
                deg=2
            ))
            left_x_start = int(poly_left(max_y))
            leftX = left_x_start
            left_x_end = int(poly_left(min_y))
            finalLines.append([left_x_start, max_y, left_x_end, min_y])

            if self.show_every_frame:
                right_eq_x = []
                right_eq_y = []
                for i in range(100, 480):
                    if i % 10 == 0:
                        x_val = int(poly_left(i))
                        print("{},{}".format(x_val, i))
                        right_eq_y.append(i)
                        right_eq_x.append(x_val)
                print("Plotting {},{}".format(right_eq_x, right_eq_y))
                plt.plot(right_eq_x, right_eq_y, 'o', color='red')


        if len(right_line_y) != 0 and len(right_line_x) != 0:
            poly_right = np.poly1d(np.polyfit(
                right_line_y,
                right_line_x,
                deg=2
            ))
            right_x_start = int(poly_right(max_y))
            rightX = right_x_start
            right_x_end = int(poly_right(min_y))
            finalLines.append([right_x_start, max_y, right_x_end, min_y])

            if self.show_every_frame:
                right_eq_x = []
                right_eq_y = []
                for i in range(100, 480):
                    if i % 10 == 0:
                        x_val = int(poly_right(i))
                        print("{},{}".format(x_val, i))
                        right_eq_y.append(i)
                        right_eq_x.append(x_val)
                print("Plotting {},{}".format(right_eq_x, right_eq_y))
                plt.plot(right_eq_x, right_eq_y, 'o', color='yellow')
                plt.imshow(image)
                plt.show()

        if leftX or rightX is not None:
            if rightX is None:
                print('rightX is None')
                deviation = self.centerOfCar - leftX
                delta = self.centerOfCar - deviation
            elif leftX is None:
                print('leftX is None')
                deviation = rightX - self.centerOfCar
                delta = self.centerOfCar - deviation
            else:
                centeroflanes = (leftX + rightX) / 2
                delta = self.centerOfCar - centeroflanes

        else:
            error = 1

        self.writeToFrame(image, "Image Center : " + str(centerofimage), (20, 150), (255, 255, 255))
        self.writeToFrame(image, "Lane Center : " + str(centeroflanes), (20, 200), (255, 255, 255))
        self.writeToFrame(image, "error : " + str(error), (20, 250), (255, 255, 255))
        self.writeToFrame(image, "Frame No. " + str(self.frame_count), (20, 100), (255, 255, 255))
        # if abs(error) > 80:
        #     self.writeToFrame(image, "ERROR", (20, 300), (0, 0, 255))

        line_image = draw_lines(
            image,
            [finalLines],
            color=[0, 0, 255],
            thickness=5,
        )
        if self.show_every_frame:
            showImage("Output image", line_image)

        if self.build_output_video:
            self.output_VideoWriter.add_image(line_image)
        return error, delta, line_image

    def writeToFrame(self, image, text, position, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        lineType = 1

        cv2.putText(image, text, position, font, fontScale, color, lineType)
        return image

    def closePipeline(self):
        if self.build_output_video:
            print("Generated output video. Closing streams.")
            self.output_VideoWriter.close()
        if self.build_input_video:
            print("Generated input video. Closing streams.")
            self.input_VideoWriter.close()
