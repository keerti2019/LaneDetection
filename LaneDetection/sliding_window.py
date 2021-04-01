from utilities.imageHelper import get_image_with_highlighted_points_from_sliding_window, get_image_with_highlighted_curves, write_text_on_image, keep_color, keep_color_red
from utilities.image_debugger import ImageDebugger
from utilities.image_warp import perspective_warp
from utilities.lane_detection_multiple_sliding_window import MultipleSlidingWindowLaneDetection
from utilities.polyfit import PolyFit
from utilities.pre_processing_pipeline import pipeline
from utilities.runpipeline_video_latest_frame import VideoCaptureNextFrame
import numpy as np
import cv2
import time
import os

from utilities.video_writer import VideoWriter

image_debugger = ImageDebugger(True)
image_debugger_false = ImageDebugger(False)
src = np.float32([(0, 0), (1280, 0), (0, 720), (1280, 720)])
# dst = np.float32([(0, 0), (1280, 0), (-800, 720), (2280, 720)])
dst = np.float32([(0, 0), (1280, 0), (-1200, 720), (2680, 720)])
is_scaled_warping = False
dir_path = os.path.dirname(os.path.realpath(__file__))
vid_capture = VideoCaptureNextFrame(dir_path+"/../input/lab-test104.mp4")
dirsw = MultipleSlidingWindowLaneDetection()
# video_writer_input = VideoWriter("lab-test104-directional-sw-output-_feb28.mp4", 5, 1280, 720)

try:
    carCenterPixelBasedOnCameraPosition = (1280 / 2)
    lane_width_in_pixels = 420
    frame_number = 0
    deviation = 0
    while True:
        frame = vid_capture.read()
        frame_number = frame_number + 1
        if frame_number <= 12:
            print("Skipping frame: {}".format(frame_number))
            continue
        if frame is None:
            print("No frame received. Closing.")
            break
        print("Processing frame: {}".format(frame_number))

        start_time = time.time()
        red_img_r = keep_color_red(frame)
        red_img = perspective_warp(red_img_r, dst, src, is_scaled_warping)
        canny_img_from_pipeline, wit = pipeline(frame, image_debugger_false)
        canny_time = time.time() - start_time
        start_time = time.time()
        warped_pipeline_image = perspective_warp(canny_img_from_pipeline, dst, src, is_scaled_warping)
        warp_time = time.time() - start_time
        start_time = time.time()
        get_points_instead_of_curves = True
        if get_points_instead_of_curves:
            image_with_points, plot_y, y_lo, y_hi, l_x_lo, l_x_hi, r_x_lo, r_x_hi, \
            lft_ln_ctr, rt_lane_ctr = dirsw.multiple_sliding_window_lane_det(warped_pipeline_image,
                                                                             image_debugger_false,
                                                                             get_points_instead_of_curves,
                                                                             poly_fit_left=PolyFit(4),
                                                                             poly_fit_right=PolyFit(4),
                                                                             window_height=55,  # 80 for camera looking down
                                                                             window_radius=35,  # 35 default
                                                                             minpix=1)
            img_out = get_image_with_highlighted_points_from_sliding_window(
                overlay_base_img=frame,
                un_warp_overlay_base_img=False,
                overlay_image_weight=0.4,
                image_with_points=image_with_points,
                do_un_warp=True,
                left_center=lft_ln_ctr,
                right_center=rt_lane_ctr,
                warp_src=src,
                warp_dst=dst,
                is_scaled_warping=is_scaled_warping,
                win_y_low=y_lo,
                win_y_high=y_hi,
                win_x_left_low=l_x_lo,
                win_x_left_high=l_x_hi,
                win_x_right_low=r_x_lo,
                win_x_right_high=r_x_hi)
        else:
            lcurve, rcurve, plot_y, y_lo, y_hi, l_x_lo, l_x_hi, r_x_lo, r_x_hi, lft_ln_ctr, rt_lane_ctr = dirsw.multiple_sliding_window_lane_det(warped_pipeline_image,
                                                                                                                                                 image_debugger_false,
                                                                                                                                                 get_points_instead_of_curves,
                                                                                                                                                 poly_fit_left=PolyFit(4),
                                                                                                                                                 poly_fit_right=PolyFit(4),
                                                                                                                                                 window_height=55,#80 for camera looking down
                                                                                                                                                 window_radius=35,
                                                                                                                                                 minpix=1)
            img_out = get_image_with_highlighted_curves(frame, lcurve, rcurve, src, dst, is_scaled_warping, y_lo, y_hi, l_x_lo, l_x_hi, r_x_lo, r_x_hi)
        lane_detection_time = time.time() - start_time

        error = False
        if lft_ln_ctr is not None and rt_lane_ctr is not None:
            lanes_center = (lft_ln_ctr + rt_lane_ctr) / 2
            print(lanes_center)
            deviation = carCenterPixelBasedOnCameraPosition - lanes_center
            lane_width_in_pixels = rt_lane_ctr - lft_ln_ctr
        elif lft_ln_ctr is None and rt_lane_ctr is None:
            error = True
        else:
            if lft_ln_ctr is not None:
                deviation = carCenterPixelBasedOnCameraPosition - (lft_ln_ctr + (lane_width_in_pixels / 2))
                print((lft_ln_ctr + (lane_width_in_pixels / 2)))
            if rt_lane_ctr is not None:
                deviation = carCenterPixelBasedOnCameraPosition - (rt_lane_ctr - (lane_width_in_pixels / 2))
                print((rt_lane_ctr - (lane_width_in_pixels / 2)))

        write_text_on_image(img_out, "Frame Number: {}".format(frame_number), (10, 30), (255, 255, 255))

        if lft_ln_ctr is not None:
            write_text_on_image(img_out, "Left lane pixel {}".format(lft_ln_ctr), (10, 140), (220, 20, 60), 1)
        if rt_lane_ctr is not None:
            write_text_on_image(img_out, "Right lane pixel {}.".format(rt_lane_ctr), (10, 180), (220, 20, 60), 1)

        if error:
            write_text_on_image(img_out, "ERROR {}".format(error), (10, 220), (60, 20, 220), 1)
        else:
            write_text_on_image(img_out, "ERROR {}".format(error), (10, 220), (220, 20, 60), 1)
            write_text_on_image(img_out, "Deviation {}".format(round(deviation, 5)), (10, 250), (220, 20, 60), 1)

        print("Time in seconds: Canny {} Warp {} LaneDetection {} Error {} Deviation {} LeftLaneCenter{}, RightLaneCenter{}".format(
            round(canny_time, 5), round(warp_time, 5), round(lane_detection_time, 5), error, deviation, lft_ln_ctr, rt_lane_ctr))

        # video_writer_input.add_image(img_out)
        cv2.imshow("Output Image", img_out)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
finally:
    # video_writer_input.close()
    cv2.destroyAllWindows()
    vid_capture.get_cap().release()
