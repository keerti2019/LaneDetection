import time

import cv2
import numpy as np
import os

from utilities import dbscanClustering
from utilities.lane_detection_clustering import ClusteringLaneDetection
from utilities.imageHelper import write_text_on_image, get_image_with_highlighted_curves, \
    keep_color_red, get_image_with_highlighted_clusters
from utilities.image_debugger import ImageDebugger
from utilities.image_warp import perspective_warp
from utilities.polyfit import PolyFit
from utilities.pre_processing_pipeline import pipeline
from utilities.runpipeline_video_latest_frame import VideoCaptureNextFrame
from utilities.video_writer import VideoWriter

image_debugger = ImageDebugger(True)
image_debugger_false = ImageDebugger(False)
src = np.float32([(0, 0), (1280, 0), (0, 720), (1280, 720)])
dst = np.float32([(0, 0), (1280, 0), (-800, 720), (2280, 720)])
is_scaled_warping = False
dir_path = os.path.dirname(os.path.realpath(__file__))
vid_capture = VideoCaptureNextFrame(dir_path+"/../input/lab-test101.mp4")
clustering_lane_detection = ClusteringLaneDetection()

try:
    carCenterPixelBasedOnCameraPosition = (1280 / 2)
    lane_width_in_pixels = 500
    # If positive, it will come towards right, car needs to go left
    frame_number = 0
    deviation = 0
    while True:
        frame = vid_capture.read()
        frame_number = frame_number + 1
        if frame_number <= 11:
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
        try:
            clustered_image, clusters, counts = dbscanClustering.perform_dbscan(warped_pipeline_image, eps=50, min_samples=3)
        except:
            print("Error in clustering")
            continue
        clustering_time = time.time() - start_time
        start_time = time.time()
        show_cluster = True
        left_out, right_out, win_y_low, win_y_high, win_x_left_low, win_x_left_high, win_x_right_low, win_x_right_high, left_lane_center, right_lane_center = clustering_lane_detection. \
            lane_detection_with_clustered_image(clustered_image,
                                                image_debugger_false,
                                                poly_fit_left=PolyFit(4, use_cache=True),
                                                poly_fit_right=PolyFit(4, use_cache=True),
                                                get_clusters_instead_of_curves=show_cluster,
                                                window_height=50,
                                                window_radius=50)
        lane_detection_time = time.time() - start_time

        error = False
        if left_lane_center is not None and right_lane_center is not None:
            lanes_center = (left_lane_center + right_lane_center) / 2
            deviation = carCenterPixelBasedOnCameraPosition - lanes_center
            lane_width_in_pixels = right_lane_center - left_lane_center
        elif left_lane_center is None and right_lane_center is None:
            error = True
        else:
            if left_lane_center is not None:
                deviation = carCenterPixelBasedOnCameraPosition - (left_lane_center + (lane_width_in_pixels / 2))
            if right_lane_center is not None:
                deviation = carCenterPixelBasedOnCameraPosition - (right_lane_center - (lane_width_in_pixels / 2))

        if show_cluster:
            img_out = get_image_with_highlighted_clusters(overlay_base_img=frame,
                                                          un_warp_overlay_base_img=False,
                                                          overlay_image_weight=0.4,
                                                          clustered_img=clustered_image,
                                                          left_cluster_value=left_out,
                                                          right_cluster_value=right_out,
                                                          do_un_warp=True,
                                                          warp_src=src,
                                                          warp_dst=dst,
                                                          is_scaled_warping=is_scaled_warping,
                                                          win_y_low=win_y_low,
                                                          win_y_high=win_y_high,
                                                          win_x_left_low=win_x_left_low,
                                                          win_x_left_high=win_x_left_high,
                                                          win_x_right_low=win_x_right_low,
                                                          win_x_right_high=win_x_right_high)
        else:
            img_out = get_image_with_highlighted_curves(frame, left_out, right_out, src, dst, is_scaled_warping,
                                                        win_y_low, win_y_high, win_x_left_low, win_x_left_high, win_x_right_low, win_x_right_high)
        write_text_on_image(img_out, str(frame_number), (10, 20), (255, 255, 255))

        if left_lane_center is not None:
            write_text_on_image(img_out, "Left lane pixel {}".format(left_lane_center), (10, 140), (220, 20, 60), 1)
        if right_lane_center is not None:
            write_text_on_image(img_out, "Right lane pixel {}.".format(right_lane_center), (10, 180), (220, 20, 60), 1)

        if error:
            write_text_on_image(img_out, "ERROR {}".format(error), (10, 220), (60, 20, 220), 1)
        else:
            write_text_on_image(img_out, "ERROR {}".format(error), (10, 220), (220, 20, 60), 1)
            write_text_on_image(img_out, "Deviation {}".format(round(deviation, 5)), (10, 250), (220, 20, 60), 1)

        print("Time in seconds: Canny {} Warp {} Clustering {} LaneDetection {} Error {} Deviation {} LeftLaneCenter{}, RightLaneCenter{}".format(
            round(canny_time, 5), round(warp_time, 5), round(clustering_time, 5), round(lane_detection_time, 5), error, deviation, left_lane_center, right_lane_center))

        # video_writer_input.add_image(img_out)
        cv2.imshow("Output Image", img_out)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
finally:
    # video_writer_input.close()
    cv2.destroyAllWindows()
    vid_capture.get_cap().release()
