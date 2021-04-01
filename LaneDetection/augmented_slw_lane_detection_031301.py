import math

import matplotlib.image as mpimg
from utilities.PyrUp import pyrUp
from utilities.tracking_box import TrackingBox
from utilities import dbscanClustering
from utilities.Utilities import get_color, get_left_right_windows
from utilities.imageHelper import write_text_on_image
from utilities.image_debugger import ImageDebugger
from utilities.image_warp import perspective_warp
from utilities.lane_detection_augmented_sliding_window import AugmentedSlidingWindow
from utilities.pre_processing_pipeline import pipeline
from utilities.runpipeline_video_latest_frame import VideoCaptureNextFrame
import numpy as np
import cv2
import time
import os

from utilities.video_writer import VideoWriter

image_debugger_split = ImageDebugger(False)
image_debugger_per_box = ImageDebugger(False)
image_debugger = ImageDebugger(False)
image_debugger_pipeline = ImageDebugger(False)
src = np.float32([(0, 0), (1280, 0), (0, 720), (1280, 720)])
# dst = np.float32([(0, 0), (1280, 0), (-800, 720), (2280, 720)])
dst = np.float32([(0, 0), (1280, 0), (-1200, 720), (2680, 720)])
# dst = np.float32([(0, 0), (1280, 0), (-3680, 720), (4960, 720)])
is_scaled_warping = False
dir_path = os.path.dirname(os.path.realpath(__file__))
vid_capture = VideoCaptureNextFrame(dir_path+"/../input/lab-test031301.mp4")

# Initial tracking boxes for each lane at bottom of image are hard coded. Real time implementation should use histogram to determine these. This real time implementation works only when car has started to move on straight lines.
# Initial tracking boxes also contain information regarding total lanes.
# Cars middle position is hardcoded depending on where camera resides on car
# Information about when lanes are splitting and if lane is turning left or right is hard coded. This is retrieved from maps in real time.
# Pre-Recorded Video is read frame by frame and pre-processed
# Clustering is done as part of pre-processing, and total pixels in particular lane is used to determine weather lane is dashed or straight. Currently total pixel count is hard coded.
# Augmented sliding windows are used to track lanes in each image starting from their tracking box located at bottom
# In post processing, things line adding or removing lanes, and drawing appropriate boxes and collecting deviation from lane etc.  is performed.
# Uncomment lines containing video_writer_output to store output video
class Main:
    def __init__(self):
        self.l1 = TrackingBox(3, 318, 568, 0, get_color(1))
        self.l2 = TrackingBox(3, 524, 710, 1, get_color(4))
        self.l3 = TrackingBox(3, 798, 709, 2, get_color(6))
        self.sw = [self.l1, self.l2, self.l3]
        self.total_lanes = 3
        self.right_lane_merge_after_frame = []
        self.right_lane_split_after_frame = []
        self.left_lane_merge_after_frame = []
        self.left_lane_split_after_frame = []
        self.next_turn_is_left = 1
        self.straight_line_threshold = 400
        self.car_middle = 658

        self.tracing_window_radius_len = 40
        self.tracking_window_radius_ht = 40

        # Used to print box information in consecutive lines
        self.box_output_print = 1

        # Used for deviation calculation whenever there is only single line detected.
        # As midpoint of lane cannot be calculated with single line, we use this variable as threshold and find the center
        # Initially, it is 0, but once both the lanes' lines are detected, this value is updated continuously.
        # There is an assumption that first frame will definitely have both of the lines of lane, otherwise deviation
        #  cannot be calculated with single line detected
        self.lane_distance_in_points = 0

        self.augmented_sw = AugmentedSlidingWindow()
        # Multiple sliding window parameters
        self.window_height_radius = 60
        self.window_height_after_lane_turned_horizontal = 25
        self.bottom_window_height_radius = 35
        self.window_width_radius = 40
        self.window_width_after_lane_turned_horizontal = 20
        self.min_pixels_in_window = 10
        self.pixels_distance_on_x_axis_to_reduce_window_height = 300
        self.pixels_on_y_axis_to_reduce_window_height = 300

        # Used for lane center - Get lane center from past 3 frames average
        self.lane_center_cache = []

        # self.video_writer_output = VideoWriter("lab-test031302-final-op.mp4", 5, 1280, 720)

    def main(self):
        start_time = time.time()
        try:
            frame_number = 0
            while True:
                frame = vid_capture.read()
                frame_number = frame_number + 1
                if frame_number <= 10:
                    # Skipping frame as recorded video initial frames are dark. RealSense camera takes time to adjust.
                    print("Skipping frame: {}".format(frame_number))
                    continue
                if frame is None:
                    print("No frame received. Closing.")
                    break
                print("Processing frame: {}".format(frame_number))
                output_image = self.process_frame(frame, frame_number)

                # self.video_writer_output.add_image(output_image)
                cv2.imshow("Output Image", output_image)

                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
        finally:
            end_time = time.time()
            print("Time: {}".format(end_time - start_time))
            # self.video_writer_output.close()
            cv2.destroyAllWindows()
            vid_capture.get_cap().release()

    def process_frame(self, frame, frame_number):

        # Reset constants for every frame
        self.box_output_print = 1

        image_debugger.show_image_grey("Input image", frame)

        # warp the image
        warped_image = perspective_warp(frame, dst, src, is_scaled_warping)
        image_debugger.show_image_grey("warped image", warped_image)

        # Get canny blurred and cropped image
        warped_pipeline_image, wit = pipeline(warped_image, image_debugger_pipeline)
        image_debugger.show_image_grey("canny image", warped_pipeline_image)

        # Pyr Down image
        warped_pipeline_image_pyr_down = cv2.pyrDown(warped_pipeline_image)
        warped_pipeline_image_pyr_down = cv2.pyrDown(warped_pipeline_image_pyr_down)
        image_debugger.show_image_grey("pyr down", warped_pipeline_image_pyr_down)

        # Perform clustering
        clustered_image, clusters, counts = dbscanClustering.perform_dbscan(warped_pipeline_image_pyr_down, eps=4, min_samples=1)

        # Pyr Up image accordingly maintaining cluster values
        clustered_image = pyrUp(clustered_image, 2)
        image_debugger.show_image("Pyr up clustered image", clustered_image)
        # clustered_image_copy = np.copy(clustered_image)

        # We detect lanes in opposite order of lane turn
        if self.next_turn_is_left:
            # Parse windows right to left
            start = self.sw.__len__() - 1
            end = -1
            step = -1
        else:
            # Parse windows left to right
            start = 0
            end = self.sw.__len__()
            step = 1

        image_debugger.show_image("Image before filtering any lane points.", clustered_image)
        output_image = np.zeros_like(warped_image)
        output_image_info = np.zeros_like(warped_image)
        for index in range(start, end, step):
            window = self.sw[index]

            # Get non-zero from clustered image
            nonzero = clustered_image.nonzero()
            nonzero_y = np.array(nonzero[0])
            nonzero_x = np.array(nonzero[1])
            non_zero_coordinates = np.transpose(nonzero)

            # Construct box for tracking window
            mean_x, mean_y = window.get_mean_box()
            win_x_left_high = mean_x + self.tracing_window_radius_len
            win_x_left_low = mean_x - self.tracing_window_radius_len
            win_y_low = mean_y - self.tracking_window_radius_ht
            win_y_high = mean_y + self.tracking_window_radius_ht

            # Display window for debugging
            if image_debugger.is_enabled():
                window_display = np.zeros_like(frame)
                cv2.rectangle(window_display, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), window.get_color(), 2)
                image_debugger.show_image("Window", cv2.addWeighted(perspective_warp(frame, dst, src, is_scaled_warping), 0.5, window_display, 1, 0))

            # Get indexes of non-zero points in the box
            good_points_indexes = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                                   (nonzero_x >= win_x_left_low) & (nonzero_x < win_x_left_high)).nonzero()[0]

            if len(good_points_indexes) == 0:
                # If no points found in starting window, move window a bit up and find points
                # As sometimes due to dashed lines, starting window may be in the gap
                diff = 100
                extended_margin = 20
                good_points_indexes = ((nonzero_y >= ((win_y_low - diff - extended_margin) - extended_margin)) & (nonzero_y < (win_y_high - diff) + extended_margin) &
                                       (nonzero_x >= (win_x_left_low - extended_margin)) & (nonzero_x < (win_x_left_high + extended_margin))).nonzero()[0]

            if len(good_points_indexes) != 0:
                # Collect x and y co-ordinates of all points inside window
                good_points = np.empty((len(good_points_indexes), 2), dtype=int)
                # Collect all cluster labels for respective points in windows
                cluster_values = []
                i = 0
                for point in good_points_indexes:
                    good_points[i] = non_zero_coordinates[point]
                    cluster_values.append(clustered_image[non_zero_coordinates[point][0], non_zero_coordinates[point][1]])
                    i = i + 1

                # Get mean x and y values for positions of all points in windows
                x_mean = np.mean(np.transpose(good_points)[1])
                y_mean = np.mean(np.transpose(good_points)[0])
                # Get cluster label present for majority of points in windows
                cluster_value = max(set(cluster_values), key=cluster_values.count)

                # Determine if lane is straight or dashed lane based on total points in clustered image containing current cluster label
                is_straight_line = False
                if counts[cluster_value - 1] > self.straight_line_threshold:
                    is_straight_line = True
                # Save x and y means in cache of tracking box
                window.add(x_mean, y_mean, is_straight_line)

                # Get all points in lane falling in current tracking window, and remove that lane so next tracking windows don't track these already detected lane points
                if window.is_straight_line():
                    # Image to detect non-cluster points in MSW
                    cluster_points_img = np.where(clustered_image == cluster_value, np.uint8(1), np.uint8(0))
                    # Get points in current cluster of straight line
                    mask = np.where(clustered_image == cluster_value, clustered_image, 0).nonzero()

                    self.calculate_x_for_cluster_points_at_bottom(window, mask)

                    # Highlight lane points in output image
                    output_image[mask[0], mask[1]] = window.get_color()
                    # Draw window info on output image
                    self.draw_window_output_info(window, output_image, output_image_info, mask[0].size)
                    image_debugger_split.show_image("Output image with new points detected in solid lane", output_image)
                    # Retain lane points other than currently detected lane points (lane points containing current cluster value)
                    clustered_image = np.where(clustered_image == cluster_value, 0, clustered_image)
                    image_debugger_split.show_image("Clustered image after removing detected solid lane points", clustered_image)



                    # If lane is splitting, then start sliding window to get extra points
                    if self.right_lane_split_after_frame and frame_number == self.right_lane_split_after_frame[0]:
                        print("Got lane split")
                        xo, yo = self.find_split_via_msw(window, frame, nonzero_x, nonzero_y, cluster_points_img)
                        self.right_lane_split_after_frame.remove(self.right_lane_split_after_frame[0])
                        if self.right_lane_split_after_frame and self.right_lane_split_after_frame[0] == -10:
                            new_tracking_box = TrackingBox(3, 968, 460, 3, get_color(10))
                            new_tracking_box.make_straight_line()
                            self.sw.append(new_tracking_box)
                            self.right_lane_split_after_frame.remove(self.right_lane_split_after_frame[0])
                            window.make_dashed_line()
                            print("Made {} window Dashed".format(window.box_id))
                        if xo is not None:
                            xo_avg = np.int(np.average(xo))
                            yo_avg = np.int(np.average(yo))
                            new_tracking_box = TrackingBox(5, xo_avg, yo_avg, 10, get_color(7))
                            new_tracking_box.make_dashed_line()

                            nonzero = clustered_image.nonzero()
                            nonzero_y = np.array(nonzero[0])
                            nonzero_x = np.array(nonzero[1])

                            x_coordinates, y_coordinates = self.augmented_sw.perform_lane_detection(new_tracking_box, frame, nonzero_x, nonzero_y, self.next_turn_is_left, is_scaled_warping, dst, src, image_debugger_per_box, image_debugger,
                                                                                                    self.window_height_radius, self.bottom_window_height_radius, self.window_height_after_lane_turned_horizontal,
                                                                                                    self.window_width_after_lane_turned_horizontal, self.window_width_radius, self.min_pixels_in_window,
                                                                                                    self.pixels_distance_on_x_axis_to_reduce_window_height, self.pixels_on_y_axis_to_reduce_window_height)
                            output_image[y_coordinates, x_coordinates] = new_tracking_box.get_color()
                            clustered_image[y_coordinates, x_coordinates] = 0

                            self.draw_window_output_info(new_tracking_box, output_image, output_image_info, xo.size)
                            image_debugger_split.show_image("Output image with split lane added", output_image)
                            image_debugger_split.show_image("Clustered Image after removing split lane", clustered_image)
                else:
                    x_coordinates, y_coordinates = self.augmented_sw.perform_lane_detection(window, frame, nonzero_x, nonzero_y, self.next_turn_is_left, is_scaled_warping, dst, src, image_debugger_per_box, image_debugger, self.window_height_radius,
                                                                                            self.bottom_window_height_radius, self.window_height_after_lane_turned_horizontal, self.window_width_after_lane_turned_horizontal, self.window_width_radius,
                                                                                            self.min_pixels_in_window, self.pixels_distance_on_x_axis_to_reduce_window_height, self.pixels_on_y_axis_to_reduce_window_height)

                    output_image[y_coordinates, x_coordinates] = window.get_color()
                    # Draw window info on output image
                    self.draw_window_output_info(window, output_image, output_image_info, x_coordinates.size)
                    image_debugger_split.show_image("Output image with new points detected in dashed lane", output_image)
                    # Remove lane points detected in current lane, so next lane's sliding windows does not detect them again
                    clustered_image[y_coordinates, x_coordinates] = 0
                    image_debugger_split.show_image("Clustered image after removing detected dashed lane points", clustered_image)


            else:
                # No points found in tracking box
                self.draw_window_output_info(window, output_image, output_image_info, 0)

        # Dropping/Adding the tracking windows if lanes merged or split
        if self.right_lane_merge_after_frame and frame_number == self.right_lane_merge_after_frame[0]:
            window_that_may_become_straight_line = self.sw[self.sw.__len__() - 2]
            print("Removing dashed line which became straight line {} {}.".format(window_that_may_become_straight_line.box_id, window_that_may_become_straight_line.get_mean_box()))
            self.sw.remove(window_that_may_become_straight_line)
            self.right_lane_merge_after_frame.remove(self.right_lane_merge_after_frame[0])
            print("Yes")
        if self.right_lane_split_after_frame and frame_number >= self.right_lane_split_after_frame[0]:
            print("Yes")
        if self.left_lane_merge_after_frame and frame_number >= self.left_lane_merge_after_frame[0]:
            print("Yes")
        if self.left_lane_merge_after_frame and frame_number >= self.left_lane_merge_after_frame[0]:
            print("Yes")

        # Write frame number
        write_text_on_image(output_image_info, str(frame_number), (10, 30), get_color(8))

        # Draw car center
        cv2.rectangle(output_image, (self.car_middle - 5, 710), (self.car_middle + 5, 719), get_color(4), 2)
        cv2.rectangle(output_image, (self.car_middle - 2, 700), (self.car_middle + 2, 710), get_color(4), 2)
        write_text_on_image(output_image_info, "Car actual pos", (10, 720 - (self.box_output_print * 30)), get_color(4), thickness=2)
        self.box_output_print = self.box_output_print + 1
        write_text_on_image(output_image_info, "Lane center", (10, 720 - (self.box_output_print * 30)), get_color(5), thickness=2)
        self.box_output_print = self.box_output_print + 1

        # Add vehicle position, deviation and error information
        self.add_deviation_info(output_image_info, output_image)

        # Put output image on top of frame
        output_image = cv2.addWeighted(perspective_warp(frame, dst, src, is_scaled_warping), 0.5, output_image, 1, 0)
        # output_image = cv2.addWeighted(frame, 0.5, perspective_warp(output_image, src, dst, is_scaled_warping), 1, 0)

        # Put output information on top of output image
        output_image = cv2.addWeighted(output_image, 1, output_image_info, 1, 0)

        # image_debugger.enabled(True)
        image_debugger.show_image("output image", output_image)
        # image_debugger.enabled(False)

        return output_image

    def calculate_x_for_cluster_points_at_bottom(self, window, mask_with_cluster_points):
        x, y = window.get_mean_box()
        nonzero_mask_y = np.array(mask_with_cluster_points[0])
        nonzero_mask_x = np.array(mask_with_cluster_points[1])
        cluster_point_indexes_inside_tracking_box = ((nonzero_mask_y >= (y - self.tracking_window_radius_ht)) & (nonzero_mask_y < 720) &
                                                     (nonzero_mask_x >= (x - self.tracing_window_radius_len)) & (nonzero_mask_x < (x + self.tracing_window_radius_len))).nonzero()[0]

        if cluster_point_indexes_inside_tracking_box.size == 0:
            return

        cluster_points_inside_tracking_box = np.empty((len(cluster_point_indexes_inside_tracking_box), 2), dtype=int)
        mask_transpose = np.transpose(mask_with_cluster_points)
        i = 0
        for point in cluster_point_indexes_inside_tracking_box:
            cluster_points_inside_tracking_box[i] = mask_transpose[point]
            i = i + 1

        if cluster_point_indexes_inside_tracking_box.size > 1000:
            top_100_bottom_most_point_in_cluster_indexes = np.argpartition(np.transpose(cluster_points_inside_tracking_box)[0], -1000)[-100:]
        else:
            top_100_bottom_most_point_in_cluster_indexes = np.argpartition(np.transpose(cluster_points_inside_tracking_box)[0], -cluster_point_indexes_inside_tracking_box.size)[-100:]

        x_mean = 0
        total = 0
        for top_100_bottom_most_point_in_cluster_index in top_100_bottom_most_point_in_cluster_indexes:
            x_mean += cluster_points_inside_tracking_box[top_100_bottom_most_point_in_cluster_index][1]
            total += 1

        window.set_x_for_deviation(int(x_mean / total))

    def draw_window_output_info(self, window, output_image, output_image_info, total_points):
        mean_x, mean_y = window.get_mean_box()
        win_x_left_high = mean_x + self.tracing_window_radius_len
        win_x_left_low = mean_x - self.tracing_window_radius_len
        win_y_low = mean_y - self.tracking_window_radius_ht
        win_y_high = mean_y + self.tracking_window_radius_ht
        txt = "Dashed Lane"
        thickness = 2
        if window.is_straight_line():
            txt = "Straight Lane"
            thickness = 5
        write_text_on_image(output_image_info, str(window.box_id) + ": " + txt + ": " + str(total_points) + " points.", (10, 720 - (self.box_output_print * 30)), window.color, thickness=2)
        self.box_output_print = self.box_output_print + 1
        cv2.rectangle(output_image, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), window.color, thickness)

    # Perform multiple-sliding-window and returns X and Y co-ordinates which are not detected in cluster
    # Takes tracking window, current frame, current clustered image, non-zero X and Y of clustered image
    def find_split_via_msw(self, window, frame, nonzero_x, nonzero_y, image_with_current_cluster):
        # Multiple sliding window parameters
        window_height_radius = 60
        bottom_window_height_radius = 35
        window_height_after_lane_turned_horizontal = 25
        window_width_radius = 40
        min_pixels_in_window = 10
        pixels_distance_on_x_axis_to_reduce_window_height = 100
        pixels_on_y_axis_to_reduce_window_height = 200
        # Setup starting sliding window
        x, y = window.get_mean_box()
        starting_window_x_axis_pixel = x
        msw_tracking_box = TrackingBox(3, x, y, 1000, get_color(8))
        xw, yw = msw_tracking_box.get_mean_box()
        y_low = yw - window_height_radius
        y_high = yw + window_height_radius
        x_low = xw - window_width_radius
        x_high = xw + window_width_radius

        windows = [[x_low, x_high, y_low, y_high]]
        all_lane_points_positions = []

        if image_debugger_split.is_enabled():
            img_showing_box_per_iteration = np.zeros_like(frame)

        # For every set of multiple sliding windows
        while True:
            point_pos_in_current_windows = []
            for dsw_window in windows:
                if image_debugger_split.is_enabled():
                    cv2.rectangle(img_showing_box_per_iteration, (dsw_window[0], dsw_window[2]), (dsw_window[1], dsw_window[3]), window.get_color(), 2)

                # Get good points in sliding window
                good_point_positions = ((nonzero_y >= dsw_window[2]) & (nonzero_y < dsw_window[3]) & (nonzero_x >= dsw_window[0]) & (nonzero_x < dsw_window[1])).nonzero()[0]
                if len(good_point_positions) > min_pixels_in_window:
                    point_pos_in_current_windows.append(good_point_positions)
            if image_debugger_split.is_enabled():
                image_debugger_split.show_image("Boxes", cv2.addWeighted(perspective_warp(frame, dst, src, is_scaled_warping), 1, img_showing_box_per_iteration, 1, 0))

            # Stop future sliding windows if no points detected in current set of multiple sliding windows
            if len(point_pos_in_current_windows) <= 0:
                break

            # Gather new points detected in current windows, discarding points detected in previous set of windows
            point_pos_in_current_windows = np.concatenate(point_pos_in_current_windows)
            if len(all_lane_points_positions) > 0:
                t = np.concatenate(all_lane_points_positions)
            else:
                t = []
            new_points_pos = np.setdiff1d(point_pos_in_current_windows, t)
            all_lane_points_positions.append(new_points_pos)

            # If new points are detected, draw next set of windows based on these points OR stop proceeding further with new set of windows
            if len(new_points_pos) > 0:
                xo = nonzero_x[new_points_pos]
                yo = nonzero_y[new_points_pos]
                v = set(image_with_current_cluster[yo, xo])
                if 0 in v:
                    image_with_current_cluster_dup = np.copy(image_with_current_cluster)
                    image_debugger_split.show_image_grey("current cluster", image_with_current_cluster)
                    image_with_current_cluster_dup[yo, xo] = 1
                    image_debugger_split.show_image_grey("current cluster with msw points", image_with_current_cluster_dup)
                    new_points_not_in_cluster = np.bitwise_xor(image_with_current_cluster_dup, image_with_current_cluster)
                    image_debugger_split.show_image_grey("New points", new_points_not_in_cluster)
                    nonzero_new_points = new_points_not_in_cluster.nonzero()
                    return nonzero_new_points[1], nonzero_new_points[0]
                x_mean_for_new_points = np.int(np.mean(nonzero_x[new_points_pos]))
                y_mean_for_new_points = np.int(np.mean(nonzero_y[new_points_pos]))
                # top_win, top_left_win, top_right_win, left_win, right_win
                next_set_of_left_windows = [
                    [x_mean_for_new_points - window_width_radius, x_mean_for_new_points + window_width_radius, y_mean_for_new_points - 2 * window_height_radius, y_mean_for_new_points],
                    [x_mean_for_new_points - (3 * window_width_radius), x_mean_for_new_points - window_width_radius, y_mean_for_new_points - 2 * window_height_radius, y_mean_for_new_points],
                    [x_mean_for_new_points + window_width_radius, x_mean_for_new_points + (3 * window_width_radius), y_mean_for_new_points - 2 * window_height_radius, y_mean_for_new_points],
                    [x_mean_for_new_points - (2 * window_width_radius), x_mean_for_new_points, y_mean_for_new_points - bottom_window_height_radius, y_mean_for_new_points + bottom_window_height_radius],
                    [x_mean_for_new_points, x_mean_for_new_points + (2 * window_width_radius), y_mean_for_new_points - bottom_window_height_radius, y_mean_for_new_points + bottom_window_height_radius]]
                windows = next_set_of_left_windows

                if math.fabs(x_mean_for_new_points - starting_window_x_axis_pixel) > pixels_distance_on_x_axis_to_reduce_window_height or y_mean_for_new_points < pixels_on_y_axis_to_reduce_window_height:
                    window_height_radius = window_height_after_lane_turned_horizontal
                    bottom_window_height_radius = window_height_after_lane_turned_horizontal
            else:
                break

            if image_debugger_split.is_enabled():
                img_showing_box_per_iteration[nonzero_y[new_points_pos], nonzero_x[new_points_pos]] = get_color(9)
                image_debugger_split.show_image("Points in Boxes", cv2.addWeighted(perspective_warp(frame, dst, src, is_scaled_warping), 1, img_showing_box_per_iteration, 1, 0))
        return None, None

    def add_deviation_info(self, output_image_info, output_image):
        win_left, win_right = get_left_right_windows(self.sw, self.car_middle)

        # Calculate deviation and print info
        if win_left is None and win_right is None:
            write_text_on_image(output_image_info, "Deviation: Error no boxes found", (10, 720 - (self.box_output_print * 30)), get_color(8), thickness=2)
            self.box_output_print = self.box_output_print + 1
        else:
            if win_left is None or win_right is None:
                if win_left is None:
                    non_null_box = win_right
                else:
                    non_null_box = win_left

                x = non_null_box.x_for_deviation()
                is_right_box_available = False  # True means right box is missing else left box is missing
                text = "right"
                if non_null_box == win_right:
                    is_right_box_available = True
                    text = "left"

                write_text_on_image(output_image_info, "Vehicle is {} of box with id {}".format(text, non_null_box.box_id), (10, 720 - (self.box_output_print * 30)), get_color(8), thickness=2)
                self.box_output_print = self.box_output_print + 1
                if self.lane_distance_in_points == 0:
                    write_text_on_image(output_image_info, "Deviation: ERROR Past lane width unknown".format(non_null_box.box_id), (10, 720 - (self.box_output_print * 30)), get_color(8), thickness=2)
                    self.box_output_print = self.box_output_print + 1
                else:
                    if is_right_box_available:
                        actual_center = x - int(self.lane_distance_in_points / 2)
                    else:
                        actual_center = x + int(self.lane_distance_in_points / 2)
                    self.lane_center_cache.append(actual_center)
                    average_center = self.get_average_lane_center()
                    cv2.rectangle(output_image, (average_center - 5, 700), (average_center + 5, 719), get_color(5), 2)
                    deviation = self.car_middle - average_center
                    write_text_on_image(output_image_info, "Deviation {}".format(deviation), (10, 720 - (self.box_output_print * 30)), get_color(8), thickness=2)
                    self.box_output_print = self.box_output_print + 1
                    if deviation > 0:
                        write_text_on_image(output_image_info, "Car moving right", (10, 720 - (self.box_output_print * 30)), get_color(8), thickness=2)
                        self.box_output_print = self.box_output_print + 1
                    else:
                        write_text_on_image(output_image_info, "Car moving left", (10, 720 - (self.box_output_print * 30)), get_color(8), thickness=2)
                        self.box_output_print = self.box_output_print + 1
            else:
                x_left = win_left.x_for_deviation()
                x_right = win_right.x_for_deviation()
                self.lane_distance_in_points = x_right - x_left
                if self.lane_distance_in_points < 0:
                    raise Exception("Right and Left boxes positional difference should not be less than 0")

                write_text_on_image(output_image_info, "Car between box {} and {}".format(win_left.box_id, win_right.box_id), (10, 720 - (self.box_output_print * 30)), get_color(8), thickness=2)
                self.box_output_print = self.box_output_print + 1

                actual_center = int((x_left + x_right) / 2)
                self.lane_center_cache.append(actual_center)
                average_center = self.get_average_lane_center()

                cv2.rectangle(output_image, (average_center - 5, 700), (average_center + 5, 719), get_color(5), 2)
                deviation = self.car_middle - average_center
                write_text_on_image(output_image_info, "Deviation {}".format(deviation), (10, 720 - (self.box_output_print * 30)), get_color(8), thickness=2)
                self.box_output_print = self.box_output_print + 1
                if deviation > 0:
                    write_text_on_image(output_image_info, "Car moving right", (10, 720 - (self.box_output_print * 30)), get_color(8), thickness=2)
                    self.box_output_print = self.box_output_print + 1
                else:
                    write_text_on_image(output_image_info, "Car moving left", (10, 720 - (self.box_output_print * 30)), get_color(8), thickness=2)
                    self.box_output_print = self.box_output_print + 1

    def get_average_lane_center(self):
        return int(np.mean(self.lane_center_cache[-3:]))


main = Main()
main.main()
