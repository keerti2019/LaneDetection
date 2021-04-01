import math

from utilities.tracking_box import TrackingBox
from utilities.Utilities import get_color
import cv2
import numpy as np

from utilities.image_warp import perspective_warp


# Logic that performs augmented sliding windows. This takes image and required parameters.
class AugmentedSlidingWindow:
    def perform_lane_detection(self,
                               window,
                               frame,
                               nonzero_x,
                               nonzero_y,
                               is_next_turn_left,
                               is_scaled_warping,
                               dst,
                               src,
                               image_debugger_per_box,
                               image_debugger,
                               window_height_radius=60,
                               bottom_window_height_radius=35,
                               window_height_after_lane_turned_horizontal=25,
                               window_width_after_lane_turned_horizontal=20,
                               window_width_radius=40,
                               min_pixels_in_window=10,
                               pixels_distance_on_x_axis_to_reduce_window_height=300,
                               pixels_on_y_axis_to_reduce_window_height=300):
        # Tracking previous y-mean so when msw start moving back wards, we will stop
        # This prevents tracking back all points in case two turning lanes are merging at horizon
        previous_y_mean = 720

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

        if image_debugger_per_box.is_enabled():
            img_showing_box_per_iteration = np.zeros_like(frame)

        if image_debugger.is_enabled():
            img_showing_all_boxes = np.zeros_like(frame)

        # To track first window so that if no points found, we retry again by moving window little up
        first_window = True
        # For every set of multiple sliding windows
        while True:
            point_pos_in_current_windows = []
            for dsw_window in windows:
                if image_debugger_per_box.is_enabled():
                    cv2.rectangle(img_showing_box_per_iteration, (dsw_window[0], dsw_window[2]), (dsw_window[1], dsw_window[3]), window.get_color(), 2)

                if image_debugger.is_enabled():
                    cv2.rectangle(img_showing_all_boxes, (dsw_window[0], dsw_window[2]), (dsw_window[1], dsw_window[3]), window.get_color(), 2)

                # Get good points in sliding window
                good_point_positions = ((nonzero_y >= dsw_window[2]) & (nonzero_y < dsw_window[3]) & (nonzero_x >= dsw_window[0]) & (nonzero_x < dsw_window[1])).nonzero()[0]
                if len(good_point_positions) > min_pixels_in_window:
                    point_pos_in_current_windows.append(good_point_positions)
            if image_debugger_per_box.is_enabled():
                image_debugger_per_box.show_image("Windows", cv2.addWeighted(perspective_warp(frame, dst, src, is_scaled_warping), 1, img_showing_box_per_iteration, 1, 0))
                # image_debugger_per_box.show_image("Windows", cv2.addWeighted(np.dstack((frame, frame, frame)) * 1, 1, img_showing_box_per_iteration, 1, 0))

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
                x_mean_for_new_points = np.int(np.mean(nonzero_x[new_points_pos]))
                y_mean_for_new_points = np.int(np.mean(nonzero_y[new_points_pos]))
                if y_mean_for_new_points > (previous_y_mean + 20):
                    print("Sliding windows starting to move back. Hence stopping.")
                    break
                if y_mean_for_new_points < previous_y_mean:
                    previous_y_mean = y_mean_for_new_points

                if math.fabs(x_mean_for_new_points - starting_window_x_axis_pixel) > pixels_distance_on_x_axis_to_reduce_window_height or y_mean_for_new_points < pixels_on_y_axis_to_reduce_window_height:
                    window_height_radius = window_height_after_lane_turned_horizontal
                    bottom_window_height_radius = window_height_after_lane_turned_horizontal
                    if is_next_turn_left:
                        # reduce size of top right window width
                        top_left_win = [x_mean_for_new_points - (3 * window_width_radius), x_mean_for_new_points - window_width_radius, y_mean_for_new_points - 2 * window_height_radius, y_mean_for_new_points]
                        top_right_win = [x_mean_for_new_points + window_width_radius, x_mean_for_new_points + (3 * window_width_after_lane_turned_horizontal), y_mean_for_new_points - 2 * window_height_radius,
                                         y_mean_for_new_points]
                    else:
                        # reduce size of top left window width
                        top_left_win = [x_mean_for_new_points - (3 * window_width_after_lane_turned_horizontal), x_mean_for_new_points - window_width_radius, y_mean_for_new_points - 2 * window_height_radius,
                                        y_mean_for_new_points]
                        top_right_win = [x_mean_for_new_points + window_width_radius, x_mean_for_new_points + (3 * window_width_radius), y_mean_for_new_points - 2 * window_height_radius, y_mean_for_new_points]
                else:
                    top_left_win = [x_mean_for_new_points - (3 * window_width_radius), x_mean_for_new_points - window_width_radius, y_mean_for_new_points - 2 * window_height_radius, y_mean_for_new_points]
                    top_right_win = [x_mean_for_new_points + window_width_radius, x_mean_for_new_points + (3 * window_width_radius), y_mean_for_new_points - 2 * window_height_radius, y_mean_for_new_points]

                # top_win, top_left_win, top_right_win, left_win, right_win
                next_set_of_left_windows = [
                    [x_mean_for_new_points - window_width_radius, x_mean_for_new_points + window_width_radius, y_mean_for_new_points - 2 * window_height_radius, y_mean_for_new_points],
                    top_left_win,
                    top_right_win,
                    [x_mean_for_new_points - (2 * window_width_radius), x_mean_for_new_points, y_mean_for_new_points - bottom_window_height_radius, y_mean_for_new_points + bottom_window_height_radius],
                    [x_mean_for_new_points, x_mean_for_new_points + (2 * window_width_radius), y_mean_for_new_points - bottom_window_height_radius, y_mean_for_new_points + bottom_window_height_radius]]
                windows = next_set_of_left_windows
            else:
                break

            if image_debugger_per_box.is_enabled():
                img_showing_box_per_iteration[nonzero_y[new_points_pos], nonzero_x[new_points_pos]] = get_color(1)
        if image_debugger.is_enabled():
            image_debugger.show_image("All Boxes", cv2.addWeighted(perspective_warp(frame, dst, src, is_scaled_warping), 1, img_showing_all_boxes, 1, 0))

        # Remove lane points detected in current lane, so next lane's sliding windows does not detect them again
        x_coordinates = []
        y_coordinates = []
        if all_lane_points_positions.__len__() is not 0:
            lane_points = np.concatenate(all_lane_points_positions)
            x_coordinates = nonzero_x[lane_points]
            y_coordinates = nonzero_y[lane_points]
        return x_coordinates, y_coordinates
