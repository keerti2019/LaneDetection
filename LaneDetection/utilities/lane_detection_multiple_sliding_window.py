import numpy as np
import cv2

from utilities.imageHelper import get_hist
from utilities.polyfit import PolyFit


class MultipleSlidingWindowLaneDetection:
    def __init__(self):
        self.left_lane_center_cache = []
        self.right_lane_center_cache = []

    def multiple_sliding_window_lane_det(self, warped_canny_img, image_debugger, get_points_instead_of_curves=False, poly_fit_left=PolyFit(4), poly_fit_right=PolyFit(4), window_height=50, window_radius=150, minpix=1):
        histogram = get_hist(warped_canny_img)
        # image_debugger.show_image_histogram(histogram)

        # find peaks of left and right halves
        midpoint = int(histogram.shape[0] / 2)
        cache_size = 3

        if not self.left_lane_center_cache:
            left_lane_center = np.argmax(histogram[:midpoint])
        else:
            mean_X = int(np.mean(self.left_lane_center_cache[-cache_size:]))
            lower_bound = mean_X - window_radius
            if lower_bound < 0:
                lower_bound = 0
            upper_bound = mean_X + window_radius
            if upper_bound >= warped_canny_img.shape[1]:
                upper_bound = (warped_canny_img.shape[1] - 1)
            left_lane_center = np.argmax(histogram[lower_bound:upper_bound]) + lower_bound

        if not self.right_lane_center_cache:
            right_lane_center = np.argmax(histogram[midpoint:]) + midpoint
        else:
            mean_Y = int(np.mean(self.right_lane_center_cache[-cache_size:]))
            lower_bound = mean_Y - window_radius
            if lower_bound < 0:
                lower_bound = 0
            upper_bound = mean_Y + window_radius
            if upper_bound >= warped_canny_img.shape[1]:
                upper_bound = (warped_canny_img.shape[1] - 1)
            right_lane_center = np.argmax(histogram[lower_bound:upper_bound]) + lower_bound

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds, y_low, y_high, lx_low, lx_high, l_lane_center = self.get_lane_points(warped_canny_img, left_lane_center, window_height, window_radius, image_debugger, minpix)
        right_lane_inds, y_low, y_high, rx_low, rx_high, r_lane_center = self.get_lane_points(warped_canny_img, right_lane_center, window_height, window_radius, image_debugger, minpix)

        # Extract left and right line pixel positions
        nonzero = warped_canny_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        plot_y = np.linspace(0, warped_canny_img.shape[0] - 1, warped_canny_img.shape[0])

        left_fitx = None
        right_fitx = None

        # Concatenate the arrays of indices
        if len(left_lane_inds) > 0:
            left_lane_inds = np.concatenate(left_lane_inds)
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            left_fitx = poly_fit_left.perform_poly_fit(leftx, lefty, plot_y)
            self.left_lane_center_cache.append(l_lane_center)

        if len(right_lane_inds) > 0:
            right_lane_inds = np.concatenate(right_lane_inds)
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            right_fitx = poly_fit_right.perform_poly_fit(rightx, righty, plot_y)
            self.right_lane_center_cache.append(r_lane_center)


        if image_debugger.is_enabled() or get_points_instead_of_curves:
            img_showing_points_points = np.dstack((warped_canny_img, warped_canny_img, warped_canny_img)) * 255
            if len(left_lane_inds) > 0:
                img_showing_points_points[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
            if len(right_lane_inds) > 0:
                img_showing_points_points[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
            image_debugger.show_image("All points", img_showing_points_points)
            if get_points_instead_of_curves:
                return img_showing_points_points, plot_y, y_low, y_high, lx_low, lx_high, rx_low, rx_high, l_lane_center, r_lane_center

        return left_fitx, right_fitx, plot_y, y_low, y_high, lx_low, lx_high, rx_low, rx_high, l_lane_center, r_lane_center

    def get_lane_points(self, warped_canny_img, starting_point_for_lane, window_height, window_radius, image_debugger, min_pixels_in_window):
        nonzero = warped_canny_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        valid_points_inds = []
        y_low = warped_canny_img.shape[0] - 1 * window_height
        y_high = warped_canny_img.shape[0] - 0 * window_height
        x_low = starting_point_for_lane - window_radius
        x_high = starting_point_for_lane + window_radius
        windows = [[x_low, x_high, y_low, y_high]]
        first_window_lane_center = None

        if image_debugger.is_enabled():
            img_showing_box_all = np.dstack((warped_canny_img, warped_canny_img, warped_canny_img)) * 255
        while True:
            points_in_current_windows = []
            if image_debugger.is_enabled():
                img_showing_box_per_iteration = np.dstack((warped_canny_img, warped_canny_img, warped_canny_img)) * 255
            for win in windows:
                if image_debugger.is_enabled():
                    cv2.rectangle(img_showing_box_per_iteration, (win[0], win[2]), (win[1], win[3]), (100, 255, 0), 2)
                    cv2.rectangle(img_showing_box_all, (win[0], win[2]), (win[1], win[3]), (100, 255, 0), 2)
                good_point_inds = ((nonzeroy >= win[2]) & (nonzeroy < win[3]) & (nonzerox >= win[0]) & (nonzerox < win[1])).nonzero()[0]
                if len(good_point_inds) > min_pixels_in_window:
                    points_in_current_windows.append(good_point_inds)
                    # Store starting window points
                    if first_window_lane_center is None:
                        non_zero_coordinates = np.transpose(nonzero)
                        good_points = np.empty((len(good_point_inds), 2), dtype=int)
                        i = 0
                        for index in good_point_inds:
                            good_points[i] = non_zero_coordinates[index]
                            i = i + 1
                        first_window_lane_center = np.mean(np.transpose(good_points)[1])
            if image_debugger.is_enabled():
                image_debugger.show_image("Box", img_showing_box_per_iteration)

            if len(points_in_current_windows) > 0:
                points_in_current_windows = np.concatenate(points_in_current_windows)
            else:
                break
            if len(valid_points_inds) > 0:
                t = np.concatenate(valid_points_inds)
            else:
                t = []
            new_points = np.setdiff1d(points_in_current_windows, t)
            valid_points_inds.append(new_points)
            if len(new_points) > 0:
                x_mean_for_all_good_points = np.int(np.mean(nonzerox[new_points]))
                y_mean_for_all_good_points = np.int(np.mean(nonzeroy[new_points]))
                # top_win, top_left_win, top_right_win, left_win, right_win
                next_set_of_left_windows = [
                    [x_mean_for_all_good_points - window_radius, x_mean_for_all_good_points + window_radius, y_mean_for_all_good_points - window_height, y_mean_for_all_good_points],
                    [x_mean_for_all_good_points - (3 * window_radius), x_mean_for_all_good_points - window_radius, y_mean_for_all_good_points - window_height, y_mean_for_all_good_points],
                    [x_mean_for_all_good_points + window_radius, x_mean_for_all_good_points + (3 * window_radius), y_mean_for_all_good_points - window_height, y_mean_for_all_good_points],
                    [x_mean_for_all_good_points - (2 * window_radius), x_mean_for_all_good_points, int(y_mean_for_all_good_points - (window_height / 2)), int(y_mean_for_all_good_points + (window_height / 2))],
                    [x_mean_for_all_good_points, x_mean_for_all_good_points + (2 * window_radius), int(y_mean_for_all_good_points - (window_height / 2)), int(y_mean_for_all_good_points + (window_height / 2))]]
            else:
                break

            windows = next_set_of_left_windows
            if image_debugger.is_enabled():
                img_showing_box_per_iteration[nonzeroy[new_points], nonzerox[new_points]] = [255, 0, 100]
                image_debugger.show_image("Boxes", img_showing_box_per_iteration)
        if image_debugger.is_enabled():
            image_debugger.show_image("All Boxes", img_showing_box_all)
        return valid_points_inds, y_low, y_high, x_low, x_high, first_window_lane_center

