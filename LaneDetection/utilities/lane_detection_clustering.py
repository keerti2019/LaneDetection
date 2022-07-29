import numpy as np
import cv2
from utilities.polyfit import PolyFit
from utilities.imageHelper import get_hist


class ClusteringLaneDetection:
    def __init__(self):
        self.left_lane_center_cache = []
        self.right_lane_center_cache = []
        # boxes = [ yl, yh, xl, xh]
        self.boxes = [[0.0, 55.0, 0.0, 71.0], [0.0, 55.0, 71.0, 142.0], [0.0, 55.0, 142.0, 213.0], [0.0, 55.0, 213.0, 284.0], [0.0, 55.0, 284.0, 355.0], [0.0, 55.0, 355.0, 426.0], [0.0, 55.0, 426.0, 497.0], [0.0, 55.0, 497.0, 568.0],
                      [0.0, 55.0, 568.0, 639.0], [0.0, 55.0, 639.0, 710.0], [0.0, 55.0, 710.0, 781.0], [0.0, 55.0, 781.0, 852.0], [0.0, 55.0, 852.0, 923.0], [0.0, 55.0, 923.0, 994.0], [0.0, 55.0, 994.0, 1065.0], [0.0, 55.0, 1065.0, 1136.0],
                      [0.0, 55.0, 1136.0, 1207.0], [0.0, 55.0, 1207.0, 1278.0], [55.0, 110.0, 0.0, 71.0], [55.0, 110.0, 71.0, 142.0], [55.0, 110.0, 142.0, 213.0], [55.0, 110.0, 213.0, 284.0], [55.0, 110.0, 284.0, 355.0], [55.0, 110.0, 355.0, 426.0],
                      [55.0, 110.0, 426.0, 497.0], [55.0, 110.0, 497.0, 568.0], [55.0, 110.0, 568.0, 639.0], [55.0, 110.0, 639.0, 710.0], [55.0, 110.0, 710.0, 781.0], [55.0, 110.0, 781.0, 852.0], [55.0, 110.0, 852.0, 923.0],
                      [55.0, 110.0, 923.0, 994.0], [55.0, 110.0, 994.0, 1065.0], [55.0, 110.0, 1065.0, 1136.0], [55.0, 110.0, 1136.0, 1207.0], [55.0, 110.0, 1207.0, 1278.0], [110.0, 165.0, 0.0, 71.0], [110.0, 165.0, 71.0, 142.0],
                      [110.0, 165.0, 142.0, 213.0], [110.0, 165.0, 213.0, 284.0], [110.0, 165.0, 284.0, 355.0], [110.0, 165.0, 355.0, 426.0], [110.0, 165.0, 426.0, 497.0], [110.0, 165.0, 497.0, 568.0], [110.0, 165.0, 568.0, 639.0],
                      [110.0, 165.0, 639.0, 710.0], [110.0, 165.0, 710.0, 781.0], [110.0, 165.0, 781.0, 852.0], [110.0, 165.0, 852.0, 923.0], [110.0, 165.0, 923.0, 994.0], [110.0, 165.0, 994.0, 1065.0], [110.0, 165.0, 1065.0, 1136.0],
                      [110.0, 165.0, 1136.0, 1207.0], [110.0, 165.0, 1207.0, 1278.0], [165.0, 220.0, 0.0, 71.0], [165.0, 220.0, 71.0, 142.0], [165.0, 220.0, 142.0, 213.0], [165.0, 220.0, 213.0, 284.0], [165.0, 220.0, 284.0, 355.0],
                      [165.0, 220.0, 355.0, 426.0], [165.0, 220.0, 426.0, 497.0], [165.0, 220.0, 497.0, 568.0], [165.0, 220.0, 568.0, 639.0], [165.0, 220.0, 639.0, 710.0], [165.0, 220.0, 710.0, 781.0], [165.0, 220.0, 781.0, 852.0],
                      [165.0, 220.0, 852.0, 923.0], [165.0, 220.0, 923.0, 994.0], [165.0, 220.0, 994.0, 1065.0], [165.0, 220.0, 1065.0, 1136.0], [165.0, 220.0, 1136.0, 1207.0], [165.0, 220.0, 1207.0, 1278.0], [220.0, 275.0, 0.0, 71.0],
                      [220.0, 275.0, 71.0, 142.0], [220.0, 275.0, 142.0, 213.0], [220.0, 275.0, 213.0, 284.0], [220.0, 275.0, 284.0, 355.0], [220.0, 275.0, 355.0, 426.0], [220.0, 275.0, 426.0, 497.0], [220.0, 275.0, 497.0, 568.0],
                      [220.0, 275.0, 568.0, 639.0], [220.0, 275.0, 639.0, 710.0], [220.0, 275.0, 710.0, 781.0], [220.0, 275.0, 781.0, 852.0], [220.0, 275.0, 852.0, 923.0], [220.0, 275.0, 923.0, 994.0], [220.0, 275.0, 994.0, 1065.0],
                      [220.0, 275.0, 1065.0, 1136.0], [220.0, 275.0, 1136.0, 1207.0], [220.0, 275.0, 1207.0, 1278.0], [275.0, 330.0, 0.0, 71.0], [275.0, 330.0, 71.0, 142.0], [275.0, 330.0, 142.0, 213.0], [275.0, 330.0, 213.0, 284.0],
                      [275.0, 330.0, 284.0, 355.0], [275.0, 330.0, 355.0, 426.0], [275.0, 330.0, 426.0, 497.0], [275.0, 330.0, 497.0, 568.0], [275.0, 330.0, 568.0, 639.0], [275.0, 330.0, 639.0, 710.0], [275.0, 330.0, 710.0, 781.0],
                      [275.0, 330.0, 781.0, 852.0], [275.0, 330.0, 852.0, 923.0], [275.0, 330.0, 923.0, 994.0], [275.0, 330.0, 994.0, 1065.0], [275.0, 330.0, 1065.0, 1136.0], [275.0, 330.0, 1136.0, 1207.0], [275.0, 330.0, 1207.0, 1278.0],
                      [330.0, 385.0, 0.0, 71.0], [330.0, 385.0, 71.0, 142.0], [330.0, 385.0, 142.0, 213.0], [330.0, 385.0, 213.0, 284.0], [330.0, 385.0, 284.0, 355.0], [330.0, 385.0, 355.0, 426.0], [330.0, 385.0, 426.0, 497.0],
                      [330.0, 385.0, 497.0, 568.0], [330.0, 385.0, 568.0, 639.0], [330.0, 385.0, 639.0, 710.0], [330.0, 385.0, 710.0, 781.0], [330.0, 385.0, 781.0, 852.0], [330.0, 385.0, 852.0, 923.0], [330.0, 385.0, 923.0, 994.0],
                      [330.0, 385.0, 994.0, 1065.0], [330.0, 385.0, 1065.0, 1136.0], [330.0, 385.0, 1136.0, 1207.0], [330.0, 385.0, 1207.0, 1278.0], [385.0, 440.0, 0.0, 71.0], [385.0, 440.0, 71.0, 142.0], [385.0, 440.0, 142.0, 213.0],
                      [385.0, 440.0, 213.0, 284.0], [385.0, 440.0, 284.0, 355.0], [385.0, 440.0, 355.0, 426.0], [385.0, 440.0, 426.0, 497.0], [385.0, 440.0, 497.0, 568.0], [385.0, 440.0, 568.0, 639.0], [385.0, 440.0, 639.0, 710.0],
                      [385.0, 440.0, 710.0, 781.0], [385.0, 440.0, 781.0, 852.0], [385.0, 440.0, 852.0, 923.0], [385.0, 440.0, 923.0, 994.0], [385.0, 440.0, 994.0, 1065.0], [385.0, 440.0, 1065.0, 1136.0], [385.0, 440.0, 1136.0, 1207.0],
                      [385.0, 440.0, 1207.0, 1278.0], [440.0, 495.0, 0.0, 71.0], [440.0, 495.0, 71.0, 142.0], [440.0, 495.0, 142.0, 213.0], [440.0, 495.0, 213.0, 284.0], [440.0, 495.0, 284.0, 355.0], [440.0, 495.0, 355.0, 426.0],
                      [440.0, 495.0, 426.0, 497.0], [440.0, 495.0, 497.0, 568.0], [440.0, 495.0, 568.0, 639.0], [440.0, 495.0, 639.0, 710.0], [440.0, 495.0, 710.0, 781.0], [440.0, 495.0, 781.0, 852.0], [440.0, 495.0, 852.0, 923.0],
                      [440.0, 495.0, 923.0, 994.0], [440.0, 495.0, 994.0, 1065.0], [440.0, 495.0, 1065.0, 1136.0], [440.0, 495.0, 1136.0, 1207.0], [440.0, 495.0, 1207.0, 1278.0], [495.0, 550.0, 0.0, 71.0], [495.0, 550.0, 71.0, 142.0],
                      [495.0, 550.0, 142.0, 213.0], [495.0, 550.0, 213.0, 284.0], [495.0, 550.0, 284.0, 355.0], [495.0, 550.0, 355.0, 426.0], [495.0, 550.0, 426.0, 497.0], [495.0, 550.0, 497.0, 568.0], [495.0, 550.0, 568.0, 639.0],
                      [495.0, 550.0, 639.0, 710.0], [495.0, 550.0, 710.0, 781.0], [495.0, 550.0, 781.0, 852.0], [495.0, 550.0, 852.0, 923.0], [495.0, 550.0, 923.0, 994.0], [495.0, 550.0, 994.0, 1065.0], [495.0, 550.0, 1065.0, 1136.0],
                      [495.0, 550.0, 1136.0, 1207.0], [495.0, 550.0, 1207.0, 1278.0], [550.0, 605.0, 0.0, 71.0], [550.0, 605.0, 71.0, 142.0], [550.0, 605.0, 142.0, 213.0], [550.0, 605.0, 213.0, 284.0], [550.0, 605.0, 284.0, 355.0],
                      [550.0, 605.0, 355.0, 426.0], [550.0, 605.0, 426.0, 497.0], [550.0, 605.0, 497.0, 568.0], [550.0, 605.0, 568.0, 639.0], [550.0, 605.0, 639.0, 710.0], [550.0, 605.0, 710.0, 781.0], [550.0, 605.0, 781.0, 852.0],
                      [550.0, 605.0, 852.0, 923.0], [550.0, 605.0, 923.0, 994.0], [550.0, 605.0, 994.0, 1065.0], [550.0, 605.0, 1065.0, 1136.0], [550.0, 605.0, 1136.0, 1207.0], [550.0, 605.0, 1207.0, 1278.0], [605.0, 660.0, 0.0, 71.0],
                      [605.0, 660.0, 71.0, 142.0], [605.0, 660.0, 142.0, 213.0], [605.0, 660.0, 213.0, 284.0], [605.0, 660.0, 284.0, 355.0], [605.0, 660.0, 355.0, 426.0], [605.0, 660.0, 426.0, 497.0], [605.0, 660.0, 497.0, 568.0],
                      [605.0, 660.0, 568.0, 639.0], [605.0, 660.0, 639.0, 710.0], [605.0, 660.0, 710.0, 781.0], [605.0, 660.0, 781.0, 852.0], [605.0, 660.0, 852.0, 923.0], [605.0, 660.0, 923.0, 994.0], [605.0, 660.0, 994.0, 1065.0],
                      [605.0, 660.0, 1065.0, 1136.0], [605.0, 660.0, 1136.0, 1207.0], [605.0, 660.0, 1207.0, 1278.0], [660.0, 715.0, 0.0, 71.0], [660.0, 715.0, 71.0, 142.0], [660.0, 715.0, 142.0, 213.0], [660.0, 715.0, 213.0, 284.0],
                      [660.0, 715.0, 284.0, 355.0], [660.0, 715.0, 355.0, 426.0], [660.0, 715.0, 426.0, 497.0], [660.0, 715.0, 497.0, 568.0], [660.0, 715.0, 568.0, 639.0], [660.0, 715.0, 639.0, 710.0], [660.0, 715.0, 710.0, 781.0],
                      [660.0, 715.0, 781.0, 852.0], [660.0, 715.0, 852.0, 923.0], [660.0, 715.0, 923.0, 994.0], [660.0, 715.0, 994.0, 1065.0], [660.0, 715.0, 1065.0, 1136.0], [660.0, 715.0, 1136.0, 1207.0], [660.0, 715.0, 1207.0, 1278.0]]

    def lane_detection_with_clustered_image(self, clustered_img, image_debugger, window_height=100, window_radius=75, poly_fit_left=PolyFit(5), poly_fit_right=PolyFit(5), get_clusters_instead_of_curves=False):
        left_lane_center = None
        right_lane_center = None

        image_debugger.show_image("Clustered Image", clustered_img)
        histogram = get_hist(clustered_img)
        image_debugger.show_image_histogram(histogram)

        # find peaks of left and right halves
        midpoint = int(histogram.shape[0] / 2)
        cache_size = 5

        if not self.left_lane_center_cache:
            left_x_base = np.argmax(histogram[:midpoint])
        else:
            mean_X = int(np.mean(self.left_lane_center_cache[-cache_size:]))
            lower_bound = mean_X - window_radius
            if lower_bound < 0:
                lower_bound = 0
            upper_bound = mean_X + window_radius
            if upper_bound >= clustered_img.shape[1]:
                upper_bound = (clustered_img.shape[1] - 1)
            left_x_base = np.argmax(histogram[lower_bound:upper_bound]) + lower_bound

        if not self.right_lane_center_cache:
            right_x_base = np.argmax(histogram[midpoint:]) + midpoint
        else:
            mean_Y = int(np.mean(self.right_lane_center_cache[-cache_size:]))
            lower_bound = mean_Y - window_radius
            if lower_bound < 0:
                lower_bound = 0
            upper_bound = mean_Y + window_radius
            if upper_bound >= clustered_img.shape[1]:
                upper_bound = (clustered_img.shape[1] - 1)
            right_x_base = np.argmax(histogram[lower_bound:upper_bound]) + lower_bound

        # Get two windows for base lookup of lanes
        win_y_low = clustered_img.shape[0] - window_height
        win_y_high = clustered_img.shape[0]
        win_x_left_low = left_x_base - window_radius
        win_x_left_high = left_x_base + window_radius
        win_x_right_low = right_x_base - window_radius
        win_x_right_high = right_x_base + window_radius

        out_img = np.dstack((clustered_img, clustered_img, clustered_img)) * 255
        cv2.rectangle(out_img, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), (100, 255, 0), 2)
        cv2.rectangle(out_img, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), (100, 255, 0), 2)
        image_debugger.show_image("Clustered Image with boxes", out_img)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = clustered_img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        # Identify X,Y positions in left and right windows
        good_left_indexs = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                            (nonzero_x >= win_x_left_low) & (nonzero_x < win_x_left_high)).nonzero()[0]
        good_right_indexs = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                             (nonzero_x >= win_x_right_low) & (nonzero_x < win_x_right_high)).nonzero()[0]

        non_zero_coordinates = np.transpose(nonzero)

        left_cluster = None
        right_cluster = None
        left_clusters = []
        right_clusters = []
        if len(good_right_indexs) != 0:
            # get X and Y points from positions
            good_right_points = np.empty((len(good_right_indexs), 2), dtype=int)
            i = 0
            for index in good_right_indexs:
                good_right_points[i] = non_zero_coordinates[index]
                i = i + 1

            right_lane_center = np.mean(np.transpose(good_right_points)[1])
            self.right_lane_center_cache.append(right_lane_center)

            # Identify clusters for all points in windows
            for point in good_right_points:
                right_clusters.append(clustered_img[point[0], point[1]])
            # eliminating unnecessary clusters
            if right_clusters:
                right_cluster = max(set(right_clusters), key=right_clusters.count)

        if len(good_left_indexs) != 0:
            good_left_points = np.empty((len(good_left_indexs), 2), dtype=int)
            i = 0
            for index in good_left_indexs:
                good_left_points[i] = non_zero_coordinates[index]
                i = i + 1

            left_lane_center = np.mean(np.transpose(good_left_points)[1])
            self.left_lane_center_cache.append(left_lane_center)

            for point in good_left_points:
                left_clusters.append(clustered_img[point[0], point[1]])

            if left_clusters:
                left_cluster = max(set(left_clusters), key=left_clusters.count)

        if get_clusters_instead_of_curves:
            return left_cluster, right_cluster, win_y_low, win_y_high, win_x_left_low, win_x_left_high, win_x_right_low, win_x_right_high, left_lane_center, right_lane_center

        plot_y = np.linspace(20, clustered_img.shape[0] - 1, clustered_img.shape[0])
        left_curve = self.get_curve(left_cluster, poly_fit_left, plot_y, clustered_img, image_debugger, "left-lane")
        right_curve = self.get_curve(right_cluster, poly_fit_right, plot_y, clustered_img, image_debugger, "right-lane")

        image_debugger.show_image_with_curves("curves on original img", clustered_img, plot_y, left_curve, right_curve)

        return left_curve, right_curve, win_y_low, win_y_high, win_x_left_low, win_x_left_high, win_x_right_low, win_x_right_high, left_lane_center, right_lane_center

    def get_curve(self, cluster_number, poly_fit, plot_y, clustered_img, image_debugger, lane_name):
        if cluster_number is not None:
            if image_debugger.is_enabled():
                image_debugger.show_image("Cluster points on {} lane".format(lane_name),
                                          np.where(clustered_img == cluster_number, cluster_number, 0))
            return self.fit_average_of_points_in_boxes_to_curve(cluster_number, image_debugger, poly_fit, plot_y, clustered_img)
        else:
            image_debugger.show_image("No cluster points detected on {} lane".format(lane_name), clustered_img)
            return None

    def fit_raw_points_to_curve(self, cluster_number, poly_fit, plot_y, clustered_img):
        raw_lane_points_in_cluster = np.where(clustered_img == cluster_number)
        return poly_fit.perform_poly_fit(raw_lane_points_in_cluster[1], raw_lane_points_in_cluster[0], plot_y)

    def fit_average_of_points_in_boxes_to_curve(self, cluster_number, image_debugger, poly_fit, plot_y, clustered_img):
        min_points_in_box = 50
        nonzero_lane_img = np.where(clustered_img == cluster_number)
        nonzero_lane_img_y = np.array(nonzero_lane_img[0])
        nonzero_lane_img_x = np.array(nonzero_lane_img[1])

        box_points = [[], []]
        image_with_boxes = np.dstack((clustered_img, clustered_img, clustered_img)) * 255
        for box in self.boxes:
            good_indices = ((nonzero_lane_img_y >= box[0]) & (nonzero_lane_img_y < box[1]) &
                            (nonzero_lane_img_x >= box[2]) & (nonzero_lane_img_x < box[3])).nonzero()[0]
            if not good_indices.size:
                continue
            if good_indices.size > min_points_in_box:
                box_points[0].append(np.int(np.mean(nonzero_lane_img_y[good_indices])))
                box_points[1].append(np.int(np.mean(nonzero_lane_img_x[good_indices])))
                if image_debugger.is_enabled():
                    cv2.rectangle(image_with_boxes,
                                  (int(box[2]), int(box[0])), (int(box[3]), int(box[1])),
                                  (100, 255, 0),
                                  3)
        image_debugger.show_image("Boxes containing cluster points", image_with_boxes)
        print(box_points[0])
        print(box_points[1])
        return poly_fit.perform_poly_fit(box_points[1], box_points[0], plot_y)
