import matplotlib.pyplot as plt
import numpy as np
import cv2

from utilities.image_warp import perspective_warp

# This file contains helper methods

def showImage(filename, image):
    plt.imshow(image)
    plt.title(filename)
    plt.show()
    return


def SaveFile(filename, image):
    plt.imshow(image)
    plt.title(filename)
    plt.savefig(filename + ".jpg", bbox_inches='tight')
    return


# Takes image and polygon vertices and returns cropped image
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Takes hough lines and draws them on image
def displayAllHoughLines(lines, image):
    all_hough_line_coordinates = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            all_hough_line_coordinates.append([x1, y1, x2, y2])

    image_with_all_lines = draw_lines(
        image,
        [all_hough_line_coordinates],
        thickness=2
    )
    return image_with_all_lines


# Draw lines on provided image "img" with give color and thickness
# Also displays lines that are drawn before displaying on image - uses optionalText to display image title
def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8
    )
    # img = np.copy(img)
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img


# Takes hough lines which are categorized as left right and omitted based on line slope and displays then with different colors
def display_all_hough_lines_based_on_category(left_category, right_category, omitted_category, image):
    image_showing_all_lines_with_categories = draw_lines(
        image,
        [left_category],
        [255, 0, 0],
        thickness=2
    )
    image_showing_all_lines_with_categories = draw_lines(
        image_showing_all_lines_with_categories,
        [right_category],
        [0, 255, 0],
        thickness=2
    )
    image_showing_all_lines_with_categories = draw_lines(
        image_showing_all_lines_with_categories,
        [omitted_category],
        [0, 0, 255],
        thickness=2
    )
    showImage("Showing All Lines With Categories \n {Green: RIGHT}  {Red: LEFT}  {Blue:OMMITTED}",
              image_showing_all_lines_with_categories)
    return


def keep_color(image, sensitivity=25):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(image, image, mask=mask)

    # cv2.imshow('frame', image)
    # cv2.imshow('mask', mask)
    # cv2.imshow('res', res)
    #
    # cv2.waitKey(30000)
    # cv2.imwrite("/Users/GautamChand/PycharmProjects/car/bagfiles/mask.jpeg", mask)
    return res


def keep_color_red(image, sensitivity=25):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 150, 150])
    upper_white = np.array([200, 255, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(image, image, mask=mask)
    return res


# For Performance test videos
def keep_red_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    upper_white = np.array([20, 255, 255])  # 0,255,255
    lower_white = np.array([0, 160, 160])

    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(image, image, mask=mask)
    return res


# For Performance test videos
def keep_green_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    upper_white = np.array([70, 255, 255])  # 60,255,255
    lower_white = np.array([40, 160, 160])

    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(image, image, mask=mask)
    return res


# For Performance test videos
def keep_yellow_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    upper_white = np.array([38, 255, 255])  # 28,255,255
    lower_white = np.array([10, 160, 160])

    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(image, image, mask=mask)
    return res


# For Performance test videos
def keep_blue_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    upper_white = np.array([130, 255, 255])  # 120,255,255
    lower_white = np.array([80, 160, 160])

    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(image, image, mask=mask)
    return res


# Get image histogram from given range of height and all width of image
def get_hist(img):
    # hist = np.sum(img[img.shape[0] // 2:, :], axis=0)
    hist = np.sum(img[img.shape[0] - 120:img.shape[0] - 50, :], axis=0)
    return hist


def write_text_on_image(image, text, position, font_color=(0, 0, 0), font_size=0.5, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font,
                font_size,
                font_color, thickness)


def get_image_with_highlighted_curves(input_image, left_fit, right_fit, src_a, dst_a, is_scaled_warping,
                                      win_y_low, win_y_high, win_x_left_low, win_x_left_high, win_x_right_low, win_x_right_high):
    image_with_curves = np.zeros_like(input_image)
    cv2.rectangle(image_with_curves, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), (100, 255, 255), 2)
    cv2.rectangle(image_with_curves, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), (100, 255, 0), 2)

    if left_fit is None or right_fit is None:
        if left_fit is None:
            write_text_on_image(input_image, "No left lane detected", (10, 60), (60, 20, 220), 1)
        if right_fit is None:
            write_text_on_image(input_image, "No right lane detected", (10, 100), (60, 20, 220), 1)
    else:
        plot_y = np.linspace(0, input_image.shape[0] - 1, input_image.shape[0])

        left = np.array([np.transpose(np.vstack([left_fit, plot_y]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, plot_y])))])
        points = np.hstack((left, right))

        cv2.fillPoly(image_with_curves, np.int_(points), (0, 255, 130))

    inv_perspective_on_curves = perspective_warp(image_with_curves,
                                                 src_a,
                                                 dst_a,
                                                 is_scaled_warping
                                                 )
    input_image_overlapped_with_curves = cv2.addWeighted(input_image, 1, inv_perspective_on_curves, 0.7, 0)
    return input_image_overlapped_with_curves


def get_image_with_highlighted_clusters(overlay_base_img,
                                        un_warp_overlay_base_img,
                                        overlay_image_weight,
                                        clustered_img,
                                        left_cluster_value,
                                        right_cluster_value,
                                        do_un_warp,
                                        warp_src,
                                        warp_dst,
                                        is_scaled_warping,
                                        win_y_low,
                                        win_y_high,
                                        win_x_left_low,
                                        win_x_left_high,
                                        win_x_right_low,
                                        win_x_right_high):
    image_with_clusters = np.zeros_like(overlay_base_img)

    if left_cluster_value is not None:
        left_cluster_points = np.where(clustered_img == left_cluster_value, left_cluster_value, 0).nonzero()
        left_points_to_draw = np.transpose((left_cluster_points[1], left_cluster_points[0]))
        for point in left_points_to_draw:
            image_with_clusters[point[1], point[0]] = [0, 123, 255]

    if right_cluster_value is not None:
        right_cluster_points = np.where(clustered_img == right_cluster_value, right_cluster_value, 0).nonzero()
        right_points_to_draw = np.transpose((right_cluster_points[1], right_cluster_points[0]))
        for point in right_points_to_draw:
            image_with_clusters[point[1], point[0]] = [255, 123, 0]

    cv2.rectangle(image_with_clusters, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), (100, 255, 255), 2)
    cv2.rectangle(image_with_clusters, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), (100, 255, 0), 2)

    if un_warp_overlay_base_img:
        overlay_base_img = perspective_warp(overlay_base_img, warp_src, warp_dst, is_scaled_warping)
    if do_un_warp:
        image_with_clusters = perspective_warp(image_with_clusters, warp_src, warp_dst, is_scaled_warping)
    output_image = cv2.addWeighted(overlay_base_img, overlay_image_weight, image_with_clusters, 1, 0)
    if left_cluster_value is None:
        write_text_on_image(output_image, "No left lane detected", (10, 60), (60, 20, 220), 1)
    if right_cluster_value is None:
        write_text_on_image(output_image, "No right lane detected", (10, 100), (60, 20, 220), 1)
    return output_image


def get_image_with_highlighted_points_from_sliding_window(overlay_base_img,
                                                          un_warp_overlay_base_img,
                                                          overlay_image_weight,
                                                          image_with_points,
                                                          do_un_warp,
                                                          left_center,
                                                          right_center,
                                                          warp_src,
                                                          warp_dst,
                                                          is_scaled_warping,
                                                          win_y_low,
                                                          win_y_high,
                                                          win_x_left_low,
                                                          win_x_left_high,
                                                          win_x_right_low,
                                                          win_x_right_high):
    cv2.rectangle(image_with_points, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), (100, 255, 255), 2)
    cv2.rectangle(image_with_points, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), (100, 255, 0), 2)

    if un_warp_overlay_base_img:
        overlay_base_img = perspective_warp(overlay_base_img, warp_src, warp_dst, is_scaled_warping)

    if do_un_warp:
        image_with_points = perspective_warp(image_with_points, warp_src, warp_dst, is_scaled_warping)
    if left_center is None:
        write_text_on_image(image_with_points, "No left lane detected", (10, 60), (60, 20, 220), 1)
    if right_center is None:
        write_text_on_image(image_with_points, "No right lane detected", (10, 100), (60, 20, 220), 1)
    output_image = cv2.addWeighted(overlay_base_img, overlay_image_weight, image_with_points, 1, 0)
    return output_image
