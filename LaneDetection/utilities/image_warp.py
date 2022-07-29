import numpy as np
import cv2


# Utility method to perform image warping
def perspective_warp(input_image, src, dst, scale=True):
    input_image_width = input_image.shape[1]
    input_image_height = input_image.shape[0]
    dst_size = (input_image_width, input_image_height)

    if scale:
        img_size = np.float32([dst_size])
        src = src * img_size
        dst = dst * img_size
    # Given src and dst points, calculate the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(input_image, transform_matrix, dst_size)
    return warped
