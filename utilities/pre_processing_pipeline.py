from utilities.imageHelper import keep_color, region_of_interest
import cv2
import numpy as np


def pipeline(image, image_debugger, cropped_image=True, is_input_for_performance=False):
    image_debugger.show_image("pipeline input", image)
    if is_input_for_performance:
        gray_image = image
        white_extracted_image = image
        image_debugger.show_image("Not performing white, greyscale as its performance", gray_image)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), white_extracted_image
    else:
        white_extracted_image = keep_color(image, 70)  # 70 - for 0,1 80 for 2,3,4
        image_debugger.show_image("white", white_extracted_image)

        gray_image = cv2.cvtColor(white_extracted_image, cv2.COLOR_RGB2GRAY)
        image_debugger.show_image_grey("gray scale image", gray_image)

        binary = np.where(gray_image != 0, 1, 0)
        image_debugger.show_image_grey("binary image just for display. Its not used.", binary)

        kernel_size = 3
        blur_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
        image_debugger.show_image("blur image", blur_image)
        canny_image = cv2.Canny(blur_image, 250, 500)
        image_debugger.show_image_grey("canny image", canny_image)

        if cropped_image:
            height = image.shape[0]
            width = image.shape[1]
            # Commented out below logic that crops part of the image as we are working on lanes drawn in the laboratory. This is useful when working in real world as top part of the image is usually Sky and can be cropped out.
            # Similarly, in real world, the image can be cropped in triangle form as captured lanes are usually in that shape.
            # Curved Lanes cannot be seen in usual triangular shape. We need to rely on maps to get the shape of upcoming lane (OR other algorithms to recognize shape of curved lane) and crop the image appropriately.

            # region_of_interest_vertices = [
            #     (0, height), (width / 4, 5 * height / 10),
            #     (3 * width / 4, 5 * height / 10),
            #     (width, height)]

            # We were cropping below when camera position was a bit low and car hood is in video taken
            # region_of_interest_vertices = [
            #     (0, height-50), (0, 0),
            #     (width, 0),
            #     (width, height-50)]

            # region_of_interest_vertices = [
            #     (0, height), (0, 200),
            #     (width, 200),
            #     (width, height)]
            #

            # For hough transform
            # region_of_interest_vertices = [
            #     (0, height), (473,251), (1275, 251),
            #     (width, height)]
            #
            # canny_image = region_of_interest(
            #     canny_image,
            #     np.array(
            #         [region_of_interest_vertices],
            #         np.int32
            #     ),
            # )
            image_debugger.show_image("Cropped image", canny_image)
            return canny_image, white_extracted_image
