import matplotlib.pyplot as plt
import numpy as np
import cv2

# Contains various helper methods to debug an image
class ImageDebugger:
    def __init__(self, is_show_all_frames_enabled):
        self.is_show_all_frames_enabled = is_show_all_frames_enabled

    def show_image(self, title, image):
        if self.is_show_all_frames_enabled:
            plt.imshow(image)
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.show()

    def show_image_bgr(self, title, image):
        if self.is_show_all_frames_enabled:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.show()

    def show_image_grey(self, title, image):
        if self.is_show_all_frames_enabled:
            plt.imshow(image, cmap='gray')
            plt.title(title)
            plt.show()

    def show_image_with_curves(self, title, image, plot_y, *args):
        if self.is_show_all_frames_enabled:
            plt.imshow(image)
            notFound = False
            for arg in args:
                if arg is not None:
                    plt.plot(arg, plot_y, color='yellow', linewidth=1)
                else:
                    notFound = True
            if notFound:
                plt.title(title + "All Curves Not found")
            else:
                plt.title(title)
            plt.xlim(0,1279)
            plt.ylim(720, 0)
            plt.show()

    def show_image_histogram(self, histogram):
        if self.is_show_all_frames_enabled:
            plt.bar(np.arange(len(histogram)), histogram)
            plt.title("Histogram")
            plt.show()

    def is_enabled(self):
        return self.is_show_all_frames_enabled

    def enabled(self, val):
        self.is_show_all_frames_enabled = val
