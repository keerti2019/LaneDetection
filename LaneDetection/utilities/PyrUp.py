import math
import time

import numpy as np

from utilities.image_debugger import ImageDebugger


# cv2.pyrDown is used for rendering down the image
# This method contains logic to render up the image
# Commented logic contains custom pyr-up logic
def pyrUp(clustered_img, times):
    # copy = np.ndarray(shape=(clustered_img.shape[0] * (times * 2), clustered_img.shape[1] * (times * 2)), dtype=int)
    # pow = math.pow(2, times)
    # strat = time.time()
    # # for i in range(0, clustered_img.shape[0]):
    # #     for j in range(0, clustered_img.shape[1]):
    # #         for k in range(int(i * pow), int(i * pow + pow)):
    # #             for l in range(int(j * pow), int(j * pow + pow)):
    # #                 copy[k][l] = clustered_img[i][j]
    #
    # bla = map(get())
    # for i in range(0, clustered_img.shape[0] * (times * 2)):
    #     for j in range(0, clustered_img.shape[1] * (times * 2)):
    #         ll = get(i, j, pow, clustered_img)
    #         copy[i][j] = ll
    # print(time.time() - strat)
    v = np.ones((4, 4), dtype=np.uint8)
    copy = np.kron(clustered_img, v)
    return copy


# def get(x, y, factor, clustered_img):
#     p = int(x / factor)
#     q = int(y / factor)
#     return clustered_img[p][q]
