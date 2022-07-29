import numpy as np


# Custom class to encapsulate a bounding box to track moving lanes in image
class TrackingBox:
    def __init__(self, cache_size, init_x, init_y, box_id, color):
        self.box_id = box_id
        self.color = color
        self.x_cache = []
        self.y_cache = []
        self.cache_size = cache_size
        self.straight_line = []
        self.straight_line_cache_size = 40
        if init_x is not None:
            self.x_cache.append(init_x)
        if init_y is not None:
            self.y_cache.append(init_y)
        self.x_pos_for_deviation = 0

    def set_color(self, color):
        self.color = color

    def get_mean_box(self):
        if not self.x_cache:
            return None, None
        mean_x = int(np.mean(self.x_cache[-self.cache_size:]))
        mean_y = int(np.mean(self.y_cache[-self.cache_size:]))
        return mean_x, mean_y

    def is_straight_line(self):
        if not self.straight_line:
            return False
        mean = max(set(self.straight_line[-self.straight_line_cache_size:]), key=self.straight_line.count)
        self.straight_line = self.straight_line[-self.straight_line_cache_size:]
        return mean

    def get_box_id(self):
        return self.box_id

    def get_color(self):
        return self.color

    def add(self, x, y, is_straight_line):
        self.x_cache.append(x)
        self.y_cache.append(y)
        self.straight_line.append(is_straight_line)

    def clear_and_add(self, x, y):
        self.x_cache = [x]
        self.y_cache = [y]

    def make_straight_line(self):
        self.straight_line = [True, True, True, True, True, True, True]

    def make_dashed_line(self):
        self.straight_line = [False, False, False, False, False, False]

    def set_x_for_deviation(self, x):
        self.x_pos_for_deviation = x

    def x_for_deviation(self):
        if self.is_straight_line():
            return self.x_pos_for_deviation
        else:
            return int(np.mean(self.x_cache[-self.cache_size:]))
