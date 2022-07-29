import math

# This file contains helper methods

def get_color(cluster):
    if cluster == 1:  # RED
        return [0, 0, 255]
    elif cluster == 2:  # ORANGE
        return [0, 139, 255]
    elif cluster == 3:  # YELLOW
        return [0, 243, 255]
    elif cluster == 4:  # LIME GREEN
        return [0, 255, 175]
    elif cluster == 5:  # LIME BLUE
        return [251, 255, 0]
    elif cluster == 6:  # DARK BLUE
        return [255, 39, 0]
    elif cluster == 7:  # PING
        return [255, 0, 255]
    elif cluster == 8:  # WHITE
        return [255, 255, 255]
    elif cluster == 9:  # GREENISH BROWN
        return [57, 102, 99]
    elif cluster == 10:  # BROWN
        return [87, 87, 142]
    else:
        print("No value for cluster {}".format(cluster))
        return [102, 62, 57]


def get_left_right_windows(tracking_windows, car_middle):
    diff_right = None
    diff_left = None
    win_left = None
    win_right = None
    # TODO: Add x pos for deviation for asw
    for win in tracking_windows:
        x = win.x_for_deviation()
        raw_diff = car_middle - x
        if raw_diff > 0:
            # Dealing with window on left of car
            if diff_left is None:
                diff_left = raw_diff
                win_left = win
            else:
                if raw_diff < diff_left:
                    diff_left = raw_diff
                    win_left = win
        else:
            # Dealing with window on right of car
            abs_diff = math.fabs(raw_diff)
            if diff_right is None:
                diff_right = abs_diff
                win_right = win
            else:
                if abs_diff < diff_right:
                    diff_right = abs_diff
                    win_right = win
    return win_left, win_right
