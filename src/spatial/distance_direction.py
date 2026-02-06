def get_direction(x_center, frame_width):
    if x_center < frame_width / 3:
        return "left"
    elif x_center < (2 * frame_width) / 3:
        return "center"
    else:
        return "right"
