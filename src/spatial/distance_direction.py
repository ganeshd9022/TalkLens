def get_direction(x_center, frame_width):
    if x_center < frame_width / 3:
        return "left"
    elif x_center < (2 * frame_width) / 3:
        return "center"
    else:
        return "right"


def get_distance(box_width, frame_width):
    ratio = box_width / frame_width

    if ratio > 0.55:
        return "very close"
    elif ratio > 0.35:
        return "near"
    elif ratio > 0.2:
        return "at a moderate distance"
    else:
        return "far"
