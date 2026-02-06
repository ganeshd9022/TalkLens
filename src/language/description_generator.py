def generate_sentence(label, direction, distance):
    if distance == "near":
        dist_phrase = "near you"
    elif distance == "at a moderate distance":
        dist_phrase = "a short distance away"
    else:
        dist_phrase = "far away"

    if direction == "center":
        dir_phrase = "in front of you"
    else:
        dir_phrase = f"on your {direction}"

    sentence = f"There is a {label} {dist_phrase} {dir_phrase}."
    return sentence
