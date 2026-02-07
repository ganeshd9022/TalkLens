def reason_about_scene(scene, question):
    question = question.lower().strip()

    if not scene:
        return "It is safe to move forward."

    dangerous_objects = ["car", "bus", "truck", "bike", "motorcycle"]
    blocking_distance = ["near", "at a moderate distance"]

    # ---------------------------------
    # SAFETY & NAVIGATION
    # ---------------------------------
    if any(word in question for word in ["safe", "move", "walk", "go", "forward"]):
        # Highest priority: vehicles
        for obj in scene:
            if obj["label"] in dangerous_objects:
                return (
                    f"There is a {obj['label']} {obj['distance']} on your "
                    f"{obj['direction']}. Please stop."
                )

        # Blocking obstacle in front
        for obj in scene:
            if obj["direction"] == "center" and obj["distance"] in blocking_distance:
                # Decide safer side
                left_blocked = any(
                    o["direction"] == "left" and o["distance"] in blocking_distance
                    for o in scene
                )
                right_blocked = any(
                    o["direction"] == "right" and o["distance"] in blocking_distance
                    for o in scene
                )

                if not left_blocked:
                    return (
                        f"There is a {obj['label']} in front of you. "
                        "Please step left."
                    )
                elif not right_blocked:
                    return (
                        f"There is a {obj['label']} in front of you. "
                        "Please step right."
                    )
                else:
                    return (
                        f"There is a {obj['label']} blocking your path. "
                        "Please stop."
                    )

        return "It appears safe to move forward."

    # ---------------------------------
    # PERSON RELATED
    # ---------------------------------
    if "person" in question or "people" in question:
        for obj in scene:
            if obj["label"] == "person":
                return f"There is a person {obj['distance']} on your {obj['direction']}."
        return "I do not see any person nearby."

    # ---------------------------------
    # WHAT IS AROUND
    # ---------------------------------
    if any(word in question for word in ["see", "around", "front", "what"]):
        descriptions = [
            f"a {o['label']} {o['distance']} on your {o['direction']}"
            for o in scene
        ]
        return "I can see " + ", ".join(descriptions) + "."

    return "I am unable to answer that based on what I see."
