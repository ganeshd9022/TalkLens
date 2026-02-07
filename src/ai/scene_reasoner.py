def reason_about_scene(scene, question):
    question = question.lower().strip()

    if not scene:
        return "It is safe to move forward."

    dangerous_objects = ["car", "bus", "truck", "bike", "motorcycle"]

    # ---------------------------------
    # SAFETY & NAVIGATION
    # ---------------------------------
    if any(word in question for word in ["safe", "move", "walk", "go", "forward"]):
        # 1. Vehicles (always critical)
        for obj in scene:
            if obj["label"] in dangerous_objects:
                if obj["distance"] in ["very close", "near"]:
                    return (
                        f"A {obj['label']} is {obj['distance']} on your "
                        f"{obj['direction']}. Please stop immediately."
                    )
                else:
                    return (
                        f"A {obj['label']} is on your {obj['direction']}. "
                        "Please be careful."
                    )

        # 2. Obstacle directly ahead
        for obj in scene:
            if obj["direction"] == "center":
                if obj["distance"] == "very close":
                    return (
                        f"A {obj['label']} is very close in front of you. "
                        "Please stop."
                    )
                if obj["distance"] == "near":
                    return (
                        f"A {obj['label']} is in front of you. "
                        "Please step left or right."
                    )

        return "It appears safe to move forward."

    # ---------------------------------
    # PERSON RELATED
    # ---------------------------------
    if "person" in question or "people" in question:
        for obj in scene:
            if obj["label"] == "person":
                return (
                    f"A person is {obj['distance']} on your {obj['direction']}."
                )
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
