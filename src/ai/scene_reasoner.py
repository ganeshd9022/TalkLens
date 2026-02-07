def reason_about_scene(scene, question):
    question = question.lower().strip()

    if not scene:
        return "I do not see any obstacles in front of you."

    # Priority definitions
    dangerous_objects = ["car", "bus", "truck", "bike", "motorcycle"]
    blocking_distance = ["near", "at a moderate distance"]

    # ----------------------------
    # SAFETY & MOVEMENT QUESTIONS
    # ----------------------------
    if any(word in question for word in ["safe", "danger", "move", "walk", "go"]):
        # 1. Moving vehicles (highest priority)
        for obj in scene:
            if obj["label"] in dangerous_objects:
                return (
                    f"There is a {obj['label']} {obj['distance']} on your "
                    f"{obj['direction']}. Please stop and be careful."
                )

        # 2. Obstacle directly ahead
        for obj in scene:
            if obj["direction"] == "center" and obj["distance"] in blocking_distance:
                return (
                    f"There is a {obj['label']} in front of you. "
                    "It may not be safe to move forward."
                )

        return "It appears safe to move forward."

    # ----------------------------
    # PERSON-RELATED QUESTIONS
    # ----------------------------
    if "person" in question or "people" in question:
        for obj in scene:
            if obj["label"] == "person":
                return f"There is a person {obj['distance']} on your {obj['direction']}."
        return "I do not see any person nearby."

    # ----------------------------
    # WHAT IS AROUND / IN FRONT
    # ----------------------------
    if any(word in question for word in ["front", "around", "see", "what"]):
        descriptions = []
        for obj in scene:
            descriptions.append(
                f"a {obj['label']} {obj['distance']} on your {obj['direction']}"
            )

        if len(descriptions) == 1:
            return f"I can see {descriptions[0]}."
        else:
            return "I can see " + ", ".join(descriptions) + "."

    # ----------------------------
    # FALLBACK (CONTROLLED)
    # ----------------------------
    return "I am unable to answer that based on what I see."
