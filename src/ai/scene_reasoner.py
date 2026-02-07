def reason_about_scene(scene, question):
    question = question.lower()

    if not scene:
        return "I do not see any obstacles in front of you."

    # ----------------------------
    # Define priority categories
    # ----------------------------
    dangerous_objects = ["car", "bus", "truck", "bike", "motorcycle"]
    obstacles = ["chair", "table", "bench", "box"]

    # ----------------------------
    # 1. SAFETY CHECK
    # ----------------------------
    if "safe" in question or "danger" in question or "move" in question:
        # Highest priority: dangerous moving objects
        for obj in scene:
            if obj["label"] in dangerous_objects:
                return (
                    f"There is a {obj['label']} {obj['distance']} on your "
                    f"{obj['direction']}. Please be careful."
                )

        # Next priority: obstacles in front
        for obj in scene:
            if obj["direction"] == "center" and obj["distance"] != "far":
                return (
                    f"There is a {obj['label']} in front of you. "
                    "It may not be safe to move forward."
                )

        return "It appears safe to move forward."

    # ----------------------------
    # 2. PERSON RELATED QUESTIONS
    # ----------------------------
    if "person" in question or "people" in question:
        persons = [o for o in scene if o["label"] == "person"]
        if persons:
            p = persons[0]
            return f"There is a person {p['distance']} on your {p['direction']}."
        else:
            return "I do not see any person nearby."

    # ----------------------------
    # 3. WHAT IS AROUND / IN FRONT
    # ----------------------------
    if "front" in question or "see" in question or "what" in question:
        descriptions = []
        for o in scene:
            descriptions.append(
                f"a {o['label']} {o['distance']} on your {o['direction']}"
            )
        return "I can see " + ", ".join(descriptions) + "."

    # ----------------------------
    # 4. FALLBACK
    # ----------------------------
    return "I am unable to answer that based on what I see."
