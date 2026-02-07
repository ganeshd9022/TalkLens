def answer_question(scene, question):
    question = question.lower()

    if "person" in question:
        persons = [obj for obj in scene if obj["label"] == "person"]
        if persons:
            p = persons[0]
            return f"Yes, there is a person {p['distance']} on your {p['direction']}."
        else:
            return "No, I do not see any person around you."

    if "danger" in question or "safe" in question:
        dangerous_objects = ["car", "bus", "truck", "bike"]
        dangers = [obj for obj in scene if obj["label"] in dangerous_objects]
        if dangers:
            d = dangers[0]
            return f"Be careful. There is a {d['label']} {d['distance']} on your {d['direction']}."
        else:
            return "I do not see any immediate danger around you."

    if "what is in front" in question or "describe" in question:
        if not scene:
            return "I do not see any objects in front of you."
        response = "I can see "
        response += ", ".join(
            [f"a {obj['label']} {obj['distance']} on your {obj['direction']}" for obj in scene]
        )
        return response + "."

    return "I am not sure how to answer that yet."
