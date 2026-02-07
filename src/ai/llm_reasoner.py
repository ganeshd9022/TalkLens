import subprocess

MODEL_NAME = "phi3:mini"  # fast local model

def ask_llama(scene, question):
    # Build compact scene description
    if not scene:
        scene_text = "No objects detected."
    else:
        scene_text = ""
        for obj in scene:
            scene_text += (
                f"- {obj['label']} is {obj['distance']} on your {obj['direction']}.\n"
            )

    # STRICT answer-only prompt
    prompt = f"""
You must answer ONLY the user's question.
Do NOT introduce yourself.
Do NOT mention AI, assistant, system, or model.
Do NOT explain your reasoning.
Do NOT add extra information.

Scene:
{scene_text}

Question:
{question}

Answer with ONE short, clear sentence.
"""

    result = subprocess.run(
        ["ollama", "run", MODEL_NAME],
        input=prompt,
        text=True,
        capture_output=True
    )

    answer = result.stdout.strip()

    # Safety: trim to first sentence only
    if "." in answer:
        answer = answer.split(".")[0] + "."

    return answer
