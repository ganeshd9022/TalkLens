import subprocess

MODEL_NAME = "phi3:mini"

def polish_answer(fact_sentence):
    prompt = f"""
Rewrite the sentence below in simple, calm, natural language.
Do NOT add information.
Do NOT remove information.
Do NOT mention AI or assistant.

Sentence:
{fact_sentence}

Return ONE short sentence only.
"""

    result = subprocess.run(
        ["ollama", "run", MODEL_NAME],
        input=prompt,
        text=True,
        capture_output=True
    )

    answer = result.stdout.strip()

    # Hard safety trim
    if "." in answer:
        answer = answer.split(".")[0] + "."

    return answer
