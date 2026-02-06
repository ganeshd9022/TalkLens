import cv2

from detection.object_detection import ObjectDetector
from spatial.distance_direction import get_direction, get_distance
from audio.text_to_speech import speak
from memory.scene_memory import SceneMemory
from interface.voice_input import listen_once
from ai.llm_reasoner import ask_llama

# ---------------------------------
# Initialize system components
# ---------------------------------
camera = cv2.VideoCapture(0)
detector = ObjectDetector()
memory = SceneMemory()

print("✅ TalkLens started")
print("👉 Press SPACE to ask about surroundings")
print("👉 Press Q to quit")

# ---------------------------------
# Main loop
# ---------------------------------
while True:
    success, frame = camera.read()
    if not success:
        print("❌ Camera not accessible")
        break

    frame_width = frame.shape[1]
    detections = detector.detect(frame)

    # ---------------------------------
    # Build structured scene data
    # ---------------------------------
    scene = []

    for obj in detections:
        direction = get_direction(obj["x_center"], frame_width)
        distance = get_distance(obj["box_width"], frame_width)

        scene.append({
            "label": obj["label"],
            "direction": direction,
            "distance": distance
        })

    # Show camera feed
    cv2.imshow("TalkLens - Vision Assistance", frame)

    key = cv2.waitKey(1) & 0xFF

    # ---------------------------------
    # PUSH-TO-TALK (SPACE BAR)
    # ---------------------------------
    if key == ord(' '):
        speak("Listening")
        question = listen_once()

        if question.strip() == "":
            speak("I did not hear a question.")
        else:
            speak("Let me think.")
            answer = ask_llama(scene, question)

            print("🧠 AI Answer:", answer)

            # avoid repeating the same response
            if memory.should_speak(answer):
                speak(answer)

    # ---------------------------------
    # Quit
    # ---------------------------------
    if key == ord('q'):
        break

# ---------------------------------
# Cleanup
# ---------------------------------
camera.release()
cv2.destroyAllWindows()
print("👋 TalkLens closed")
