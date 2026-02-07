import cv2

from detection.object_detection import ObjectDetector
from spatial.distance_direction import get_direction, get_distance
from audio.text_to_speech import speak
from memory.scene_memory import SceneMemory
from interface.voice_input import listen_once
from ai.llm_reasoner import ask_llama

# ---------------------------------
# PERFORMANCE CONFIG
# ---------------------------------
FRAME_SKIP = 5          # YOLO runs every 5 frames
frame_count = 0

# ---------------------------------
# INITIALIZE COMPONENTS
# ---------------------------------
camera = cv2.VideoCapture(0)
detector = ObjectDetector()
memory = SceneMemory()

last_detections = []      # stores last YOLO output
cached_scene = []         # structured scene for LLaMA
last_scene_signature = None

print("✅ TalkLens started")
print("👉 Press SPACE to ask a question")
print("👉 Press Q to quit")

# ---------------------------------
# MAIN LOOP
# ---------------------------------
while True:
    success, frame = camera.read()
    if not success:
        print("❌ Camera not accessible")
        break

    frame_width = frame.shape[1]
    frame_count += 1

    # ---------------------------------
    # RUN YOLO (ONLY SOMETIMES)
    # ---------------------------------
    if frame_count % FRAME_SKIP == 0:
        last_detections = detector.detect(frame)

        scene = []
        scene_signature = []

        for obj in last_detections:
            direction = get_direction(obj["x_center"], frame_width)
            distance = get_distance(obj["box_width"], frame_width)

            scene.append({
                "label": obj["label"],
                "direction": direction,
                "distance": distance
            })

            scene_signature.append(
                f"{obj['label']}-{direction}-{distance}"
            )

        signature = "|".join(scene_signature)

        if signature != last_scene_signature:
            cached_scene = scene
            last_scene_signature = signature

    # ---------------------------------
    # DRAW BOUNDING BOXES (EVERY FRAME)
    # ---------------------------------
    for obj in last_detections:
        x1, y1, x2, y2 = obj["bbox"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            obj["label"],
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # ---------------------------------
    # DISPLAY CAMERA FEED
    # ---------------------------------
    cv2.imshow("TalkLens - Vision Assistance", frame)
    key = cv2.waitKey(1) & 0xFF

    # ---------------------------------
    # PUSH-TO-TALK (SPACE)
    # ---------------------------------
    if key == ord(' '):
        speak("Listening")
        question = listen_once()

        if question.strip() == "":
            speak("I did not hear a question.")
        else:
            answer = ask_llama(cached_scene, question)

            if memory.should_speak(answer):
                print("🧠 Answer:", answer)
                speak(answer)

    # ---------------------------------
    # QUIT
    # ---------------------------------
    if key == ord('q'):
        break

# ---------------------------------
# CLEANUP
# ---------------------------------
camera.release()
cv2.destroyAllWindows()
print("👋 TalkLens closed")
