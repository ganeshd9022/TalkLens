import cv2
from detection.object_detection import ObjectDetector
from spatial.distance_direction import get_direction, get_distance
from audio.text_to_speech import speak
from memory.scene_memory import SceneMemory
from interface.voice_input import listen_command

camera = cv2.VideoCapture(0)
detector = ObjectDetector()
memory = SceneMemory()

latest_messages = []

while True:
    success, frame = camera.read()
    if not success:
        break

    frame_width = frame.shape[1]
    detections = detector.detect(frame)

    latest_messages.clear()

    for obj in detections:
        direction = get_direction(obj["x_center"], frame_width)
        distance = get_distance(obj["box_width"], frame_width)
        message = f"{obj['label']} detected on the {direction}, {distance}"
        latest_messages.append(message)

    cv2.imshow("TalkLens - Voice Controlled", frame)

    # Listen for voice command (press 'v' to activate listening)
    if cv2.waitKey(1) & 0xFF == ord('v'):
        command = listen_command()

        if "what is in front of me" in command:
            if latest_messages:
                for msg in latest_messages:
                    if memory.should_speak(msg):
                        speak(msg)
            else:
                speak("I do not see any objects")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
