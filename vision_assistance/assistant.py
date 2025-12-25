from detector import ObjectDetector
from camera import Camera
from scene_memory import SceneMemory
from voice_output import VoiceOutput
import time

detector = ObjectDetector()
camera = Camera()
memory = SceneMemory()
voice = VoiceOutput()

print("Scanning environment...")

while True:
    frame = camera.get_frame()
    if frame is None:
        continue

    objects = detector.detect(frame)
    memory.update(objects)

    time.sleep(2)

    user_input = input("Ask about environment (type 'q' to quit): ")

    if user_input.lower() == 'q':
        break

    response = memory.describe()
    voice.speak(response)

camera.release()
