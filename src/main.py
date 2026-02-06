import cv2
from detection.object_detection import ObjectDetector
from spatial.distance_direction import get_direction
from audio.text_to_speech import speak

camera = cv2.VideoCapture(0)
detector = ObjectDetector()

while True:
    success, frame = camera.read()
    if not success:
        break

    frame_width = frame.shape[1]
    detections = detector.detect(frame)

    for obj in detections:
        direction = get_direction(obj["x_center"], frame_width)
        message = f"{obj['label']} detected on the {direction}"
        print(message)
        speak(message)

    cv2.imshow("TalkLens - Speaking Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
