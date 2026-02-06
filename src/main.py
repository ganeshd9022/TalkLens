import cv2
from detection.object_detection import ObjectDetector

# start camera
camera = cv2.VideoCapture(0)

# load AI detector
detector = ObjectDetector()

while True:
    success, frame = camera.read()

    if not success:
        print("Camera not working")
        break

    # AI detects objects
    objects = detector.detect(frame)

    # show detected objects in terminal
    if objects:
        print("Detected:", objects)

    cv2.imshow("TalkLens - Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
