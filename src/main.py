import cv2

# open camera (0 = default laptop camera)
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        print("Camera not working")
        break

    # show the camera frame
    cv2.imshow("TalkLens Camera", frame)

    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
