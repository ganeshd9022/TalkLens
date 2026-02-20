import cv2


class Camera:
    """
    Handles camera initialization and frame retrieval.
    """

    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera.")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera.")
        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()