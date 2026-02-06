class SceneMemory:
    def __init__(self):
        self.last_message = None

    def should_speak(self, current_message):
        if current_message != self.last_message:
            self.last_message = current_message
            return True
        return False
