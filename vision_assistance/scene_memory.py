class SceneMemory:
    def __init__(self):
        self.last_scene = []

    def update(self, objects):
        self.last_scene = objects

    def describe(self):
        if not self.last_scene:
            return "I do not see anything clearly."
        
        summary = set([obj[0] for obj in self.last_scene])
        return "I can see " + ", ".join(summary)
