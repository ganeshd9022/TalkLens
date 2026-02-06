import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 150)  # speech speed

def speak(text):
    engine.say(text)
    engine.runAndWait()

