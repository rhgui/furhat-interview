from furhat_realtime_api import FurhatClient
import time
import logging
import os
from dotenv import load_dotenv

load_dotenv()

class FurhatRobot:
    def __init__(self):
        host = os.getenv("FURHAT_HOST")
        self.client = FurhatClient(host)
        self.host = host
        self.connected = False
        logging.basicConfig(level=logging.INFO)
        self.client.set_logging_level(logging.INFO)

    def connect(self):
        self.client.connect()
        self.connected = True
        print(f"Connected to Furhat at {self.host}")
        time.sleep(1)

    def disconnect(self):
        if self.connected:
            self.client.disconnect()
            self.connected = False
            print("Disconnected from Furhat")

    def attend_user(self):
        self.client.request_attend_user("closest")
        time.sleep(0.5)

    def smile_gesture(self):
        self.client.request_gesture_start(name="Smile", wait=True)
        time.sleep(0.5)

    def speak(self, text: str):
        self.client.request_speak_text(text=text, wait=True)
        time.sleep(1)

    def execute_sequence(self, message: str):
        self.attend_user()
        self.smile_gesture()
        self.speak(message)

    def ask_question(self, question_text: str):
        """
        Simple helper: attend the user and ask a single question.
        """
        self.attend_user()
        self.speak(question_text)

    def comfort_and_ask_easy(self, comfort_text: str, easy_question_text: str):
        """
        Used when module detects nervousness (returns 'Yes').

        The robot first says a comfort sentence, then asks an easier question.
        """
        self.attend_user()
        # Comfort message
        self.speak(comfort_text)
        # Then ask the easy question
        self.speak(easy_question_text)
