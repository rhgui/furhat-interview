from furhat_realtime_api import FurhatClient
import time
import logging
import os
from dotenv import load_dotenv

import asyncio
import websockets
import json
import base64
from typing import Optional, Callable

import numpy as np

load_dotenv()

class FurhatRobot:
    """
    High-level wrapper for controlling the Furhat robot.

    This class combines:
      1) The original FurhatClient-based API (speak, gesture, attend)
      2) A WebSocket-based Realtime API client for:
           - camera frames
           - raw microphone audio
           - ASR partial/final results

    You can keep using the old methods (execute_sequence, speak, etc)
    AND additionally call the new async realtime methods for streaming.
    """
    def __init__(self):
        host = os.getenv("FURHAT_HOST")
        self.client = FurhatClient(host)
        self.host = host
        self.connected = False
        logging.basicConfig(level=logging.INFO)
        self.client.set_logging_level(logging.INFO)

        # ===== New fields for Realtime WebSocket =====
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.realtime_connected: bool = False
        self._event_loop_task: Optional[asyncio.Task] = None
        self._running: bool = False

        # Streaming state flags
        self.camera_running: bool = False
        self.audio_running: bool = False
        self.listening: bool = False

        # Optional callbacks (set by user)
        #   frame:    decoded camera frame (np.ndarray, BGR)
        #   audio:    raw PCM16 bytes from microphone
        #   partial:  ASR partial text
        #   final:    ASR final text
        self.on_frame: Optional[Callable[[np.ndarray], None]] = None
        self.on_audio: Optional[Callable[[bytes], None]] = None
        self.on_partial_text: Optional[Callable[[str], None]] = None
        self.on_final_text: Optional[Callable[[str], None]] = None

    def connect(self):
        """
        Connect using the high-level FurhatClient (non-realtime).
        This is typically used for speak/gesture/attend actions.
        """
        self.client.connect()
        self.connected = True
        print(f"Connected to Furhat at {self.host}")
        time.sleep(1)

    def disconnect(self):
        """
        Disconnect the high-level FurhatClient.
        Does NOT close the realtime WebSocket (see disconnect_realtime()).
        """
        if self.connected:
            self.client.disconnect()
            self.connected = False
            print("Disconnected from Furhat")

    def attend_user(self):
        """Attend the closest user using FurhatClient."""
        self.client.request_attend_user("closest")
        time.sleep(0.5)

    def smile_gesture(self):
        """Play a 'Smile' gesture using FurhatClient."""
        self.client.request_gesture_start(name="Smile", wait=True)
        time.sleep(0.5)

    def speak(self, text: str):
        """
        Make Furhat speak the given text using the built-in TTS.

        Note: This uses the high-level FurhatClient, not the realtime API.
        """
        self.client.request_speak_text(text=text, wait=True)

    def execute_sequence(self, message: str):
        """
        Original helper: attend the user, smile, and speak a message.
        """
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

    async def connect_realtime(self):
        """
        Open a WebSocket connection to the Furhat Realtime API.
        """
        if self.realtime_connected:
            return

        port = int(os.getenv("FURHAT_RT_PORT", "9000"))
        route = os.getenv("FURHAT_RT_ROUTE", "/v1/events")
        url = f"ws://{self.host}:{port}{route}"
        print(f"[Realtime] Connecting to {url}")

        self.ws = await websockets.connect(url)
        self.realtime_connected = True
        self._running = True

        # ---- optional auth (only if required by Furhat web settings) ----
        auth_key = os.getenv("FURHAT_AUTH_KEY", "").strip()
        if auth_key:
            await self._send_event("request.auth", key=auth_key)
            # 这里不强制等待 response.auth；如需严格校验可在 _event_loop 里处理 response.auth

        self._event_loop_task = asyncio.create_task(self._event_loop())
        print("[Realtime] Connected to WebSocket")

    async def disconnect_realtime(self):
        """
        Close the WebSocket connection to the Realtime API.
        """
        self._running = False
        if self.ws is not None:
            await self.ws.close()
            self.ws = None
        self.realtime_connected = False
        print("[Realtime] Disconnected from WebSocket")

    async def _send_event(self, event_type: str, **params):
        """
        Helper for sending a JSON event over the Realtime WebSocket.
        """
        if not self.ws:
            return
        msg = {"type": event_type}
        msg.update(params)
        await self.ws.send(json.dumps(msg))

        # -------------------- Camera control --------------------

    async def start_camera(self):
        """
        Start receiving camera frames (response.camera.data).

        NOTE:
          You must set self.on_frame callback to actually use the frames:
              robot.on_frame = your_frame_handler
        """
        if not self.realtime_connected:
            raise RuntimeError("Realtime not connected. Call connect_realtime() first.")

        if not self.camera_running:
            await self._send_event("request.camera.start")
            self.camera_running = True
            print("[Realtime] Camera started")

    async def stop_camera(self):
        """
        Stop receiving camera frames.
        """
        if self.camera_running:
            await self._send_event("request.camera.stop")
            self.camera_running = False
            print("[Realtime] Camera stopped")

        # -------------------- Audio control --------------------

    async def start_audio(self, sample_rate: int = 16000):
        """
        Start receiving raw microphone audio (response.audio.data).

        NOTE:
          - Audio data is PCM16 mono bytes in 'microphone' field.
          - Set self.on_audio to handle the data.
        """
        if not self.realtime_connected:
            raise RuntimeError("Realtime not connected. Call connect_realtime() first.")

        if not self.audio_running:
            await self._send_event(
                "request.audio.start",
                sample_rate=sample_rate,
                microphone=True,
                speaker=False,
            )
            self.audio_running = True
            print(f"[Realtime] Audio started (mic, {sample_rate} Hz)")

    async def stop_audio(self):
        """
        Stop receiving microphone audio.
        """
        if self.audio_running:
            await self._send_event("request.audio.stop")
            self.audio_running = False
            print("[Realtime] Audio stopped")

        # -------------------- ASR (listen) control --------------------

    async def start_listen(self, languages=None, phrases=None, end_speech_timeout: float = 2.0):
        """
        Start ASR listening (response.hear.* events).
        """
        if not self.realtime_connected:
            raise RuntimeError("Realtime not connected. Call connect_realtime() first.")

        if languages is None:
            languages = ["en-US"]
        if phrases is None:
            phrases = []

        if not self.listening:
            # Configure ASR
            await self._send_event("request.listen.config", languages=languages, phrases=phrases)
            # Start listening with explicit end_speech_timeout
            await self._send_event(
                "request.listen.start",
                partial=True,
                concat=True,
                end_speech_timeout=float(end_speech_timeout),
            )
            self.listening = True
            print(f"[Realtime] Listen started (languages={languages}, end_speech_timeout={end_speech_timeout})")

    async def stop_listen(self):
        """
        Stop ASR listening.
        """
        if self.listening:
            await self._send_event("request.listen.stop")
            self.listening = False
            print("[Realtime] Listen stopped")

        # =====================================================================
        # Event loop: handle all incoming realtime events from Furhat
        # =====================================================================

    async def _event_loop(self):
        """
        Internal background loop that receives all realtime events and
        dispatches them to the registered callbacks.

        You usually do not call this yourself; it is spawned as a task
        inside connect_realtime().
        """
        while self._running and self.ws is not None:
            try:
                raw = await self.ws.recv()
                event = json.loads(raw)
                etype = event.get("type")

                # -------------- Camera frames --------------
                if etype == "response.camera.data":
                    image_b64 = event.get("image")
                    if image_b64 and self.on_frame:
                        frame = self._decode_frame(image_b64)
                        if frame is not None:
                            self.on_frame(frame)

                # -------------- Raw microphone audio --------------
                elif etype == "response.audio.data":
                    mic_b64 = event.get("microphone")
                    if mic_b64 and self.on_audio:
                        pcm = base64.b64decode(mic_b64)
                        self.on_audio(pcm)

                # -------------- ASR partial text --------------
                elif etype == "response.hear.partial":
                    text = event.get("text", "")
                    if self.on_partial_text:
                        self.on_partial_text(text)

                # -------------- ASR final text --------------
                elif etype == "response.hear.end":
                    text = event.get("text", "")
                    if self.on_final_text:
                        self.on_final_text(text)

                # You can also handle other response.* types here if needed

            except websockets.ConnectionClosed:
                print("[Realtime] Connection closed by server.")
                self.realtime_connected = False
                break
            except Exception as e:
                print(f"[Realtime] Error in event loop: {e}")
                await asyncio.sleep(0.1)

        # =====================================================================
        # Helper: decode camera frame
        # =====================================================================

    def _decode_frame(self, image_b64: str):
        """
        Decode base64 JPEG from response.camera.data into a BGR image.
        """
        try:
            data = base64.b64decode(image_b64)
            arr = np.frombuffer(data, np.uint8)
            import cv2  # local import to avoid mandatory OpenCV if not needed

            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

