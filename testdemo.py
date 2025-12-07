import base64
import json
import threading
import time
from typing import List

import numpy as np
import websocket

from nervous_fusion import NervousFusion, FusionConfig

ROBOT_IP = "192.168.1.110"
WS_URL = f"ws://{ROBOT_IP}:9000/v1/events"

BASELINE_SECONDS = 8
ANSWER_SECONDS = 10
AUDIO_SAMPLE_RATE = 16000
EXPECTED_ANSWER_TIME_S = 10.0

fusion: NervousFusion | None = None
current_phase = "idle"          # "idle" / "baseline" / "answer"

baseline_frames: List[np.ndarray] = []
answer_frames: List[np.ndarray] = []
baseline_audio_chunks: List[bytes] = []
answer_audio_chunks: List[bytes] = []


# ========== 小工具：解码 base64 图像 & 音频 ==========
def decode_frame_base64(image_b64: str):
    try:
        arr = np.frombuffer(base64.b64decode(image_b64), np.uint8)
        import cv2
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def decode_audio_base64(mic_b64: str) -> bytes:
    try:
        return base64.b64decode(mic_b64)
    except Exception:
        return b""


# ========== Furhat 辅助 ==========
def furhat_say(ws, text: str):
    ws.send(json.dumps({
        "type": "request.speak.text",
        "text": text,
        "request_id": "say-001",
    }))


def start_camera(ws, request_id: str):
    ws.send(json.dumps({
        "type": "request.camera.start",
        "request_id": request_id,
    }))


def stop_camera(ws, request_id: str):
    ws.send(json.dumps({
        "type": "request.camera.stop",
        "request_id": request_id,
    }))


def start_audio(ws):
    ws.send(json.dumps({
        "type": "request.audio.start",
        "sample_rate": AUDIO_SAMPLE_RATE,
        "microphone": True,
        "speaker": False,
    }))


def stop_audio(ws):
    ws.send(json.dumps({
        "type": "request.audio.stop",
    }))


# ========== WebSocket 回调 ==========
def on_message(ws, message: str):
    global current_phase
    global baseline_frames, answer_frames
    global baseline_audio_chunks, answer_audio_chunks

    msg = json.loads(message)
    msg_type = msg.get("type")

    # 摄像头帧
    if msg_type in ("response.camera.data", "event.sense.image"):
        image_b64 = msg.get("image") or msg.get("data")
        if not image_b64:
            return
        frame = decode_frame_base64(image_b64)
        if frame is None:
            return

        if current_phase == "baseline":
            baseline_frames.append(frame)
        elif current_phase == "answer":
            answer_frames.append(frame)

    # 音频帧
    elif msg_type == "response.audio.data":
        mic_b64 = msg.get("microphone")
        if not mic_b64:
            return
        audio_bytes = decode_audio_base64(mic_b64)
        if not audio_bytes:
            return

        if current_phase == "baseline":
            baseline_audio_chunks.append(audio_bytes)
        elif current_phase == "answer":
            answer_audio_chunks.append(audio_bytes)

    else:
        print("[EVENT]", msg_type, msg)


def on_error(ws, error):
    print("[ERROR]", error)


def on_close(ws, code, msg):
    print("WebSocket closed:", code, msg)


# ========== 主流程：只调用 NervousFusion ==========
def run_fusion_demo(ws):
    global fusion, current_phase
    global baseline_frames, answer_frames
    global baseline_audio_chunks, answer_audio_chunks

    # 1) 初始化 fusion
    config = FusionConfig(
        nervous_threshold=0.60,
        audio_weight=0.7,
        video_weight=0.3,
    )
    fusion = NervousFusion(config=config)

    furhat_say(ws, "Hello! I will test your nervousness using both audio and video.")
    time.sleep(1.0)

    # 整个流程开启音频流
    start_audio(ws)

    # 2) Baseline：音频 + 视频
    furhat_say(
        ws,
        "First, please answer in a relaxed way for baseline, "
        "for example tell me about your day."
    )
    time.sleep(2.0)

    baseline_frames = []
    baseline_audio_chunks = []

    current_phase = "baseline"
    start_camera(ws, "camera-baseline-start")

    time.sleep(BASELINE_SECONDS)

    stop_camera(ws, "camera-baseline-stop")
    current_phase = "idle"

    baseline_audio_bytes = b"".join(baseline_audio_chunks)
    baseline_text = ""   # 暂时不接 ASR

    print("\n[Baseline] frames =", len(baseline_frames),
          "audio bytes =", len(baseline_audio_bytes))

    fusion.build_baseline(
        first_audio_bytes=baseline_audio_bytes,
        sample_rate=AUDIO_SAMPLE_RATE,
        first_text=baseline_text,
        baseline_frames=baseline_frames,
    )

    furhat_say(ws, "Baseline completed, thank you.")
    time.sleep(1.5)

    # 3) 正式回答
    furhat_say(
        ws,
        "Now, please introduce yourself as in a real interview."
    )
    time.sleep(1.5)

    answer_frames = []
    answer_audio_chunks = []

    current_phase = "answer"
    start_camera(ws, "camera-answer-start")

    time.sleep(ANSWER_SECONDS)

    stop_camera(ws, "camera-answer-stop")
    current_phase = "idle"

    answer_audio_bytes = b"".join(answer_audio_chunks)
    answer_text = ""          # 未来接 ASR
    reaction_time_s = 0.0     # 未来用 speak.end / hear.start 算
    extra_filler_count = 0    # 未来用 partial ASR 算

    furhat_say(ws, "Great. I am analyzing your nervousness now.")
    time.sleep(1.0)

    # 4) 真正只用 NervousFusion 做融合判断
    is_nervous, scores = fusion.evaluate_answer(
        audio_bytes=answer_audio_bytes,
        sample_rate=AUDIO_SAMPLE_RATE,
        text=answer_text,
        reaction_time_s=reaction_time_s,
        expected_answer_time_s=EXPECTED_ANSWER_TIME_S,
        extra_filler_count=extra_filler_count,
        segment_frames=answer_frames,
    )

    print("\n========== Multimodal Nervousness Result ==========")
    print("Audio score :", scores.audio_score)
    print("Video score :", scores.video_score)
    print("Fused score :", scores.fused_score)
    print("Video reliab:", scores.video_reliability)
    print("Nervous?    :", is_nervous)

    # 5) Furhat 反馈
    if scores.video_reliability < 0.3:
        furhat_say(
            ws,
            "I could not clearly see your face, "
            "please look at me more directly next time."
        )
    else:
        if is_nervous:
            furhat_say(
                ws,
                "Based on your audio and video, you seem a bit nervous. "
                "That is completely normal."
            )
        else:
            furhat_say(
                ws,
                "Based on your audio and video, you look quite relaxed. Well done."
            )

    time.sleep(3.0)
    stop_audio(ws)
    ws.close()


def on_open(ws):
    print("Connected to Furhat (Fusion Nervousness Demo)")
    t = threading.Thread(target=run_fusion_demo, args=(ws,))
    t.daemon = True
    t.start()


if __name__ == "__main__":
    ws_app = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws_app.run_forever()
