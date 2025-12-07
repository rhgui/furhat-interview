import asyncio
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from cv_parser import CVParser
from question_generator import QuestionGenerator
from followup_generator import FollowupGenerator
from transition_generator import TransitionGenerator
from furhat_robot import FurhatRobot
from nervous_fusion import NervousFusion, FusionConfig
from nervous_detector_voice import FillerCounter

# ============================================================
# 配置
# ============================================================

FURHAT_IP = "192.168.1.110"
AUDIO_SAMPLE_RATE = 16000

# 这里只保留一个“大超时”，真正的回答结束由 Furhat 的 end_speech_timeout 决定
GLOBAL_LISTEN_TIMEOUT_EXTRA = 10.0


@dataclass
class QuestionResult:
    question_id: str
    main_text: str
    answer_text: str
    audio_bytes: bytes
    frames: List[np.ndarray]
    reaction_time_s: float
    extra_filler_count: int
    is_nervous: Optional[bool] = None
    comfort_given: bool = False
    fused_score: Optional[float] = None


# ============================================================
# 访谈 Session：负责整个流程 + Realtime 回调
# ============================================================

class InterviewSession:
    def __init__(self, robot: FurhatRobot, questions: List[Dict]):
        self.robot = robot
        # 按 id 建字典，方便直接 Q1/Q2/Q3...
        self.questions: Dict[str, Dict] = {q["id"]: q for q in questions}

        # NervousFusion（音频 + 视频）
        config = FusionConfig(
            nervous_threshold=0.60,
            audio_weight=0.7,
            video_weight=0.3,
        )
        self.fusion = NervousFusion(config=config)

        # LLM 模块
        self.followup_gen = FollowupGenerator()
        self.transition_gen = TransitionGenerator()

        # 语音相关状态
        self.sample_rate = AUDIO_SAMPLE_RATE
        self.filler_counter = FillerCounter()

        # 当前回答的缓冲区 & 状态
        self.current_question_id: Optional[str] = None
        self.current_question_text: str = ""
        self.current_expected_time_s: float = 10.0

        self.is_collecting_answer: bool = False
        self.collect_video: bool = False
        self.is_audio_recording: bool = False

        self.answer_audio_chunks: List[bytes] = []
        self.answer_frames: List[np.ndarray] = []
        self.partial_text: str = ""
        self.final_text: str = ""

        self.t_question_end: float = 0.0
        self.t_speech_start: Optional[float] = None
        self.t_last_activity: Optional[float] = None
        self.reaction_time_s: float = 0.0

        self.answer_done_event: asyncio.Event = asyncio.Event()

        # baseline 是否已建立（Q2 之后）
        self.baseline_built: bool = False

        # Realtime 回调绑定
        self.robot.on_frame = self._on_frame
        self.robot.on_audio = self._on_audio
        self.robot.on_partial_text = self._on_partial_text
        self.robot.on_final_text = self._on_final_text

    # --------------------------------------------------------
    # Realtime 回调
    # --------------------------------------------------------

    def _on_frame(self, frame: np.ndarray):
        if self.is_collecting_answer and self.collect_video:
            self.answer_frames.append(frame)

    def _on_audio(self, pcm: bytes):
        """
        音频流一直开，但只有 is_audio_recording=True 时才写到当前回答 buffer。
        这样传给 NervousFusion 的音频只包含“从用户开口到说完”的部分。
        """
        if self.is_collecting_answer and self.is_audio_recording:
            self.answer_audio_chunks.append(pcm)

    def _on_partial_text(self, text: str):
        """
        streaming partial：
          - 用来统计 filler（um/uh 等）
          - 检测第一次开口时刻（reaction time）
        回答结束本身，交给 Furhat 的 end_speech_timeout + response.hear.end。
        """
        if not self.is_collecting_answer:
            return

        now = time.monotonic()
        t = text or ""

        # filler 统计
        self.filler_counter.on_partial(t)

        stripped = t.strip()

        # 第一次出现非空 partial → 用户开始说话
        if self.t_speech_start is None and stripped:
            self.t_speech_start = now
            self.reaction_time_s = self.t_speech_start - self.t_question_end
            # 从现在开始才把音频写进 buffer
            self.is_audio_recording = True

        self.partial_text = t

    def _on_final_text(self, text: str):
        """
        Furhat 的 response.hear.end。
        Furhat 内部已经根据 end_speech_timeout（例如 2 秒）判断回答结束。
        这里只负责：
          - 记录最终文本
          - 通知主协程：回答结束了
        """
        if not self.is_collecting_answer:
            return

        self.final_text = text or ""
        if not self.answer_done_event.is_set():
            self.is_collecting_answer = False
            self.is_audio_recording = False
            self.answer_done_event.set()

    # --------------------------------------------------------
    # 问问题 + 收回答（统一封装）
    # --------------------------------------------------------

    async def ask_and_record_answer(
        self,
        question_id: str,
        main_text: str,
        expected_time_s: float,
        collect_video: bool,
    ) -> Tuple[bytes, str, List[np.ndarray], float, int]:

        self.current_question_id = question_id
        self.current_question_text = main_text
        self.current_expected_time_s = expected_time_s
        self.collect_video = collect_video

        # 清空状态
        self.answer_audio_chunks = []
        self.answer_frames = []
        self.partial_text = ""
        self.final_text = ""
        self.t_speech_start = None
        self.t_last_activity = None
        self.reaction_time_s = 0.0
        self.filler_counter.reset()
        self.is_audio_recording = False
        self.is_collecting_answer = False
        self.answer_done_event.clear()

        # 机器人发问（阻塞到说完）
        self.robot.attend_user()
        self.robot.speak(main_text)

        # 问题结束时间（reaction time 用）
        self.t_question_end = time.monotonic()

        # 启动监听（Furhat 内部 listen.config 应该已经把 end_speech_timeout 设成 2 秒）
        self.is_collecting_answer = True
        await self.robot.start_listen(
            languages=["en-US"],
            phrases=["um", "uh", "er", "ah", "eh"],  # 可选：提高这些词不被ASR吃掉的概率
            end_speech_timeout=2.0,
        )

        start_time = time.monotonic()

        # 等待结束：
        #   1）Furhat 的 response.hear.end（_on_final_text 设置 answer_done_event）
        #   2）expected_time + GLOBAL_LISTEN_TIMEOUT_EXTRA 超时兜底
        while not self.answer_done_event.is_set():
            now = time.monotonic()

            # 超时保护：防止 ASR 出错导致永远等不到 hear.end
            if now - start_time >= expected_time_s + GLOBAL_LISTEN_TIMEOUT_EXTRA:
                if not self.answer_done_event.is_set():
                    self.is_collecting_answer = False
                    self.is_audio_recording = False
                    # 如果没 final_text，就退而求其次用最后一个 partial
                    if not self.final_text:
                        self.final_text = self.partial_text
                    self.answer_done_event.set()
                    break

            await asyncio.sleep(0.1)

        await self.robot.stop_listen()

        audio_bytes = b"".join(self.answer_audio_chunks)
        text = self.final_text or self.partial_text or ""
        frames = list(self.answer_frames)
        reaction_time = self.reaction_time_s
        extra_fillers = self.filler_counter.count

        return audio_bytes, text, frames, reaction_time, extra_fillers

    # --------------------------------------------------------
    # Q2：构建 NervousFusion baseline
    # --------------------------------------------------------

    async def run_q2_baseline(self) -> QuestionResult:
        q2 = self.questions["Q2"]
        audio, text, frames, rt, fillers = await self.ask_and_record_answer(
            question_id="Q2",
            main_text=q2["text"],
            expected_time_s=q2["expected_time_s"],
            collect_video=True,
        )

        # 用 Q2 建立多模态 baseline
        self.fusion.build_baseline(
            first_audio_bytes=audio,
            sample_rate=self.sample_rate,
            first_text=text,
            baseline_frames=frames,
        )
        self.baseline_built = True

        self.robot.speak("Thank you for the introduction.")

        return QuestionResult(
            question_id="Q2",
            main_text=q2["text"],
            answer_text=text,
            audio_bytes=audio,
            frames=frames,
            reaction_time_s=rt,
            extra_filler_count=fillers,
        )

    # --------------------------------------------------------
    # Q3–Q5：完整逻辑（紧张度 + 安慰 + follow-up）
    # --------------------------------------------------------

    async def run_main_question(
        self,
        qid: str,
        mode: str,
        is_last: bool,
    ) -> QuestionResult:
        assert self.baseline_built, "Baseline must be built before Q3–Q5."

        q = self.questions[qid]
        if mode == "guided" and q.get("guided_main"):
            main_text = q["guided_main"]
        else:
            main_text = q["text"]

        expected_time = float(q.get("expected_time_s", 30.0))

        # 1) 问主问题 + 收回答（含视频）
        audio, text, frames, rt, fillers = await self.ask_and_record_answer(
            question_id=qid,
            main_text=main_text,
            expected_time_s=expected_time,
            collect_video=True,
        )

        # 2) NervousFusion 评估
        is_nervous, scores = self.fusion.evaluate_answer(
            audio_bytes=audio,
            sample_rate=self.sample_rate,
            text=text,
            reaction_time_s=rt,
            expected_answer_time_s=expected_time,
            extra_filler_count=fillers,
            segment_frames=frames,
        )

        comfort_given = False
        if is_nervous:
            self.robot.speak(
                "Thank you for sharing that. It's completely normal to feel a bit nervous in interviews. "
                "Please take your time."
            )
            comfort_given = True
        else:
            self.robot.speak("Thank you, that was clear.")

        # 3) Follow-up 决策
        follow = self.followup_gen.generate_followup(
            main_question=main_text,
            answer_text=text,
            is_nervous=is_nervous,
        )

        if follow.get("need_followup") and follow.get("followup_question"):
            fu_q = follow["followup_question"]
            audio_fu, text_fu, _, _, _ = await self.ask_and_record_answer(
                question_id=f"{qid}_FU",
                main_text=fu_q,
                expected_time_s=min(expected_time, 40.0),
                collect_video=False,
            )
            full_text = text.strip() + "\n\nFollow-up: " + text_fu.strip()
        else:
            full_text = text

        return QuestionResult(
            question_id=qid,
            main_text=main_text,
            answer_text=full_text,
            audio_bytes=audio,
            frames=frames,
            reaction_time_s=rt,
            extra_filler_count=fillers,
            is_nervous=is_nervous,
            comfort_given=comfort_given,
            fused_score=scores.fused_score,
        )

    # --------------------------------------------------------
    # 整个访谈：Preparation → Q1 → Q2 → Q3〜Q5 → Closing
    # --------------------------------------------------------

    async def run(self):
        # 连接 Realtime，开启音频+相机
        await self.robot.connect_realtime()
        await self.robot.start_audio(sample_rate=self.sample_rate)
        await self.robot.start_camera()

        # 1. 准备阶段
        self.robot.execute_sequence(
            "Hi, welcome to the interview. Please make yourself comfortable and take a seat."
        )
        self.robot.speak(
            "I will ask you a few questions about your background and experience. "
            "There are no right or wrong answers. Just answer in your own way."
        )

        # 2. Q1：简单问候
        q1 = self.questions["Q1"]
        await self.ask_and_record_answer(
            question_id="Q1",
            main_text=q1["text"],
            expected_time_s=q1["expected_time_s"],
            collect_video=False,
        )
        self.robot.speak("Thanks for sharing that.")

        # 3. Q2：自我介绍 + baseline
        q2_result = await self.run_q2_baseline()

        # 4. Q3：第一道 CV-based 问题（必定 original）
        q3_result = await self.run_main_question(
            qid="Q3",
            mode="original",
            is_last=False,
        )

        # Q3 → Q4 过渡
        q4 = self.questions["Q4"]
        trans34 = self.transition_gen.generate_transition(
            was_nervous=bool(q3_result.is_nervous),
            comfort_given=bool(q3_result.comfort_given),
            answer_text=q3_result.answer_text,
            next_original_main=q4["text"],
            next_guided_main=q4.get("guided_main", q4["text"]),
        )

        if trans34["comfort_text"]:
            self.robot.speak(trans34["comfort_text"])
        if trans34["connector_text"]:
            self.robot.speak(trans34["connector_text"])

        q4_mode = trans34.get("target_mode", "original")

        # 5. Q4：进一步行为/经验问题
        q4_result = await self.run_main_question(
            qid="Q4",
            mode=q4_mode,
            is_last=False,
        )

        # Q4 → Q5 过渡
        q5 = self.questions["Q5"]
        trans45 = self.transition_gen.generate_transition(
            was_nervous=bool(q4_result.is_nervous),
            comfort_given=bool(q4_result.comfort_given),
            answer_text=q4_result.answer_text,
            next_original_main=q5["text"],
            next_guided_main=q5.get("guided_main", q5["text"]),
        )

        if trans45["comfort_text"]:
            self.robot.speak(trans45["comfort_text"])
        if trans45["connector_text"]:
            self.robot.speak(trans45["connector_text"])

        q5_mode = trans45.get("target_mode", "original")

        # 6. Q5：最后一个深度问题
        q5_result = await self.run_main_question(
            qid="Q5",
            mode=q5_mode,
            is_last=True,
        )

        # 7. 访谈总结
        nervous_scores = [
            s for s in [
                q3_result.fused_score,
                q4_result.fused_score,
                q5_result.fused_score,
            ] if s is not None
        ]
        avg_nervous = sum(nervous_scores) / len(nervous_scores) if nervous_scores else 0.0

        if avg_nervous >= 0.6:
            summary = (
                "Thank you very much for taking the time to talk with me today. "
                "I know interviews can feel stressful, but you handled it well and kept going. "
                "I really appreciate your effort."
            )
        else:
            summary = (
                "Thank you very much for your time today. "
                "You explained your experience clearly, and it was great to hear about your background. "
                "That was all from my side for now."
            )

        self.robot.speak(summary)

        # 关闭 Realtime 流
        await self.robot.stop_audio()
        await self.robot.stop_camera()
        await self.robot.disconnect_realtime()


# ============================================================
# 主入口：CV → 生成问题 → 连接 Furhat → 跑访谈
# ============================================================

async def main():
    # 强制指定 Furhat IP
    os.environ["FURHAT_HOST"] = FURHAT_IP

    # 1) 解析 CV（如果 dummy_cv.pdf 不存在，就走 fallback）
    cv_parser = CVParser()
    cv_path = os.getenv("CV_FILE", "dummy_cv.pdf")
    cv_data = cv_parser.parse_cv(uploaded_file_path=cv_path)
    if "error" in cv_data:
        cv_data = {
            "name": "Candidate",
            "email": "",
            "education": ["Master's student"],
            "experience": ["Student projects"],
            "skills": ["Python", "Machine Learning", "Robotics"],
        }

    # 2) 生成 Q1–Q5
    qgen = QuestionGenerator()
    questions = qgen.generate_full_question_set(
        cv_data=cv_data,
        position="Graduate interview",
    )

    # 3) 连接 Furhat 高层 API（speak/attend）
    robot = FurhatRobot()
    robot.connect()

    # 4) 跑整套 Session
    session = InterviewSession(robot=robot, questions=questions)
    try:
        await session.run()
    finally:
        # Realtime 在 session.run 里已经关闭，这里只断开高层连接
        robot.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
