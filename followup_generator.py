# followup_generator.py
# ============================================================
# Dynamic Follow-Up Question Generator (Boolean nervous flag)
#
# This module is called AFTER the candidate answers a question.
# It uses:
#   - the main question text
#   - the candidate's answer (ASR transcript)
#   - a boolean nervousness flag (is_nervous: bool)
#
# to:
#   1) Decide whether a follow-up question is needed.
#   2) Generate ONE follow-up question if needed.
#
# Nervousness control:
#   - is_nervous = True
#       • Use a GUIDED style:
#           - soft, supportive, encouraging
#           - gentle, anxiety-aware wording
#
#   - is_nervous = False
#       • Use a NEUTRAL style:
#           - professional, neutral, content-focused
#           - no emotional / therapeutic language
#
# The module does NOT speak the follow-up itself. Your dialogue
# controller / Furhat layer should:
#   - Decide whether to use the follow-up (need_followup flag)
#   - Speak it if needed
#   - Then go back to listening / next main question
# ============================================================

from __future__ import annotations

import os
import json
from typing import Dict, Any

from dotenv import load_dotenv
from google import genai


class FollowupGenerator:
    """
    Dynamic follow-up generator using Gemini.

    Typical usage:

        gen = FollowupGenerator()

        result = gen.generate_followup(
            main_question=main_q_text,
            answer_text=asr_answer_text,
            is_nervous=nervous_bool,  # True / False from your fusion module
        )

        if result["need_followup"]:
            followup_text = result["followup_question"]
            # robot.say(followup_text)

    Returned dictionary:

        {
            "need_followup": bool,         # whether to ask a follow-up
            "reason": "string",            # explanation for logging/debugging
            "followup_question": "string"  # empty string if no follow-up is needed
        }
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the LLM client for follow-up generation.

        The GEMINI_API_KEY is loaded from the environment via .env.
        """
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in .env")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    # ========================================================
    # Small utility: strip ```json ... ``` code fences
    # ========================================================
    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """
        Remove markdown-style code fences if the model wraps JSON in:

            ```json
            {...}
            ```

        This makes json.loads() more robust.
        """
        s = text.strip()
        if s.startswith("```"):
            s = s.strip("`").strip()
            if s.lower().startswith("json"):
                s = s[4:].strip()
        return s

    # ========================================================
    # Prompt for GUIDED mode (is_nervous = True)
    # ========================================================
    def _build_prompt_guided(self, main_question: str, answer_text: str) -> str:
        """
        Build the prompt for the GUIDED mode.

        This mode is used when the candidate is detected as nervous.
        The follow-up question should be:
            - Gentle and supportive.
            - Encouraging the candidate to elaborate at their own pace.
            - Open-ended (no yes/no questions).
        """
        return (
            "You are an interview assistant generating a GUIDED follow-up question.\n"
            "The candidate appears nervous, so your question must be:\n"
            "  - Gentle and supportive.\n"
            "  - Encouraging them to elaborate in a safe way.\n"
            "  - Directly based on their answer.\n"
            "  - Open-ended (no yes/no questions).\n"
            "  - Short (no more than 25 words).\n\n"
            "Your tasks:\n"
            "  1) Decide whether a follow-up question is needed.\n"
            "     A follow-up IS needed if the answer is incomplete, vague,\n"
            "     missing important details, or hints at something that should\n"
            "     be explored further.\n"
            "  2) If a follow-up is needed, generate ONE supportive follow-up\n"
            "     question.\n"
            "  3) If NO follow-up is needed, return an empty string for\n"
            "     followup_question.\n\n"
            "Return ONLY a JSON object with EXACTLY these fields:\n"
            "  \"need_followup\": true or false,\n"
            "  \"reason\": \"string\",\n"
            "  \"followup_question\": \"string\"\n\n"
            "Main interview question:\n"
            f"{main_question}\n\n"
            "Candidate's answer (ASR transcript, may contain errors):\n"
            f"{answer_text}\n\n"
            "Now return ONLY the JSON object. No extra text."
        )

    # ========================================================
    # Prompt for NEUTRAL mode (is_nervous = False)
    # ========================================================
    def _build_prompt_neutral(self, main_question: str, answer_text: str) -> str:
        """
        Build the prompt for the NEUTRAL mode.

        This mode is used when the candidate is confident and not nervous.
        The follow-up question should be:
            - Professional and neutral in tone.
            - Focused on clarifying or deepening content.
            - Open-ended (no yes/no questions).
        """
        return (
            "You are a professional interviewer generating a follow-up question.\n"
            "The candidate appears confident (not nervous).\n\n"
            "Your follow-up question should be:\n"
            "  - Neutral and professional in tone.\n"
            "  - Focused on clarification, depth, or missing details.\n"
            "  - Directly based on the candidate's answer.\n"
            "  - Open-ended (avoid yes/no questions).\n"
            "  - Short (no more than 25 words).\n"
            "  - NOT emotional, comforting, or therapeutic.\n\n"
            "Your tasks:\n"
            "  1) Decide whether a follow-up question is needed.\n"
            "  2) If a follow-up is needed, generate ONE professional follow-up question.\n"
            "  3) If NO follow-up is needed, return an empty string for\n"
            "     followup_question.\n\n"
            "Return ONLY a JSON object with EXACTLY these fields:\n"
            "  \"need_followup\": true or false,\n"
            "  \"reason\": \"string\",\n"
            "  \"followup_question\": \"string\"\n\n"
            "Main interview question:\n"
            f"{main_question}\n\n"
            "Candidate's answer (ASR transcript, may contain errors):\n"
            f"{answer_text}\n\n"
            "Now return ONLY the JSON object. No extra text."
        )

    # ========================================================
    # Unified public API (now using is_nervous: bool)
    # ========================================================
    def generate_followup(
        self,
        main_question: str,
        answer_text: str,
        is_nervous: bool,
    ) -> Dict[str, Any]:
        """
        Generate a dynamic follow-up question (if needed).

        Args:
            main_question : str
                The main interview question that was just asked
                (could be original_main or guided_main).
            answer_text   : str
                The candidate's answer (ASR transcript).
            is_nervous    : bool
                Nervousness flag from your fusion module:
                    True  → use GUIDED style (soft, supportive).
                    False → use NEUTRAL style (professional).

        Returns:
            dict with:
                - need_followup: bool
                - reason: str
                - followup_question: str (empty if no follow-up needed)
        """

        # Map the boolean nervous flag to internal mode
        if is_nervous:
            prompt = self._build_prompt_guided(main_question, answer_text)
        else:
            prompt = self._build_prompt_neutral(main_question, answer_text)

        # Call the Gemini model
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        raw = response.text or ""
        raw = self._strip_code_fence(raw)

        # Try to parse output as JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback behavior if JSON cannot be parsed:
            # treat as "no follow-up" to avoid breaking the dialogue.
            return {
                "need_followup": False,
                "reason": "JSON parse error; fallback to no follow-up",
                "followup_question": "",
            }

        # Extract and normalize the fields
        need = bool(data.get("need_followup", False))
        reason = str(data.get("reason", "")).strip()
        followup_question = str(data.get("followup_question", "")).strip()

        # If no follow-up is needed, ensure the question is empty
        if not need:
            followup_question = ""

        return {
            "need_followup": need,
            "reason": reason,
            "followup_question": followup_question,
        }


# ============================================================
# Optional local test (manual debugging)
# ============================================================
# if __name__ == "__main__":
#     gen = FollowupGenerator()
#
#     main_q = "Can you describe a project where you used Python?"
#     ans = "I used Python in several school projects but I am not sure what details to mention."
#
#     print("\n--- Guided mode (is_nervous=True) ---")
#     out_guided = gen.generate_followup(main_q, ans, is_nervous=True)
#     print(out_guided)
#
#     print("\n--- Neutral mode (is_nervous=False) ---")
#     out_neutral = gen.generate_followup(main_q, ans, is_nervous=False)
#     print(out_neutral)
