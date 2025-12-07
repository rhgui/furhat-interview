# ======================================================================
# transition_generator.py
# ----------------------------------------------------------------------
# This module generates:
#   1) An optional *comfort_text* (only when the user was nervous and
#      has not yet received comfort for the previous question).
#   2) A *connector_text* that smoothly transitions from the user's
#      previous answer to the NEXT main question.
#
# Nervousness logic:
#   - Case 1: was_nervous = True,  comfort_given = False
#       → Provide comfort_text AND a gentle guided connector.
#
#   - Case 2: was_nervous = True,  comfort_given = True
#       → Provide ONLY guided connector (comfort_text must be empty).
#
#   - Case 3: was_nervous = False
#       → Provide ONLY neutral connector to original_main.
#
# Output:
#   {
#       "comfort_text": "...",
#       "connector_text": "...",
#       "target_mode": "guided"  or "original"
#   }
#
# The model MUST output valid JSON; markdown code fences are removed safely.
# ======================================================================

from __future__ import annotations
import os
import json
from typing import Dict, Any

from dotenv import load_dotenv
from google import genai


class TransitionGenerator:
    """
    Generates comfort + connector utterances for interview transitions.
    Uses a Gemini model to produce short natural language outputs.

    Public API:
        result = generate_transition(
            was_nervous: bool,
            comfort_given: bool,
            answer_text: str,
            next_original_main: str,
            next_guided_main: str
        )
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in .env")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    # ==================================================================
    # Utility: remove ```json ... ``` wrappers if present
    # ==================================================================
    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """
        If the LLM wraps JSON in ```json ... ```
        this function removes those markdown fences.
        """
        s = text.strip()
        if s.startswith("```"):
            s = s.strip("`").strip()
            # Remove a leading 'json'
            if s.lower().startswith("json"):
                s = s[4:].strip()
        return s

    # ==================================================================
    # Prompt builder: Case 1 — nervous & no comfort given
    # ==================================================================
    def _build_prompt_comfort_and_link(
        self,
        answer_text: str,
        next_guided_main: str,
    ) -> str:
        """
        Case 1:
          - User was nervous
          - No comfort was given yet
        The model must output:
          * a short comfort_text (reassurance)
          * a connector_text leading softly to the NEXT guided question
        """
        return f"""
You are a social robot interviewer interacting with a student who appears nervous.
Below is the student's previous answer:

USER_ANSWER:
{answer_text}

TASK
----
1. Provide a short *comfort_text*:
   - Supportive, kind, validating effort.
   - No clinical or diagnostic language.
   - ONE short sentence.

2. Provide a short *connector_text* that gently leads to the next guided question.
   - Do NOT repeat the full question.
   - Use a soft transition phrase.
   - ONE sentence.

NEXT_GUIDED_MAIN:
{next_guided_main}

OUTPUT FORMAT
-------------
Return ONLY a JSON object:

{{
  "comfort_text": "your warm reassurance sentence",
  "connector_text": "your gentle transition sentence"
}}

Your answer MUST be valid JSON ONLY (no markdown, no commentary).
        """.strip()

    # ==================================================================
    # Prompt builder: Case 2 — nervous & already comforted
    # ==================================================================
    def _build_prompt_link_guided(
        self,
        answer_text: str,
        next_guided_main: str,
    ) -> str:
        """
        Case 2:
          - User was nervous
          - Comfort was already given earlier
        The model must output:
          * NO comfort_text (empty string)
          * ONLY a gentle connector leading to next guided question
        """
        return f"""
You are a social robot interviewer. The student already received comfort earlier.
Below is the student's answer:

USER_ANSWER:
{answer_text}

TASK
----
Generate only a *connector_text*:
  - Warm but not overly repetitive.
  - ONE sentence.
  - Leads gently toward the next guided question.
  - comfort_text MUST be empty ("").

NEXT_GUIDED_MAIN:
{next_guided_main}

OUTPUT FORMAT
-------------
Return ONLY a JSON object:

{{
  "comfort_text": "",
  "connector_text": "your gentle transition sentence"
}}

Your answer MUST be valid JSON ONLY.
        """.strip()

    # ==================================================================
    # Prompt builder: Case 3 — not nervous
    # ==================================================================
    def _build_prompt_link_neutral(
        self,
        answer_text: str,
        next_original_main: str,
    ) -> str:
        """
        Case 3:
          - User was NOT nervous
        The model must output:
          * NO comfort_text
          * A neutral, professional connector_text
        """
        return f"""
You are a social robot interviewer in a professional but friendly setting.
Below is the user's previous answer:

USER_ANSWER:
{answer_text}

TASK
----
Generate ONLY a neutral *connector_text* (ONE sentence):
  - Brief acknowledgment of the answer.
  - Smooth transition to the next original_main question.
  - comfort_text MUST be empty ("").

NEXT_ORIGINAL_MAIN:
{next_original_main}

OUTPUT FORMAT
-------------
Return ONLY a JSON object:

{{
  "comfort_text": "",
  "connector_text": "your neutral transition sentence"
}}

Your answer MUST be valid JSON ONLY.
        """.strip()

    # ==================================================================
    # MAIN PUBLIC FUNCTION
    # ==================================================================
    def generate_transition(
        self,
        was_nervous: bool,
        comfort_given: bool,
        answer_text: str,
        next_original_main: str,
        next_guided_main: str,
    ) -> Dict[str, Any]:
        """
        Generate transition utterances according to nervousness state.

        Args:
            was_nervous       : bool — previous answer classified as nervous?
            comfort_given     : bool — has comfort been given already?
            answer_text       : str  — previous user answer
            next_original_main: str  — next question's original version
            next_guided_main  : str  — next question's guided version

        Returns:
            Dict with:
                comfort_text
                connector_text
                target_mode       ("guided" or "original")
        """
        # ---------- Decide which prompt to use ----------
        if was_nervous:
            target_mode = "guided"
            if not comfort_given:
                # Case 1: nervous + no comfort → comfort + guided connector
                prompt = self._build_prompt_comfort_and_link(
                    answer_text=answer_text,
                    next_guided_main=next_guided_main,
                )
            else:
                # Case 2: nervous + comfort already given → guided connector only
                prompt = self._build_prompt_link_guided(
                    answer_text=answer_text,
                    next_guided_main=next_guided_main,
                )
        else:
            # Case 3: not nervous → neutral connector
            target_mode = "original"
            prompt = self._build_prompt_link_neutral(
                answer_text=answer_text,
                next_original_main=next_original_main,
            )

        # ---------- Query Gemini ----------
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        raw = response.text or ""
        raw = self._strip_code_fence(raw)

        # ---------- Parse JSON safely ----------
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback behavior if LLM output is malformed
            if was_nervous and not comfort_given:
                return {
                    "comfort_text": "Thank you for sharing that. You're doing really well.",
                    "connector_text": "When you feel ready, I'd like to continue with the next question.",
                    "target_mode": "guided",
                }
            else:
                return {
                    "comfort_text": "",
                    "connector_text": "Thank you. Let's continue with the next question.",
                    "target_mode": target_mode,
                }

        # ---------- Extract fields ----------
        comfort_text = str(data.get("comfort_text", "")).strip()
        connector_text = str(data.get("connector_text", "")).strip()

        # ---------- Enforce rule: no comfort_text except Case 1 ----------
        if not was_nervous or (was_nervous and comfort_given):
            comfort_text = ""

        # ---------- Ensure connector_text always exists ----------
        if not connector_text:
            if target_mode == "guided":
                connector_text = (
                    "When you feel ready, I'd like to move on to the next question."
                )
            else:
                connector_text = (
                    "Thank you for your answer. Let's continue with the next question."
                )

        # ---------- Final output ----------
        return {
            "comfort_text": comfort_text,
            "connector_text": connector_text,
            "target_mode": target_mode,
        }



# ============================================================
# Local test (for debugging)
# ============================================================

# gen = TransitionGenerator()
#
# out = gen.generate_transition(
#     was_nervous=nervous_bool,
#     comfort_given=already_comforted_bool,
#     answer_text=transcribed_answer,
#     next_original_main=next_question["text"],
#     next_guided_main=next_question["guided_main"],
# )
#
# robot_says(out["comfort_text"] + " " + out["connector_text"])
# mode = out["target_mode"]
