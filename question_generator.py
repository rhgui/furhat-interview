"""
question_generator.py

Generate interview questions and follow-ups from a CV using Google Gemini
(gemini-2.0-flash), with simple nervous / not_nervous control.

Public entry points (for app.py / furhat_robot.py / others):

- generate_main_questions(
      cv_text: str,
      position: str,
      nervous_flag: str,   # "Yes" / "No"  (Yes = nervous)
      num_questions: int = 3,
  ) -> list[dict]

    Returns a list of question dicts:
    {
        "id": "Q3",
        "text": "...",
        "category": "technical_experience",
        "difficulty": "easy|medium|hard",
        "expected_time_seconds": 25,
        "followup_suggestions": ["...", "..."]
    }

- evaluate_answer_for_followup(
      question: dict,
      answer_text: str,
      nervous_flag: str,   # "Yes" / "No"
  ) -> dict

    Returns:
    {
        "is_complete": bool,
        "need_followup": bool,
        "followup_question": str | None,
        "expected_time_seconds": int | None
    }

Both functions internally normalize nervous_flag:
    "Yes"  -> "nervous"
    "No"   -> "not_nervous"
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Literal

import google.generativeai as genai

# ---------------------------------------------------------------------------
# Gemini configuration 
# ---------------------------------------------------------------------------

GEMINI_MODEL_NAME = "gemini-2.0-flash"


def _configure_gemini() -> genai.GenerativeModel:
    """
    Configure Gemini client once and return a GenerativeModel object.
    This uses same model name.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Please export it in your environment."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL_NAME)


# Create a single model instance for this module.
_gemini_model: genai.GenerativeModel = _configure_gemini()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class InterviewQuestion:
    id: str
    text: str
    category: str
    difficulty: str
    expected_time_seconds: int
    followup_suggestions: List[str]


@dataclass
class FollowupDecision:
    is_complete: bool
    need_followup: bool
    followup_question: Optional[str]
    expected_time_seconds: Optional[int]


EmotionStr = Literal["nervous", "not_nervous"]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _normalize_nervous_flag(flag: str) -> EmotionStr:
    """
    Convert external Yes / No (or other variants) into
    internal 'nervous' / 'not_nervous'.
    """
    if flag is None:
        return "not_nervous"

    f = str(flag).strip().lower()

    if f in {"yes", "y", "true", "1", "nervous", "anxious"}:
        return "nervous"

    if f in {"no", "n", "false", "0", "not_nervous", "calm", "relaxed"}:
        return "not_nervous"

    # Fallback: be conservative and assume not nervous
    return "not_nervous"


def _extract_json_array(text: str) -> Any:
    """
    Extract the first JSON array from a model response.
    - Gemini often wraps JSON in backticks or extra text.
    """
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")

    if first_bracket == -1 or last_bracket == -1 or last_bracket <= first_bracket:
        raise ValueError("No JSON array found in model output.")

    json_str = text[first_bracket : last_bracket + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from model output. Raw slice:\n{json_str}"
        ) from e


def _extract_json_object(text: str) -> Any:
    """
    Extract the first JSON object ({ ... }) from model output.
    Used for followup decisions.
    """
    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        raise ValueError("No JSON object found in model output.")

    json_str = text[first_brace : last_brace + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON object from model output. Raw slice:\n{json_str}"
        ) from e


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_questions_prompt(
    cv_text: str,
    position: str,
    emotion: EmotionStr,
    num_questions: int,
) -> str:
    """
    Build the prompt for generating main questions Q3/Q4/Q5.
    LLM role: HR interviewer. It must infer the specific domain
    (robotics, AI, mechanical engineering, etc.) from the CV itself.
    """

    if emotion == "nervous":
        emotion_instructions = (
            "- The candidate appears NERVOUS based on voice and video.\n"
            "- Start from easier, more concrete questions.\n"
            "- Avoid aggressive or overly challenging questions.\n"
        )
    else:
        emotion_instructions = (
            "- The candidate does NOT appear nervous.\n"
            "- You may use medium difficulty questions.\n"
        )

    return f"""
You are an HR interviewer for the position: "{position}".

You have access to the candidate's full CV text below. 
From the CV, infer their professional domain and technical background 
(e.g., robotics & AI, mechanical engineering, software engineering, etc.),
and adapt your questions accordingly.

Candidate CV (verbatim text):
-----------------------------
{cv_text}
-----------------------------

Interview context:
- The first two warm-up questions are already asked (Q1: feeling; Q2: self-introduction).
- Now you must generate the next {num_questions} questions: Q3, Q4, Q5, ...
- These questions should dig into the candidate's background, projects and skills.

Emotion state:
{emotion_instructions}

Output requirements:
- Return a JSON array of EXACTLY {num_questions} objects.
- Each object MUST contain the following fields:
  - "id": string, like "Q3", "Q4", "Q5" (sequential).
  - "text": the full interview question in English.
  - "category": a short category, e.g. 
      "technical_experience", "project_reflection", "motivation",
      "behavioral", "teamwork", etc.
  - "difficulty": one of "easy", "medium", "hard".
  - "expected_time_seconds": integer, the EXPECTED time (in seconds)
      that a good candidate answer would take to speak.
      For example: 
        easy ~ 15-25 seconds,
        medium ~ 25-40 seconds,
        hard ~ 40-60 seconds.
      Choose a reasonable value depending on question complexity.
  - "followup_suggestions": an array of 2-4 short follow-up question candidates
      that could be asked later to go deeper on the same topic.

IMPORTANT:
- The JSON array must NOT contain any comments, no trailing commas, no markdown.
- Do not add explanations before or after the JSON.
    """.strip()


def _build_followup_prompt(
    question: InterviewQuestion,
    answer_text: str,
    emotion: EmotionStr,
) -> str:
    """
    Build prompt for evaluating an answer and deciding whether to ask a follow-up.
    """

    if emotion == "nervous":
        emotion_instructions = (
            "- The candidate is NERVOUS. Be supportive and gentle in your judgement.\n"
        )
    else:
        emotion_instructions = (
            "- The candidate is NOT nervous. You can expect a reasonably detailed answer.\n"
        )

    return f"""
You are an HR interviewer evaluating a candidate's answer.

You asked the following question:
Q: "{question.text}"

The candidate answered:
A: "{answer_text}"

Context:
- The question difficulty was labeled as "{question.difficulty}".
- The candidate is speaking in a {"NERVOUS" if emotion == "nervous" else "NOT NERVOUS"} state.
{emotion_instructions}
- You must decide whether:
  (a) the answer is complete enough, and
  (b) a follow-up question would add meaningful depth or clarification.

Output format:
Return a single JSON object with the fields:
- "is_complete": boolean, is the answer reasonably complete?
- "need_followup": boolean, should we ask a follow-up question for this topic?
- "followup_question": string | null
    - If "need_followup" is true, propose ONE concise follow-up question.
    - If "need_followup" is false, set this to null.
- "expected_time_seconds": integer | null
    - If "need_followup" is true, give the expected time (seconds) for 
      answering the follow-up question (similar scale as main questions).
    - Otherwise set this to null.

IMPORTANT:
- The JSON object must NOT contain comments, no trailing commas, no markdown.
- Do not add explanations before or after the JSON.
    """.strip()


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------


def generate_main_questions(
    cv_text: str,
    position: str,
    nervous_flag: str,
    num_questions: int = 3,
) -> List[Dict[str, Any]]:
    """
    Generate main interview questions (Q3/Q4/Q5) from CV.

    Parameters
    ----------
    cv_text : str
        Raw extracted text from the candidate's CV.
    position : str
        Target position title, e.g. "Robotics Intern".
    nervous_flag : str
        "Yes" if the candidate is nervous, "No" if not.
        (Other variants are normalized internally.)
    num_questions : int
        Number of questions to generate (usually 3 for Q3/Q4/Q5).

    Returns
    -------
    List[Dict[str, Any]]
        List of question dicts ready to be saved or sent to Furhat.
    """
    emotion = _normalize_nervous_flag(nervous_flag)

    prompt = _build_questions_prompt(
        cv_text=cv_text,
        position=position,
        emotion=emotion,
        num_questions=num_questions,
    )

    response = _gemini_model.generate_content(prompt)
    raw_text = response.text or ""

    data = _extract_json_array(raw_text)

    questions: List[InterviewQuestion] = []
    for item in data:
        # Robust extraction with defaults.
        q_id = str(item.get("id", ""))
        text = str(item.get("text", "")).strip()
        category = str(item.get("category", "general")).strip()
        difficulty = str(item.get("difficulty", "medium")).strip().lower()
        try:
            expected_time = int(item.get("expected_time_seconds", 30))
        except Exception:
            expected_time = 30
        followups = item.get("followup_suggestions") or []
        if not isinstance(followups, list):
            followups = []

        questions.append(
            InterviewQuestion(
                id=q_id,
                text=text,
                category=category,
                difficulty=difficulty,
                expected_time_seconds=expected_time,
                followup_suggestions=[str(f).strip() for f in followups],
            )
        )

    # Convert dataclasses to simple dicts for outside use.
    return [asdict(q) for q in questions]


def evaluate_answer_for_followup(
    question_dict: Dict[str, Any],
    answer_text: str,
    nervous_flag: str,
) -> Dict[str, Any]:
    """
    Evaluate a candidate's answer and decide whether to ask a follow-up.

    Parameters
    ----------
    question_dict : Dict[str, Any]
        One question dict as produced by generate_main_questions(...).
    answer_text : str
        Candidate's spoken answer (converted to text by ASR).
    nervous_flag : str
        "Yes" if nervous, "No" if not.

    Returns
    -------
    Dict[str, Any]
        {
          "is_complete": bool,
          "need_followup": bool,
          "followup_question": str | None,
          "expected_time_seconds": int | None
        }
    """
    emotion = _normalize_nervous_flag(nervous_flag)

    # Convert dict back to dataclass for convenient access
    question = InterviewQuestion(
        id=str(question_dict.get("id", "")),
        text=str(question_dict.get("text", "")),
        category=str(question_dict.get("category", "general")),
        difficulty=str(question_dict.get("difficulty", "medium")),
        expected_time_seconds=int(question_dict.get("expected_time_seconds", 30)),
        followup_suggestions=[
            str(f).strip() for f in question_dict.get("followup_suggestions", [])
        ],
    )

    prompt = _build_followup_prompt(
        question=question,
        answer_text=answer_text,
        emotion=emotion,
    )

    response = _gemini_model.generate_content(prompt)
    raw_text = response.text or ""

    obj = _extract_json_object(raw_text)

    is_complete = bool(obj.get("is_complete", False))
    need_followup = bool(obj.get("need_followup", False))
    followup_question = obj.get("followup_question")
    if followup_question is not None:
        followup_question = str(followup_question).strip() or None

    expected_time = obj.get("expected_time_seconds")
    try:
        expected_time_int = int(expected_time) if expected_time is not None else None
    except Exception:
        expected_time_int = None

    decision = FollowupDecision(
        is_complete=is_complete,
        need_followup=need_followup,
        followup_question=followup_question,
        expected_time_seconds=expected_time_int,
    )

    return {
        "is_complete": decision.is_complete,
        "need_followup": decision.need_followup,
        "followup_question": decision.followup_question,
        "expected_time_seconds": decision.expected_time_seconds,
    }
