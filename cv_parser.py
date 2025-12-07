import PyPDF2
import json
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
import os
from openai import OpenAI

# Load .env once at import time
load_dotenv()


class CVParser:
    """
    Parse CV PDF files into structured JSON using DeepSeek (OpenAI-compatible API).
    """

    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not set in .env")

        # DeepSeek uses OpenAI-compatible API
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        # Default model name (same as used in QuestionGenerator)
        self.model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")

        # Default file path (overwritten when a user uploads a file)
        self.cv_file = "dummy_cv.pdf"

    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract plain text from a PDF file for LLM parsing.
        """
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += "\n" + page_text
        return text

    def parse_with_llm(self, text: str) -> Dict:
        """
        Parse CV text using DeepSeek and return structured JSON:

        {
          "name": "Full name",
          "email": "email or empty string",
          "education": ["degree, university, years"],
          "experience": ["job title, company, years"],
          "skills": ["skill1", "skill2", "skill3"]
        }
        """

        prompt = (
            "You are a CV parser. Read the CV text and return ONLY valid JSON, "
            "with no markdown, no comments, and no code fences. "
            "Use EXACTLY the following schema:\n"
            "{\n"
            '  \"name\": \"Full name\",\n'
            '  \"email\": \"email or empty string\",\n'
            '  \"education\": [\"degree, university, years\"],\n'
            '  \"experience\": [\"job title, company, years\"],\n'
            '  \"skills\": [\"skill1\", \"skill2\", \"skill3\"]\n'
            "}\n\n"
            "CV text:\n"
            f"{text}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            raw = resp.choices[0].message.content.strip()

            # Debug save: inspect raw output if parsing fails
            with open("deepseek_cv_raw_debug.txt", "w", encoding="utf-8") as dbg:
                dbg.write(raw)

            # Remove ```json ... ``` wrappers if present
            if raw.startswith("```"):
                cleaned = raw.strip().strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
                raw = cleaned

            return json.loads(raw)

        except Exception as e:
            # Fallback structure: prevents crash in QuestionGenerator
            return {
                "name": "Unknown",
                "email": "",
                "education": [],
                "experience": [],
                "skills": [],
                "error": f"DeepSeek CV parsing failed: {e}",
            }

    def parse_cv(self, uploaded_file_path: str | None = None) -> Dict:
        """
        Main external entry point.

        If an uploaded file path is provided, parse that file.
        Otherwise, use the default self.cv_file.
        """
        file_path = uploaded_file_path or self.cv_file

        if not Path(file_path).exists():
            return {"error": f"File not found: {file_path}"}

        text = self.extract_text_from_pdf(file_path)
        return self.parse_with_llm(text)
