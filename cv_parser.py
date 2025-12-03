import PyPDF2
import json
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
import os
from google import genai

# Load .env once at import time
load_dotenv()

class CVParser:
    def __init__(self):
        self.cv_file = "dummy_cv.pdf"  # default, overridden by upload
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.0-flash"

    def extract_text_from_pdf(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += "\n" + t
        return text

    def parse_with_gemini(self, text: str) -> Dict:
        prompt = (
            "You are a CV parser. Read the CV text and return ONLY valid JSON, "
            "no markdown, no explanations. Use exactly this schema:\n"
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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            raw = response.text.strip()

            # Optional debug
            with open("gemini_raw_debug.txt", "w", encoding="utf-8") as dbg:
                dbg.write(raw)

            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()

            return json.loads(raw)
        except Exception as e:
            return {"name": "Unknown", "skills": [], "error": f"Gemini parse failed: {e}"}

    def parse_cv(self, uploaded_file_path: str | None = None) -> Dict:
        file_path = uploaded_file_path or self.cv_file
        if not Path(file_path).exists():
            return {"error": f"File {file_path} not found"}
        text = self.extract_text_from_pdf(file_path)
        return self.parse_with_gemini(text)
