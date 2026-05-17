import base64
import json
import io
import pandas as pd
from groq import Groq
import google.generativeai as genai

MAX_PAGES = 5


# ── Dynamic Personalized Prompt ──────────────────────────────────────────────
def get_analysis_prompt(user_profile=None):
    """
    Generates a personalized prompt by injecting user demographics.
    This enables the 'Predictive' aspect of your project.
    """
    profile_context = ""
    if user_profile is not None:
        profile_context = f"""
PATIENT CONTEXT:
- Age: {user_profile.get('age', '21')}
- Gender: {user_profile.get('gender', 'Male')}
- BMI: {user_profile.get('bmi', 'N/A')}
- Blood Group: {user_profile.get('blood_group', 'O+')}
- Location: Ahmedabad, Gujarat

Please interpret the following lab results considering this specific patient profile.
"""

    return f"""
You are an expert clinical diagnostician and lab report analyst.
{profile_context}

Analyze the provided lab report images/text. Your goal is to provide a predictive health analysis, identifying patterns across different biomarkers.

Return ONLY valid JSON.

STRUCTURE:
{{
  "counts": {{
    "normal": 0, "high": 0, "low": 0, "borderline": 0, "total_tests": 0
  }},
  "categories": {{
    "CBC": [], "Lipid Profile": [], "Liver Function": [], 
    "Kidney Function": [], "Thyroid": [], "Blood Sugar": [], "Other": []
  }},
  "overall_interpretation": "A 2-3 sentence clinical summary of the patient's state.",
  "advice": "Clear, actionable steps the patient should take next based on these results.",
  "critical_flags": ["List any findings requiring immediate attention"],
  "possible_conditions": ["List potential health risks or conditions to monitor based on patterns (not a final diagnosis)"],
  "dietary_recommendations": ["Specific foods to increase or avoid"],
  "lifestyle_changes": ["Habits like sleep, exercise, or stress management"]
}}

Note: Each test in 'categories' must include: {{"name": "...", "value": 0.0, "unit": "...", "status": "Normal/High/Low/Borderline", "meaning": "..."}}
"""


class LabInterpreter:
    def __init__(self, api_key, lab_schema=None, gemini_api_key=None):
        self.groq_client = Groq(api_key=api_key) if api_key else None
        self.gemini_key = gemini_api_key
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)

    def interpret(self, file_bytes, filename, user_profile=None):
        """
        Main entry point for report analysis.
        """
        is_pdf = filename.lower().endswith(".pdf")
        images_base64 = []

        # 1. Prepare Data
        if is_pdf:
            images_base64 = self._pdf_to_images(file_bytes)
        else:
            images_base64 = [base64.b64encode(file_bytes).decode("utf-8")]

        # Generate the personalized prompt
        prompt = get_analysis_prompt(user_profile)

        # 2. Try Gemini (Vision & PDF Support)
        if self.gemini_key:
            try:
                print(f"--- Attempting Gemini Analysis: {filename} ---")
                result = self._call_gemini(prompt, images_base64)
                if result: return result
            except Exception as e:
                print(f"⚠️ Gemini Error: {e}")

        # 3. Fallback to Groq (Llama 3.3 70B)
        if self.groq_client:
            try:
                print(f"--- Falling back to Groq: {filename} ---")
                # For Groq, we extract text if it's a PDF for better Llama-3 processing
                if is_pdf:
                    text_content = self._extract_text_from_pdf(file_bytes)
                    return self._call_groq_text(text_content, prompt)
                else:
                    return self._call_groq_vision(images_base64, prompt)
            except Exception as e:
                print(f"❌ Groq Error: {e}")

        return None

    def _call_gemini(self, prompt, images_b64):
        model = genai.GenerativeModel('gemini-1.5-pro')
        content = [prompt]
        for img in images_b64:
            content.append({"mime_type": "image/jpeg", "data": img})

        response = model.generate_content(content)
        return self._clean_json(response.text)

    def _call_groq_text(self, text, prompt):
        full_query = f"{prompt}\n\nLAB REPORT TEXT:\n{text}"
        chat_completion = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": full_query}],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(chat_completion.choices[0].message.content)

    def _clean_json(self, text):
        # Removes markdown code blocks if present
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)

    def _pdf_to_images(self, file_bytes):
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        images = []
        for i in range(min(len(doc), MAX_PAGES)):
            page = doc.load_page(i)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("jpeg")
            images.append(base64.b64encode(img_bytes).decode("utf-8"))
        doc.close()
        return images

    def _extract_text_from_pdf(self, file_bytes):
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text