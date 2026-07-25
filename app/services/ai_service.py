"""
AI Service — connects to Grok (primary) and OpenRouter (fallback) for:
  • Medical report generation
  • Urdu translation
  • Chatbot conversations about eye health
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ── API endpoints ────────────────────────────────────────────────────────────
GROK_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = "grok-3-mini-fast"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "deepseek/deepseek-chat-v3-0324:free"

TIMEOUT = 120.0  # seconds


# ─────────────────────────────────────────────────────────────────────────────
#  LOW-LEVEL CALL
# ─────────────────────────────────────────────────────────────────────────────

async def _call_llm(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """
    Try Grok first; on any failure fall back to OpenRouter.
    Returns the assistant's reply text.
    """
    settings = get_settings()

    # ── Attempt 1: Grok / Groq Cloud ─────────────────────────────────────
    if settings.GROK_API_KEY:
        try:
            api_url = GROK_URL
            model_name = GROK_MODEL
            if settings.GROK_API_KEY.startswith("gsk_"):
                api_url = "https://api.groq.com/openai/v1/chat/completions"
                model_name = "llama-3.3-70b-versatile"

            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(
                    api_url,
                    headers={
                        "Authorization": f"Bearer {settings.GROK_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_name,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                logger.info("AI response received (%d chars) from %s", len(content), model_name)
                return content.strip()
        except Exception as exc:
            logger.warning("Grok/Groq call failed: %s — falling back to OpenRouter", exc)

    # ── Attempt 2: OpenRouter ────────────────────────────────────────────
    if settings.OPENROUTER_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://eye-cared-bot.vercel.app",
                        "X-Title": "Eye Cared Bot",
                    },
                    json={
                        "model": OPENROUTER_MODEL,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                logger.info("OpenRouter response received (%d chars)", len(content))
                return content.strip()
        except Exception as exc:
            logger.error("OpenRouter call also failed: %s", exc)

    raise RuntimeError("All AI providers failed. Check API keys and network connectivity.")


# ─────────────────────────────────────────────────────────────────────────────
#  REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

_REPORT_SYSTEM_PROMPT = """You are an expert ophthalmologist AI assistant for Eye Cared Bot. 
Generate a comprehensive, professional medical report based on the AI scan results provided.

The report MUST include ALL of the following sections in order. Use the exact section titles shown:

1. **Patient Details** — Summarise the patient info provided.
2. **AI Scan Results** — State the detected disease and confidence score.
3. **Disease Explanation** — A clear, patient-friendly explanation of the condition.
4. **Common Symptoms** — Bullet list of symptoms associated with this condition.
5. **Possible Causes** — Bullet list of known causes / contributing factors.
6. **Risk Factors** — Bullet list of risk factors.
7. **Prevention & Lifestyle Tips** — Actionable advice to slow or prevent progression.
8. **Recommended Medical Tests** — List of clinical tests a doctor may order.
9. **Treatment Options** — Overview of available treatments (medication, surgery, etc.).
10. **Emergency Warning Signs** — When the patient should seek urgent care.
11. **Disclaimer** — State that this is an AI-generated report and does NOT replace professional medical advice. Advise the patient to consult a qualified ophthalmologist.

Format each section with the section title on its own line followed by the content.
Use clear, compassionate language appropriate for a patient audience.
Do NOT use markdown headers (#), just use the section titles in bold or plain text."""


async def generate_report(
    disease: str,
    confidence: float,
    patient_name: str = "Patient",
    patient_age: Optional[int] = None,
    patient_gender: Optional[str] = None,
    language: str = "en",
) -> List[Dict[str, str]]:
    """
    Generate a structured medical report.

    Returns a list of dicts: [{"title": "...", "content": "..."}, ...]
    """
    patient_info = f"Name: {patient_name}"
    if patient_age:
        patient_info += f", Age: {patient_age}"
    if patient_gender:
        patient_info += f", Gender: {patient_gender}"

    lang_instruction = ""
    if language == "ur":
        lang_instruction = "\n\nIMPORTANT: Write the ENTIRE report in Urdu (اردو). Use Urdu script throughout."

    user_prompt = (
        f"Generate a complete medical eye report for the following scan results:\n\n"
        f"Patient Information: {patient_info}\n"
        f"Detected Disease: {disease}\n"
        f"AI Confidence Score: {confidence * 100:.1f}%\n"
        f"{lang_instruction}"
    )

    messages = [
        {"role": "system", "content": _REPORT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_text = await _call_llm(messages, temperature=0.4, max_tokens=4096)
        sections = _parse_report_sections(raw_text)
        return sections
    except Exception as exc:
        logger.warning("LLM report generation failed, using mock report: %s", exc)
        return get_mock_report(
            disease=disease,
            confidence=confidence,
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            language=language,
        )


def _parse_report_sections(text: str) -> List[Dict[str, str]]:
    """
    Best-effort parsing of the LLM output into titled sections.
    Falls back to a single section if parsing fails.
    """
    section_titles = [
        "Patient Details",
        "AI Scan Results",
        "Disease Explanation",
        "Common Symptoms",
        "Possible Causes",
        "Risk Factors",
        "Prevention & Lifestyle Tips",
        "Recommended Medical Tests",
        "Treatment Options",
        "Emergency Warning Signs",
        "Disclaimer",
    ]

    sections: List[Dict[str, str]] = []
    remaining = text

    for i, title in enumerate(section_titles):
        # Find this section title (case-insensitive, with optional ** wrappers)
        import re
        pattern = re.compile(
            r"(?:\*{0,2})\s*" + re.escape(title) + r"\s*(?:\*{0,2})\s*[:\-—]?\s*",
            re.IGNORECASE,
        )
        match = pattern.search(remaining)
        if match is None:
            continue

        start = match.end()

        # Find the start of the next section
        next_start = len(remaining)
        for next_title in section_titles[i + 1:]:
            next_pattern = re.compile(
                r"(?:\*{0,2})\s*" + re.escape(next_title) + r"\s*(?:\*{0,2})\s*[:\-—]?",
                re.IGNORECASE,
            )
            next_match = next_pattern.search(remaining, pos=start)
            if next_match:
                next_start = next_match.start()
                break

        content = remaining[start:next_start].strip()
        if content:
            sections.append({"title": title, "content": content})

    # Fallback: if parsing found nothing, wrap the whole text
    if not sections:
        sections = [{"title": "Medical Report", "content": text.strip()}]

    return sections


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSLATION
# ─────────────────────────────────────────────────────────────────────────────

async def translate_report_to_urdu(sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Translate all report sections to Urdu."""
    combined = "\n\n".join(f"**{s['title']}**\n{s['content']}" for s in sections)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional medical translator. Translate the following "
                "medical eye report from English to Urdu (اردو). Keep the section "
                "structure intact. Translate section titles too. Maintain medical "
                "accuracy. Use simple Urdu suitable for patients."
            ),
        },
        {"role": "user", "content": combined},
    ]

    try:
        translated = await _call_llm(messages, temperature=0.3, max_tokens=5000)
        parsed = _parse_urdu_sections(translated)
        return parsed if parsed else [{"title": "طبی رپورٹ", "content": translated}]
    except Exception as exc:
        logger.warning("LLM report translation failed, using mock translation: %s", exc)
        return get_mock_report_urdu_translation(sections)


def _parse_urdu_sections(text: str) -> List[Dict[str, str]]:
    """
    Parse translated Urdu text into sections by splitting on bold markers.
    """
    import re
    parts = re.split(r"\*\*(.+?)\*\*", text)
    sections: List[Dict[str, str]] = []
    i = 1
    while i < len(parts) - 1:
        title = parts[i].strip()
        content = parts[i + 1].strip()
        if title and content:
            sections.append({"title": title, "content": content})
        i += 2
    return sections


# ─────────────────────────────────────────────────────────────────────────────
#  CHATBOT
# ─────────────────────────────────────────────────────────────────────────────

_CHAT_SYSTEM_PROMPT = """You are Eye Cared Bot's friendly AI assistant specializing in eye health and ophthalmology.

Rules:
1. Answer questions about eye diseases, symptoms, treatments, prevention, and general eye care.
2. Be empathetic, informative, and use simple language.
3. If the user writes in Urdu or Roman Urdu, reply in the same language.
4. If the user writes in English, reply in English.
5. NEVER provide definitive diagnoses — always recommend consulting an ophthalmologist for medical decisions.
6. You can discuss the diseases detected by Eye Cared Bot: Normal, Diabetic Retinopathy, Glaucoma, Cataract, Age-related Macular Degeneration, Hypertensive Retinopathy, Pathological Myopia.
7. For non-eye-health topics, politely redirect the conversation to eye care.
8. Keep responses concise but thorough (2-4 paragraphs max unless asked for detail).
9. Use bullet points for lists of symptoms, tips, etc."""


async def chat(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, str]:
    """
    Handle a chatbot message.
    Returns {"reply": "...", "detected_language": "en" | "ur"}
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _CHAT_SYSTEM_PROMPT},
    ]

    # Add conversation history
    if history:
        for msg in history[-10:]:  # keep last 10 turns
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": message})

    try:
        reply = await _call_llm(messages, temperature=0.7, max_tokens=1500)
    except Exception as exc:
        logger.warning("LLM chat failed, using mock chatbot reply: %s", exc)
        reply = get_mock_chat_reply(message)

    # Simple language detection
    detected = _detect_language(message)

    return {"reply": reply, "detected_language": detected}


def _detect_language(text: str) -> str:
    """Heuristic language detection: Urdu vs English."""
    urdu_range = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF" or "\uFB50" <= ch <= "\uFDFF")
    if urdu_range > len(text) * 0.3:
        return "ur"
    return "en"


def get_mock_report(
    disease: str,
    confidence: float,
    patient_name: str = "Patient",
    patient_age: Optional[int] = None,
    patient_gender: Optional[str] = None,
    language: str = "en",
) -> List[Dict[str, str]]:
    if language == "ur":
        return [
            {"title": "مریض کی تفصیلات", "content": f"نام: {patient_name}\nعمر: {patient_age or 'منسلک نہیں'}\nجنس: {patient_gender or 'منسلک نہیں'}"},
            {"title": "مصنوعی ذہانت کے نتائج", "content": f"تشخیص شدہ بیماری: {disease}\nاعتماد کا اسکور: {confidence * 100:.1f}%"},
            {"title": "بیماری کی وضاحت", "content": f"اس اسکین نے اشارہ کیا ہے کہ مریض میں {disease} کی علامات ہو سکتی ہیں۔ یہ ایک طبی معائنہ کی ضرورت پیش کرتا ہے۔"},
            {"title": "عام علامات", "content": "• بینائی میں دھندلاپن\n• نظر کی کمزوری\n• آنکھوں میں درد یا دباؤ کا احساس"},
            {"title": "ممکنہ وجوہات", "content": "• خاندانی تاریخ اور جینیات\n• عمر کی زیادتی\n• ذیابیطس یا ہائی بلڈ پریشر"},
            {"title": "خطرے کے عوامل", "content": "• عمر کا 50 سال سے زیادہ ہونا\n• آنکھ کی پچھلی چوٹ\n• ذیابیطس کے مریض"},
            {"title": "روک تھام اور طرز زندگی کی تجاویز", "content": "• آنکھوں کا باقاعدگی سے معائنہ کروائیں۔\n• متوازن غذا کھائیں جس میں سبز پتوں والی سبزیاں شامل ہوں۔\n• سگریٹ نوشی سے پرہیز کریں۔"},
            {"title": "تجویز کردہ طبی ٹیسٹ", "content": "• بصری میدان کا ٹیسٹ (Visual Field Test)\n• او سی ٹی اسکین (OCT Scan)\n• ٹونومیٹری (آنکھ کا دباؤ)"},
            {"title": "علاج کے اختیارات", "content": "• آئی ڈراپس (آنکھ کے قطرے)\n• لیزر علاج (Laser Treatment)\n• سرجری (اگر ضرورت ہو)"},
            {"title": "ہنگامی انتباہی علامات", "content": "• اچانک نظر کا چلے جانا\n• شدید درد یا سرخی\n• روشنی کے جھماکے نظر آنا"},
            {"title": "دستبرداری", "content": "یہ رپورٹ ایک خودکار نظام (AI) کے ذریعے تیار کی گئی ہے اور یہ کسی مستند ڈاکٹر کے معائنے کا نعم البدل نہیں ہے۔ براہ کرم اپنے ماہر چشم سے رجوع کریں۔"},
        ]
    else:
        return [
            {"title": "Patient Details", "content": f"Name: {patient_name}\nAge: {patient_age or 'N/A'}\nGender: {patient_gender or 'N/A'}"},
            {"title": "AI Scan Results", "content": f"Detected Disease: {disease}\nConfidence Score: {confidence * 100:.1f}%"},
            {"title": "Disease Explanation", "content": f"The AI scan has detected signs of {disease}. This is a condition that affects eye health and requires clinical correlation by an ophthalmologist."},
            {"title": "Common Symptoms", "content": "• Gradual loss or blurring of vision\n• Distortions in central vision\n• Eye fatigue or mild discomfort"},
            {"title": "Possible Causes", "content": "• Age-related changes in the retina\n• Genetic predisposition\n• Systemic conditions like diabetes or hypertension"},
            {"title": "Risk Factors", "content": "• Age over 50 years\n• Family history of eye diseases\n• Prolonged exposure to blue light or UV rays"},
            {"title": "Prevention & Lifestyle Tips", "content": "• Maintain a diet rich in green leafy vegetables and omega-3.\n• Protect eyes with sunglasses in bright sunlight.\n• Stop smoking and control blood pressure/blood sugar."},
            {"title": "Recommended Medical Tests", "content": "• Optical Coherence Tomography (OCT scan)\n• Tonometry (eye pressure measurement)\n• Visual Field (perimetry) testing"},
            {"title": "Treatment Options", "content": "• Prescription eye drops or medication\n• Laser photocoagulation\n• Surgical intervention if recommended by your doctor"},
            {"title": "Emergency Warning Signs", "content": "• Sudden, complete loss of vision in one or both eyes\n• Severe eye pain accompanied by redness or nausea\n• Sudden appearance of floaters or flashes of light"},
            {"title": "Disclaimer", "content": "This AI-generated report is for educational purposes only and does NOT constitute professional medical advice. Always consult a qualified ophthalmologist."},
        ]


def get_mock_report_urdu_translation(sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
    translations = {
        "Patient Details": "مریض کی تفصیلات",
        "AI Scan Results": "مصنوعی ذہانت کے نتائج",
        "Disease Explanation": "بیماری کی وضاحت",
        "Common Symptoms": "عام علامات",
        "Possible Causes": "ممکنہ وجوہات",
        "Risk Factors": "خطرے کے عوامل",
        "Prevention & Lifestyle Tips": "روک تھام اور طرز زندگی کی تجاویز",
        "Recommended Medical Tests": "تجویز کردہ طبی ٹیسٹ",
        "Treatment Options": "علاج کے اختیارات",
        "Emergency Warning Signs": "ہنگامی انتباہی علامات",
        "Disclaimer": "دستبرداری"
    }
    
    translated_sections = []
    for s in sections:
        title_ur = translations.get(s["title"], "تفصیلات")
        content_ur = s["content"]
        if s["title"] == "Common Symptoms":
            content_ur = "• بینائی میں دھندلاپن\n• نظر کی کمزوری\n• آنکھوں میں درد یا دباؤ کا احساس"
        elif s["title"] == "Disclaimer":
            content_ur = "یہ رپورٹ ایک خودکار نظام (AI) کے ذریعے تیار کی گئی ہے اور یہ کسی مستند ڈاکٹر کے معائنے کا نعم البدل نہیں ہے۔ براہ کرم اپنے ماہر چشم سے رجوع کریں۔"
        elif s["title"] == "Emergency Warning Signs":
            content_ur = "• اچانک نظر کا چلے جانا\n• شدید درد یا سرخی\n• روشنی کے جھماکے نظر آنا"
        translated_sections.append({"title": title_ur, "content": content_ur})
        
    return translated_sections


def get_mock_chat_reply(message: str) -> str:
    is_urdu = any("\u0600" <= ch <= "\u06FF" for ch in message)
    if is_urdu:
        return (
            "پیارے صارف! میں آئی کیئرڈ بوٹ کا اے آئی اسسٹنٹ ہوں۔\n\n"
            "فی الحال، اے آئی سروس ڈیمو موڈ میں چل رہی ہے کیونکہ اے آئی اے پی آئی کیز (Grok/OpenRouter) کنفیگر نہیں کی گئیں۔\n\n"
            "آنکھ کی حفاظت کے عمومی مشورے:\n"
            "• اپنی آنکھوں کو تیز روشنی اور سگریٹ کے دھوئیں سے بچائیں۔\n"
            "• ہر 20 منٹ بعد 20 سیکنڈ کے لیے 20 فٹ دور دیکھیں۔\n"
            "• آنکھوں کے کسی بھی بڑے مسئلے کی صورت میں فوراً ماہر چشم سے رابطہ کریں۔"
        )
    else:
        return (
            "Hello! I am the Eye Cared Bot AI Assistant.\n\n"
            "Note: The AI chatbot is currently running in **Demo Mode** because the API keys (Grok or OpenRouter) have not been configured in the backend `.env` file.\n\n"
            "General Eye Care Tips:\n"
            "• Follow the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds.\n"
            "• Maintain regular eye exams and eat foods rich in Vitamin A and Omega-3.\n"
            "• Always consult a qualified ophthalmologist for any visual issues or emergency warnings."
        )
