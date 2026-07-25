import io
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from gtts import gTTS
from typing import Any, Dict

from app.core.security import get_current_user

router = APIRouter(prefix="/api/tts", tags=["TTS"])


@router.get("/speak", summary="Convert text to speech audio (EN/UR)")
async def text_to_speech(
    text: str = Query(..., description="Text to convert to speech"),
    lang: str = Query("en", description="Language code: 'en' for English, 'ur' for Urdu"),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Generates an MP3 audio stream from the given text using Google TTS.
    Works on any device and deployment — no language pack needed.
    """
    # Sanitize lang to only allow supported codes
    supported = {"en": "en", "ur": "ur"}
    lang_code = supported.get(lang, "en")

    # Generate audio using gTTS (calls Google's TTS API server-side)
    tts = gTTS(text=text[:3000], lang=lang_code, slow=False)  # cap at 3000 chars

    # Write to an in-memory buffer
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    return StreamingResponse(
        audio_buffer,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=report.mp3"},
    )
