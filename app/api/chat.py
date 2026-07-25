"""
POST /api/chat — Conversational AI assistant for eye health questions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends

from app.core.security import get_current_user_optional
from app.models.schemas import ChatRequest, ChatResponse
from app.services.ai_service import chat

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Chat"])


@router.post(
    "/chat/message",
    response_model=ChatResponse,
    summary="Chat with Eye Cared Bot AI assistant (alias)",
)
@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with Eye Cared Bot AI assistant",
)
async def chat_endpoint(
    body: ChatRequest,
    user: Dict[str, Any] | None = Depends(get_current_user_optional),
):
    """
    Send a message to the AI eye-health assistant.
    Accepts optional conversation history for multi-turn chat.
    Authentication is optional — unauthenticated users can still chat.
    """
    history = [{"role": m.role, "content": m.content} for m in body.history]

    result = await chat(message=body.message, history=history)

    return ChatResponse(
        reply=result["reply"],
        detected_language=result.get("detected_language"),
    )


@router.get(
    "/chat/history",
    summary="Get user chat history",
)
async def chat_history(
    user: Dict[str, Any] | None = Depends(get_current_user_optional),
):
    return {"history": []}
