"""
Notification endpoints:
  GET  /api/notifications           — list user's notifications
  POST /api/notifications/{id}/read — mark a notification as read
  POST /api/notifications/read-all  — mark all notifications as read
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, Query

from app.core.security import get_current_user
from app.models.schemas import NotificationListResponse, NotificationItem, MessageResponse
from app.services.notification_service import (
    get_user_notifications,
    mark_notification_read,
    mark_all_read,
    get_unread_count,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/notifications", tags=["Notifications"])


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/notifications
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=NotificationListResponse,
    summary="Get current user's notifications",
)
async def list_notifications(
    limit: int = Query(50, ge=1, le=100),
    unread_only: bool = Query(False),
    user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = user["sub"]

    notifications = await get_user_notifications(
        user_id=user_id,
        limit=limit,
        unread_only=unread_only,
    )
    unread = await get_unread_count(user_id)

    return NotificationListResponse(
        notifications=[NotificationItem(**n) for n in notifications],
        unread_count=unread,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/notifications/{id}/read
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/{notification_id}/read",
    response_model=MessageResponse,
    summary="Mark a notification as read",
)
async def mark_read(
    notification_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
):
    success = await mark_notification_read(notification_id, user["sub"])
    if not success:
        return MessageResponse(message="Notification not found or already read.")
    return MessageResponse(message="Notification marked as read.")


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/notifications/read-all
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/read-all",
    response_model=MessageResponse,
    summary="Mark all notifications as read",
)
async def mark_all_notifications_read(
    user: Dict[str, Any] = Depends(get_current_user),
):
    count = await mark_all_read(user["sub"])
    return MessageResponse(message=f"{count} notifications marked as read.")
