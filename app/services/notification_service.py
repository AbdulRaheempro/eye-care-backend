"""
Notification Service — helpers to create, fetch, and manage notifications
stored in the Supabase `notifications` table.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from app.core.database import get_supabase_client

logger = logging.getLogger(__name__)


async def create_notification(
    user_id: str,
    title: str,
    message: str,
    notification_type: str = "info",
    link: Optional[str] = None,
) -> Dict:
    """
    Insert a new notification row.

    Parameters
    ----------
    user_id : target user's UUID
    title : short heading
    message : body text
    notification_type : info | report | review | system
    link : optional deep-link path (e.g. /reports/<id>)
    """
    supabase = get_supabase_client()

    payload = {
        "id": str(uuid4()),
        "user_id": user_id,
        "title": title,
        "message": message,
        "type": notification_type,
        "read": False,
        "link": link,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        result = supabase.table("notifications").insert(payload).execute()
        logger.info("Notification created for user %s: %s", user_id, title)
        data = result.data[0] if result.data else payload
        data["is_read"] = data.get("read", False)
        return data
    except Exception as exc:
        logger.error("Failed to create notification: %s", exc)
        payload["is_read"] = False
        return payload


async def get_user_notifications(
    user_id: str,
    limit: int = 50,
    unread_only: bool = False,
) -> List[Dict]:
    """Fetch notifications for a user, newest first."""
    supabase = get_supabase_client()

    try:
        query = (
            supabase.table("notifications")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
        )
        if unread_only:
            query = query.eq("read", False)

        result = query.execute()
        rows = result.data or []
        for r in rows:
            r["is_read"] = r.get("read", False)
        return rows
    except Exception as exc:
        logger.error("Failed to fetch notifications: %s", exc)
        return []


async def mark_notification_read(notification_id: str, user_id: str) -> bool:
    """Mark a single notification as read. Returns True on success."""
    supabase = get_supabase_client()

    try:
        result = (
            supabase.table("notifications")
            .update({"read": True})
            .eq("id", notification_id)
            .eq("user_id", user_id)
            .execute()
        )
        return bool(result.data)
    except Exception as exc:
        logger.error("Failed to mark notification read: %s", exc)
        return False


async def mark_all_read(user_id: str) -> int:
    """Mark all unread notifications as read for a user. Returns count updated."""
    supabase = get_supabase_client()

    try:
        result = (
            supabase.table("notifications")
            .update({"read": True})
            .eq("user_id", user_id)
            .eq("read", False)
            .execute()
        )
        count = len(result.data) if result.data else 0
        logger.info("Marked %d notifications as read for user %s", count, user_id)
        return count
    except Exception as exc:
        logger.error("Failed to mark all as read: %s", exc)
        return 0


async def get_unread_count(user_id: str) -> int:
    """Return the number of unread notifications for a user."""
    supabase = get_supabase_client()

    try:
        result = (
            supabase.table("notifications")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .eq("read", False)
            .execute()
        )
        return result.count or 0
    except Exception as exc:
        logger.error("Failed to count unread notifications: %s", exc)
        return 0
