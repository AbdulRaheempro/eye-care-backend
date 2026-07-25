"""
Admin endpoints:
  GET   /api/admin/stats            — platform statistics
  GET   /api/admin/users            — list all users
  PATCH /api/admin/users/{id}/role  — update a user's role
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Depends, Query

from app.core.security import require_role
from app.core.database import get_supabase_client
from app.models.schemas import (
    PlatformStats,
    AdminUserItem,
    RoleUpdateRequest,
    MessageResponse,
)
from app.services.notification_service import create_notification

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["Admin"])


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/admin/stats
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/stats",
    response_model=PlatformStats,
    summary="Platform-wide statistics",
)
async def get_stats(
    user: Dict[str, Any] = Depends(require_role("admin")),
):
    supabase = get_supabase_client()

    # Total users
    try:
        users_result = supabase.table("profiles").select("id", count="exact").execute()
        total_users = users_result.count or 0
    except Exception:
        total_users = 0

    # Total scans
    try:
        scans_result = supabase.table("scans").select("id", count="exact").execute()
        total_scans = scans_result.count or 0
    except Exception:
        total_scans = 0

    # Total reports
    try:
        reports_result = supabase.table("reports").select("id", count="exact").execute()
        total_reports = reports_result.count or 0
    except Exception:
        total_reports = 0

    # Total doctors
    try:
        doctors_result = (
            supabase.table("profiles")
            .select("id", count="exact")
            .eq("role", "doctor")
            .execute()
        )
        total_doctors = doctors_result.count or 0
    except Exception:
        total_doctors = 0

    # Disease distribution
    disease_distribution: Dict[str, int] = {}
    try:
        scans_all = supabase.table("scans").select("disease").execute()
        for scan in (scans_all.data or []):
            disease = scan.get("disease", "Unknown")
            disease_distribution[disease] = disease_distribution.get(disease, 0) + 1
    except Exception:
        pass

    # Scans in last 7 days
    scans_last_7 = 0
    try:
        seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        recent_result = (
            supabase.table("scans")
            .select("id", count="exact")
            .gte("created_at", seven_days_ago)
            .execute()
        )
        scans_last_7 = recent_result.count or 0
    except Exception:
        pass

    return PlatformStats(
        total_users=total_users,
        total_scans=total_scans,
        total_reports=total_reports,
        total_doctors=total_doctors,
        disease_distribution=disease_distribution,
        scans_last_7_days=scans_last_7,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/admin/users
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/users",
    response_model=List[AdminUserItem],
    summary="List all users",
)
async def list_users(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    role: str | None = Query(None, pattern="^(user|doctor|admin)$"),
    user: Dict[str, Any] = Depends(require_role("admin")),
):
    supabase = get_supabase_client()
    offset = (page - 1) * per_page

    try:
        query = (
            supabase.table("profiles")
            .select("id, email, full_name, role, created_at")
            .order("created_at", desc=True)
            .range(offset, offset + per_page - 1)
        )
        if role:
            query = query.eq("role", role)

        result = query.execute()
        profiles = result.data or []
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    # Enrich with scan counts
    items: List[AdminUserItem] = []
    for p in profiles:
        scan_count = 0
        try:
            sc = (
                supabase.table("scans")
                .select("id", count="exact")
                .eq("user_id", p["id"])
                .execute()
            )
            scan_count = sc.count or 0
        except Exception:
            pass

        items.append(AdminUserItem(
            id=p["id"],
            email=p.get("email", ""),
            full_name=p.get("full_name"),
            role=p.get("role", "user"),
            created_at=p.get("created_at"),
            scan_count=scan_count,
        ))

    return items


# ─────────────────────────────────────────────────────────────────────────────
#  PATCH /api/admin/users/{id}/role
# ─────────────────────────────────────────────────────────────────────────────

@router.patch(
    "/users/{user_id}/role",
    response_model=MessageResponse,
    summary="Update a user's role",
)
async def update_user_role(
    user_id: str,
    body: RoleUpdateRequest,
    admin: Dict[str, Any] = Depends(require_role("admin")),
):
    supabase = get_supabase_client()

    # Prevent self-demotion
    if user_id == admin["sub"]:
        raise HTTPException(status_code=400, detail="Cannot change your own role.")

    # Verify user exists
    try:
        user_result = supabase.table("profiles").select("id, role").eq("id", user_id).execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not user_result.data:
        raise HTTPException(status_code=404, detail="User not found.")

    old_role = user_result.data[0].get("role", "user")

    # Update role
    try:
        supabase.table("profiles").update({"role": body.role}).eq("id", user_id).execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    # Notify user of role change
    await create_notification(
        user_id=user_id,
        title="Role Updated",
        message=f"Your role has been changed from '{old_role}' to '{body.role}'.",
        notification_type="system",
    )

    return MessageResponse(message=f"User role updated from '{old_role}' to '{body.role}'.")
