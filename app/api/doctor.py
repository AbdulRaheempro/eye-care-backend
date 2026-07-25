"""
Doctor portal endpoints:
  GET  /api/doctor/reports          — list reports assigned to the doctor
  GET  /api/doctor/reports/{id}     — get a specific assigned report
  POST /api/doctor/reports/{id}/review — submit a review
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Depends

from app.core.security import require_role
from app.core.database import get_supabase_client
from app.models.schemas import (
    DoctorReportResponse,
    DoctorReviewRequest,
    MessageResponse,
)
from app.services.notification_service import create_notification

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/doctor", tags=["Doctor Portal"])


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/doctor/reports
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/reports",
    response_model=List[DoctorReportResponse],
    summary="List reports assigned or submitted to the doctor portal",
)
async def list_doctor_reports(
    user: Dict[str, Any] = Depends(require_role("doctor", "admin")),
):
    supabase = get_supabase_client()

    try:
        result = (
            supabase.table("reports")
            .select("*, scans(image_url)")
            .eq("status", "pending")
            .order("created_at", desc=True)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    reports = result.data or []
    # Filter for doctor role: show ONLY reports explicitly assigned to them
    if user.get("role") != "admin":
        reports = [
            r for r in reports
            if r.get("doctor_id") == user["sub"]
        ]

    return [_map_doctor_report(r) for r in reports]


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/doctor/reports/{id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/reports/{report_id}",
    response_model=DoctorReportResponse,
    summary="Get a specific assigned report",
)
async def get_doctor_report(
    report_id: str,
    user: Dict[str, Any] = Depends(require_role("doctor", "admin")),
):
    supabase = get_supabase_client()

    try:
        result = (
            supabase.table("reports")
            .select("*, scans(image_url)")
            .eq("id", report_id)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not result.data:
        raise HTTPException(status_code=404, detail="Report not found.")

    report = result.data[0]
    return _map_doctor_report(report)


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/doctor/reports/{id}/review
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/reports/{report_id}/review",
    response_model=MessageResponse,
    summary="Submit a doctor's review for a report",
)
async def submit_review(
    report_id: str,
    body: DoctorReviewRequest,
    user: Dict[str, Any] = Depends(require_role("doctor", "admin")),
):
    supabase = get_supabase_client()

    # Fetch report
    try:
        result = (
            supabase.table("reports")
            .select("*")
            .eq("id", report_id)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not result.data:
        raise HTTPException(status_code=404, detail="Report not found.")

    report = result.data[0]

    # Update report with review and assign to this doctor if unassigned
    now = datetime.now(timezone.utc).isoformat()
    final_notes = body.review_notes or body.notes or ""
    if body.recommendations:
        final_notes = f"{final_notes}\n\nRecommendations: {body.recommendations}" if final_notes else f"Recommendations: {body.recommendations}"
    if not final_notes:
        final_notes = "Clinically reviewed and verified by ophthalmologist."

    status_val = body.status or ("follow_up" if body.follow_up_required else "approved")
    if status_val == "verified":
        status_val = "approved"

    update_data = {
        "doctor_id": user["sub"],
        "doctor_review": final_notes,
        "severity": body.severity,
        "follow_up_required": body.follow_up_required,
        "status": status_val,
        "reviewed_at": now,
        "updated_at": now,
        "flagged_for_retraining": body.flagged_for_retraining,
    }

    # Save translation of the review to content_en and content_ur
    content_en = report.get("content_en") or {}
    content_ur = report.get("content_ur") or {}
    
    content_en["doctor_review"] = final_notes
    
    try:
        from app.services.ai_service import _call_llm
        messages = [
            {"role": "system", "content": "Translate the following doctor's review notes from English to Urdu. Use simple language for the patient."},
            {"role": "user", "content": final_notes}
        ]
        final_notes_ur = await _call_llm(messages, temperature=0.3, max_tokens=1000)
    except Exception:
        final_notes_ur = final_notes
        
    content_ur["doctor_review"] = final_notes_ur
    
    update_data["content_en"] = content_en
    update_data["content_ur"] = content_ur

    try:
        supabase.table("reports").update(update_data).eq("id", report_id).execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    # Notify patient
    short_notes = (final_notes[:80] + '...') if len(final_notes) > 80 else final_notes
    status_display = "Approved" if status_val == "approved" else "Rejected" if status_val == "rejected" else "Needs Follow-up"
    await create_notification(
        user_id=report["user_id"],
        title=f"Doctor Review: {status_display}",
        message=f"Status: {status_display}. Doctor says: '{short_notes}'",
        notification_type="review",
        link=f"/reports/{report_id}",
    )

    return MessageResponse(message="Review submitted successfully.")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _map_doctor_report(row: Dict) -> DoctorReportResponse:
    """Map a database row to the DoctorReportResponse schema."""
    sections = row.get("sections", [])
    if isinstance(sections, list):
        sections = [{"title": s.get("title", ""), "content": s.get("content", "")} for s in sections]
    else:
        sections = []

    return DoctorReportResponse(
        id=row["id"],
        scan_id=row.get("scan_id", ""),
        patient_name=row.get("patient_name"),
        patient_age=row.get("patient_age"),
        patient_gender=row.get("patient_gender"),
        disease=row.get("disease", ""),
        confidence=row.get("confidence", 0.0),
        sections=sections,
        image_url=row.get("scans", {}).get("image_url") if isinstance(row.get("scans"), dict) else None,
        doctor_review=row.get("doctor_review"),
        severity=row.get("severity"),
        follow_up_required=row.get("follow_up_required", False),
        status=row.get("status", "pending"),
        created_at=row.get("created_at", ""),
        reviewed_at=row.get("reviewed_at"),
        flagged_for_retraining=row.get("flagged_for_retraining", False),
    )
