"""
Report endpoints:
  POST /api/reports/generate       — generate an AI medical report
  GET  /api/reports                — list current user's reports
  GET  /api/reports/{id}           — get a single report
  GET  /api/reports/{id}/pdf/{lang}— download report as PDF
  POST /api/reports/{id}/send-to-doctor — assign report to a doctor
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import Response

from app.core.security import get_current_user
from app.core.database import get_supabase_client
from app.models.schemas import (
    ReportGenerateRequest,
    ReportData,
    ReportListItem,
    ReportSection,
    SendToDoctorRequest,
    MessageResponse,
)
from app.services.ai_service import generate_report, translate_report_to_urdu
from app.services.pdf_service import generate_pdf
from app.services.notification_service import create_notification

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/reports", tags=["Reports"])


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/reports/generate
# ─────────────────────────────────────────────────────────────────────────────

def map_sections_to_content(
    sections: List[Any], disease: str, confidence: float
) -> Dict[str, Any]:
    content = {
        "disease_name": disease,
        "confidence_score": f"{round(confidence * 100)}%",
        "severity_level": "Moderate",
        "explanation": "",
        "symptoms": [],
        "causes": [],
        "risk_factors": [],
        "preventive_measures": [],
        "recommended_tests": [],
        "suggested_treatment": [],
        "emergency_warning": "",
        "disclaimer": "",
    }

    import re

    def parse_list(text: str) -> List[str]:
        if not text:
            return []
        items = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r"^[•\-\*\d+\.\u2022]\s*", "", line).strip()
            if cleaned:
                items.append(cleaned)
        return items

    for sec in sections:
        if isinstance(sec, dict):
            title = sec.get("title", "")
            body = sec.get("content", "")
        else:
            title = getattr(sec, "title", "")
            body = getattr(sec, "content", "")
        
        if not title:
            continue
        title_lower = title.strip().lower()
        body_str = body.strip() if body else ""

        if "explanation" in title_lower or "وضاحت" in title_lower:
            content["explanation"] = body_str
        elif "symptom" in title_lower or "علامات" in title_lower:
            content["symptoms"] = parse_list(body_str)
        elif "cause" in title_lower or "وجوہات" in title_lower:
            content["causes"] = parse_list(body_str)
        elif "risk" in title_lower or "عوامل" in title_lower:
            content["risk_factors"] = parse_list(body_str)
        elif (
            "prevention" in title_lower
            or "روک تھام" in title_lower
            or "lifestyle" in title_lower
        ):
            content["preventive_measures"] = parse_list(body_str)
        elif "test" in title_lower or "ٹیسٹ" in title_lower:
            content["recommended_tests"] = parse_list(body_str)
        elif "treatment" in title_lower or "علاج" in title_lower or "option" in title_lower:
            content["suggested_treatment"] = parse_list(body_str)
        elif "warning" in title_lower or "انتباہ" in title_lower:
            content["emergency_warning"] = body_str
        elif "disclaimer" in title_lower or "دستبرداری" in title_lower:
            content["disclaimer"] = body_str

    return content


def _content_to_sections(content: Dict[str, Any], disease: str, lang: str = "en") -> List[Dict[str, str]]:
    """Convert a stored content_en / content_ur dict back into a sections list for PDF."""
    if not content:
        return []

    if lang == "ur":
        mapping = [
            ("بیماری کی تفصیل",   "explanation"),
            ("علامات",            "symptoms"),
            ("وجوہات",            "causes"),
            ("خطرہ کے عوامل",    "risk_factors"),
            ("احتیاطی تدابیر",   "preventive_measures"),
            ("تجویز کردہ ٹیسٹ",  "recommended_tests"),
            ("علاج کے اختیارات", "suggested_treatment"),
            ("ہنگامی انتباہ",    "emergency_warning"),
            ("دستبرداری",        "disclaimer"),
        ]
    else:
        mapping = [
            ("Disease Explanation",  "explanation"),
            ("Symptoms",             "symptoms"),
            ("Causes",               "causes"),
            ("Risk Factors",         "risk_factors"),
            ("Preventive Measures",  "preventive_measures"),
            ("Recommended Tests",    "recommended_tests"),
            ("Treatment Options",    "suggested_treatment"),
            ("Emergency Warning",    "emergency_warning"),
            ("Disclaimer",           "disclaimer"),
        ]

    sections = []
    for title, key in mapping:
        value = content.get(key)
        if not value:
            continue
        if isinstance(value, list):
            body = "\n".join(f"• {item}" for item in value if item)
        else:
            body = str(value)
        if body.strip():
            sections.append({"title": title, "content": body})
    return sections


@router.post(
    "/generate",
    response_model=ReportData,
    summary="Generate AI medical report from a scan",
)
async def generate_report_endpoint(
    body: ReportGenerateRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Generate a comprehensive AI report for a completed scan."""
    supabase = get_supabase_client()
    user_id = user["sub"]

    # ── Fetch the scan ───────────────────────────────────────────────────
    try:
        scan_result = (
            supabase.table("scans")
            .select("*")
            .eq("id", body.scan_id)
            .eq("user_id", user_id)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not scan_result.data:
        raise HTTPException(status_code=404, detail="Scan not found or access denied.")

    scan = scan_result.data[0]
    disease = scan["disease"]
    confidence = scan["confidence"]

    # Auto-extract name, age, gender from scan or profile
    preds = scan.get("all_predictions") if isinstance(scan.get("all_predictions"), dict) else {}
    p_name = body.patient_name or scan.get("patient_name") or preds.get("_patient_name")
    p_age = body.patient_age if body.patient_age is not None else (scan.get("patient_age") if scan.get("patient_age") is not None else preds.get("_patient_age"))
    p_gender = body.patient_gender or scan.get("patient_gender") or preds.get("_patient_gender")

    if not p_name or p_name == "Patient" or p_age is None or not p_gender:
        try:
            profile_res = (
                supabase.table("profiles")
                .select("full_name, age, gender")
                .eq("id", user_id)
                .execute()
            )
            if profile_res.data:
                p_data = profile_res.data[0]
                if not p_name or p_name == "Patient":
                    p_name = p_data.get("full_name") or user.get("email", "Patient")
                if p_age is None:
                    p_age = p_data.get("age")
                if not p_gender:
                    p_gender = p_data.get("gender")
        except Exception as exc:
            logger.warning("Could not fetch profile info for report: %s", exc)

    if not p_name:
        p_name = "Patient"

    # ── Generate AI report in English ─────────────────────────────────────
    try:
        sections_en = await generate_report(
            disease=disease,
            confidence=confidence,
            patient_name=p_name,
            patient_age=p_age,
            patient_gender=p_gender,
            language="en",
        )
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"AI report generation failed: {exc}",
        )

    # ── Translate to Urdu ────────────────────────────────────────────────
    try:
        sections_ur = await translate_report_to_urdu(sections_en)
    except Exception as exc:
        logger.warning("Urdu translation failed: %s", exc)
        sections_ur = []

    # Map content formats
    content_en = map_sections_to_content(sections_en, disease, confidence)
    content_ur = map_sections_to_content(sections_ur, disease, confidence) if sections_ur else None

    # Determine requested sections language
    sections = sections_ur if body.language == "ur" else sections_en

    # ── Persist report ───────────────────────────────────────────────────
    report_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()

    report_record = {
        "id": report_id,
        "scan_id": body.scan_id,
        "user_id": user_id,
        "patient_name": p_name,
        "patient_age": p_age,
        "patient_gender": p_gender,
        "disease": disease,
        "confidence": confidence,
        "sections": sections,
        "language": body.language,
        "status": "pending",
        "content_en": content_en,
        "content_ur": content_ur,
        "created_at": now,
        "updated_at": now,
    }

    try:
        supabase.table("reports").insert(report_record).execute()
    except Exception as exc:
        logger.error("Failed to save report record: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save report record to database: {exc}",
        )

    return ReportData(
        id=report_id,
        scan_id=body.scan_id,
        user_id=user_id,
        patient_name=body.patient_name,
        patient_age=body.patient_age,
        patient_gender=body.patient_gender,
        disease=disease,
        confidence=confidence,
        sections=[ReportSection(**s) for s in sections],
        language=body.language,
        status="pending",
        content_en=content_en,
        content_ur=content_ur,
        created_at=now,
        updated_at=now,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/reports
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=List[ReportListItem],
    summary="List current user's reports",
)
async def list_reports(user: Dict[str, Any] = Depends(get_current_user)):
    supabase = get_supabase_client()
    try:
        result = (
            supabase.table("reports")
            .select("id, disease, confidence, patient_name, status, language, created_at, doctor_review")
            .eq("user_id", user["sub"])
            .order("created_at", desc=True)
            .execute()
        )
        return result.data or []
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/reports/{id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{report_id}",
    response_model=ReportData,
    summary="Get a single report",
)
async def get_report(
    report_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
):
    supabase = get_supabase_client()
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

    # Access control: owner, assigned doctor, or admin
    user_role = user.get("role", "user")
    if (
        report["user_id"] != user["sub"]
        and report.get("doctor_id") != user["sub"]
        and user_role != "admin"
    ):
        raise HTTPException(status_code=403, detail="Access denied.")

    # Fallback mapping for older database records missing content_en/content_ur
    if not report.get("content_en") and not report.get("content_ur"):
        lang = report.get("language", "en")
        sections = report.get("sections", [])
        mapped_content = map_sections_to_content(
            sections, report.get("disease", "Unknown"), report.get("confidence", 0.0)
        )
        if lang == "ur":
            report["content_ur"] = mapped_content
            report["content_en"] = None
        else:
            report["content_en"] = mapped_content
            report["content_ur"] = None

    return report


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/reports/{id}/pdf/{lang}
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{report_id}/download",
    summary="Download report as PDF (default English)",
    responses={200: {"content": {"application/pdf": {}}}},
)
@router.get(
    "/{report_id}/pdf/{lang}",
    summary="Download report as PDF",
    responses={200: {"content": {"application/pdf": {}}}},
)
async def download_pdf(
    report_id: str,
    lang: str = "en",
    user: Dict[str, Any] = Depends(get_current_user),
):
    lang = (lang or "en").strip().lower()
    if lang not in ("en", "ur"):
        lang = "en"  # Graceful fallback instead of error

    # Fetch report
    supabase = get_supabase_client()
    try:
        result = supabase.table("reports").select("*").eq("id", report_id).execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not result.data:
        raise HTTPException(status_code=404, detail="Report not found.")

    report = result.data[0]

    # Access control
    user_role = user.get("role", "user")
    if (
        report["user_id"] != user["sub"]
        and report.get("doctor_id") != user["sub"]
        and user_role != "admin"
    ):
        raise HTTPException(status_code=403, detail="Access denied.")

    # Get English sections
    sections_en = report.get("sections", [])
    if report.get("language", "en") != "en":
        # stored sections are Urdu — fetch English from content_en
        content_en = report.get("content_en") or {}
        sections_en = _content_to_sections(content_en, report.get("disease", ""))

    # Get Urdu sections — try content_ur first, else translate
    sections_ur = []
    content_ur = report.get("content_ur")
    if content_ur:
        sections_ur = _content_to_sections(content_ur, report.get("disease", ""), lang="ur")
    else:
        try:
            sections_ur = await translate_report_to_urdu(sections_en)
        except Exception as exc:
            logger.warning("Urdu translation failed for PDF: %s", exc)
            sections_ur = []

    # Fallback to profile if report lacks name/age/gender
    p_name = report.get("patient_name")
    p_age = report.get("patient_age")
    p_gender = report.get("patient_gender")
    if not p_name or p_name == "Patient" or p_age is None or not p_gender:
        try:
            profile_res = (
                supabase.table("profiles")
                .select("full_name, age, gender")
                .eq("id", report["user_id"])
                .execute()
            )
            if profile_res.data:
                p_data = profile_res.data[0]
                if not p_name or p_name == "Patient":
                    p_name = p_data.get("full_name") or "Patient"
                if p_age is None:
                    p_age = p_data.get("age")
                if not p_gender:
                    p_gender = p_data.get("gender")
        except Exception as exc:
            logger.warning("Could not fetch profile info for download_pdf: %s", exc)

    # Generate bilingual PDF
    try:
        pdf_bytes = generate_pdf(
            disease=report["disease"],
            confidence=report["confidence"],
            sections_en=sections_en,
            sections_ur=sections_ur,
            patient_name=p_name or "Patient",
            patient_age=p_age,
            patient_gender=p_gender,
            doctor_review=report.get("doctor_review"),
        )
    except Exception as exc:
        logger.error("PDF generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")

    filename = f"eye_cared_report_{report_id[:8]}_bilingual.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/reports/{id}/send-to-doctor
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/{report_id}/send",
    response_model=MessageResponse,
    summary="Assign report to a doctor for review (alias)",
)
@router.post(
    "/{report_id}/send-to-doctor",
    response_model=MessageResponse,
    summary="Assign report to a doctor for review",
)
async def send_to_doctor(
    report_id: str,
    body: SendToDoctorRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    supabase = get_supabase_client()

    # Verify report belongs to user
    try:
        result = (
            supabase.table("reports")
            .select("*")
            .eq("id", report_id)
            .eq("user_id", user["sub"])
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if not result.data:
        raise HTTPException(status_code=404, detail="Report not found or access denied.")

    # Verify target is a doctor or general portal
    target_doc_id = body.doctor_id
    target_doc_name = "Medical Portal"
    if not target_doc_id or target_doc_id == "general":
        try:
            doc_res = (
                supabase.table("profiles")
                .select("id, role, full_name")
                .eq("role", "doctor")
                .limit(1)
                .execute()
            )
            if doc_res.data:
                target_doc_id = doc_res.data[0]["id"]
                target_doc_name = doc_res.data[0].get("full_name") or "Available Physician"
            else:
                target_doc_id = None
                target_doc_name = "General Medical Pool"
        except Exception as exc:
            target_doc_id = None
            target_doc_name = "General Medical Pool"
    else:
        try:
            doctor_result = (
                supabase.table("profiles")
                .select("id, role, full_name")
                .eq("id", target_doc_id)
                .execute()
            )
            if doctor_result.data and doctor_result.data[0].get("role") in ("doctor", "admin"):
                target_doc_name = doctor_result.data[0].get("full_name") or "Physician"
            else:
                # Fallback if specific doctor ID invalid
                target_doc_id = None
                target_doc_name = "General Medical Pool"
        except Exception:
            target_doc_id = None
            target_doc_name = "General Medical Pool"

    # Update report status to pending/in_review
    try:
        update_payload = {
            "status": "pending",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if target_doc_id:
            update_payload["doctor_id"] = target_doc_id
        supabase.table("reports").update(update_payload).eq("id", report_id).execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    # Notify doctor if assigned
    if target_doc_id:
        report = result.data[0]
        await create_notification(
            user_id=target_doc_id,
            title="New Report Assigned",
            message=f"A patient has sent you a report for review. Disease: {report.get('disease', 'Eye Condition')}",
            notification_type="report",
            link=f"/doctor/review/{report_id}",
        )

    return MessageResponse(message=f"Report sent to {target_doc_name} for review!")
