"""
POST /api/scans/upload (alias /api/predict) — Upload a fundus image and get disease predictions.
GET  /api/scans — Get user's scan history.
GET  /api/scans/{id} — Get specific scan result.
DELETE /api/scans/{id} — Delete scan.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, status

from app.core.security import get_current_user
from app.core.database import get_supabase_client
from app.models.schemas import PredictionResponse, PredictionResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/scans", tags=["Prediction"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# ── POST /api/scans/upload ───────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=PredictionResponse,
    summary="Predict eye disease from fundus image (Upload)",
    responses={400: {"description": "Invalid image"}, 413: {"description": "File too large"}},
)
async def predict_upload(
    file: UploadFile = File(..., description="Fundus / retinal image (JPEG, PNG, WebP)"),
    patient_age: Optional[int] = Form(None),
    patient_gender: Optional[str] = Form(None),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Accept a fundus image, run the OcuNetV4 model, save to database, and return prediction."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image type '{file.content_type}'. Allowed: JPEG, PNG, WebP, BMP, TIFF.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 10 MB limit.",
        )
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Run inference
    try:
        from app.services.ml_service import predict_image
        top_disease, top_confidence, all_preds, heatmap_bytes = predict_image(image_bytes)
    except ValueError as ve:
        if str(ve) == "INVALID_IMAGE":
            logger.warning("User uploaded an out-of-distribution image (e.g. a chair).")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This image is not valid. Please upload a real medical eye scan.",
            )
        raise ve
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {exc}",
        )

    scan_id = str(uuid4())
    user_id = user["sub"]
    image_url = None
    heatmap_url = None

    # Upload to Supabase Storage
    try:
        supabase = get_supabase_client()
        storage_path = f"scans/{user_id}/{scan_id}.jpg"
        supabase.storage.from_("eye-scans").upload(
            storage_path,
            image_bytes,
            file_options={"content-type": file.content_type or "image/jpeg"},
        )
        public_url = supabase.storage.from_("eye-scans").get_public_url(storage_path)
        image_url = public_url
        
        # Upload Heatmap if available
        if heatmap_bytes:
            heatmap_storage_path = f"scans/{user_id}/{scan_id}_heatmap.jpg"
            supabase.storage.from_("eye-scans").upload(
                heatmap_storage_path,
                heatmap_bytes,
                file_options={"content-type": "image/jpeg"},
            )
            heatmap_url = supabase.storage.from_("eye-scans").get_public_url(heatmap_storage_path)
            
    except Exception as exc:
        logger.warning("Image/Heatmap upload to storage failed (non-critical): %s", exc)

    # Extract patient name, age, gender from profile if not provided via form
    profile_name = user.get("email", "Patient")
    profile_age = patient_age
    profile_gender = patient_gender
    try:
        profile_res = (
            get_supabase_client()
            .table("profiles")
            .select("full_name, age, gender")
            .eq("id", user_id)
            .execute()
        )
        if profile_res.data:
            p_data = profile_res.data[0]
            if p_data.get("full_name"):
                profile_name = p_data["full_name"]
            if profile_age is None and p_data.get("age"):
                profile_age = p_data["age"]
            if profile_gender is None and p_data.get("gender"):
                profile_gender = p_data["gender"]
    except Exception as exc:
        logger.warning("Could not fetch profile info for scan: %s", exc)

    # Store scan record
    now = datetime.now(timezone.utc).isoformat()
    all_preds_dict = {d: c for d, c in all_preds}
    all_preds_dict["_patient_name"] = profile_name
    all_preds_dict["_patient_age"] = profile_age
    all_preds_dict["_patient_gender"] = profile_gender
    if heatmap_url:
        all_preds_dict["_heatmap_url"] = heatmap_url

    scan_db_record = {
        "id": scan_id,
        "user_id": user_id,
        "disease": top_disease,
        "confidence": top_confidence,
        "all_predictions": all_preds_dict,
        "image_url": image_url,
        "created_at": now,
    }

    try:
        supabase = get_supabase_client()
        supabase.table("scans").insert(scan_db_record).execute()
    except Exception as exc:
        logger.error("Failed to save scan record: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save scan record to database: {exc}",
        )

    return PredictionResponse(
        scan_id=scan_id,
        top_prediction=PredictionResult(disease=top_disease, confidence=top_confidence),
        all_predictions=[
            PredictionResult(disease=d, confidence=c) for d, c in all_preds
        ],
        image_url=image_url,
        heatmap_url=heatmap_url,
        created_at=now,
    )

# ── GET /api/scans ───────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=List[Dict[str, Any]],
    summary="Get user's scan history",
)
async def get_scans(
    page: int = 1,
    limit: int = 10,
    user: Dict[str, Any] = Depends(get_current_user),
):
    supabase = get_supabase_client()
    user_id = user["sub"]
    offset = (page - 1) * limit

    try:
        res = (
            supabase.table("scans")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        scans = res.data or []
        for s in scans:
            preds = s.get("all_predictions") or {}
            if isinstance(preds, dict):
                s["patient_name"] = preds.get("_patient_name")
                s["patient_age"] = preds.get("_patient_age")
                s["patient_gender"] = preds.get("_patient_gender")
                s["heatmap_url"] = preds.get("_heatmap_url")
        return scans
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {exc}",
        )

# ── GET /api/scans/{id} ───────────────────────────────────────────────────────

@router.get(
    "/{id}",
    response_model=Dict[str, Any],
    summary="Get specific scan result",
)
async def get_scan_by_id(
    id: str,
    user: Dict[str, Any] = Depends(get_current_user),
):
    supabase = get_supabase_client()
    user_id = user["sub"]

    try:
        res = supabase.table("scans").select("*").eq("id", id).execute()
        if not res.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Scan result not found",
            )
        scan = res.data[0]
        # Allow owner, doctors, or admins to view scan results
        if scan.get("user_id") != user_id and user.get("role") not in ("doctor", "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to view this scan",
            )
        preds = scan.get("all_predictions") or {}
        if isinstance(preds, dict):
            scan["patient_name"] = preds.get("_patient_name")
            scan["patient_age"] = preds.get("_patient_age")
            scan["patient_gender"] = preds.get("_patient_gender")
            scan["heatmap_url"] = preds.get("_heatmap_url")
        return scan
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {exc}",
        )

# ── DELETE /api/scans/{id} ───────────────────────────────────────────────────

@router.delete(
    "/{id}",
    summary="Delete scan",
)
async def delete_scan(
    id: str,
    user: Dict[str, Any] = Depends(get_current_user),
):
    supabase = get_supabase_client()
    user_id = user["sub"]

    try:
        # Check ownership
        res = supabase.table("scans").select("user_id").eq("id", id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Scan not found")
        
        if res.data[0].get("user_id") != user_id and user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Unauthorized")

        supabase.table("scans").delete().eq("id", id).execute()
        return {"status": "success", "message": "Scan deleted"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")
