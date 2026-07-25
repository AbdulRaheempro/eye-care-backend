"""
All Pydantic models / schemas used across the application.
Organised by domain: Auth, Prediction, Report, Chat, Doctor, Admin, Notification.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, EmailStr


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERIC
# ═══════════════════════════════════════════════════════════════════════════════

class MessageResponse(BaseModel):
    """Generic JSON message."""
    message: str


class ErrorResponse(BaseModel):
    """Standard error payload."""
    detail: str


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ═══════════════════════════════════════════════════════════════════════════════

class TokenData(BaseModel):
    sub: str
    email: str
    role: str = "user"


class UserProfile(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    role: str = "user"
    created_at: Optional[str] = None


class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

class PredictionResult(BaseModel):
    """Single-class prediction."""
    disease: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Full prediction response returned by POST /api/predict."""
    scan_id: str
    top_prediction: PredictionResult
    all_predictions: List[PredictionResult]
    image_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    created_at: str


# ═══════════════════════════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════════════════════════

class ReportGenerateRequest(BaseModel):
    """Body for POST /api/reports/generate."""
    scan_id: str
    patient_name: Optional[str] = "Patient"
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    language: str = Field(default="en", pattern="^(en|ur)$")


class ReportSection(BaseModel):
    title: str
    content: str


class ReportData(BaseModel):
    id: str
    scan_id: str
    user_id: str
    patient_name: Optional[str] = None
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    disease: str
    confidence: float
    sections: List[ReportSection]
    language: str = "en"
    doctor_id: Optional[str] = None
    doctor_review: Optional[str] = None
    status: str = "pending"  # pending | approved | rejected | follow_up
    content_en: Optional[Dict[str, Any]] = None
    content_ur: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: Optional[str] = None


class ReportListItem(BaseModel):
    id: str
    disease: str
    confidence: float
    patient_name: Optional[str] = None
    status: str
    language: str
    created_at: str
    doctor_review: Optional[str] = None


class SendToDoctorRequest(BaseModel):
    doctor_id: Optional[str] = "general"


# ═══════════════════════════════════════════════════════════════════════════════
#  CHAT
# ═══════════════════════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: List[ChatMessage] = Field(default_factory=list)
    language: Optional[str] = None  # auto-detect if None


class ChatResponse(BaseModel):
    reply: str
    detected_language: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  DOCTOR
# ═══════════════════════════════════════════════════════════════════════════════

class DoctorReviewRequest(BaseModel):
    review_notes: Optional[str] = None
    notes: Optional[str] = None
    recommendations: Optional[str] = None
    status: Optional[str] = "approved"
    severity: Optional[str] = None
    follow_up_required: bool = False
    flagged_for_retraining: bool = False


class DoctorReportResponse(BaseModel):
    id: str
    scan_id: str
    patient_name: Optional[str] = None
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    disease: str
    confidence: float
    sections: List[ReportSection]
    image_url: Optional[str] = None
    doctor_review: Optional[str] = None
    severity: Optional[str] = None
    follow_up_required: bool = False
    status: str
    created_at: str
    reviewed_at: Optional[str] = None
    flagged_for_retraining: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
#  ADMIN
# ═══════════════════════════════════════════════════════════════════════════════

class PlatformStats(BaseModel):
    total_users: int = 0
    total_scans: int = 0
    total_reports: int = 0
    total_doctors: int = 0
    disease_distribution: Dict[str, int] = Field(default_factory=dict)
    scans_last_7_days: int = 0


class AdminUserItem(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    role: str
    created_at: Optional[str] = None
    scan_count: int = 0


class RoleUpdateRequest(BaseModel):
    role: str = Field(pattern="^(user|doctor|admin)$")


# ═══════════════════════════════════════════════════════════════════════════════
#  NOTIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class NotificationItem(BaseModel):
    id: str
    user_id: str
    title: str
    message: str
    type: str = "info"  # info | report | review | system
    is_read: bool = False
    link: Optional[str] = None
    created_at: str


class NotificationListResponse(BaseModel):
    notifications: List[NotificationItem]
    unread_count: int
