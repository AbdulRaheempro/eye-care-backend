"""
Auth helper endpoints:
  POST /api/auth/register — register a new user in Supabase and create a profile
  POST /api/auth/login    — log in a user and return a JWT access token
  GET  /api/auth/me       — get current user profile
  POST /api/auth/profile  — create / update profile after Supabase auth
  GET  /api/auth/doctors  — list available doctors (for send-to-doctor UI)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr

from app.core.security import get_current_user, create_access_token
from app.core.database import get_supabase_client, get_supabase_admin, get_supabase_auth_client
from app.models.schemas import UserProfile, UserProfileUpdate, MessageResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["Auth"])

# ── Request/Response Schemas ──────────────────────────────────────────────────

class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str

class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    phone: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    role: Optional[str] = "patient"

class AuthResponse(BaseModel):
    user: UserProfile
    access_token: str
    token_type: str = "bearer"

# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/auth/register
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=AuthResponse,
    summary="Register a new user",
)
async def register_user(body: UserRegisterRequest):
    auth_client = get_supabase_auth_client()
    admin_client = get_supabase_admin()

    try:
        # 1. Sign up the user in Supabase Auth with metadata options
        auth_res = auth_client.auth.sign_up({
            "email": body.email,
            "password": body.password,
            "options": {
                "data": {
                    "display_name": body.full_name,
                    "full_name": body.full_name,
                    "phone": body.phone,
                }
            }
        })
    except Exception as exc:
        logger.error("Supabase Auth sign up failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {exc}",
        )

    if not auth_res.user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration failed: User not created.",
        )

    user_id = auth_res.user.id
    now = datetime.now(timezone.utc).isoformat()
    assigned_role = body.role if body.role in ("patient", "doctor", "admin") else "patient"

    # 2. Insert profile row in the database using the admin client (bypasses RLS locks on registration)
    profile_data = {
        "id": user_id,
        "email": body.email,
        "full_name": body.full_name,
        "phone": body.phone,
        "age": body.age,
        "gender": body.gender,
        "role": assigned_role,
        "created_at": now,
        "updated_at": now,
    }

    try:
        admin_client.table("profiles").insert(profile_data).execute()
    except Exception as exc:
        logger.error("Failed to create profile row: %s", exc)
        # We don't fail registration if profile insert fails (the user can update it later),
        # but we log it.

    # 3. Create access token
    token_data = {"sub": user_id, "email": body.email, "role": assigned_role}
    token = create_access_token(token_data)

    user_profile = UserProfile(
        id=user_id,
        email=body.email,
        full_name=body.full_name,
        age=body.age,
        gender=body.gender,
        phone=body.phone,
        role=assigned_role,
        created_at=now,
    )

    return AuthResponse(user=user_profile, access_token=token)

# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/auth/login
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Log in a user",
)
async def login_user(body: UserLoginRequest):
    auth_client = get_supabase_auth_client()
    supabase = get_supabase_client()

    try:
        # 1. Sign in with password using Supabase Auth
        auth_res = auth_client.auth.sign_in_with_password({
            "email": body.email,
            "password": body.password,
        })
    except Exception as exc:
        logger.error("Supabase Auth sign in failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not auth_res.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
        )

    user_id = auth_res.user.id

    # 2. Fetch the user's profile
    try:
        profile_res = supabase.table("profiles").select("*").eq("id", user_id).execute()
        if profile_res.data:
            profile = profile_res.data[0]
            role = profile.get("role", "patient")
            full_name = profile.get("full_name")
            age = profile.get("age")
            gender = profile.get("gender")
            phone = profile.get("phone")
            created_at = profile.get("created_at")
        else:
            role = "patient"
            full_name = None
            age = None
            gender = None
            phone = None
            created_at = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        logger.warning("Failed to fetch profile: %s", exc)
        role = "patient"
        full_name = None
        age = None
        gender = None
        phone = None
        created_at = datetime.now(timezone.utc).isoformat()

    # 3. Create access token
    token_data = {"sub": user_id, "email": body.email, "role": role}
    token = create_access_token(token_data)

    user_profile = UserProfile(
        id=user_id,
        email=body.email,
        full_name=full_name,
        age=age,
        gender=gender,
        phone=phone,
        role=role,
        created_at=created_at,
    )

    return AuthResponse(user=user_profile, access_token=token)

# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/auth/me
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/me",
    response_model=UserProfile,
    summary="Get the current authenticated user's profile",
)
async def get_me(user: Dict[str, Any] = Depends(get_current_user)):
    supabase = get_supabase_client()
    user_id = user["sub"]

    try:
        result = supabase.table("profiles").select("*").eq("id", user_id).execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if result.data:
        profile = result.data[0]
        return UserProfile(
            id=profile["id"],
            email=profile.get("email", user.get("email", "")),
            full_name=profile.get("full_name"),
            age=profile.get("age"),
            gender=profile.get("gender"),
            phone=profile.get("phone"),
            role=profile.get("role", "patient"),
            created_at=profile.get("created_at"),
        )

    return UserProfile(
        id=user_id,
        email=user.get("email", ""),
        full_name=None,
        role=user.get("role", "patient"),
    )

# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/auth/profile
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/profile",
    response_model=MessageResponse,
    summary="Create or update user profile",
)
async def upsert_profile(
    body: UserProfileUpdate,
    user: Dict[str, Any] = Depends(get_current_user),
):
    supabase = get_supabase_client()
    user_id = user["sub"]

    profile_data: Dict[str, Any] = {
        "id": user_id,
        "email": user.get("email", ""),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if body.full_name is not None:
        profile_data["full_name"] = body.full_name
    if body.avatar_url is not None:
        profile_data["avatar_url"] = body.avatar_url

    try:
        existing = supabase.table("profiles").select("id").eq("id", user_id).execute()
        if existing.data:
            supabase.table("profiles").update(profile_data).eq("id", user_id).execute()
        else:
            profile_data["role"] = "patient"
            profile_data["created_at"] = datetime.now(timezone.utc).isoformat()
            supabase.table("profiles").insert(profile_data).execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    return MessageResponse(message="Profile saved successfully.")

# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/auth/doctors
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/doctors",
    response_model=List[UserProfile],
    summary="List available doctors",
)
async def list_doctors(user: Dict[str, Any] = Depends(get_current_user)):
    supabase = get_supabase_client()

    try:
        result = (
            supabase.table("profiles")
            .select("id, email, full_name, role, created_at")
            .eq("role", "doctor")
            .order("full_name")
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    return [
        UserProfile(
            id=d["id"],
            email=d.get("email", ""),
            full_name=d.get("full_name"),
            role="doctor",
            created_at=d.get("created_at"),
        )
        for d in (result.data or [])
    ]
