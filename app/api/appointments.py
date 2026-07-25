from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime, timezone

from app.core.security import get_current_user, require_role
from app.core.database import get_supabase_client
from app.services.notification_service import create_notification

router = APIRouter(prefix="/api/appointments", tags=["Appointments"])

class AppointmentRequest(BaseModel):
    doctor_id: str
    appointment_date: str # ISO format
    notes: str = ""

class AppointmentStatusUpdate(BaseModel):
    status: str # "Approved", "Declined", "Completed"

@router.post("/", summary="Patient requests an appointment")
async def create_appointment(
    body: AppointmentRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    supabase = get_supabase_client()
    patient_id = user["sub"]
    
    # 1. Verify doctor exists
    try:
        doc_res = supabase.table("profiles").select("id, full_name").eq("id", body.doctor_id).eq("role", "doctor").execute()
        if not doc_res.data:
            raise HTTPException(status_code=404, detail="Doctor not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # 2. Get patient name
    try:
        pat_res = supabase.table("profiles").select("full_name").eq("id", patient_id).execute()
        patient_name = pat_res.data[0].get("full_name") or "A patient"
    except:
        patient_name = "A patient"

    # 3. Create appointment
    appt_data = {
        "patient_id": patient_id,
        "doctor_id": body.doctor_id,
        "appointment_date": body.appointment_date,
        "notes": body.notes,
        "status": "Pending"
    }

    try:
        res = supabase.table("appointments").insert(appt_data).execute()
        new_appt = res.data[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # 4. Notify doctor
    await create_notification(
        user_id=body.doctor_id,
        title="New Appointment Request",
        message=f"{patient_name} has requested an appointment on {body.appointment_date.split('T')[0]}.",
        notification_type="appointment",
        link="/doctor/appointments"
    )

    return {"message": "Appointment requested successfully", "appointment": new_appt}


@router.get("/patient", summary="Get patient's appointments")
async def get_patient_appointments(user: Dict[str, Any] = Depends(get_current_user)):
    supabase = get_supabase_client()
    try:
        # Since PostgREST joins work if foreign keys are set up. If not, we do it manually.
        res = supabase.table("appointments").select("*, doctor:profiles!doctor_id(full_name, email)").eq("patient_id", user["sub"]).order("appointment_date", desc=True).execute()
        return res.data or []
    except Exception as exc:
        # Fallback if the join fails due to no foreign key in PostgREST
        res = supabase.table("appointments").select("*").eq("patient_id", user["sub"]).order("appointment_date", desc=True).execute()
        return res.data or []


@router.get("/doctor", summary="Get doctor's appointments")
async def get_doctor_appointments(user: Dict[str, Any] = Depends(require_role("doctor", "admin"))):
    supabase = get_supabase_client()
    try:
        res = supabase.table("appointments").select("*, patient:profiles!patient_id(full_name, email, age, gender)").eq("doctor_id", user["sub"]).order("appointment_date", desc=True).execute()
        return res.data or []
    except Exception as exc:
        res = supabase.table("appointments").select("*").eq("doctor_id", user["sub"]).order("appointment_date", desc=True).execute()
        return res.data or []


@router.patch("/{id}/status", summary="Doctor updates appointment status")
async def update_appointment_status(
    id: str,
    body: AppointmentStatusUpdate,
    user: Dict[str, Any] = Depends(require_role("doctor", "admin"))
):
    supabase = get_supabase_client()
    try:
        res = supabase.table("appointments").update({"status": body.status}).eq("id", id).eq("doctor_id", user["sub"]).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Appointment not found or not assigned to you")
            
        appt = res.data[0]
        
        # Notify patient
        await create_notification(
            user_id=appt["patient_id"],
            title=f"Appointment {body.status}",
            message=f"Your appointment has been {body.status.lower()}.",
            notification_type="appointment",
            link="/appointments"
        )
        return {"message": f"Appointment {body.status}", "appointment": appt}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
