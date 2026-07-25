import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status, File, UploadFile, Depends
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.security import get_current_user
from app.api import auth, predict, reports, chat, doctor, admin, notifications, appointments, tts

# ── Configure Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lifespan Management ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start up and shutdown events for the application."""
    logger.info("Initializing Eye Cared Bot Backend...")
    # Pre-import/load model logic here if we want to warm it up
    try:
        from app.services.ml_service import load_model
        load_model()
    except Exception as exc:
        logger.warning("Could not warm up model during startup (non-critical): %s", exc)
    yield
    logger.info("Shutting down Eye Cared Bot Backend...")

# ── FastAPI App ─────────────────────────────────────────────────────────────
settings = get_settings()
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# ── CORS Middleware ─────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register API Routers ────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(reports.router)
app.include_router(chat.router)
app.include_router(doctor.router)
app.include_router(admin.router)
app.include_router(notifications.router)
app.include_router(appointments.router)
app.include_router(tts.router)

# ── Predict Alias ───────────────────────────────────────────────────────────
@app.post("/api/predict", tags=["Prediction"])
async def predict_alias(
    file: UploadFile = File(..., description="Fundus / retinal image"),
    user: Dict[str, Any] = Depends(get_current_user),
):
    from app.api.predict import predict_upload
    return await predict_upload(file, user)

# ── Root / Health Check ─────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    """Verify application health and database connection."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }

@app.get("/", tags=["System"])
async def root():
    """Simple root welcome message."""
    return {
        "message": f"Welcome to the {settings.APP_NAME} API v{settings.APP_VERSION}.",
        "docs_url": "/docs",
    }

# ── Error Handlers ──────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."},
    )
