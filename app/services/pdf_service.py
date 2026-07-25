"""
PDF Service — generate bilingual (English + Urdu) medical report PDFs using ReportLab.
English section first, then a divider, then the Urdu section below.
"""

from __future__ import annotations

import io
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak,
    KeepTogether,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)

# ── Colours ───────────────────────────────────────────────────────────────────
PRIMARY   = colors.HexColor("#2563EB")
DARK      = colors.HexColor("#1E293B")
GRAY      = colors.HexColor("#64748B")
LIGHT_BG  = colors.HexColor("#F1F5F9")
ACCENT    = colors.HexColor("#DBEAFE")
DIVIDER   = colors.HexColor("#10B981")   # green divider between EN / UR sections
DIVIDER_BG= colors.HexColor("#D1FAE5")

# ── Font registration ─────────────────────────────────────────────────────────
_FONTS_DIR        = os.path.join(os.path.dirname(__file__), "..", "fonts")
_URDU_FONT        = "NotoNaskhArabic"
_URDU_FONT_PATH   = os.path.join(_FONTS_DIR, "NotoNaskhArabic-Regular.ttf")
_URDU_FONT_READY  = False


def _ensure_urdu_font() -> bool:
    global _URDU_FONT_READY
    if _URDU_FONT_READY:
        return True
    try:
        pdfmetrics.registerFont(TTFont(_URDU_FONT, _URDU_FONT_PATH))
        _URDU_FONT_READY = True
        logger.info("NotoNastaliqUrdu font registered.")
        return True
    except Exception as exc:
        logger.warning("Could not register Urdu font: %s", exc)
        return False


def _reshape(text: str) -> str:
    """Reshape + bidi-reorder Urdu text for correct RTL PDF rendering."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(str(text)))
    except ImportError:
        return str(text)


# ── Page callback ─────────────────────────────────────────────────────────────

def _header_footer(canvas, doc):
    canvas.saveState()

    # Blue header bar
    canvas.setFillColor(PRIMARY)
    canvas.rect(0, A4[1] - 50, A4[0], 50, fill=1, stroke=0)

    canvas.setFont("Helvetica-Bold", 15)
    canvas.setFillColor(colors.white)
    canvas.drawCentredString(A4[0] / 2, A4[1] - 32, "Eye Cared Bot — AI Clinical Report")

    # Footer line + text
    canvas.setStrokeColor(LIGHT_BG)
    canvas.setLineWidth(1)
    canvas.line(40, 34, A4[0] - 40, 34)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRAY)
    canvas.drawCentredString(
        A4[0] / 2, 20,
        f"Eye Cared Bot  |  AI-Powered Eye Disease Detection  |  Page {doc.page}  |  Confidential"
    )

    canvas.restoreState()


# ── Style helpers ─────────────────────────────────────────────────────────────

def _en_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "sub":      ParagraphStyle("EnSub",     fontName="Helvetica",      fontSize=9,  textColor=GRAY,    alignment=TA_CENTER, spaceAfter=10, spaceBefore=8),
        "heading":  ParagraphStyle("EnHeading", fontName="Helvetica-Bold", fontSize=13, textColor=PRIMARY, spaceBefore=12, spaceAfter=5, alignment=TA_LEFT),
        "body":     ParagraphStyle("EnBody",    fontName="Helvetica",      fontSize=10, leading=15, textColor=DARK, alignment=TA_LEFT, spaceAfter=3),
        "label":    ParagraphStyle("EnLabel",   fontName="Helvetica-Bold", fontSize=10, textColor=GRAY,  alignment=TA_LEFT),
        "value":    ParagraphStyle("EnValue",   fontName="Helvetica",      fontSize=10, textColor=DARK,  alignment=TA_LEFT),
        "disc":     ParagraphStyle("EnDisc",    fontName="Helvetica",      fontSize=9,  textColor=colors.HexColor("#DC2626"), leading=13, alignment=TA_LEFT, spaceBefore=6),
        "divider":  ParagraphStyle("Divider",   fontName="Helvetica-Bold", fontSize=13, textColor=DIVIDER, alignment=TA_CENTER, spaceBefore=20, spaceAfter=10),
    }


def _ur_styles(urdu_ok: bool) -> Dict[str, ParagraphStyle]:
    base  = getSampleStyleSheet()
    font  = _URDU_FONT if urdu_ok else "Helvetica"
    bfont = _URDU_FONT if urdu_ok else "Helvetica-Bold"
    return {
        "heading":  ParagraphStyle("UrHeading", fontName=bfont,         fontSize=13, textColor=PRIMARY, spaceBefore=12, spaceAfter=5,  alignment=TA_RIGHT),
        "body":     ParagraphStyle("UrBody",    fontName=font,          fontSize=13, leading=24, textColor=DARK, alignment=TA_RIGHT, spaceAfter=4),
        "label":    ParagraphStyle("UrLabel",   fontName=bfont,         fontSize=11, textColor=GRAY,  alignment=TA_RIGHT),
        # Values are Latin (names, disease) — use Helvetica so they render correctly
        "value":    ParagraphStyle("UrValue",   fontName="Helvetica",   fontSize=11, textColor=DARK,  alignment=TA_LEFT),
        "disc":     ParagraphStyle("UrDisc",    fontName=font,          fontSize=11, textColor=colors.HexColor("#DC2626"), leading=20, alignment=TA_RIGHT, spaceBefore=6),
        "divider":  ParagraphStyle("UrDivider", fontName=font,          fontSize=13, textColor=DIVIDER, alignment=TA_RIGHT, spaceBefore=0, spaceAfter=0),
        "divider_en": ParagraphStyle("UrDivEn", fontName="Helvetica-Bold", fontSize=13, textColor=DIVIDER, alignment=TA_LEFT,  spaceBefore=0, spaceAfter=0),
    }


# ── Info table builder ────────────────────────────────────────────────────────

def _info_table(rows, col_widths=(170, 330)):
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), ACCENT),
        ("BACKGROUND",    (1, 0), (1, -1), colors.white),
        ("TEXTCOLOR",     (0, 0), (0, -1), PRIMARY),
        ("TEXTCOLOR",     (1, 0), (1, -1), DARK),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING",    (0, 0), (-1, -1), 9),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# ── Public API ────────────────────────────────────────────────────────────────

def generate_pdf(
    disease: str,
    confidence: float,
    sections_en: List[Dict[str, str]],
    sections_ur: List[Dict[str, str]],
    patient_name: str = "Patient",
    patient_age: Optional[int] = None,
    patient_gender: Optional[str] = None,
    doctor_review: Optional[str] = None,
    # kept for backwards-compat; ignored — always bilingual
    sections: Optional[List] = None,
    language: str = "en",
) -> bytes:
    """
    Build a bilingual (English on top, Urdu below) medical report PDF.
    Returns raw PDF bytes.
    """
    urdu_ok = _ensure_urdu_font()
    ens     = _en_styles()
    urs     = _ur_styles(urdu_ok)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=68,
        bottomMargin=50,
        leftMargin=42,
        rightMargin=42,
        title=f"Eye Cared Bot - Medical Report - {patient_name}",
        author="Eye Cared Bot",
    )

    story: list = []
    date_str = datetime.now(timezone.utc).strftime("%B %d, %Y  •  %H:%M UTC")

    # ════════════════════════════════════════════════════════════════════
    #  ENGLISH SECTION
    # ════════════════════════════════════════════════════════════════════
    story.append(Paragraph(f"Generated on {date_str}", ens["sub"]))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY, spaceAfter=12))

    # English patient info table
    en_rows = [
        [Paragraph("Patient Name",       ens["label"]), Paragraph(patient_name or "—",                      ens["value"])],
        [Paragraph("Age",                ens["label"]), Paragraph(str(patient_age) if patient_age else "—", ens["value"])],
        [Paragraph("Gender",             ens["label"]), Paragraph(patient_gender or "—",                    ens["value"])],
        [Paragraph("Detected Condition", ens["label"]), Paragraph(disease,                                  ens["value"])],
        [Paragraph("AI Confidence",      ens["label"]), Paragraph(f"{confidence * 100:.1f}%",               ens["value"])],
    ]
    story.append(_info_table(en_rows))
    story.append(Spacer(1, 16))

    # English sections
    for sec in sections_en:
        title   = sec.get("title", "")
        content = sec.get("content", "")
        is_disc = "disclaimer" in title.lower()
        story.append(Paragraph(title, ens["heading"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_BG, spaceAfter=3))
        for line in content.split("\n"):
            line = line.strip()
            if line:
                if line.startswith(("- ", "• ", "* ")):
                    line = f"● {line[2:]}"
                story.append(Paragraph(line, ens["disc"] if is_disc else ens["body"]))
        story.append(Spacer(1, 8))

    # English doctor review
    if doctor_review:
        story.append(Paragraph("Doctor's Review", ens["heading"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_BG, spaceAfter=3))
        story.append(Paragraph(doctor_review, ens["body"]))

    # ════════════════════════════════════════════════════════════════════
    #  BILINGUAL DIVIDER
    # ════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 24))

    # Green divider banner — two-column: English left, Urdu right
    div_en = Paragraph("— Urdu Version Below —", urs["divider_en"])
    div_ur = Paragraph(_reshape("ذیل میں اردو ورژن") if urdu_ok else "اردو ورژن", urs["divider"])
    divider_table = Table(
        [[div_en, div_ur]],
        colWidths=[(A4[0] - 84) * 0.5, (A4[0] - 84) * 0.5],
    )
    divider_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), DIVIDER_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 16),
        ("BOX",           (0, 0), (-1, -1), 1.5, DIVIDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(KeepTogether([divider_table]))
    story.append(Spacer(1, 24))

    # ════════════════════════════════════════════════════════════════════
    #  URDU SECTION
    # ════════════════════════════════════════════════════════════════════
    def ur(text):
        return _reshape(text) if urdu_ok else str(text)

    # Urdu patient info table
    # Label col uses Urdu font (RTL), Value col uses Helvetica (Latin text renders correctly)
    ur_rows = [
        [Paragraph(ur("مریض کا نام"),      urs["label"]), Paragraph(patient_name or "—",                      urs["value"])],
        [Paragraph(ur("عمر"),               urs["label"]), Paragraph(str(patient_age) if patient_age else "—", urs["value"])],
        [Paragraph(ur("جنس"),               urs["label"]), Paragraph(patient_gender or "—",                    urs["value"])],
        [Paragraph(ur("تشخیص شدہ بیماری"), urs["label"]), Paragraph(disease,                                  urs["value"])],
        [Paragraph(ur("اے آئی اعتماد"),     urs["label"]), Paragraph(f"{confidence * 100:.1f}%",               urs["value"])],
    ]
    story.append(_info_table(ur_rows))
    story.append(Spacer(1, 16))

    # Urdu sections
    for sec in sections_ur:
        title   = ur(sec.get("title", ""))
        content = ur(sec.get("content", ""))
        is_disc = "اعلان" in sec.get("title", "") or "disclaimer" in sec.get("title", "").lower()
        story.append(Paragraph(title, urs["heading"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_BG, spaceAfter=3))
        for line in content.split("\n"):
            line = line.strip()
            if line:
                story.append(Paragraph(line, urs["disc"] if is_disc else urs["body"]))
        story.append(Spacer(1, 8))

    # Urdu doctor review
    if doctor_review:
        story.append(Paragraph(ur("ڈاکٹر کا جائزہ"), urs["heading"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_BG, spaceAfter=3))
        story.append(Paragraph(ur(doctor_review), urs["body"]))

    # ── Build ─────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    result = buf.getvalue()
    buf.close()
    return result
