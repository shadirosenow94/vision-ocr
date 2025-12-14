from fastapi import FastAPI
import pytesseract
import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from io import BytesIO
import re

# HEIC support
from pillow_heif import register_heif_opener
register_heif_opener()

app = FastAPI(
    title="Smart OCR with Fallback Form Extraction",
    version="0.1.0"
)

# ---------------------------
# Image Utilities
# ---------------------------

def normalize_orientation(img: Image.Image) -> Image.Image:
    """
    Fix EXIF orientation (critical for phone photos)
    """
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img


def preprocess_photo(img_np: np.ndarray) -> np.ndarray:
    """
    Robust preprocessing for photographed documents
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.bilateralFilter(gray, 11, 75, 75)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )

    return thresh


# ---------------------------
# Road Tax Parsing (Malaysia)
# ---------------------------

def parse_malaysia_roadtax(text: str) -> dict:
    plate = re.search(r'\b[A-Z]{1,3}\d{1,4}[A-Z]?\b', text)
    date = re.search(
        r'\b\d{2}\s(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s\d{4}\b',
        text
    )
    amount = re.search(r'RM\s?\d+(\.\d{2})?', text)
    receipt = re.search(r'\bVEL\d{5}\b', text)

    return {
        "plate_number": plate.group(0) if plate else None,
        "expiry_date": date.group(0) if date else None,
        "amount": amount.group(0) if amount else None,
        "receipt_code": receipt.group(0) if receipt else None
    }


# ---------------------------
# Fallback Form Extraction
# ---------------------------

def fallback_form_extraction(text: str) -> dict:
    """
    Generic form key-value inference when document type is unknown
    """
    fields = {}
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for i in range(len(lines) - 1):
        if lines[i].endswith(":"):
            key = lines[i].replace(":", "").strip().lower().replace(" ", "_")
            value = lines[i + 1].strip()
            fields[key] = value

    return fields


# ---------------------------
# Document Classification
# ---------------------------

def classify_document(text: str) -> tuple[str, float]:
    score = 0.0

    keywords = [
        "LESEN KENDERAAN MOTOR",
        "MOTOSIKAL",
        "SELAIN MOTOSIKAL",
        "JPJ",
        "ROAD TAX"
    ]

    for k in keywords:
        if k.lower() in text.lower():
            score += 0.2

    if score >= 0.6:
        return "malaysia_road_tax", min(score, 1.0)

    return "unknown_form", score


# ---------------------------
# API Endpoint
# ---------------------------

@app.post("/ocr/document")
def ocr_document(payload: dict):
    image_url = payload.get("image_url")
    if not image_url:
        return {"error": "image_url is required"}

    try:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()

        img = Image.open(BytesIO(resp.content))
        img = normalize_orientation(img)
        img = img.convert("RGB")

        img_np = np.array(img)
        processed = preprocess_photo(img_np)

        text = pytesseract.image_to_string(
            processed,
            lang="eng+msa",
            config="--psm 6"
        )

        doc_type, confidence = classify_document(text)

        if doc_type == "malaysia_road_tax":
            fields = parse_malaysia_roadtax(text)

            # Confidence refinement
            filled = sum(1 for v in fields.values() if v)
            confidence = min(confidence + (filled * 0.1), 1.0)

            return {
                "document_type": doc_type,
                "confidence": round(confidence, 2),
                "fields": fields
            }

        # Fallback
        return {
            "document_type": "unknown_form",
            "confidence": round(confidence, 2),
            "fields": fallback_form_extraction(text),
            "note": "Fallback form extraction used"
        }

    except Exception as e:
        return {
            "error": "OCR processing failed",
            "details": str(e)
        }