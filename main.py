from fastapi import FastAPI, HTTPException
import pytesseract
import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from io import BytesIO
import re
import os
import tempfile

# Optional HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except Exception:
    HEIC_SUPPORTED = False

app = FastAPI(
    title="Smart OCR with Tiered Quality Pipeline",
    version="2.0.0"
)

# =========================================================
# Stage 0 — Image Loader & Normalisation
# =========================================================
def load_image_from_url(url: str) -> np.ndarray:
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch image")

    try:
        img = Image.open(BytesIO(resp.content))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    # Resize very large images (safety)
    max_dim = 1600
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS
        )

    return np.array(img)


# =========================================================
# Stage 1 — Quality Gate (Honesty First)
# =========================================================
def assess_image_quality(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast_score = gray.std()

    quality_score = min(
        (blur_score / 150.0) * 0.5 + (contrast_score / 60.0) * 0.5,
        1.0
    )

    if blur_score < 40 or contrast_score < 20:
        return "unreadable", round(quality_score, 2)

    if blur_score < 90 or contrast_score < 35:
        return "degraded", round(quality_score, 2)

    return "good", round(quality_score, 2)


# =========================================================
# Stage 2 — Dual OCR Preprocessing
# =========================================================
def preprocess_scan(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )
    return thresh


def preprocess_photo(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    return gray


def run_ocr(img: np.ndarray) -> str:
    return pytesseract.image_to_string(
        img,
        lang="eng+msa",
        config="--oem 3 --psm 6"
    )


# =========================================================
# Stage 3 — ML Text Detection (Region-Based OCR)
# =========================================================
def ml_text_detection(img: np.ndarray) -> str:
    """
    Lightweight contour-based text region detection
    (CPU-safe, deterministic fallback when OCR is weak)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    text_chunks = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 40 or h < 15:
            continue

        roi = img[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        text = pytesseract.image_to_string(
            roi_gray,
            lang="eng+msa",
            config="--oem 3 --psm 7"
        )

        if text.strip():
            text_chunks.append(text)

    return "\n".join(text_chunks)


# =========================================================
# Stage 4 — Malaysia JPJ Classification
# =========================================================
def classify_jpj(text: str):
    t = text.upper()

    if "LESEN KENDERAAN MOTOR" not in t:
        return None

    plate = re.search(r"\b[A-Z]{1,3}\d{1,4}[A-Z]?\b", t)
    expiry = re.search(
        r"\b\d{2}\s(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s\d{4}\b",
        t
    )
    amount = re.search(r"RM\s?\d+(?:\.\d{2})?", t)
    receipt = re.search(r"\bVEL\d{5}\b", t)

    vehicle_class = None
    if "MOTOSIKAL" in t:
        vehicle_class = "MOTOSIKAL"
    elif "SELAIN MOTOSIKAL" in t:
        vehicle_class = "SELAIN MOTOSIKAL"

    confidence = 0.0
    confidence += 0.3 if plate else 0
    confidence += 0.3 if expiry else 0
    confidence += 0.2 if amount else 0
    confidence += 0.2 if vehicle_class else 0

    return {
        "document_type": "malaysia_roadtax_jpj",
        "confidence": round(min(confidence, 1.0), 2),
        "fields": {
            "plate_number": plate.group(0) if plate else None,
            "expiry_date": expiry.group(0) if expiry else None,
            "vehicle_class": vehicle_class,
            "amount": amount.group(0) if amount else None,
            "receipt_code": receipt.group(0) if receipt else None
        }
    }


# =========================================================
# Stage 5 — Fallback Form Extraction (Safe)
# =========================================================
def fallback_form_extraction(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    fields = {}

    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1)
            if len(k) < 40 and len(v) < 120:
                fields[k.strip()] = v.strip()

    return {
        "document_type": "unknown_form",
        "confidence": 0.2 if fields else 0.1,
        "fields": fields,
        "note": "Fallback form extraction used"
    }


# =========================================================
# API Endpoint
# =========================================================
@app.post("/ocr/document")
def ocr_document(payload: dict):
    image_url = payload.get("image_url")
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url required")

    img = load_image_from_url(image_url)

    readability, quality_score = assess_image_quality(img)
    if readability == "unreadable":
        return {
            "document_type": "unreadable",
            "confidence": 0.0,
            "quality_score": quality_score,
            "reason": "Image too blurry or poorly lit for OCR",
            "action_required": "Please upload a clearer photo"
        }

    # Dual OCR
    scan_text = run_ocr(preprocess_scan(img))
    photo_text = run_ocr(preprocess_photo(img))
    combined_text = scan_text + "\n" + photo_text

    # JPJ detection
    jpj = classify_jpj(combined_text)
    if jpj:
        jpj["quality_score"] = quality_score
        return jpj

    # ML-assisted text detection fallback
    ml_text = ml_text_detection(img)
    if ml_text:
        combined_text += "\n" + ml_text
        jpj = classify_jpj(combined_text)
        if jpj:
            jpj["quality_score"] = quality_score
            jpj["note"] = "Detected via ML text regions"
            return jpj

    return fallback_form_extraction(combined_text)


# =========================================================
# Startup Check
# =========================================================
@app.on_event("startup")
def startup_check():
    langs = pytesseract.get_languages(config="")
    print("Tesseract languages:", langs)
    if "msa" not in langs:
        print("WARNING: Malay language pack (msa) missing")