from fastapi import FastAPI, HTTPException
import pytesseract
import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from io import BytesIO
import re
import os
import math

# HEIC support (CRITICAL)
import pillow_heif
pillow_heif.register_heif_opener()

app = FastAPI(
    title="Smart OCR with Photo-First Pipeline (Malaysia)",
    version="2.0.0"
)

# -------------------------------------------------
# Image loader (HEIC-safe + orientation safe)
# -------------------------------------------------
def load_image_from_url(url: str) -> np.ndarray:
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch image")

    try:
        img = Image.open(BytesIO(resp.content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {str(e)}")

    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return np.array(img)


# -------------------------------------------------
# Photo-first enhancement pipeline
# -------------------------------------------------
def enhance_photo(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.normalize(gray, None, 30, 255, cv2.NORM_MINMAX)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)

    gray = cv2.fastNlMeansDenoising(gray, h=10)

    return gray


# -------------------------------------------------
# OCR runner
# -------------------------------------------------
def run_ocr(img: np.ndarray) -> str:
    return pytesseract.image_to_string(
        img,
        lang="eng+msa",
        config="--oem 3 --psm 6"
    )


# -------------------------------------------------
# OCR quality estimator
# -------------------------------------------------
def ocr_quality(text: str) -> float:
    if not text.strip():
        return 0.0

    alnum = sum(c.isalnum() for c in text)
    printable = sum(c.isprintable() for c in text)
    ratio = alnum / max(len(text), 1)

    words = [w for w in re.split(r"\s+", text) if len(w) > 2]
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)

    score = 0.0
    score += 0.4 if ratio > 0.35 else 0.0
    score += 0.3 if printable / len(text) > 0.9 else 0.0
    score += 0.3 if avg_word_len > 3 else 0.0

    return round(score, 2)


# -------------------------------------------------
# Malaysia JPJ classifier
# -------------------------------------------------
def classify_jpj(text: str):
    t = text.upper()

    if "LESEN" not in t and "KENDERAAN" not in t:
        return None

    plate = re.search(r"\b[A-Z]{1,3}\d{1,4}[A-Z]?\b", t)
    expiry = re.search(
        r"\b\d{2}\s(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s\d{4}\b", t
    )
    amount = re.search(r"RM\s?\d+(?:\.\d{2})?", t)
    receipt = re.search(r"\bVEL\d{5}\b", t)

    vehicle_class = None
    if "MOTOSIKAL" in t:
        vehicle_class = "MOTOSIKAL"
    elif "SELAIN MOTOSIKAL" in t:
        vehicle_class = "SELAIN MOTOSIKAL"

    confidence = 0.0
    confidence += 0.35 if plate else 0
    confidence += 0.25 if expiry else 0
    confidence += 0.2 if amount else 0
    confidence += 0.2 if vehicle_class else 0

    if confidence < 0.6:
        return None

    return {
        "document_type": "malaysia_roadtax_jpj",
        "confidence": round(confidence, 2),
        "fields": {
            "plate_number": plate.group(0) if plate else None,
            "expiry_date": expiry.group(0) if expiry else None,
            "vehicle_class": vehicle_class,
            "amount": amount.group(0) if amount else None,
            "receipt_code": receipt.group(0) if receipt else None
        }
    }


# -------------------------------------------------
# Safe fallback
# -------------------------------------------------
def fallback_form_extraction(text: str, quality: float):
    if quality < 0.4:
        return {
            "document_type": "unreadable",
            "confidence": 0.0,
            "reason": "Photo quality too low for structured extraction",
            "suggestion": "Retake photo with better lighting or closer framing"
        }

    fields = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            if len(k.strip()) > 2 and len(v.strip()) > 2:
                fields[k.strip()] = v.strip()

    if not fields:
        return {
            "document_type": "unknown_form",
            "confidence": round(quality, 2),
            "note": "Text detected but structure unclear"
        }

    return {
        "document_type": "unknown_form",
        "confidence": round(quality, 2),
        "fields": fields
    }


# -------------------------------------------------
# API endpoint
# -------------------------------------------------
@app.post("/ocr/document")
def ocr_document(payload: dict):
    image_url = payload.get("image_url")
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url required")

    img = load_image_from_url(image_url)

    raw_text = run_ocr(img)
    enhanced_img = enhance_photo(img)
    enhanced_text = run_ocr(enhanced_img)

    raw_q = ocr_quality(raw_text)
    enh_q = ocr_quality(enhanced_text)

    text = enhanced_text if enh_q >= raw_q else raw_text
    quality = max(raw_q, enh_q)

    jpj = classify_jpj(text)
    if jpj:
        return jpj

    return fallback_form_extraction(text, quality)


# -------------------------------------------------
# Startup sanity check
# -------------------------------------------------
@app.on_event("startup")
def check_tesseract():
    langs = pytesseract.get_languages(config="")
    print("Available Tesseract languages:", langs)
    if "msa" not in langs:
        print("WARNING: Malay language pack (msa) missing!")