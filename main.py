from fastapi import FastAPI, HTTPException
import pytesseract
import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from io import BytesIO
import re
import os

app = FastAPI(
    title="Smart OCR with Fallback Form Extraction",
    version="1.0.0"
)

# -------------------------
# Utility: download image
# -------------------------
def load_image_from_url(url: str) -> np.ndarray:
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch image")

    img = Image.open(BytesIO(resp.content))
    img = ImageOps.exif_transpose(img)  # fix orientation
    img = img.convert("RGB")
    return np.array(img)


# -------------------------
# Preprocessing (JPJ-safe)
# -------------------------
def preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast normalization (CRITICAL for JPJ pink docs)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Noise reduction without killing text
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive threshold handles photos + scans
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )
    return thresh


# -------------------------
# OCR
# -------------------------
def run_ocr(img: np.ndarray) -> str:
    return pytesseract.image_to_string(
        img,
        lang="eng+msa",
        config="--oem 3 --psm 6"
    )


# -------------------------
# Malaysia JPJ classifier
# -------------------------
def classify_jpj(text: str):
    text_upper = text.upper()

    if "LESEN KENDERAAN MOTOR" not in text_upper:
        return None

    plate = re.search(r"\b[A-Z]{1,3}\d{1,4}[A-Z]?\b", text_upper)
    expiry = re.search(r"\b\d{2}\s(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s\d{4}\b", text_upper)
    amount = re.search(r"RM\s?\d+(?:\.\d{2})?", text_upper)
    receipt = re.search(r"\bVEL\d{5}\b", text_upper)

    vehicle_class = None
    if "MOTOSIKAL" in text_upper:
        vehicle_class = "MOTOSIKAL"
    elif "SELAIN MOTOSIKAL" in text_upper:
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


# -------------------------
# Generic fallback extractor
# -------------------------
def fallback_form_extraction(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    fields = {}

    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()

    return {
        "document_type": "unknown_form",
        "confidence": 0.2 if fields else 0.1,
        "fields": fields,
        "note": "Fallback form extraction used"
    }


# -------------------------
# API endpoint
# -------------------------
@app.post("/ocr/document")
def ocr_document(payload: dict):
    image_url = payload.get("image_url")
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url required")

    img = load_image_from_url(image_url)
    processed = preprocess(img)
    text = run_ocr(processed)

    jpj = classify_jpj(text)
    if jpj:
        return jpj

    return fallback_form_extraction(text)


# -------------------------
# Startup sanity check
# -------------------------
@app.on_event("startup")
def check_tesseract():
    langs = pytesseract.get_languages(config="")
    print("Available Tesseract languages:", langs)
    if "msa" not in langs:
        print("WARNING: Malay language pack (msa) missing!")