from fastapi import FastAPI, HTTPException
import pytesseract
import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from io import BytesIO
import re
from datetime import datetime

# HEIC support (CRITICAL)
import pillow_heif
pillow_heif.register_heif_opener()

app = FastAPI(
    title="Smart OCR with Photo-First Pipeline (Malaysia)",
    version="2.1.0"
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
# Enhanced Malaysia JPJ classifier with better patterns
# -------------------------------------------------
def classify_jpj(text: str):
    t = text.upper()
    
    # More flexible document identification
    has_lesen = "LESEN" in t
    has_kenderaan = "KENDERAAN" in t or "MOTOR" in t
    
    if not (has_lesen or has_kenderaan):
        return None

    # Enhanced plate number pattern (more flexible)
    # Matches: WRU7352, WRU 7352, ABC1234, etc.
    plate_patterns = [
        r"\b[A-Z]{2,4}\s?\d{3,4}\s?[A-Z]?\b",  # Standard format
        r"(?:^|\n)([A-Z]{2,4}\s?\d{3,4})(?:\s|$)",  # Line-based
    ]
    plate = None
    for pattern in plate_patterns:
        match = re.search(pattern, t)
        if match:
            plate = match.group(0).strip()
            # Clean up the plate
            plate = re.sub(r'\s+', '', plate)
            break
    
    # Enhanced expiry date pattern
    # Matches: 16 SEP 2026, 16SEP2026, etc.
    expiry_patterns = [
        r"\b(\d{1,2})\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*(\d{4})\b",
        r"\b(\d{1,2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{4})\b"
    ]
    expiry = None
    for pattern in expiry_patterns:
        match = re.search(pattern, t)
        if match:
            day, month, year = match.groups()
            expiry = f"{day.zfill(2)} {month} {year}"
            break
    
    # Amount pattern (RM90.00, RM 90.00, etc.)
    amount_match = re.search(r"RM\s?(\d+(?:\.\d{2})?)", t)
    amount = f"RM{amount_match.group(1)}" if amount_match else None
    
    # Receipt/Reference codes
    # VEL02504, 0014322, etc.
    receipt_patterns = [
        r"\bVEL\d{5,}\b",
        r"\b\d{7,}\b"
    ]
    receipts = []
    for pattern in receipt_patterns:
        matches = re.findall(pattern, t)
        receipts.extend(matches)
    
    # Vehicle class detection
    vehicle_class = None
    if "MOTOSIKAL" in t and "SELAIN" not in t:
        vehicle_class = "MOTOSIKAL"
    elif "SELAIN MOTOSIKAL" in t or "SELAIN MOTO" in t:
        vehicle_class = "SELAIN MOTOSIKAL"
    
    # Location extraction (SEMENANJUNG, PERSENDIRIAN, etc.)
    location = None
    if "SEMENANJUNG" in t:
        location = "SEMENANJUNG"
    elif "PERSENDIRIAN" in t:
        location = "PERSENDIRIAN"
    
    # Build confidence score (more lenient)
    confidence = 0.0
    confidence += 0.3 if has_lesen or has_kenderaan else 0
    confidence += 0.3 if plate else 0
    confidence += 0.2 if expiry else 0
    confidence += 0.1 if amount else 0
    confidence += 0.1 if vehicle_class else 0
    
    # Lower threshold to 0.4 (was 0.6)
    if confidence < 0.4:
        return None

    # Extract all numeric codes that might be useful
    numeric_codes = re.findall(r"\b\d{7,}\b", t)
    
    return {
        "document_type": "malaysia_roadtax_jpj",
        "confidence": round(confidence, 2),
        "fields": {
            "plate_number": plate,
            "expiry_date": expiry,
            "vehicle_class": vehicle_class,
            "amount": amount,
            "location": location,
            "receipt_codes": receipts[:3] if receipts else None,  # Limit to top 3
            "reference_numbers": numeric_codes[:3] if numeric_codes else None,
            "raw_text_length": len(text)
        },
        "debug": {
            "has_lesen": has_lesen,
            "has_kenderaan": has_kenderaan,
            "text_preview": text[:200] if len(text) > 200 else text
        }
    }


# -------------------------------------------------
# Safe fallback
# -------------------------------------------------
def fallback_form_extraction(text: str, quality: float):
    if quality < 0.3:
        return {
            "document_type": "unreadable",
            "confidence": 0.0,
            "reason": "Photo quality too low for structured extraction",
            "suggestion": "Retake photo with better lighting or closer framing",
            "quality_score": quality
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
            "note": "Text detected but structure unclear",
            "text_preview": text[:300] if len(text) > 300 else text
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

    # Try multiple OCR passes
    raw_text = run_ocr(img)
    enhanced_img = enhance_photo(img)
    enhanced_text = run_ocr(enhanced_img)

    raw_q = ocr_quality(raw_text)
    enh_q = ocr_quality(enhanced_text)

    text = enhanced_text if enh_q >= raw_q else raw_text
    quality = max(raw_q, enh_q)

    # Try JPJ classification on both versions
    jpj = classify_jpj(enhanced_text)
    if not jpj:
        jpj = classify_jpj(raw_text)
    
    if jpj:
        jpj["ocr_quality"] = quality
        jpj["method"] = "enhanced" if enh_q >= raw_q else "raw"
        return jpj

    return fallback_form_extraction(text, quality)


@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "2.1.0"}


# -------------------------------------------------
# Startup sanity check
# -------------------------------------------------
@app.on_event("startup")
def check_tesseract():
    langs = pytesseract.get_languages(config="")
    print("Available Tesseract languages:", langs)
    if "msa" not in langs:
        print("WARNING: Malay language pack (msa) missing!")
    if "eng" not in langs:
        print("WARNING: English language pack missing!")