from fastapi import FastAPI, HTTPException
import pytesseract
import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from io import BytesIO
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import asyncio
from functools import partial

# HEIC support (CRITICAL)
import pillow_heif
pillow_heif.register_heif_opener()

app = FastAPI(
    title="Smart OCR with Photo-First Pipeline (Malaysia)",
    version="2.1.0"
)

# Executor for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=2)

# -------------------------------------------------
# Image loader with size limits and timeout
# -------------------------------------------------
def load_image_from_url(url: str, max_size_mb: int = 10) -> np.ndarray:
    try:
        resp = requests.get(url, timeout=30, stream=True)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch image")

        # Check content length
        content_length = resp.headers.get('content-length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"Image too large (max {max_size_mb}MB)")

        # Load image
        content = resp.content
        img = Image.open(BytesIO(content))
        
    except requests.Timeout:
        raise HTTPException(status_code=408, detail="Image download timeout")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {str(e)}")

    # Handle orientation
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    
    # Resize if too large (OCR doesn't need huge images)
    max_dimension = 2000
    if max(img.size) > max_dimension:
        ratio = max_dimension / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        print(f"Resized image from {img.size} to {new_size}")
    
    return np.array(img)


# -------------------------------------------------
# Photo-first enhancement pipeline (optimized)
# -------------------------------------------------
def enhance_photo(img: np.ndarray) -> np.ndarray:
    """Lighter enhancement pipeline for faster processing"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive histogram equalization (lighter)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Normalize
    gray = cv2.normalize(gray, None, 30, 255, cv2.NORM_MINMAX)

    # Sharpen (lighter kernel)
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]]) / 1.5
    gray = cv2.filter2D(gray, -1, kernel)

    return gray


# -------------------------------------------------
# OCR runner with timeout
# -------------------------------------------------
def run_ocr(img: np.ndarray, timeout: int = 30) -> str:
    """Run OCR with timeout protection"""
    try:
        # Run in thread pool with timeout
        future = executor.submit(
            pytesseract.image_to_string,
            img,
            lang="eng+msa",
            config="--oem 3 --psm 6"
        )
        result = future.result(timeout=timeout)
        return result
    except TimeoutError:
        raise HTTPException(status_code=504, detail="OCR processing timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


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
    plate_patterns = [
        r"\b[A-Z]{2,4}\s?\d{3,4}\s?[A-Z]?\b",
        r"(?:^|\n)([A-Z]{2,4}\s?\d{3,4})(?:\s|$)",
    ]
    plate = None
    for pattern in plate_patterns:
        match = re.search(pattern, t)
        if match:
            plate = match.group(0).strip()
            plate = re.sub(r'\s+', '', plate)
            break
    
    # Enhanced expiry date pattern
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
    
    # Amount pattern
    amount_match = re.search(r"RM\s?(\d+(?:\.\d{2})?)", t)
    amount = f"RM{amount_match.group(1)}" if amount_match else None
    
    # Receipt/Reference codes
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
    
    # Location extraction
    location = None
    if "SEMENANJUNG" in t:
        location = "SEMENANJUNG"
    elif "PERSENDIRIAN" in t:
        location = "PERSENDIRIAN"
    
    # Build confidence score
    confidence = 0.0
    confidence += 0.3 if has_lesen or has_kenderaan else 0
    confidence += 0.3 if plate else 0
    confidence += 0.2 if expiry else 0
    confidence += 0.1 if amount else 0
    confidence += 0.1 if vehicle_class else 0
    
    if confidence < 0.4:
        return None

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
            "receipt_codes": receipts[:3] if receipts else None,
            "reference_numbers": numeric_codes[:3] if numeric_codes else None,
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
# API endpoint (async with timeout)
# -------------------------------------------------
@app.post("/ocr/document")
async def ocr_document(payload: dict):
    image_url = payload.get("image_url")
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url required")

    print(f"Processing image: {image_url}")
    
    # Load image (with timeout and size limit)
    try:
        img = await asyncio.get_event_loop().run_in_executor(
            executor,
            partial(load_image_from_url, image_url)
        )
        print(f"Image loaded: {img.shape}")
    except Exception as e:
        print(f"Image load failed: {e}")
        raise

    # Strategy: Try raw first (faster), only enhance if needed
    print("Running raw OCR...")
    try:
        raw_text = await asyncio.get_event_loop().run_in_executor(
            executor,
            partial(run_ocr, img, 30)
        )
        raw_q = ocr_quality(raw_text)
        print(f"Raw OCR quality: {raw_q}")
    except Exception as e:
        print(f"Raw OCR failed: {e}")
        raise

    # Try classification on raw first
    jpj = classify_jpj(raw_text)
    if jpj and jpj['confidence'] > 0.6:
        print("Document classified from raw OCR")
        jpj["ocr_quality"] = raw_q
        jpj["method"] = "raw"
        return jpj

    # If raw wasn't confident enough, try enhanced
    print("Raw OCR insufficient, trying enhanced...")
    try:
        enhanced_img = await asyncio.get_event_loop().run_in_executor(
            executor,
            partial(enhance_photo, img)
        )
        enhanced_text = await asyncio.get_event_loop().run_in_executor(
            executor,
            partial(run_ocr, enhanced_img, 30)
        )
        enh_q = ocr_quality(enhanced_text)
        print(f"Enhanced OCR quality: {enh_q}")
    except Exception as e:
        print(f"Enhanced OCR failed: {e}")
        # Fall back to raw results
        return fallback_form_extraction(raw_text, raw_q)

    # Try classification on enhanced
    jpj = classify_jpj(enhanced_text)
    if jpj:
        print("Document classified from enhanced OCR")
        jpj["ocr_quality"] = enh_q
        jpj["method"] = "enhanced"
        return jpj

    # Use better of the two
    text = enhanced_text if enh_q >= raw_q else raw_text
    quality = max(raw_q, enh_q)
    
    print(f"No JPJ classification, falling back. Quality: {quality}")
    return fallback_form_extraction(text, quality)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "2.1.0",
        "tesseract_version": pytesseract.get_tesseract_version()
    }


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
    print("OCR service ready")