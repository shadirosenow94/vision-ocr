from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import pytesseract
import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from io import BytesIO
import re
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor

# HEIC support (CRITICAL)
import pillow_heif
pillow_heif.register_heif_opener()

app = FastAPI(
    title="Smart OCR with Photo-First Pipeline (Malaysia)",
    version="3.0.0"
)

# In-memory job storage (use Redis/DB in production)
jobs: Dict[str, dict] = {}
executor = ThreadPoolExecutor(max_workers=3)

# Job cleanup every hour
async def cleanup_old_jobs():
    while True:
        await asyncio.sleep(3600)
        cutoff = datetime.now() - timedelta(hours=24)
        to_delete = [
            job_id for job_id, job in jobs.items()
            if job.get('created_at', datetime.now()) < cutoff
        ]
        for job_id in to_delete:
            del jobs[job_id]
        print(f"Cleaned up {len(to_delete)} old jobs")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_old_jobs())
    check_tesseract()

# -------------------------------------------------
# Image loader with aggressive optimization
# -------------------------------------------------
def load_image_from_url(url: str, max_size_mb: int = 15) -> np.ndarray:
    try:
        print(f"Downloading image from: {url}")
        resp = requests.get(url, timeout=60, stream=True)
        if resp.status_code != 200:
            raise Exception(f"Failed to fetch image: {resp.status_code}")

        content_length = resp.headers.get('content-length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            raise Exception(f"Image too large (max {max_size_mb}MB)")

        content = resp.content
        print(f"Downloaded {len(content)} bytes")
        
        img = Image.open(BytesIO(content))
        print(f"Image format: {img.format}, size: {img.size}, mode: {img.mode}")
        
    except requests.Timeout:
        raise Exception("Image download timeout")
    except Exception as e:
        raise Exception(f"Image decode failed: {str(e)}")

    # Handle orientation
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    
    # AGGRESSIVE resize for OCR (OCR doesn't need huge images)
    # Target: 1200px on longest side
    max_dimension = 1200
    if max(img.size) > max_dimension:
        ratio = max_dimension / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        print(f"Resized image to {new_size}")
    
    return np.array(img)


# -------------------------------------------------
# Auto-rotation and deskewing
# -------------------------------------------------
def auto_rotate_image(img: np.ndarray) -> np.ndarray:
    """Detect and correct image rotation"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Use Tesseract's OSD (Orientation and Script Detection)
    try:
        osd = pytesseract.image_to_osd(gray)
        rotation = int(re.search(r'Rotate: (\d+)', osd).group(1))
        
        if rotation != 0:
            print(f"Auto-rotating image by {rotation} degrees")
            # Rotate to correct orientation
            if rotation == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotation == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif rotation == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    except Exception as e:
        print(f"Could not detect rotation: {e}")
    
    return img


def deskew_image(img: np.ndarray) -> np.ndarray:
    """Correct slight skew/angle in image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is not None and len(lines) > 0:
        # Calculate average angle
        angles = []
        for rho, theta in lines[:20, 0]:  # Use top 20 lines
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)
        
        median_angle = np.median(angles)
        
        # Only correct if angle is significant but not too extreme
        if abs(median_angle) > 0.5 and abs(median_angle) < 45:
            print(f"Deskewing by {median_angle:.2f} degrees")
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return img


# -------------------------------------------------
# Lightweight enhancement
# -------------------------------------------------
def enhance_photo(img: np.ndarray) -> np.ndarray:
    """Super lightweight enhancement for speed"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Just CLAHE + light sharpen
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Simple unsharp mask
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    
    return gray


# -------------------------------------------------
# OCR runner (no timeout - let it finish)
# -------------------------------------------------
def run_ocr(img: np.ndarray) -> str:
    """Run OCR without timeout - runs in background"""
    try:
        print(f"Starting OCR on image shape: {img.shape}")
        result = pytesseract.image_to_string(
            img,
            lang="eng+msa",
            config="--oem 3 --psm 6"
        )
        print(f"OCR completed, extracted {len(result)} characters")
        return result
    except Exception as e:
        print(f"OCR error: {e}")
        raise Exception(f"OCR failed: {str(e)}")


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
# Enhanced Malaysia JPJ classifier
# -------------------------------------------------
def classify_jpj(text: str):
    t = text.upper()
    
    has_lesen = "LESEN" in t
    has_kenderaan = "KENDERAAN" in t or "MOTOR" in t
    
    if not (has_lesen or has_kenderaan):
        return None

    # Enhanced plate number pattern - MUST NOT be a date
    # Malaysian plates: ABC1234, WRU7352, V1234, etc.
    # Exclude patterns that look like dates (03 JAN 2025, JAN2025, etc.)
    plate_patterns = [
        r"\b[A-Z]{1,3}\d{1,4}[A-Z]?\b",  # Standard format: 1-3 letters, then digits
    ]
    
    plate = None
    plate_candidates = []
    
    for pattern in plate_patterns:
        matches = re.findall(pattern, t)
        for match in matches:
            # Filter out date-like patterns
            if re.match(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{4}", match):
                continue  # This is a date, not a plate
            if match.startswith(('RM', 'VEL')):
                continue  # This is money or receipt code
            # Valid plate should have at least 1 letter and 1 digit
            if re.search(r'[A-Z]', match) and re.search(r'\d', match):
                plate_candidates.append(match)
    
    # Prefer plates that appear near "LESEN" or at the start/middle of text
    if plate_candidates:
        # Simple heuristic: choose the first valid candidate
        plate = plate_candidates[0]
    
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
    
    # Receipt codes
    receipt_patterns = [r"\bVEL\d{5,}\b", r"\b\d{7,}\b"]
    receipts = []
    for pattern in receipt_patterns:
        matches = re.findall(pattern, t)
        receipts.extend(matches)
    
    # Vehicle class
    vehicle_class = None
    if "MOTOSIKAL" in t and "SELAIN" not in t:
        vehicle_class = "MOTOSIKAL"
    elif "SELAIN MOTOSIKAL" in t or "SELAIN MOTO" in t:
        vehicle_class = "SELAIN MOTOSIKAL"
    
    # Location
    location = None
    if "SEMENANJUNG" in t:
        location = "SEMENANJUNG"
    elif "PERSENDIRIAN" in t:
        location = "PERSENDIRIAN"
    
    # Confidence
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
# Fallback
# -------------------------------------------------
def fallback_form_extraction(text: str, quality: float):
    if quality < 0.3:
        return {
            "document_type": "unreadable",
            "confidence": 0.0,
            "reason": "Photo quality too low",
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
            "note": "Text detected but structure unclear"
        }

    return {
        "document_type": "unknown_form",
        "confidence": round(quality, 2),
        "fields": fields
    }


# -------------------------------------------------
# Background OCR processor
# -------------------------------------------------
def process_ocr_job(job_id: str, image_url: str, text_only: bool = False):
    """Process OCR in background - NO TIMEOUTS"""
    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['updated_at'] = datetime.now()
        
        # Load image
        print(f"Job {job_id}: Loading image")
        img = load_image_from_url(image_url)
        
        # Auto-rotate if needed
        img = auto_rotate_image(img)
        
        # Deskew if angled
        img = deskew_image(img)
        
        # Run OCR
        print(f"Job {job_id}: Running OCR")
        text = run_ocr(img)
        
        # If text_only mode, return just the text
        if text_only:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['result'] = {"text": text.strip()}
            jobs[job_id]['updated_at'] = datetime.now()
            print(f"Job {job_id}: Completed (text only)")
            return
        
        # Otherwise, do full processing
        raw_q = ocr_quality(text)
        print(f"Job {job_id}: OCR quality {raw_q}")
        
        # Try classification
        jpj = classify_jpj(text)
        if jpj and jpj['confidence'] > 0.6:
            print(f"Job {job_id}: Classified from OCR")
            jpj["ocr_quality"] = raw_q
            jpj["method"] = "raw"
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['result'] = jpj
            jobs[job_id]['updated_at'] = datetime.now()
            return
        
        # Try enhanced
        print(f"Job {job_id}: Trying enhanced OCR")
        enhanced_img = enhance_photo(img)
        enhanced_text = run_ocr(enhanced_img)
        enh_q = ocr_quality(enhanced_text)
        print(f"Job {job_id}: Enhanced quality {enh_q}")
        
        # Try classification on enhanced
        jpj = classify_jpj(enhanced_text)
        if jpj:
            print(f"Job {job_id}: Classified from enhanced OCR")
            jpj["ocr_quality"] = enh_q
            jpj["method"] = "enhanced"
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['result'] = jpj
            jobs[job_id]['updated_at'] = datetime.now()
            return
        
        # Fallback
        best_text = enhanced_text if enh_q >= raw_q else text
        quality = max(raw_q, enh_q)
        result = fallback_form_extraction(best_text, quality)
        
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = result
        jobs[job_id]['updated_at'] = datetime.now()
        print(f"Job {job_id}: Completed with fallback")
        
    except Exception as e:
        print(f"Job {job_id}: Failed - {e}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['updated_at'] = datetime.now()


# -------------------------------------------------
# API: Submit job (instant response)
# -------------------------------------------------
@app.post("/ocr/document")
async def submit_ocr_job(payload: dict, background_tasks: BackgroundTasks):
    """Submit OCR job - returns immediately with job_id"""
    image_url = payload.get("image_url")
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url required")
    
    # Create job
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "image_url": image_url,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    # Start processing in background
    background_tasks.add_task(process_ocr_job, job_id, image_url, False)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "OCR job submitted. Poll /ocr/status/{job_id} for results",
        "status_url": f"/ocr/status/{job_id}"
    }


# -------------------------------------------------
# API: Text-only extraction (no structure)
# -------------------------------------------------
@app.post("/ocr/text")
async def submit_text_ocr_job(payload: dict, background_tasks: BackgroundTasks):
    """Submit text-only OCR job - just extracts raw text"""
    image_url = payload.get("image_url")
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url required")
    
    # Create job
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "image_url": image_url,
        "text_only": True,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    # Start processing in background (text_only=True)
    background_tasks.add_task(process_ocr_job, job_id, image_url, True)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Text extraction job submitted. Poll /ocr/status/{job_id} for results",
        "status_url": f"/ocr/status/{job_id}"
    }


# -------------------------------------------------
# API: Check job status
# -------------------------------------------------
@app.get("/ocr/status/{job_id}")
async def get_job_status(job_id: str):
    """Get OCR job status and results"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    response = {
        "job_id": job_id,
        "status": job['status'],
        "created_at": job['created_at'].isoformat(),
        "updated_at": job['updated_at'].isoformat()
    }
    
    if job['status'] == 'completed':
        response['result'] = job.get('result')
    elif job['status'] == 'failed':
        response['error'] = job.get('error')
    
    return response


# -------------------------------------------------
# API: Synchronous endpoint (with longer timeout)
# -------------------------------------------------
@app.post("/ocr/document/sync")
async def ocr_document_sync(payload: dict):
    """Synchronous OCR - waits up to 90s for result"""
    image_url = payload.get("image_url")
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url required")
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "image_url": image_url,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    # Run in executor
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, process_ocr_job, job_id, image_url)
    
    # Wait for completion (up to 90s)
    for _ in range(180):  # 180 * 0.5s = 90s
        await asyncio.sleep(0.5)
        if jobs[job_id]['status'] in ['completed', 'failed']:
            break
    
    job = jobs[job_id]
    
    if job['status'] == 'completed':
        return job['result']
    elif job['status'] == 'failed':
        raise HTTPException(status_code=500, detail=job.get('error'))
    else:
        raise HTTPException(status_code=504, detail="Processing timeout - use async endpoint")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "active_jobs": len([j for j in jobs.values() if j['status'] == 'processing']),
        "queued_jobs": len([j for j in jobs.values() if j['status'] == 'queued'])
    }


def check_tesseract():
    langs = pytesseract.get_languages(config="")
    print("Available Tesseract languages:", langs)
    if "msa" not in langs:
        print("WARNING: Malay language pack (msa) missing!")
    if "eng" not in langs:
        print("WARNING: English language pack missing!")
    print("OCR service ready")