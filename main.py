from fastapi import FastAPI, HTTPException
import pytesseract
import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from io import BytesIO
import re

# HEIC support (CRITICAL)
import pillow_heif
pillow_heif.register_heif_opener()

app = FastAPI(
    title="Malaysian Road Tax OCR",
    version="3.1.0"
)


# -------------------------------------------------
# Image loader with optimization
# -------------------------------------------------
def load_image_from_url(url: str) -> np.ndarray:
    print(f"Downloading: {url}")
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch image")

    try:
        img = Image.open(BytesIO(resp.content))
        print(f"Loaded {img.format} image: {img.size}, mode: {img.mode}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {str(e)}")

    # Fix orientation
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    
    # Resize for faster OCR (max 1600px on longest side)
    max_dim = 1600
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        print(f"Resized to: {new_size}")
    
    return np.array(img)


# -------------------------------------------------
# Aggressive preprocessing for better OCR
# -------------------------------------------------
def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """Heavy preprocessing to maximize OCR accuracy"""
    print("Preprocessing image...")
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Increase contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Binarization - Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(binary, -1, kernel)
    
    # Morphological operations to clean up
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
    
    print("Preprocessing complete")
    return cleaned


# -------------------------------------------------
# Try multiple OCR strategies
# -------------------------------------------------
def run_ocr_multi_strategy(img: np.ndarray) -> tuple[str, float]:
    """Try multiple OCR approaches and return the best one"""
    results = []
    
    # Strategy 1: Original image with PSM 6 (uniform block of text)
    try:
        text1 = pytesseract.image_to_string(
            img,
            lang="eng+msa",
            config="--oem 3 --psm 6"
        )
        results.append(("PSM6", text1))
        print(f"Strategy 1 (PSM6): {len(text1)} chars")
    except Exception as e:
        print(f"Strategy 1 failed: {e}")
    
    # Strategy 2: Preprocessed with PSM 6
    try:
        preprocessed = preprocess_for_ocr(img)
        text2 = pytesseract.image_to_string(
            preprocessed,
            lang="eng+msa",
            config="--oem 3 --psm 6"
        )
        results.append(("Preprocessed+PSM6", text2))
        print(f"Strategy 2 (Preprocessed): {len(text2)} chars")
    except Exception as e:
        print(f"Strategy 2 failed: {e}")
    
    # Strategy 3: PSM 11 (sparse text, no layout)
    try:
        text3 = pytesseract.image_to_string(
            img,
            lang="eng+msa",
            config="--oem 3 --psm 11"
        )
        results.append(("PSM11", text3))
        print(f"Strategy 3 (PSM11): {len(text3)} chars")
    except Exception as e:
        print(f"Strategy 3 failed: {e}")
    
    # Strategy 4: PSM 3 (fully automatic)
    try:
        text4 = pytesseract.image_to_string(
            img,
            lang="eng+msa",
            config="--oem 3 --psm 3"
        )
        results.append(("PSM3", text4))
        print(f"Strategy 4 (PSM3): {len(text4)} chars")
    except Exception as e:
        print(f"Strategy 4 failed: {e}")
    
    if not results:
        raise HTTPException(status_code=500, detail="All OCR strategies failed")
    
    # Score each result
    def score_text(text: str) -> float:
        if not text.strip():
            return 0.0
        
        # Count alphanumeric characters
        alnum = sum(c.isalnum() for c in text)
        total = len(text)
        
        # Look for key road tax terms
        keywords = ["LESEN", "KENDERAAN", "MOTOR", "SEP", "WRU", "RM"]
        keyword_score = sum(1 for kw in keywords if kw in text.upper())
        
        # Prefer longer text with more keywords
        score = (alnum / total) * 0.5 + (keyword_score / len(keywords)) * 0.5
        return score
    
    # Find best result
    scored = [(method, text, score_text(text)) for method, text in results]
    scored.sort(key=lambda x: x[2], reverse=True)
    
    best_method, best_text, best_score = scored[0]
    print(f"Best strategy: {best_method} (score: {best_score:.2f})")
    
    return best_text, best_score


# -------------------------------------------------
# Enhanced plate extraction
# -------------------------------------------------
def extract_plate_number(text: str) -> str | None:
    """Extract Malaysian plate number from text"""
    t = text.upper()
    
    # Try to find plate patterns
    # Malaysian format: 1-3 letters + 1-4 digits + optional letter
    patterns = [
        r"\b([A-Z]{1,3}\s?\d{1,4}\s?[A-Z]?)\b",
    ]
    
    candidates = []
    for pattern in patterns:
        matches = re.findall(pattern, t)
        for match in matches:
            clean = re.sub(r'\s+', '', match)
            
            # Filter out false positives
            if re.match(r"^(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{4}$", clean):
                continue  # Date
            if clean.startswith(('RM', 'VEL', 'AB')):
                continue  # Amount or code
            if len(clean) < 3 or len(clean) > 8:
                continue  # Invalid length
            
            # Must have both letters and numbers
            if re.search(r'[A-Z]', clean) and re.search(r'\d', clean):
                candidates.append(clean)
    
    # Return first valid candidate
    return candidates[0] if candidates else None


# -------------------------------------------------
# Enhanced date extraction
# -------------------------------------------------
def extract_expiry_date(text: str) -> str | None:
    """Extract expiry date"""
    patterns = [
        r"\b(\d{1,2})\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*(\d{4})\b",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.upper())
        if match:
            day, month, year = match.groups()
            return f"{day.zfill(2)} {month} {year}"
    
    return None


# -------------------------------------------------
# Main OCR endpoint - SYNCHRONOUS
# -------------------------------------------------
@app.post("/ocr/extract")
def extract_text(payload: dict):
    """Extract raw text from image (synchronous)"""
    image_url = payload.get("image_url")
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url required")
    
    # Load image
    img = load_image_from_url(image_url)
    
    # Run multi-strategy OCR
    text, confidence = run_ocr_multi_strategy(img)
    
    return {
        "text": text.strip(),
        "confidence": round(confidence, 2),
        "char_count": len(text.strip())
    }


# -------------------------------------------------
# Main OCR endpoint - SYNCHRONOUS (original endpoint name)
# -------------------------------------------------
@app.post("/ocr/document")
def ocr_document(payload: dict):
    """Extract structured road tax data (synchronous) - ORIGINAL ENDPOINT"""
    image_url = payload.get("image_url")
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url required")
    
    # Load image
    img = load_image_from_url(image_url)
    
    # Run multi-strategy OCR
    text, confidence = run_ocr_multi_strategy(img)
    
    # Extract fields
    t = text.upper()
    
    # Check if it's a road tax document
    is_roadtax = "LESEN" in t and ("KENDERAAN" in t or "MOTOR" in t)
    
    if not is_roadtax:
        return {
            "document_type": "unknown",
            "confidence": 0.0,
            "raw_text": text.strip()
        }
    
    # Extract fields
    plate = extract_plate_number(text)
    expiry = extract_expiry_date(text)
    
    # Vehicle class
    vehicle_class = None
    if "SELAIN MOTOSIKAL" in t:
        vehicle_class = "SELAIN MOTOSIKAL"
    elif "MOTOSIKAL" in t:
        vehicle_class = "MOTOSIKAL"
    
    # Amount
    amount_match = re.search(r"RM\s?(\d+(?:\.\d{2})?)", t)
    amount = f"RM{amount_match.group(1)}" if amount_match else None
    
    # Location
    location = None
    if "SEMENANJUNG" in t:
        location = "SEMENANJUNG"
    if "PERSENDIRIAN" in t:
        location = "PERSENDIRIAN"
    
    # Receipt codes
    receipts = re.findall(r"\bVEL\d{5,}\b", t)
    
    return {
        "document_type": "malaysia_roadtax",
        "confidence": round(confidence, 2),
        "fields": {
            "plate_number": plate,
            "expiry_date": expiry,
            "vehicle_class": vehicle_class,
            "amount": amount,
            "location": location,
            "receipt_codes": receipts[:3] if receipts else None
        },
        "raw_text": text.strip()
    }


@app.post("/ocr/extract")
def extract_text(payload: dict):
    """Extract raw text from image (synchronous)"""
    image_url = payload.get("image_url")
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url required")
    
    # Load image
    img = load_image_from_url(image_url)
    
    # Run multi-strategy OCR
    text, confidence = run_ocr_multi_strategy(img)
    
    return {
        "text": text.strip(),
        "confidence": round(confidence, 2),
        "char_count": len(text.strip())
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "3.1.0",
        "tesseract": pytesseract.get_tesseract_version()
    }


@app.on_event("startup")
def check_tesseract():
    langs = pytesseract.get_languages(config="")
    print("Tesseract languages:", langs)
    if "eng" not in langs or "msa" not in langs:
        print("WARNING: Missing required language packs!")
    print("OCR service ready")