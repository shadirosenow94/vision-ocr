FROM python:3.11-slim

# System dependencies for OCR + OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-msa \
    libgl1 \
    libglib2.0-0 \
    libheif1 \
    libheif-dev \
    && rm -rf /var/lib/apt/lists/*

# Ensure consistent OCR language behavior
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
