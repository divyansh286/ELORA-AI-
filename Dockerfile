FROM python:3.10-slim

# ---------------------------------------------------------
# System dependencies (OpenCV, FAISS, Tesseract, build tools)
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    libffi-dev \
    tesseract-ocr \
    ffmpeg \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*


# ---------------------------------------------------------
# Work directory
# ---------------------------------------------------------
WORKDIR /app


# ---------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel


# Install PyTorch CPU (needed for transformers + diffusers)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Now install remaining project dependencies
RUN pip install --no-cache-dir -r requirements.txt


# ---------------------------------------------------------
# Copy the project
# ---------------------------------------------------------
COPY . /app


# ---------------------------------------------------------
# Expose API port
# ---------------------------------------------------------
EXPOSE 8000


# ---------------------------------------------------------
# Start the API server
# ---------------------------------------------------------
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
