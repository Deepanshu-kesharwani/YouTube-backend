# -------------------------------
# Base image
# -------------------------------
FROM python:3.11-slim

# -------------------------------
# Environment variables
# -------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------------------------------
# System dependencies
# -------------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Working directory
# -------------------------------
WORKDIR /app

# -------------------------------
# Install Python dependencies
# -------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Copy source code
# -------------------------------
COPY . .

# -------------------------------
# Expose port
# -------------------------------
EXPOSE 8000

# -------------------------------
# Run FastAPI
# -------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
