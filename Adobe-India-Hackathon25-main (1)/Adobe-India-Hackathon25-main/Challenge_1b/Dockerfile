# Use Python 3.9 slim image for CPU-only processing
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables to ensure CPU-only operation
ENV CUDA_VISIBLE_DEVICES=""
ENV NVIDIA_VISIBLE_DEVICES=""
ENV PYTORCH_CUDA_ALLOC_CONF=""

# Install system dependencies (excluding any NVIDIA/CUDA packages)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && apt-get purge -y '*cuda*' '*nvidia*' || true \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with explicit CPU-only flags
RUN pip install --no-cache-dir --index-url https://pypi.org/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy application code
COPY main.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set environment variables
ENV INPUT_DIR=/app/input
ENV OUTPUT_DIR=/app/output

# Run the application
CMD ["python", "main.py"]