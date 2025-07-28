FROM --platform=linux/amd64 python:3.10

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir PyMuPDF==1.23.14

# Copy the processing script
COPY process_pdfs_improved.py .

# Run the script
CMD ["python", "process_pdfs_improved.py"] 
