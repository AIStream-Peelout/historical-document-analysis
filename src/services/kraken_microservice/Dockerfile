FROM python:3.9-slim

# Install system dependencies required by Kraken and Pillow/OpenCV etc.
RUN apt-get update && apt-get install -y \
    libgeos-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the microservice source code
COPY main.py .

# Expose the API port (8002)
EXPOSE 8002

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
