# Kraken 7.0.3 test service (kraken-service:k7). Orli commit 7996f29 pins kraken~=7.0.2.
# Run on a spare host port, e.g. -p 8003:8002, --cpus 6, KRAKEN_SEGMENTER=default|<blla.mlmodel>|<orli.safetensors>.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git libgeos-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY main.py .
ENV KRAKEN_SEGMENTER=default KRAKEN_THREADS=6
EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
