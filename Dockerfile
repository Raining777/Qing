FROM python:3.12-slim

WORKDIR /app

# System deps for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY app/ ./app/

# Data dirs
RUN mkdir -p /app/data/uploads /app/data/chroma /app/data/checkpoints /app/data/models

# .env placeholder
COPY .env.example ./.env

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0

CMD ["python", "-m", "app.main"]
