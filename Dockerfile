FROM python:3.11-slim-bullseye

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    curl gcc g++ libxrender1 libxext6 libgomp1 libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU + neural network potentials
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Application code
COPY app/ app/
COPY main.py .

# Non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/scratch /home/appuser/.cache && \
    chown -R appuser:appuser /app /home/appuser
USER appuser

ENV PORT=8032
ENV PYTHONUNBUFFERED=1
EXPOSE 8032

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8032/health || exit 1

CMD ["python", "main.py"]
