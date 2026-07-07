FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml .
COPY requirements-docker.txt .

RUN pip install --no-cache-dir --no-deps -r requirements-docker.txt

RUN pip install --no-cache-dir --no-deps .

COPY src/ ./src/

COPY model/ ./model/

WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]