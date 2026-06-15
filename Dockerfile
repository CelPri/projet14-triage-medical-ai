FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV INFERENCE_BACKEND=mock
ENV MODEL_BASE=Qwen/Qwen3-1.7B-Base
ENV LORA_ADAPTER=outputs/qwen3-dpo
ENV MODEL_VERSION=qwen3-dpo-v1

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app ./app
COPY outputs/qwen3-dpo ./outputs/qwen3-dpo

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
