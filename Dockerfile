FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV INFERENCE_BACKEND=mock
ENV MODEL_PATH=outputs/qwen3-dpo-merged
ENV MODEL_VERSION=qwen3-dpo-merged-v1
ENV VLLM_BASE_URL=http://127.0.0.1:8000
ENV VLLM_MODEL_NAME=qwen3-dpo-merged

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
