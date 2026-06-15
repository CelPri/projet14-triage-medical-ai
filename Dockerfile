FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV INFERENCE_BACKEND=mock
ENV MODEL_PATH=outputs/qwen3-dpo-merged
ENV MODEL_VERSION=qwen3-dpo-merged-v1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# vLLM est utile seulement quand le conteneur tourne sur la VM GPU.
# En mode mock, il n'est pas utilise.
RUN pip install --no-cache-dir vllm==0.6.6.post1

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
