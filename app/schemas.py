from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    backend: str
    model_version: str


class MetadataResponse(BaseModel):
    model_path: str
    model_version: str
    training_method: str
    backend: str
    vllm_base_url: str
    vllm_model_name: str
    limitation: str


class TriageRequest(BaseModel):
    patient_text: str = Field(..., min_length=5)
    max_tokens: int = Field(default=256, ge=32, le=1024)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    request_id: Optional[str] = None


class TriageResponse(BaseModel):
    request_id: str
    response: str
    model_version: str
    backend: str
    latency_ms: float
    timestamp: str
