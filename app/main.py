import time
import uuid

from fastapi import FastAPI

from app.audit import utc_now, write_audit_log
from app.inference import INFERENCE_BACKEND, MODEL_PATH, MODEL_VERSION, InferenceEngine
from app.schemas import HealthResponse, MetadataResponse, TriageRequest, TriageResponse


app = FastAPI(title="CHSA Triage AI API", version="0.1.0")

# Le moteur est cree au demarrage de l'application.
# En mode mock, aucun modele lourd n'est charge.
engine = InferenceEngine()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        backend=INFERENCE_BACKEND,
        model_version=MODEL_VERSION,
    )


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    return MetadataResponse(
        model_path=MODEL_PATH,
        model_version=MODEL_VERSION,
        training_method="SFT LoRA + DPO + merge",
        backend=INFERENCE_BACKEND,
        limitation="POC academique : ne remplace pas un avis medical ou une decision clinique.",
    )


@app.post("/triage", response_model=TriageResponse)
def triage(request: TriageRequest) -> TriageResponse:
    start = time.perf_counter()
    request_id = request.request_id or str(uuid.uuid4())

    response_text = engine.generate_triage(
        patient_text=request.patient_text,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    timestamp = utc_now()

    # Log minimal pour garder une trace sans stocker le texte patient complet.
    write_audit_log(
        {
            "request_id": request_id,
            "timestamp": timestamp,
            "endpoint": "/triage",
            "backend": INFERENCE_BACKEND,
            "model_version": MODEL_VERSION,
            "latency_ms": latency_ms,
            "input_length": len(request.patient_text),
        }
    )

    return TriageResponse(
        request_id=request_id,
        response=response_text,
        model_version=MODEL_VERSION,
        backend=INFERENCE_BACKEND,
        latency_ms=latency_ms,
        timestamp=timestamp,
    )
