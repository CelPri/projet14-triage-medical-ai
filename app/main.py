import time
import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.audit import read_audit_log, utc_now, write_audit_log
from app.inference import (
    INFERENCE_BACKEND,
    MODEL_PATH,
    MODEL_VERSION,
    VLLM_BASE_URL,
    VLLM_MODEL_NAME,
    InferenceEngine,
)
from app.schemas import HealthResponse, MetadataResponse, TriageRequest, TriageResponse


app = FastAPI(title="CHSA Triage AI API", version="0.1.0")

# Le moteur est cree au demarrage de l'application.
# En mode mock, aucun modele lourd n'est charge.
engine = InferenceEngine()


@app.get("/", response_class=HTMLResponse)
def demo_page() -> str:
    return """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CHSA Triage AI</title>
  <style>
    :root { color-scheme: light; font-family: Arial, sans-serif; }
    body { margin: 0; background: #f7f9fb; color: #17202a; }
    main { max-width: 920px; margin: 0 auto; padding: 32px 20px; }
    h1 { margin: 0 0 8px; font-size: 30px; }
    p { color: #4a5568; line-height: 1.5; }
    textarea { width: 100%; min-height: 150px; padding: 12px; border: 1px solid #b8c2cc; border-radius: 6px; font: inherit; box-sizing: border-box; }
    button { margin-top: 12px; padding: 10px 16px; border: 0; border-radius: 6px; background: #1769aa; color: white; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .65; cursor: wait; }
    pre { white-space: pre-wrap; background: white; border: 1px solid #d6dde5; border-radius: 6px; padding: 16px; min-height: 120px; }
    .meta { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }
    .pill { background: #e8f1fa; border: 1px solid #c9dced; border-radius: 999px; padding: 6px 10px; font-size: 14px; }
  </style>
</head>
<body>
  <main>
    <h1>CHSA Triage AI</h1>
    <p>Prototype academique de triage medical. La reponse ne remplace pas un avis medical.</p>
    <div class="meta">
      <span class="pill">FastAPI: /triage</span>
      <span class="pill">vLLM: OpenAI-compatible</span>
      <span class="pill">Audit: /audit</span>
    </div>
    <textarea id="patient">Patient de 62 ans avec douleur thoracique brutale, essoufflement et sueurs.</textarea>
    <button id="submit">Analyser</button>
    <h2>Reponse</h2>
    <pre id="result">En attente...</pre>
  </main>
  <script>
    const button = document.querySelector("#submit");
    const result = document.querySelector("#result");
    button.addEventListener("click", async () => {
      button.disabled = true;
      result.textContent = "Analyse en cours...";
      try {
        const response = await fetch("/triage", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            patient_text: document.querySelector("#patient").value,
            max_tokens: 180,
            temperature: 0.2
          })
        });
        const data = await response.json();
        result.textContent = response.ok
          ? `${data.response}\\n\\nLatence: ${data.latency_ms} ms\\nRequete: ${data.request_id}`
          : JSON.stringify(data, null, 2);
      } catch (error) {
        result.textContent = String(error);
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


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
        vllm_base_url=VLLM_BASE_URL,
        vllm_model_name=VLLM_MODEL_NAME,
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


@app.get("/audit")
def audit(limit: int = 50) -> dict[str, Any]:
    bounded_limit = max(1, min(limit, 200))
    events = read_audit_log(limit=bounded_limit)
    return {"count": len(events), "events": events}
