import os

import httpx


MODEL_PATH = os.getenv("MODEL_PATH", "outputs/qwen3-dpo-merged")
MODEL_VERSION = os.getenv("MODEL_VERSION", "qwen3-dpo-merged-v1")
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "mock")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "qwen3-dpo-merged")


def build_triage_prompt(patient_text: str) -> str:
    return f"""Analyse le cas patient ci-dessous pour une demonstration de triage medical.

Regles obligatoires :
- Reponds uniquement en francais.
- Ne pose pas de diagnostic definitif.
- Ne prescris pas de traitement medicamenteux.
- Si le cas mentionne douleur thoracique, essoufflement, malaise, sueurs, deficit neurologique, confusion, detresse respiratoire ou saignement important, recommande les urgences / 15 / 112.
- Ne repete pas la question.
- Ne genere pas de texte apres la ligne Limite.

Cas patient :
{patient_text}

Format exact attendu, en 4 lignes :
Priorite :
Raison :
Conduite a tenir :
Limite :
"""


class InferenceEngine:
    def __init__(self) -> None:
        self.backend = INFERENCE_BACKEND
        self.model_version = MODEL_VERSION

    def generate_triage(
        self,
        patient_text: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        prompt = build_triage_prompt(patient_text)

        if self.backend == "mock":
            return self._mock_response()

        if self.backend in {"vllm", "vllm_http"}:
            return self._vllm_http_generate(prompt, max_tokens, temperature)

        raise ValueError(f"Backend d'inference non supporte: {self.backend}")

    def _mock_response(self) -> str:
        return (
            "Priorite : Demonstration.\n"
            "Raison : Reponse generee en mode mock pour tester l'API sans charger le modele.\n"
            "Conduite a tenir : Utiliser le backend vLLM pour une inference reelle.\n"
            "Limite : Cette reponse ne correspond pas a une evaluation medicale reelle."
        )

    def _vllm_http_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        payload = {
            "model": VLLM_MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant de triage medical prudent. "
                        "Reponds uniquement en francais, sans diagnostic definitif."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["Initialized", "Intialized", "\n\n\n"],
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{VLLM_BASE_URL}/v1/chat/completions", json=payload)
            response.raise_for_status()

        data = response.json()
        return clean_triage_response(data["choices"][0]["message"]["content"])


def clean_triage_response(text: str) -> str:
    for marker in ("Intialized", "Initialized", "Initial"):
        marker_index = text.find(marker)
        if marker_index != -1:
            text = text[:marker_index]
            break

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    fields = {
        "Priorite": "",
        "Raison": "",
        "Conduite a tenir": "",
        "Limite": "",
    }
    current_field: str | None = None

    field_aliases = {
        "Priorite": "Priorite",
        "Priorité": "Priorite",
        "Urgence": "Priorite",
        "Raison": "Raison",
        "Conduite": "Conduite a tenir",
        "Conduite a tenir": "Conduite a tenir",
        "Limite": "Limite",
    }

    for line in lines:
        matched_field = None
        for alias, field_name in field_aliases.items():
            if line.startswith(f"{alias} :") or line.startswith(f"{alias}:"):
                matched_field = field_name
                value = line.split(":", 1)[1].strip()
                fields[field_name] = value
                current_field = field_name
                break

        if matched_field is None and current_field and not fields[current_field]:
            fields[current_field] = line

    reason_lower = fields["Raison"].lower()
    severe_terms = (
        "douleur thoracique",
        "essoufflement",
        "sueurs",
        "malaise",
        "detresse respiratoire",
        "détresse respiratoire",
    )
    if any(term in reason_lower for term in severe_terms):
        fields["Priorite"] = fields["Priorite"] or "Urgence vitale possible"
        fields["Conduite a tenir"] = "Appeler le 15 ou le 112 sans attendre."
        fields["Limite"] = "Evaluation immediate par un professionnel de sante."

    noisy_terms = ("ontvangst", "intialized", "initialized")
    for field, value in fields.items():
        if any(term in value.lower() for term in noisy_terms):
            fields[field] = ""

    fallback = {
        "Priorite": "A evaluer",
        "Raison": "Signes cliniques a analyser avec prudence.",
        "Conduite a tenir": "Demander un avis medical adapte au contexte.",
        "Limite": "Prototype academique, ne remplace pas une decision clinique.",
    }

    return "\n".join(
        f"{field} : {fields[field] or fallback[field]}"
        for field in ("Priorite", "Raison", "Conduite a tenir", "Limite")
    )
