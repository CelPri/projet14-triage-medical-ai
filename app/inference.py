import os


MODEL_BASE = os.getenv("MODEL_BASE", "Qwen/Qwen3-1.7B-Base")
LORA_ADAPTER = os.getenv("LORA_ADAPTER", "outputs/qwen3-dpo")
MODEL_VERSION = os.getenv("MODEL_VERSION", "qwen3-dpo-v1")
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "mock")


def build_triage_prompt(patient_text: str) -> str:
    return f"""Tu es un assistant de triage médical.
Tu ne poses pas de diagnostic définitif.
Tu ne prescris pas de traitement médicamenteux.
Tu aides uniquement à estimer le niveau d'urgence et à recommander une conduite prudente.

Cas patient :
{patient_text}

Réponds en français avec ce format :
Priorité :
Raison :
Conduite à tenir :
Limite :
"""


class InferenceEngine:
    def __init__(self) -> None:
        self.backend = INFERENCE_BACKEND
        self.model_version = MODEL_VERSION
        self.model = None

        if self.backend == "vllm":
            self._load_vllm()

    def _load_vllm(self) -> None:
        from vllm import LLM

        self.model = LLM(
            model=MODEL_BASE,
            enable_lora=True,
        )

    def generate_triage(
        self,
        patient_text: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        prompt = build_triage_prompt(patient_text)

        if self.backend == "mock":
            return self._mock_response(patient_text)

        return self._vllm_generate(prompt, max_tokens, temperature)

    def _mock_response(self, patient_text: str) -> str:
        return (
            "Priorité : Démonstration.\n"
            "Raison : Réponse générée en mode mock pour tester l'API sans charger le modèle.\n"
            "Conduite à tenir : Utiliser le backend vLLM pour une inférence réelle.\n"
            "Limite : Cette réponse ne correspond pas à une évaluation médicale réelle."
        )

    def _vllm_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )

        outputs = self.model.generate(
            [prompt],
            sampling_params,
            lora_request=LoRARequest("qwen3-dpo", 1, LORA_ADAPTER),
        )

        return outputs[0].outputs[0].text.strip()
