import os


MODEL_PATH = os.getenv("MODEL_PATH", "outputs/qwen3-dpo-merged")
MODEL_VERSION = os.getenv("MODEL_VERSION", "qwen3-dpo-merged-v1")
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "mock")


def build_triage_prompt(patient_text: str) -> str:
    return f"""Tu es un assistant de triage medical.
Tu ne poses pas de diagnostic definitif.
Tu ne prescris pas de traitement medicamenteux.
Tu aides uniquement a estimer le niveau d'urgence et a recommander une conduite prudente.

Cas patient :
{patient_text}

Reponds en francais avec ce format :
Priorite :
Raison :
Conduite a tenir :
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

        # En mode vLLM, on charge directement le modele merge.
        self.model = LLM(model=MODEL_PATH)

    def generate_triage(
        self,
        patient_text: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        prompt = build_triage_prompt(patient_text)

        if self.backend == "mock":
            return self._mock_response()

        return self._vllm_generate(prompt, max_tokens, temperature)

    def _mock_response(self) -> str:
        return (
            "Priorite : Demonstration.\n"
            "Raison : Reponse generee en mode mock pour tester l'API sans charger le modele.\n"
            "Conduite a tenir : Utiliser le backend vLLM pour une inference reelle.\n"
            "Limite : Cette reponse ne correspond pas a une evaluation medicale reelle."
        )

    def _vllm_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )

        outputs = self.model.generate([prompt], sampling_params)

        return outputs[0].outputs[0].text.strip()
