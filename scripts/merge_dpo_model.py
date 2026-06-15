from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
DPO_ADAPTER_PATH = Path("outputs/qwen3-dpo")
MERGED_MODEL_PATH = Path("outputs/qwen3-dpo-merged")


def main():
    # Le tokenizer doit etre sauvegarde avec le modele final.
    print("Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    # On recharge le modele de base, puis on lui applique l'adapter DPO.
    print("Chargement du modele Qwen de base...")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # L'adapter DPO contient les poids entraines avec LoRA.
    print("Chargement de l'adapter LoRA DPO...")
    model = PeftModel.from_pretrained(
        base_model,
        DPO_ADAPTER_PATH,
        is_trainable=False,
    )

    # Le merge cree un modele complet, plus simple a charger en deploiement.
    print("Fusion Qwen + adapter LoRA DPO...")
    merged_model = model.merge_and_unload()

    print(f"Sauvegarde du modele merge dans : {MERGED_MODEL_PATH}")
    MERGED_MODEL_PATH.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(
        MERGED_MODEL_PATH,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(MERGED_MODEL_PATH)

    print("Merge termine.")


if __name__ == "__main__":
    main()
