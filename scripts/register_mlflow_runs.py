from pathlib import Path

import mlflow


EXPERIMENT_NAME = "projet14_qwen3_medical_triage"
MLRUNS_DIR = Path("mlruns").resolve()

SFT_LOGS = Path("outputs/qwen3-sft-3epochs/training_logs.csv")
DPO_LOGS = Path("outputs/qwen3-dpo/dpo_training_logs.csv")


def log_artifact_if_exists(path: Path):
    if path.exists():
        mlflow.log_artifact(str(path))
    else:
        print(f"Fichier non trouvé, non enregistré : {path}")


def main():
    # MLflow va écrire ses fichiers dans le dossier local mlruns.
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Run SFT : on enregistre les métriques déjà obtenues dans le notebook 02.
    with mlflow.start_run(run_name="sft_lora_qwen3_3epochs"):
        mlflow.log_param("base_model", "Qwen/Qwen3-1.7B-Base")
        mlflow.log_param("method", "SFT LoRA")
        mlflow.log_param("epochs", 3)
        mlflow.log_param("seed", 42)
        mlflow.log_param("selected_checkpoint", "outputs/qwen3-sft-3epochs/checkpoint-4762")

        mlflow.log_metric("train_loss_epoch_1", 2.162260)
        mlflow.log_metric("validation_loss_epoch_1", 2.248884)
        mlflow.log_metric("train_loss_epoch_2", 1.909145)
        mlflow.log_metric("validation_loss_epoch_2", 2.204758)
        mlflow.log_metric("train_loss_epoch_3", 1.671062)
        mlflow.log_metric("validation_loss_epoch_3", 2.202288)
        mlflow.log_metric("test_loss_selected_checkpoint", 2.236565)

        log_artifact_if_exists(SFT_LOGS)

    # Run DPO : on enregistre les métriques déjà obtenues dans le notebook 04.
    with mlflow.start_run(run_name="dpo_lora_qwen3_1epoch"):
        mlflow.log_param("base_model", "Qwen/Qwen3-1.7B-Base")
        mlflow.log_param("starting_checkpoint", "outputs/qwen3-sft-3epochs/checkpoint-4762")
        mlflow.log_param("method", "DPO LoRA")
        mlflow.log_param("epochs", 1)
        mlflow.log_param("final_adapter", "outputs/qwen3-dpo")

        mlflow.log_metric("train_loss", 0.578294)
        mlflow.log_metric("validation_loss", 0.531719)
        mlflow.log_metric("validation_reward_accuracy", 0.81)
        mlflow.log_metric("validation_reward_margin", 1.695979)

        log_artifact_if_exists(DPO_LOGS)

    print("Runs MLflow enregistrés.")
    print(f"Interface locale : mlflow ui --backend-store-uri {MLRUNS_DIR}")


if __name__ == "__main__":
    main()
