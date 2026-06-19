# Projet 14 - Assistant de triage medical

Ce projet adapte le modele `Qwen/Qwen3-1.7B-Base` pour un cas d'usage de triage medical.

Le travail couvre :

- preparation des datasets SFT et DPO ;
- fine-tuning supervise avec LoRA ;
- alignement par preferences avec DPO ;
- suivi des metriques avec logs, checkpoints et MLflow ;
- API FastAPI ;
- conteneur Docker ;
- pipeline GitHub Actions ;
- preparation du deploiement avec vLLM.

## Installation

Creer ou activer l'environnement Python, puis installer les dependances :

```powershell
python -m pip install -r requirements.txt
```

## Donnees et entrainement

Les donnees finales sont dans `data/`.

Les notebooks principaux sont :

- `notebooks/dataset_building/build_datasets.ipynb` : construction des datasets SFT et DPO ;
- `notebooks/training/01_sft_lora_test_pipeline.ipynb` : test court du pipeline SFT ;
- `notebooks/training/02_sft_lora_full_run.ipynb` : entrainement SFT complet ;
- `notebooks/training/03_sft_test_evaluation.ipynb` : evaluation des checkpoints SFT ;
- `notebooks/training/04_dpo_training.ipynb` : entrainement DPO.

## Dataset versionne

Le dataset final est aussi versionne au format JSONL dans un depot Hugging Face :

[PCelia/projet14-medical-triage-dataset](https://huggingface.co/datasets/PCelia/projet14-medical-triage-dataset)

Le depot contient les splits SFT, les splits DPO et le jeu d'evaluation de securite clinique.
Sa dataset card documente les schemas, les langues, les transformations, les sources et les limites d'usage.

Le depot est prive afin de respecter les conditions de redistribution des datasets sources.

## Resultats principaux

SFT LoRA :

- epoch 1 : train loss 2.162260, validation loss 2.248884 ;
- epoch 2 : train loss 1.909145, validation loss 2.204758 ;
- epoch 3 : train loss 1.671062, validation loss 2.202288 ;
- meilleur checkpoint retenu : `outputs/qwen3-sft-3epochs/checkpoint-4762` ;
- test loss du checkpoint retenu : 2.236565.

DPO :

- train loss : 0.578294 ;
- validation loss : 0.531719 ;
- validation reward accuracy : 0.81 ;
- validation reward margin : 1.695979 ;
- adapter final : `outputs/qwen3-dpo`.

## MLflow

Les metriques SFT et DPO peuvent etre enregistrees dans MLflow sans relancer l'entrainement :

```powershell
python scripts/register_mlflow_runs.py
python -m mlflow ui --backend-store-uri .\mlruns
```

Interface locale :

```text
http://127.0.0.1:5000
```

## Modele final merge

Pendant l'entrainement, LoRA permet de limiter l'empreinte GPU.
Pour simplifier le deploiement, l'adapter DPO est fusionne avec le modele Qwen de base.

Commande :

```powershell
python scripts/merge_dpo_model.py
```

Sortie :

```text
outputs/qwen3-dpo-merged
```

Le modele de deploiement est donc :

```text
outputs/qwen3-dpo-merged
```

## API locale

Par defaut, l'API demarre en mode `mock`.
Ce mode sert a tester l'API sans charger le modele.

```powershell
$env:INFERENCE_BACKEND="mock"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Documentation interactive :

```text
http://127.0.0.1:8000/docs
```

Endpoints :

- `GET /` : interface web minimale de demonstration ;
- `GET /health` : verification simple de l'API ;
- `GET /metadata` : informations sur le modele et le backend ;
- `POST /triage` : generation d'une reponse de triage.
- `GET /audit` : dernieres traces d'interaction sans texte patient complet.

## Docker

Une seule image Docker est utilisee pour le projet.
Elle peut tourner en mode `mock` pour les tests ou en mode `vllm` sur une VM GPU.

```powershell
docker build -t projet14-triage-api .
docker run --rm -p 8080:8080 -v "${PWD}\logs:/app/logs" projet14-triage-api
```

## vLLM

Le deploiement GPU utilise deux services :

- vLLM sert le modele merge via une API compatible OpenAI sur le port `8000` ;
- FastAPI expose l'interface de demonstration et l'endpoint metier `/triage` sur le port `8080`.

Lancement vLLM sur la VM GPU :

```bash
docker run -d --name qwen-vllm \
  --gpus all \
  -p 8000:8000 \
  -v ~/outputs/qwen3-dpo-merged:/model \
  vllm/vllm-openai:latest \
  --model /model \
  --served-model-name qwen3-dpo-merged \
  --dtype auto \
  --max-model-len 2048
```

Test direct de vLLM :

```bash
curl http://localhost:8000/v1/models
```

Lancement FastAPI contre vLLM :

```bash
docker build -t projet14-triage-api .
docker run -d --name triage-api \
  --network host \
  -e INFERENCE_BACKEND=vllm \
  -e VLLM_BASE_URL=http://127.0.0.1:8000 \
  -e VLLM_MODEL_NAME=qwen3-dpo-merged \
  -v ~/logs:/app/logs \
  projet14-triage-api
```

Les tests GitHub Actions restent en mode `mock`, car l'environnement CI ne dispose pas de GPU.
L'image Docker FastAPI n'inclut pas vLLM : elle appelle le serveur vLLM par HTTP.

## CI/CD

Le pipeline GitHub Actions execute :

```text
Push GitHub
   |
   v
Installation des dependances
   |
   v
Tests API en mode mock
   |
   v
Build et push de l'image Docker vers GHCR
   |
   v
Deploiement SSH optionnel vers une VM GPU
```

Le deploiement pilote reel doit etre lance sur une machine GPU avec Docker et NVIDIA runtime.

## Tracabilite

Chaque appel a `/triage` ecrit une ligne dans :

```text
logs/audit.jsonl
```

Les logs contiennent :

- identifiant de requete ;
- horodatage ;
- endpoint appele ;
- backend utilise ;
- version du modele ;
- latence ;
- longueur de l'entree.

Le texte medical complet n'est pas stocke dans les logs.

## Limites

Ce prototype ne remplace pas un professionnel de sante.

Le modele :

- ne doit pas poser de diagnostic definitif ;
- ne doit pas prescrire de traitement medicamenteux ;
- doit recommander une conduite prudente ;
- doit orienter vers les urgences en cas de signes graves.

## Checklist go / no-go

Avant un deploiement pilote reel :

- tests API reussis ;
- image Docker construite ;
- modele merge disponible ;
- backend `vllm` teste sur GPU ;
- latence mesuree ;
- logs d'audit actifs ;
- acces endpoint protege ;
- limites d'usage documentees ;
- controles de securite clinique valides.
