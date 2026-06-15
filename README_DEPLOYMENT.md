# Déploiement du prototype

Ce document décrit le déploiement prévu pour l'API de triage médical.

L'objectif est d'exposer le modèle fine-tuné derrière une API FastAPI, avec deux modes :

- `mock` : utilisé pour les tests locaux, Docker et GitHub Actions ;
- `vllm` : utilisé pour l'inférence réelle sur un environnement GPU.

## Architecture

```text
Utilisateur
   |
   v
Endpoint FastAPI /triage
   |
   v
InferenceEngine
   |
   v
vLLM
   |
   v
Qwen/Qwen3-1.7B-Base + adapter LoRA DPO
```

Le modèle de base reste `Qwen/Qwen3-1.7B-Base`.
L'adaptation médicale est chargée avec l'adapter LoRA entraîné :

```text
outputs/qwen3-dpo
```

## Endpoints disponibles

L'API expose trois endpoints principaux :

- `GET /health` : vérifie que l'API répond ;
- `GET /metadata` : retourne la version du modèle et le backend utilisé ;
- `POST /triage` : génère une réponse de triage à partir d'un cas patient.

## Mode mock

Le mode `mock` permet de tester l'API sans charger le modèle.

Il est utilisé pour :

- vérifier que FastAPI fonctionne ;
- lancer les tests unitaires ;
- construire l'image Docker ;
- exécuter la CI GitHub Actions sans GPU.

Commande locale :

```powershell
$env:INFERENCE_BACKEND="mock"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Commande Docker :

```powershell
docker build -t projet14-triage-api .
docker run --rm -p 8000:8000 -v "${PWD}\logs:/app/logs" projet14-triage-api
```

Dans ce mode, la réponse n'est pas une vraie réponse médicale.
Elle sert uniquement à tester le fonctionnement technique de l'API.

## Mode vLLM

Le mode `vllm` est prévu pour un environnement avec GPU.

Il permet de charger :

- le modèle de base `Qwen/Qwen3-1.7B-Base` ;
- l'adapter LoRA DPO situé dans `outputs/qwen3-dpo`.

Exemple de lancement :

```powershell
$env:INFERENCE_BACKEND="vllm"
$env:MODEL_BASE="Qwen/Qwen3-1.7B-Base"
$env:LORA_ADAPTER="outputs/qwen3-dpo"
$env:MODEL_VERSION="qwen3-dpo-v1"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Ce mode doit être exécuté sur une machine compatible avec vLLM.
Sur Windows classique, vLLM n'est pas le choix le plus simple.
Le déploiement réel est plutôt prévu sur Linux avec GPU NVIDIA.

## Variables d'environnement

| Variable | Rôle | Valeur utilisée |
|---|---|---|
| `INFERENCE_BACKEND` | Choix du backend d'inférence | `mock` ou `vllm` |
| `MODEL_BASE` | Modèle de base Hugging Face | `Qwen/Qwen3-1.7B-Base` |
| `LORA_ADAPTER` | Chemin vers l'adapter LoRA | `outputs/qwen3-dpo` |
| `MODEL_VERSION` | Nom de version exposé par l'API | `qwen3-dpo-v1` |
| `AUDIT_LOG_PATH` | Chemin du fichier de logs | `logs/audit.jsonl` |

## CI/CD

Le pipeline GitHub Actions vérifie automatiquement le projet à chaque push.

Il effectue les étapes suivantes :

```text
Push GitHub
   |
   v
GitHub Actions
   |
   v
Installation des dépendances API
   |
   v
Tests en mode mock
   |
   v
Build Docker
```

Le mode mock est volontairement utilisé dans la CI.
GitHub Actions ne fournit pas de GPU adapté pour charger Qwen avec vLLM.

Le déploiement pilote complet doit être effectué sur un serveur GPU.

## Traçabilité

Chaque appel à `/triage` écrit une ligne dans :

```text
logs/audit.jsonl
```

Les informations enregistrées sont :

- identifiant de requête ;
- horodatage ;
- endpoint appelé ;
- backend utilisé ;
- version du modèle ;
- latence ;
- longueur de l'entrée.

Le texte médical complet n'est pas enregistré dans les logs afin de limiter les risques liés aux données sensibles.

## Mesure de latence

La réponse de l'API contient un champ :

```text
latency_ms
```

Ce champ permet de suivre le temps de réponse.

En mode mock, la latence vérifie surtout le fonctionnement technique de l'API.
En mode vLLM, elle servira à mesurer la performance réelle du modèle.

## Limites d'usage

Ce prototype ne remplace pas un professionnel de santé.

Le modèle :

- ne doit pas poser de diagnostic définitif ;
- ne doit pas prescrire de traitement médicamenteux ;
- doit recommander une conduite prudente ;
- doit orienter vers les urgences en cas de signes graves.

Les réponses générées doivent rester supervisées avant tout usage réel.

## Checklist go / no-go

Avant un déploiement pilote réel, il faut vérifier :

- tests API réussis ;
- image Docker construite ;
- adapter LoRA disponible sur le serveur ;
- backend `vllm` testé sur GPU ;
- latence mesurée ;
- logs d'audit actifs ;
- accès endpoint protégé ;
- limites d'usage documentées ;
- contrôles de sécurité clinique validés.

