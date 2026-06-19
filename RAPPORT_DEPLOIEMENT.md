# Rapport final - Deploiement du prototype de triage medical

## 1. Objectif

Ce projet vise a deployer un prototype d'assistant de triage medical base sur un modele Qwen fine-tune, aligne par DPO, puis merge pour l'inference.

Le deploiement cloud repose sur deux composants :

- vLLM pour servir le modele merge avec une API compatible OpenAI ;
- FastAPI pour exposer une API metier de triage et une interface web de demonstration.

Endpoint de demonstration :

```text
http://34.56.9.145:8080/
```

Endpoints testes :

```text
GET  /
GET  /docs
GET  /health
GET  /metadata
GET  /audit
POST /triage
```

## 2. Architecture de deploiement

```text
Utilisateur / navigateur
        |
        v
FastAPI - port 8080
        |
        v
vLLM OpenAI-compatible - port 8000
        |
        v
Modele merge : /home/cenouv43/outputs/qwen3-dpo-merged
```

Le port `8080` est ouvert dans Google Cloud pour acceder a l'interface et a l'API FastAPI.
Le port `8000` sert a vLLM et reste utilise comme service interne par FastAPI.

## 3. Modele deploye

Modele final :

```text
qwen3-dpo-merged
```

Chemin sur la VM :

```text
/home/cenouv43/outputs/qwen3-dpo-merged
```

Fichiers principaux :

```text
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
chat_template.jinja
```

Le modele a ete charge avec succes par vLLM. L'endpoint `/v1/models` a retourne le modele `qwen3-dpo-merged`.

## 4. CI/CD

Le pipeline GitHub Actions execute :

1. installation des dependances ;
2. tests API en mode `mock` ;
3. build de l'image Docker FastAPI ;
4. push de l'image vers GitHub Container Registry ;
5. deploiement SSH sur la VM GPU ;
6. redemarrage du conteneur FastAPI ;
7. verification de `/health`.

Le mode `mock` est conserve pour permettre les tests CI/CD sans GPU.
Le mode `vllm` est utilise sur la VM GPU pour l'inference reelle.

Statut final :

```text
GitHub Actions : vert
Job deploy : vert
```

## 5. Metriques de performance

Test realise sur 5 appels successifs a `/triage` avec le cas :

```text
Patient de 62 ans avec douleur thoracique brutale, sueurs et essoufflement.
```

Latences observees :

```text
2655.20 ms
2657.16 ms
876.57 ms
2668.14 ms
2660.38 ms
```

Synthese :

```text
Moyenne approximative : 2303.49 ms
Mediane : 2657.16 ms
Latence typique observee : environ 2.6 s
```

Interpretation :

- la latence est acceptable pour une demonstration cloud ;
- la majorite des appels se situe autour de 2.6 secondes ;
- un appel plus rapide a ete observe a 0.88 seconde ;
- les temps incluent l'appel FastAPI, l'appel HTTP vers vLLM et la generation modele.

## 6. Robustesse

### Cas urgent

Entree :

```text
Patient de 62 ans avec douleur thoracique brutale, sueurs et essoufflement.
```

Sortie obtenue :

```text
Priorite : 1
Raison : douleur thoracique brutale, sueurs et essoufflement
Conduite a tenir : Appeler le 15 ou le 112 sans attendre.
Limite : Evaluation immediate par un professionnel de sante.
```

Le comportement est prudent pour un cas potentiellement urgent.

### Cas non urgent

Entree :

```text
Patient de 25 ans avec rhume leger depuis deux jours, sans fievre ni essoufflement.
```

Sortie obtenue :

```text
Priorite : A evaluer
Raison : Signes cliniques a analyser avec prudence.
Conduite a tenir : Demander un avis medical adapte au contexte.
Limite : Prototype academique, ne remplace pas une decision clinique.
```

### Entree invalide

Entree :

```json
{"patient_text":"mal"}
```

Resultat :

```text
HTTP 422 Unprocessable Entity
```

L'API rejette les entrees trop courtes grace a la validation Pydantic.

## 7. Audit et tracabilite

Chaque appel a `/triage` genere une trace dans :

```text
logs/audit.jsonl
```

L'endpoint `/audit` expose les derniers evenements.

Champs journalises :

```text
request_id
timestamp
endpoint
backend
model_version
latency_ms
input_length
```

Le texte patient complet n'est pas stocke dans les logs, afin de limiter les risques de confidentialite.

Exemple de trace :

```json
{
  "request_id": "eb3b439a-9fd1-40c8-ab2c-ef01717e30c2",
  "timestamp": "2026-06-16T14:57:50.939785+00:00",
  "endpoint": "/triage",
  "backend": "vllm",
  "model_version": "qwen3-dpo-merged-v1",
  "latency_ms": 2659.07,
  "input_length": 75
}
```

## 8. Limites identifiees

Le prototype ne remplace pas un professionnel de sante.

Limites principales :

- le modele peut produire des reponses imparfaites ou instables ;
- un post-traitement est necessaire pour stabiliser la sortie ;
- l'endpoint public n'est pas encore protege par authentification ;
- le trafic n'est pas encore expose en HTTPS ;
- aucune validation clinique externe n'a ete realisee ;
- la supervision humaine reste indispensable.

## 9. Roadmap de deploiement

Ameliorations prioritaires :

1. Ajouter une authentification sur l'endpoint public.
2. Passer l'exposition en HTTPS.
3. Restreindre les IP autorisees dans les regles firewall.
4. Ajouter un monitoring applicatif et systeme.
5. Ajouter des quotas et limites de debit.
6. Versionner le modele et les donnees de deploiement.
7. Ajouter des tests cliniques plus larges avec cas urgents, non urgents et ambigus.
8. Ajouter une validation humaine systematique avant usage reel.
9. Mettre en place un pipeline de reentrainement et reevaluation.
10. Tester un modele plus grand si le budget GPU le permet.

## 10. Checklist go / no-go

### Go

- API FastAPI disponible sur le cloud ;
- vLLM charge le modele merge ;
- `/triage` repond avec le backend `vllm` ;
- logs d'audit actifs ;
- CI/CD GitHub Actions vert ;
- deploiement automatique teste ;
- latence mesuree et documentee ;
- cas urgent oriente vers le 15 / 112 ;
- entree invalide rejetee.

### No-go

- endpoint expose sans controle dans un contexte reel ;
- absence de supervision clinique ;
- hallucinations ou recommandations non prudentes ;
- absence de monitoring ;
- absence de HTTPS ;
- absence de politique de confidentialite et de retention des logs.

## 11. Commandes utiles de reprise

Relancer la VM, puis se connecter :

```bash
gcloud compute instances start projet14-triage-vm --zone=us-central1-a
gcloud compute ssh projet14-triage-vm --zone=us-central1-a
```

Relancer vLLM :

```bash
docker start qwen-vllm
```

Verifier vLLM :

```bash
curl http://127.0.0.1:8000/v1/models
```

Verifier FastAPI :

```bash
curl http://127.0.0.1:8080/health
```

Arreter la VM apres demonstration :

```bash
gcloud compute instances stop projet14-triage-vm --zone=us-central1-a
```
