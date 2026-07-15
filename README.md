# vLLM — Qwen3.6-27B-AWQ (serveur d'inférence)

Serveur d'inférence haute performance pour le modèle **QuantTrio/Qwen3.6-27B-AWQ**,
déployé via [vLLM](https://github.com/vllm-project/vllm) en conteneur Docker.

> **Objectif atteint :** contexte maximal (256 K tokens) **100 % GPU, 0 % CPU** —
> aucun spillover du KV cache vers la RAM, débit soutenu ~37 tok/s sur 2× RTX 3090.

---

## 📋 Sommaire

- [Vue d'ensemble](#vue-densemble)
- [Spécifications matérielles & modèle](#spécifications-matérielles--modèle)
- [Architecture de la solution](#architecture-de-la-solution)
- [Prérequis](#prérequis)
- [Installation & lancement](#installation--lancement)
- [Configuration détaillée](#configuration-détaillée)
- [Optimisations (100 % GPU, max contexte)](#optimisations-100--gpu-max-contexte)
- [Utilisation de l'API](#utilisation-de-lapi)
- [Performances mesurées](#performances-mesurées)
- [Dépannage](#dépannage)
- [Notes & limitations](#notes--limitations)

---

## Vue d'ensemble

Ce dépôt fournit un stack d'inférence prêt à l'emploi pour servir un modèle de
langage de 27 milliards de paramètres (quantifié AWQ 4-bit) avec :

- **Contexte maximal** de `262 144` tokens (256 K) — la limite native du modèle.
- **Exécution 100 % GPU** : poids + KV cache intégralement en VRAM, aucun recours
  au CPU pour le cache (pas de `--kv-offloading`, pas de `--swap-space`).
- **Tensor Parallelism = 2** sur deux RTX 3090 (PCIe, sans NVLink).
- **Sampling et attention sur GPU** (FlashInfer) — zéro pré/post-traitement sur CPU.
- **Modèle hybride** (Mamba linear-attention + full-attention) qui rend le contexte
  256 K viable en VRAM.

---

## Spécifications matérielles & modèle

| Élément | Valeur |
|---|---|
| GPU | 2 × NVIDIA RTX 3090 (compute capability **8.6**) |
| VRAM | 24 Go × 2 = **48 Go** total |
| Interconnexion | PCIe (pas de NVLink) |
| Modèle | `QuantTrio/Qwen3.6-27B-AWQ` |
| Architecture | `Qwen3_5ForConditionalGeneration` (hybride Mamba + GDN linear-attention) |
| Quantification | AWQ 4-bit (`awq_marlin` kernel) |
| Contexte max | `262 144` tokens (`max_position_embeddings`) |
| vLLM | `0.22.1+7b9cb5b7.dev` (image `nvcr.io/nvidia/vllm:26.06-py3`) |

**Pourquoi un contexte 256 K tient-il en VRAM ?**
Le modèle est **hybride** : sur 64 couches, seules **16** sont en full-attention
(les 48 autres utilisent une linear-attention de type Mamba qui n'alloue pas de
KV cache classique). Le KV cache full-attention représente donc ~64 KiB/token →
~17 GiB à 262 K tokens, largement dans la VRAM libre (~30 Go sur 2 GPU après les
poids). Un modèle *dense* complet à 256 K aurait nécessité ~47 GiB de KV et aurait
exigé un offload CPU — ici évité.

---

## Architecture de la solution

```
┌─────────────────────────────────────────────────────────────┐
│                      Hôte (Linux)                            │
│                                                              │
│   docker-compose.yml                                         │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Conteneur vllm-qwen36-27b-awq                       │   │
│   │  (nvcr.io/nvidia/vllm:26.06-py3)                     │   │
│   │                                                       │   │
│   │   API Server  ──►  EngineCore  ──►  Worker TP0 (GPU0)│   │
│   │   :1234                  │            Worker TP1 (GPU1)│   │
│   │                          └─ NCCL (P2P/IB off) ────────┘   │
│   │                                                       │   │
│   │   KV cache : 100% VRAM (gpu-memory-utilization=0.95) │   │
│   └─────────────────────────────────────────────────────┘   │
│        │ volume                                              │
│        ▼                                                     │
│   ./models  ──►  /root/.cache/huggingface/hub               │
└─────────────────────────────────────────────────────────────┘
```

- **Tensor Parallelism (TP=2)** : les couches du modèle sont scindées sur les 2 GPU.
- **NCCL** en mode P2P/IB désactivé (stabilise le TP sur PCIe sans NVLink).
- **Volumes** : les poids du modèle sont montés en lecture depuis `./models`
  (cache HF local, pas de téléchargement réseau au démarrage).

---

## Prérequis

- Linux avec NVIDIA driver récent (≥ 550) et **NVIDIA Container Toolkit** installé.
- Docker + Docker Compose v2.
- 2 GPUs NVIDIA visibles (`nvidia-smi` liste 2 cartes).
- Les poids du modèle présents dans `./models` (ou un `HF_TOKEN` valide dans `.env`).
- ~22 Go de VRAM libres au démarrage (pour 2× 3090 : OK).

```bash
# Vérification rapide
nvidia-smi -L
docker info | grep -i runtime   # doit mentionner nvidia
```

---

## Installation & lancement

```bash
# 1. (Optionnel) Fournir un token Hugging Face si le modèle est privé
cp .env.example .env
#   éditer .env pour renseigner HF_TOKEN=hf_xxx

# 2. Démarrer le serveur
docker compose up -d

# 3. Suivre le démarrage (compilation torch.compile + capture CUDA graph ~3-4 min)
docker compose logs -f

# 4. Vérifier la santé
curl -s http://localhost:1234/health   # -> {"...ok..."} / HTTP 200

# 5. Confirmer le modèle et le contexte max
curl -s http://localhost:1234/v1/models | jq '.data[0].max_model_len'
# -> 262144
```

**Arrêt :**
```bash
docker compose down        # stoppe et supprime le conteneur
```

---

## Configuration détaillée

### `docker-compose.yml` — parties clés

| Paramètre | Valeur | Rôle |
|---|---|---|
| `image` | `nvcr.io/nvidia/vllm:26.06-py3` | vLLM 0.22.1 pré-build NVIDIA |
| `ipc: host` | — | shared memory NCCL (évite *"No available shared memory broadcast block"*) |
| `deploy.resources.devices` | `count: 2` | expose les 2 GPU au conteneur |
| `--tensor-parallel-size` | `2` | parallélisme sur 2 GPU |
| `--max-model-len` | `262144` | contexte max natif du modèle |
| `--max-num-seqs` | `1` | 1 seule séquence (contexte 256 K monopolise le KV) |
| `--gpu-memory-utilization` | `0.95` | pousse le KV cache au max en VRAM |
| `--no-disable-hybrid-kv-cache-manager` | — | requis pour les modèles hybrides |
| `--enable-prefix-caching` | — | aligne les block sizes (obligatoire ici) |
| `--trust-remote-code` | — | charge le code custom du modèle |
| `--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder` | — | function calling |
| `--reasoning-parser qwen3` | — | parsing du raisonnement natif Qwen3 |

### Variables d'environnement

| Variable | Valeur | Effet |
|---|---|---|
| `LANG` / `LC_ALL` | `C.UTF-8` | évite les `UnicodeDecodeError` du conteneur NVIDIA (LANG=POSIX) |
| `NCCL_P2P_DISABLE` / `NCCL_IB_DISABLE` | `1` | stabilise NCCL sur PCIe sans NVLink |
| `VLLM_USE_FLASHINFER_SAMPLER` | `1` | sampling top-p/top-k **sur GPU** |
| `VLLM_USE_DEEP_GEMM` | `0` | désactivé (non pertinent pour AWQ) |
| `OMP_NUM_THREADS` | `8` | oversubscription validée empiriquement (+15 % tok/s) |
| `HF_TOKEN` | depuis `.env` | auth HF si besoin |

> ⚠️ **`VLLM_SLEEP_WHEN_IDLE` a volontairement été supprimé.** Il endormait le
> moteur après inactivité, causant un *hang* de plusieurs minutes et un pic de
> latence (~2 tok/s) au premier appel suivant. Le moteur reste désormais **chaud**.

---

## Optimisations (100 % GPU, max contexte)

1. **Aucun offload CPU** — suppression de `--kv-offloading-size` /
   `--kv-offloading-backend` et de `--swap-space`. Tout le KV cache vit en VRAM.
2. **`gpu-memory-utilization=0.95`** — maximise l'espace KV en VRAM tout en gardant
   la marge pour la capture des CUDA graphs à 256 K.
3. **FlashInfer partout** — sampler **et** attention exécutés sur GPU (RTX 3090
   CC 8.6 supporté), éliminant la régression CPU du sampler PyTorch.
4. **Modèle hybride exploité** — la linear-attention Mamba ne consomme pas de KV
   classique, ce qui rend le contexte 256 K tenable en VRAM pure.
5. **Moteur toujours chaud** — pas de mise en veille, débit immédiat à chaque requête.

---

## Utilisation de l'API

Le serveur expose l'API OpenAI-compatible sur **`http://localhost:1234`**.

### Chat completion

```bash
curl -s http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b-awq",
    "messages": [
      {"role": "user", "content": "Explique le parallélisme de données en 3 phrases."}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }' | jq '.choices[0].message'
```

### Avec un client Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="vllm")

resp = client.chat.completions.create(
    model="qwen3.6-27b-awq",
    messages=[{"role": "user", "content": "Bonjour, qui es-tu ?"}],
    max_tokens=512,
    temperature=0.6,
)
print(resp.choices[0].message.content)
```

### Test de charge utile (contexte long)

Pour valider le contexte 256 K, envoyez un prompt de ~250 K tokens ; le serveur
répondra tant que `max-num-seqs=1` est respecté (une séquence à la fois).

---

## Performances mesurées

| Métrique | Résultat |
|---|---|
| Débit soutenu (chaud) | **~37 tok/s** (150 tokens en ~4.1 s, stable sur N req) |
| Débit au 1er appel (cold) | ~20 tok/s (capture CUDA graph initiale) |
| Utilisation GPU | **100 %** (les 2 cartes) |
| VRAM utilisée | **~21.9 / 24.6 Go** par GPU (poids AWQ ~10 Go + KV ~11 Go) |
| Offload CPU du KV | **Aucun** (0 % CPU pour le cache) |
| Contexte max servi | 262 144 tokens |

> Note : le modèle répond en mode *reasoning* (parser `qwen3`). Le contenu
> apparaît dans le champ `reasoning` / en streaming — comportement attendu, pas un bug.

---

## Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| *"No available shared memory broadcast block"* | `/dev/shm` Docker trop petit | Vérifier `ipc: host` dans le compose (déjà présent) |
| *"Hybrid KV cache manager is disabled but failed to convert…"* | manager hybride désactivé | `--no-disable-hybrid-kv-cache-manager` (présent) |
| `AssertionError: gpu_block_size=… not divisible by hash_block_size=…` | block sizes mal alignés | `--enable-prefix-caching` (présent) |
| 1er appel très lent / hang | `VLLM_SLEEP_WHEN_IDLE` actif | Variable supprimée de la config |
| OOM au démarrage | `gpu-memory-utilization` trop haut | baisser à `0.90` (ou réduire `max-model-len`) |
| `Custom allreduce is disabled` (warning) | Pas de P2P NVLink sur 3090 | Normal ; NCCL utilisé à la place |

---

## Notes & limitations

- **`max-num-seqs=1`** : avec un contexte 256 K, une seule séquence peut être
  traitée à la fois (le KV cache sature la VRAM). Pour du batching, réduisez
  `--max-model-len`.
- **Redémarrage lent** : la compilation `torch.compile` + capture CUDA graph
  prennent ~3-4 min au premier lancement (cache persistant dans le conteneur).
- **RTX 3090 (CC 8.6)** : `SymmMemCommunicator` et `custom_all_reduce` sont
  désactivés (non supportés) — NCCL classique est utilisé, sans impact majeur.
- **Image NVIDIA** : `VLLM_USE_FLASHINFER_MOE_FP16` est déprécié (remplacé par
  `--moe-backend` en v0.23) — warning sans conséquence ici (modèle non-MoE).
- Modèle **multimodal** (vision + texte) : le chat texte seul fonctionne ;
  l'encodeur vision est initialisé (budget 16384 tokens) mais non sollicité.

---

## Structure du dépôt

```
.
├── docker-compose.yml   # Stack d'inférence vLLM (config optimisée)
├── .env.example         # Template de variables (HF_TOKEN)
├── .env                 # Variables réelles (non versionné)
├── models/              # Cache HF local des poids (monté en volume)
└── README.md            # Ce document