# vLLM — Qwen3.8-27B-INT8-W8A16-MTP (serveur d'inférence)

Serveur d'inférence haute performance pour le modèle
**`lued/Qwen3.8-27B-INT8-W8A16-MTP`**, déployé via
[vLLM](https://github.com/vllm-project/vllm) (image nightly épinglée) en
conteneur Docker.

> **Objectif :** contexte maximal (256 K tokens) en VRAM pure sur 2× RTX 3090,
> avec **MTP speculative decoding** (3 tokens spéculatifs, acceptation ~65 %,
> débit **~58-70 tok/s**).

---

## 📋 Sommaire

- [Vue d'ensemble](#vue-densemble)
- [Spécifications matérielles & modèle](#spécifications-matérielles--modèle)
- [Architecture de la solution](#architecture-de-la-solution)
- [Prérequis](#prérequis)
- [Installation & lancement](#installation--lancement)
- [Configuration détaillée](#configuration-détaillée)
- [Optimisations](#optimisations)
- [Utilisation de l'API](#utilisation-de-lapi)
- [Performances mesurées](#performances-mesurées)
- [Dépannage](#dépannage)
- [Notes & limitations](#notes--limitations)
- [Structure du dépôt](#structure-du-dépôt)

---

## Vue d'ensemble

Ce dépôt fournit une stack d'inférence prêt à l'emploi pour servir un modèle de
langage de 27 milliards de paramètres (quantification **INT8 W8A16**) avec :

- **Contexte maximal** de `262 144` tokens (256 K) — la limite native du modèle.
- **KV cache en FP8** (`fp8_e4m3`, 1 octet/valeur) — indispensable pour tenir
  256 K en VRAM sur 2× RTX 3090.
- **MTP speculative decoding** — 3 tokens spéculatifs, acceptation mesurée
  ~65 %, débit ~58-70 tok/s.
- **Tensor Parallelism = 2** sur deux RTX 3090 (PCIe, sans NVLink).
- **Modèle hybride** (Mamba linear-attention + GDN + MTP) — le chemin
  `Qwen3.8` requiert `--mamba-cache-mode align` et le spawn multiprocess.
- **Thinking désactivé par défaut** (`enable_thinking: false`) — gain ~3-5× en
  vitesse ; réactivable par requête via `reasoning_effort` ou
  `chat_template_kwargs`.

---

## Spécifications matérielles & modèle

| Élément | Valeur |
|---|---|
| GPU | 2 × NVIDIA RTX 3090 (compute capability **8.6**, Ampere) |
| VRAM | 24 Go × 2 = **48 Go** total |
| Interconnexion | PCIe (pas de NVLink) |
| Modèle | `lued/Qwen3.8-27B-INT8-W8A16-MTP` |
| Quantification | INT8 W8A16 (poids INT8 reconstruits en BF16, pas de FP8 natif sur sm_86) |
| `dtype` servi | `bfloat16` |
| KV cache dtype | `fp8_e4m3` (1 octet/valeur) |
| Contexte max | `262 144` tokens (`max_position_embeddings`) |
| vLLM | image nightly `vllm/vllm-openai:nightly-ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9` |
| MTP | 3 tokens spéculatifs (`--speculative-config`) |

**Pourquoi un contexte 256 K tient-il en VRAM ?**
Le modèle est **hybride** : sur 64 couches, seules 16 sont en full-attention
(les 48 autres utilisent une linear-attention de type Mamba qui n'alloue pas de
KV cache classique). Combiné au **KV cache FP8** (2× plus compact que le FP16),
le cache full-attention tient largement dans la VRAM libre après les poids.

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
│   │  Conteneur vllm-qwen38-27b-int8                      │   │
│   │  (vllm/vllm-openai:nightly-ac7509e2…)                │   │
│   │                                                       │   │
│   │   API Server  ──►  EngineCore  ──►  Worker TP0 (GPU0)│   │
│   │   :1234                  │            Worker TP1 (GPU1)│   │
│   │                          └─ NCCL (P2P/IB off) ────────┘   │
│   │                                                       │   │
│   │   KV cache FP8  ·  MTP (3 tokens spéculatifs)         │   │
│   │   gpu-memory-utilization = 0.92                       │   │
│   └─────────────────────────────────────────────────────┘   │
│        │ volumes                                              │
│        ▼                                                     │
│   ./models  ──►  /root/.cache/huggingface/hub               │
│   ./cache/vllm  ──►  /root/.cache/vllm (torch.compile)      │
│   ./cache/triton ──►  /root/.triton/cache                   │
└─────────────────────────────────────────────────────────────┘
```

- **Tensor Parallelism (TP=2)** : les couches du modèle sont scindées sur les 2 GPU.
- **Pipeline Parallelism = 1** (single stage).
- **NCCL** en mode P2P/IB désactivé (stabilise le TP sur PCIe sans NVLink) ;
  `custom_all_reduce` également désactivé (non supporté sans P2P).
- **Volumes** :
  - `./models` → cache Hugging Face (poids, montés en lecture/écriture).
  - `./cache/vllm` → artefacts de compilation `torch.compile`/Inductor
    (évite de recompiler à chaque redémarrage, ~3-4 min économisées).
  - `./cache/triton` → cache Triton (kernels).

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

# 3. Suivre le démarrage (compilation torch.compile + capture CUDA graph ~3-4 min
#    au premier lancement ; ~30 s ensuite grâce aux caches persistants)
docker compose logs -f

# 4. Vérifier la santé
curl -s http://localhost:1234/health   # -> HTTP 200

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
| `image` | `vllm/vllm-openai:nightly-ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9` | nightly épinglée (seule validée pour Qwen3.8 : Marlin + Mamba cache aligné + MTP) |
| `container_name` | `vllm-qwen38-27b-int8` | — |
| `ipc: host` | — | shared memory NCCL (évite *"No available shared memory broadcast block"*) |
| `deploy.resources.devices` | `count: 2` | expose les 2 GPU au conteneur |
| `--model` | `lued/Qwen3.8-27B-INT8-W8A16-MTP` | modèle servi |
| `--served-model-name` | `qwen3.8-27b-int8-w8a16` | nom exposé via l'API |
| `--tensor-parallel-size` | `2` | parallélisme sur 2 GPU |
| `--pipeline-parallel-size` | `1` | single stage |
| `--dtype` | `bfloat16` | Ampere sm_86 : W8A16 reconstruit en BF16 (pas de FP8 natif) |
| `--performance-mode` | `balanced` | profil de performance vLLM |
| `--max-model-len` | `262144` | contexte max natif du modèle |
| `--gpu-memory-utilization` | `0.92` | marge pour la capture CUDA graph à 256 K |
| `--max-num-seqs` | `2` | 2 séquences concurrentes (commande recommandée) |
| `--max-num-batched-tokens` | `8192` | chunking des longs prompts (prefill) |
| `--kv-cache-dtype` | `fp8_e4m3` | KV cache 1 octet/valeur — requis pour 256 K en VRAM |
| `--no-enable-prefix-caching` | — | prefix caching OFF (corrompt le KV récurrent GDN+MTP, vllm#48375) |
| `--enable-chunked-prefill` | — | prefill par lots pour les longs prompts |
| `--mamba-cache-mode` | `align` | requis par le chemin MTP/GDN de Qwen3.8 |
| `--prefix-match-unit` | `16` | alignement des block sizes |
| `--enable-prompt-tokens-details` | — | détails de tokens par requête |
| `--enable-per-request-metrics` | — | métriques par requête |
| `--reasoning-parser` | `qwen3` | parsing du raisonnement natif Qwen3 |
| `--tool-call-parser` | `qwen3_coder` | function calling |
| `--enable-auto-tool-choice` | — | active le tool calling |
| `--disable-custom-all-reduce` | — | désactive custom all-reduce (non supporté sans P2P/NVLink) |
| `--trust-remote-code` | — | charge le code custom du modèle |
| `--default-chat-template-kwargs` | `{"enable_thinking":false}` | thinking OFF par défaut (gain ~3-5×) |
| `--override-generation-config` | `{"temperature":1.0,"top_p":0.95,"top_k":20,...}` | defaults de génération |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":3}` | MTP, 3 tokens spéculatifs |

### Variables d'environnement

| Variable | Valeur | Effet |
|---|---|---|
| `LANG` / `LC_ALL` / `PYTHONIOENCODING` | `C.UTF-8` / `utf-8` | évite les `UnicodeDecodeError` (conteneur en LANG=POSIX) |
| `NCCL_P2P_DISABLE` / `NCCL_IB_DISABLE` | `1` | stabilise NCCL sur PCIe sans NVLink |
| `NCCL_CUMEM_ENABLE` | `0` | désactive CUMEM (allocation CUDA classique) |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | requis par le chemin MTP/GDN de Qwen3.8 |
| `OMP_NUM_THREADS` | `1` | 1 thread OMP (sampler FlashInfer désactivé pour ce pin) |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | sampler FlashInfer désactivé (recette modèle) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True,max_split_size_mb:512` | allocation CUDA par segments extensibles |
| `HF_TOKEN` | depuis `.env` | auth HF si besoin |
| `VLLM_LOGGING_LEVEL` | `INFO` | niveau de log |

> ⚠️ **`max-num-seqs=2`** est la commande recommandée, mais le multi-stream peut
> encore crasher le moteur sur ce pin (`vllm#50021` non mergé). En cas
> d'instabilité, descendre à `1` (= config pleinement validée).

---

## Optimisations

1. **KV cache FP8** (`fp8_e4m3`) — 1 octet/valeur au lieu de 2 (FP16), divisant
   par 2 l'empreinte du KV cache et rendant le contexte 256 K tenable en VRAM.
2. **MTP speculative decoding** — 3 tokens spéculatifs, acceptation ~65 %,
   débit ~58-70 tok/s (vs ~37 tok/s sans MTP).
3. **Caches de compilation persistants** — `./cache/vllm` et `./cache/triton`
   évitent de recompiler `torch.compile`/Triton à chaque redémarrage.
4. **Chunked prefill** (`--max-num-batched-tokens 8192`) — découpe les longs
   prompts pour éviter l'OOM au prefill.
5. **`gpu-memory-utilization=0.92`** — maximise l'espace KV en VRAM tout en
   gardant la marge pour la capture des CUDA graphs à 256 K.
6. **Modèle hybride exploité** — la linear-attention Mamba ne consomme pas de KV
   classique, ce qui rend le contexte 256 K tenable en VRAM pure.
7. **Thinking OFF par défaut** — `enable_thinking: false` pour un débit maximal ;
   réactivable par requête via `reasoning_effort` ou `chat_template_kwargs`.
8. **Allocation CUDA par segments extensibles** —
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512`.

---

## Utilisation de l'API

Le serveur expose l'API OpenAI-compatible sur **`http://localhost:1234`**.

### Chat completion

```bash
curl -s http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-int8-w8a16",
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
    model="qwen3.8-27b-int8-w8a16",
    messages=[{"role": "user", "content": "Bonjour, qui es-tu ?"}],
    max_tokens=512,
    temperature=0.6,
)
print(resp.choices[0].message.content)
```

### Réactiver le thinking par requête

Le thinking est désactivé par défaut (`--default-chat-template-kwargs
{"enable_thinking":false}`). Pour le réactiver sur une requête donnée :

```bash
curl -s http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-int8-w8a16",
    "messages": [{"role": "user", "content": "Résous x^2 + 5x + 6 = 0."}],
    "reasoning_effort": "low",
    "chat_template_kwargs": {"enable_thinking": true}
  }' | jq '.choices[0].message'
```

### Test de charge utile (contexte long)

Pour valider le contexte 256 K, envoyez un prompt de ~250 K tokens ; le serveur
répondra tant que `max-num-seqs` est respecté (1-2 séquences à la fois selon la
config).

---

## Performances mesurées

| Métrique | Résultat |
|---|---|
| Débit soutenu (chaud, MTP) | **~58-70 tok/s** (3 tokens spéculatifs, acceptation ~65 %) |
| Débit sans MTP | ~37 tok/s |
| 1er appel (cold) | plus lent (capture CUDA graph initiale, ~3-4 min au premier lancement) |
| Redémarrages suivants | ~30 s (caches `./cache/vllm` + `./cache/triton` persistants) |
| Utilisation GPU | **100 %** (les 2 cartes) |
| KV cache | **FP8** (`fp8_e4m3`, 1 octet/valeur) |
| Contexte max servi | 262 144 tokens |

> Note : le modèle répond en mode *reasoning* (parser `qwen3`) si le thinking est
> activé. Le contenu apparaît alors dans le champ `reasoning` / en streaming —
> comportement attendu, pas un bug.

---

## Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| *"No available shared memory broadcast block"* | `/dev/shm` Docker trop petit | Vérifier `ipc: host` dans le compose (déjà présent) |
| Crash / instabilité en multi-stream | `vllm#50021` non mergé | Descendre `--max-num-seqs` à `1` |
| Corruption du KV récurrent (GDN+MTP) | prefix caching activé | `--no-enable-prefix-caching` (déjà présent) |
| OOM au démarrage | `gpu-memory-utilization` trop haut | Baisser à `0.90` (ou réduire `--max-model-len`) |
| OOM au prefill d'un long prompt | `max-num-batched-tokens` trop grand | Réduire `--max-num-batched-tokens` (8192 par défaut) |
| `UnicodeDecodeError` dans `torch/_ops.py` | Locale POSIX dans le conteneur | `LANG=C.UTF-8` (déjà présent) |
| Recompilation lente à chaque redémarrage | caches `./cache/vllm` absents | Vérifier les volumes `./cache/vllm` et `./cache/triton` |
| `Custom allreduce is disabled` (warning) | Pas de P2P NVLink sur 3090 | Normal ; `--disable-custom-all-reduce` + NCCL utilisé |

---

## Notes & limitations

- **`max-num-seqs=2`** : la commande recommandée, mais le multi-stream peut
  encore crasher le moteur sur ce pin (`vllm#50021` non mergé). En cas
  d'instabilité, descendre à `1` (= config pleinement validée).
- **Prefix caching désactivé** : sur GDN+MTP, le prefix caching peut corrompre
  le KV récurrent (`vllm#48375`) sauf patch source. On le laisse désactivé.
- **RTX 3090 (CC 8.6, Ampere)** : W8A16 = poids INT8 reconstruits en BF16 (pas
  de FP8 natif) ; `custom_all_reduce` désactivé (non supporté) — NCCL classique
  utilisé, sans impact majeur.
- **Thinking OFF par défaut** : gain de ~3-5× en vitesse. Réactivable par
  requête via `reasoning_effort` ou `chat_template_kwargs`.
- **Image nightly épinglée** : `vllm/vllm-openai:nightly-ac7509e2…` est la seule
  version validée pour Qwen3.8 (compressed-tensors Marlin + Mamba cache aligné +
  MTP réunis). Ne pas changer d'image sans revalidation.
- **Redémarrage** : ~3-4 min au premier lancement (compilation
  `torch.compile` + capture CUDA graph) ; ~30 s ensuite grâce aux caches
  persistants (`./cache/vllm`, `./cache/triton`).

---

## Structure du dépôt

```
.
├── docker-compose.yml   # Stack d'inférence vLLM (config optimisée Qwen3.8)
├── .env.example         # Template de variables (HF_TOKEN)
├── .env                 # Variables réelles (non versionné)
├── models/              # Cache HF local des poids (monté en volume)
├── cache/vllm/          # Cache torch.compile / Inductor (persistant)
├── cache/triton/        # Cache Triton (kernels, persistant)
└── README.md            # Ce document