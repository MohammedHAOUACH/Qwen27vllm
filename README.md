# vLLM — Qwen3.8-27B-INT8-W8A16-MTP + DFlash2 (serveur d'inférence)

Serveur d'inférence haute performance pour le modèle
**`lued/Qwen3.8-27B-INT8-W8A16-MTP`** (cible) avec drafter
**`z-lab/Qwen3.8-27B-DFlash2`**, déployé via
[vLLM](https://github.com/vllm-project/vllm) (image locale avec PR #52816) en
conteneur Docker.

> **Objectif :** **DFlash2 speculative decoding** (block-diffusion, 7 tokens
> spéculatifs, décodage lossless) sur 2× RTX 3090. ⚠️ Le drafter séparé
> (~3.9 Go BF16) réduit le contexte max à **~98 K tokens** (mesuré, VRAM 2× 24 Go)
> au lieu des 256 K natifs du modèle.

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
- [Interface web (Open WebUI)](#interface-web-open-webui)
- [Performances mesurées](#performances-mesurées)
- [Dépannage](#dépannage)
- [Notes & limitations](#notes--limitations)
- [Structure du dépôt](#structure-du-dépôt)

---

## Vue d'ensemble

Ce dépôt fournit une stack d'inférence prêt à l'emploi pour servir un modèle de
langage de 27 milliards de paramètres (quantification **INT8 W8A16**) avec :

- **Contexte maximal** : `98 304` tokens (limite native 262 144, abaissée car le
  drafter DFlash2 séparé ne laisse que ~2.7 GiB de KV cache par GPU — voir la
  section DFlash2).
- **KV cache en FP8** (`fp8_e4m3`, 1 octet/valeur) — indispensable pour tenir
  un grand contexte en VRAM sur 2× RTX 3090.
- **DFlash2 speculative decoding** — block-diffusion, 7 tokens spéculatifs par
  étape (remplace le MTP à 3 tokens) ; décodage lossless.
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
| vLLM | image locale `vllm-openai:dflash2-pr52816` (construite via `./build-dflash-image.sh`, PR #52816) |
| Spec. decoding | DFlash2, 7 tokens spéculatifs (`--speculative-config`) |

**Pourquoi le contexte est-il limité à ~98 K avec DFlash2 ?**
Le modèle est **hybride** : sur 64 couches, seules 16 sont en full-attention
(KV cache classique), les 48 autres utilisent une linear-attention Mamba. En
théorie le contexte 256 K tient donc en VRAM avec les poids seuls — mais le
**drafter DFlash2 séparé (~3.9 Go BF16, chargé en plus)** réduit la VRAM libre
au point qu'il ne reste que ~2.7 GiB de KV cache par GPU : vLLM estime le max à
~98 880 tokens. Sans drafter (ou avec un drafter intégré type MTP), les 256 K
redeviennent possibles.

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
│   │  (vllm-openai:dflash2-pr52816, construit localement) │   │
│   │                                                       │   │
│   │   API Server  ──►  EngineCore  ──►  Worker TP0 (GPU0)│   │
│   │   :1234                  │            Worker TP1 (GPU1)│   │
│   │                          └─ NCCL (P2P/IB off) ────────┘   │
│   │                                                       │   │
│   │   KV cache FP8  ·  DFlash2 (7 tokens spéculatifs)     │   │
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

## 🧪 DFlash2 — spéculative decoding par block-diffusion (expérimental)

Sur cette branche, le décodage spéculatif **MTP** est remplacé par
**[DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)**, un drafter à
*block-diffusion* : il propose un bloc entier de tokens en un seul passage, puis
un sélecteur trace un chemin cohérent parmi les candidats. Le décodage est
**lossless** (sortie greedy/sampling identique au modèle cible).

- **Cible (vérifieur)** : `lued/Qwen3.8-27B-INT8-W8A16-MTP` (inchangé — ses têtes
  MTP sont chargées mais inutilisées).
- **Drafter** : `z-lab/Qwen3.8-27B-DFlash2` (~1.9B param BF16, ~3.9 Go, public).
- **Config** : `{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}`.
  (`method=dflash` : le speculator DFlash2 est choisi automatiquement car
  l'architecture du drafter est `DFlash2DraftModel`.)
- **vLLM** : la méthode n'est pas encore mergée ; elle vit dans la PR
  [#52816](https://github.com/vllm-project/vllm/pull/52816). Le build épingle le
  head `19c9351` et cherry-pick deux bugfixes empilés dessus :
  - « probabilistic drafting safety » de benchislett (corrige les sorties
    NaN/garbage `!!!!!!…`),
  - [#52883](https://github.com/vllm-project/vllm/pull/52883) (accepte une LM
    head linéaire non quantifiée — sinon `ValueError` au démarrage).
  Le script patch aussi **FA3 désactivé** dans `setup.py` : vllm-flash-attn
  compile FA3 (Hopper sm90 uniquement) dès que CUDA ≥ 12, même pour une cible
  sm_86/8.9 — inutile ET fait OOM `cc1plus` à la compilation (45 Go RAM).
  Il faut donc construire une image locale :

```bash
# Build unique (~1-2 h) de l'image avec le support DFlash2
./build-dflash-image.sh

# Puis démarrer la stack normalement
docker compose up -d
```

> Résultats de référence (carte modèle, H200, 7 tokens spéculatifs/étape) :
> DFlash2 ≈ **3.1-3.4×** le débit autoregressif à concurrency 1 (vs ~2.2-2.6×
> pour MTP). Sur ce matériel (2× RTX 3090, TP=2) : **~49 tok/s mesurés** sur une
> génération de 300 tokens, acceptance moyenne ~3.15 tokens (cf. Performances).

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
# -> 98304 (limité par la VRAM avec le drafter DFlash2, voir section DFlash2)
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
| `image` | `vllm-openai:dflash2-pr52816` | image locale construite par `./build-dflash-image.sh` (vLLM + PR #52816 DFlash2) |
| `container_name` | `vllm-qwen38-27b-int8` | — |
| `ipc: host` | — | shared memory NCCL (évite *"No available shared memory broadcast block"*) |
| `deploy.resources.devices` | `count: 2` | expose les 2 GPU au conteneur |
| `--model` | `lued/Qwen3.8-27B-INT8-W8A16-MTP` | modèle servi |
| `--served-model-name` | `qwen3.8-27b-int8-w8a16` | nom exposé via l'API |
| `--tensor-parallel-size` | `2` | parallélisme sur 2 GPU |
| `--pipeline-parallel-size` | `1` | single stage |
| `--dtype` | `bfloat16` | Ampere sm_86 : W8A16 reconstruit en BF16 (pas de FP8 natif) |
| `--performance-mode` | `balanced` | profil de performance vLLM |
| `--max-model-len` | `98304` | contexte max (abaissé de 262144 : le drafter DFlash2 ne laisse que ~2.7 GiB de KV cache/GPU) |
| `--gpu-memory-utilization` | `0.92` | marge pour la capture CUDA graph |
| `--max-num-seqs` | `2` | 2 séquences concurrentes (commande recommandée) |
| `--max-num-batched-tokens` | `8192` | chunking des longs prompts (prefill) |
| `--kv-cache-dtype` | `fp8_e4m3` | KV cache 1 octet/valeur — requis pour 256 K en VRAM |
| `--no-enable-prefix-caching` | — | prefix caching OFF (corrompt le KV récurrent GDN+MTP, vllm#48375) |
| `--enable-chunked-prefill` | — | prefill par lots pour les longs prompts |
| `--mamba-cache-mode` | `align` | requis par le chemin GDN/Mamba de Qwen3.8 |
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
| `--speculative-config` | `{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}` | DFlash2 (block-diffusion) : 7 tokens spéculatifs, drafter séparé ~1.9B |

### Variables d'environnement

| Variable | Valeur | Effet |
|---|---|---|
| `LANG` / `LC_ALL` / `PYTHONIOENCODING` | `C.UTF-8` / `utf-8` | évite les `UnicodeDecodeError` (conteneur en LANG=POSIX) |
| `NCCL_P2P_DISABLE` / `NCCL_IB_DISABLE` | `1` | stabilise NCCL sur PCIe sans NVLink |
| `NCCL_CUMEM_ENABLE` | `0` | désactive CUMEM (allocation CUDA classique) |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | requis par le chemin GDN/Mamba de Qwen3.8 |
| `OMP_NUM_THREADS` | `1` | 1 thread OMP (sampler FlashInfer désactivé pour ce pin) |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | sampler FlashInfer désactivé (recette modèle ; A/B testé : `=1` ne change rien au débit, ~42-45 tok/s) |
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
2. **DFlash2 speculative decoding** — block-diffusion, 7 tokens spéculatifs par
   étape (vs 3 pour MTP). Décodage lossless : même sortie que le modèle cible.
   Référence H200 : ~3.1-3.4× l'autoregressif à concurrency 1 (vs ~2.2-2.6× MTP).
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

## Interface web (Open WebUI)

En complément de l'API brute, la stack embarque [Open WebUI](https://github.com/open-webui/open-webui),
une interface web auto-hébergée (chat, RAG, multi-utilisateurs) branchée sur
l'API OpenAI-compatible vLLM.

- **Accès :** <http://localhost:3000>
- **API utilisée :** `http://vllm:1234/v1` (résolue via le réseau Docker interne).
- **Modèle servi :** `qwen3.8-27b-int8-w8a16` (détecté automatiquement depuis
  `/v1/models`).
- **Données :** persistées dans le volume Docker `open-webui` (comptes,
  conversations, fichiers RAG).

```bash
# Démarre vLLM + Open WebUI
docker compose up -d

# Ouvrir http://localhost:3000
```

> ⚠️ **Démarrage différé :** vLLM met ~3-4 min à se compiler au premier
> lancement. Open WebUI démarre immédiatement mais le modèle n'apparaît dans la
> liste qu'une fois vLLM prêt (Open WebUI interroge `/v1/models` en boucle).
> Suivre la progression avec `docker compose logs -f vllm`.

**Première connexion :** créez un compte local (le premier utilisateur devient
administrateur). Open WebUI est 100 % hors-ligne — aucune donnée ne sort du
réseau local.

---

## Performances mesurées

| Métrique | Résultat |
|---|---|
| Débit soutenu (chaud, DFlash2) | **~49 tok/s** mesuré (génération 300 tokens, concurrency 1) ; acceptance ~3.15 tokens, taux draft ~31 % |
| Débit MTP (ancienne config) | ~42-45 tok/s (3 tokens, acceptation ~65 %), ~37 tok/s sans spéculation |
| 1er appel (cold) | plus lent : compilation ~66 s + capture CUDA graph initiale (~4 min jusqu'à l'API) |
| Redémarrages suivants | init moteur **~43 s** (compilation 0.6 s grâce à `./cache/vllm`), API prête ~3 min (chargement poids ~60 s inclus) |
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
- **Image locale (DFlash2)** : `vllm-openai:dflash2-pr52816` est construite à
  partir de la PR non mergée [#52816](https://github.com/vllm-project/vllm/pull/52816).
  Ne pas repasser sur une image nightly standard sans revalidation (DFlash2 y est absent).
- **Redémarrage** : ~3-4 min au premier lancement (compilation
  `torch.compile` + capture CUDA graph) ; ~30 s ensuite grâce aux caches
  persistants (`./cache/vllm`, `./cache/triton`).

---

## Structure du dépôt

```
.
├── docker-compose.yml   # Stack d'inférence vLLM + interface Open WebUI
├── .env.example         # Template de variables (HF_TOKEN)
├── .env                 # Variables réelles (non versionné)
├── models/              # Cache HF local des poids (monté en volume)
├── cache/vllm/          # Cache torch.compile / Inductor (persistant)
├── cache/triton/        # Cache Triton (kernels, persistant)
└── README.md            # Ce document
```

> Le volume nommé `open-webui` (données de l'interface web) est géré par Docker
> (`docker volume ls | grep open-webui`) — il n'apparaît pas comme dossier du
> dépôt.