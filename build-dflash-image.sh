#!/usr/bin/env bash
# =============================================================================
# Construit une image vLLM avec le support DFlash2 (PR vllm#52816, NON mergée).
#
# DFlash2 = speculative decoding par block-diffusion : un drafter séparé
# (z-lab/Qwen3.8-27B-DFlash2, ~1.9B param BF16) propose un bloc de tokens en
# un seul passage, que le modèle cible (vérifieur) valide. Décodage lossless.
#
# Ce script clone vLLM, épingle la PR #52816, applique deux bugfixes connus
# empilés dessus, puis construit l'image `vllm-openai` (ENTRYPOINT "vllm serve")
# via le Dockerfile officiel.
#
# ⚠️ Build lourd : ~1-2 h, ~50+ Go de disque, RAM >= 32 Go. À lancer une seule
#    fois (l'image est ensuite référencée par docker-compose.yml).
#
# Usage :
#   ./build-dflash-image.sh
#
# Variables surchargeables :
#   IMAGE_TAG             tag de l'image produite (défaut : vllm-openai:dflash2-pr52816)
#   CUDA_VERSION          version CUDA du build (défaut : 13.0.3 — driver >= 580)
#   TORCH_CUDA_ARCH_LIST  archs GPU compilées (défaut : "8.6 8.9")
#   MAX_JOBS              parallélisme Ninja (défaut : 12, ~45 Go RAM)
# =============================================================================
set -euo pipefail

PR_NUMBER=52816
# Head de la PR au moment de l'épinglage (force-pushée pendant la review).
# Épingler le commit rend le build reproductible : les cherry-picks ci-dessous
# s'appliquent proprement sur cette base.
PR_HEAD="19c9351904df4c63042671bc67a866ca48dc7d6f"
# Bugfix 1 (fork benchislett) : "probabilistic drafting safety" — corrige les
# sorties NaN/garbage ("!!!!!!…") et respecte draft_sample_method.
# Réf : commentaire de benchislett sur vllm#52816.
FIX_SAFETY="31840cf3ead3632f3c99db4a24e4aba39ad54ef6"
# Bugfix 2 (PR vllm#52883) : accepte une LM head linéaire non quantifiée
# (le drafter a tie_word_embeddings=false → ParallelLMHead), sinon ValueError
# "DFlash2 requires an unquantized target LM head" au démarrage.
FIX_LM_HEAD_PR=52883
FIX_LM_HEAD="ed34bf91959407b2b1b3eef4c64738b4df76ad72"

IMAGE_TAG="${IMAGE_TAG:-vllm-openai:dflash2-pr${PR_NUMBER}}"
# RTX 3090 = Ampere sm_86. Restreindre l'arch accélère fortement la compilation.
# 8.9 couvre les RTX 40xx si besoin ; retirer pour un build minimal.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6 8.9}"
# CUDA 13.0.3 = défaut vLLM main ; driver ici = 610 (CUDA UMD 13.3) → OK.
# Descendre à 12.8.0 si le driver est plus ancien (< 580).
CUDA_VERSION="${CUDA_VERSION:-13.0.3}"
# Borné pour éviter l'OOM à la compilation (host : 45 Go RAM, 12 threads).
# vLLM calcule nb_jobs_effectifs = MAX_JOBS // NVCC_THREADS ; avec 12/4 -> 3 jobs.
MAX_JOBS="${MAX_JOBS:-12}"
NVCC_THREADS="${NVCC_THREADS:-4}"
WORKDIR="${TMPDIR:-/tmp}/vllm-dflash-pr${PR_NUMBER}"

echo "==> Clonage de vLLM + PR #${PR_NUMBER} dans ${WORKDIR}"
rm -rf "${WORKDIR}"
git clone https://github.com/vllm-project/vllm.git "${WORKDIR}"
cd "${WORKDIR}"

echo "==> Épinglage du head de la PR #${PR_NUMBER} (${PR_HEAD})"
git fetch origin "pull/${PR_NUMBER}/head"
git checkout --detach "${PR_HEAD}"

echo "==> Cherry-pick des bugfixes empilés sur #${PR_NUMBER}"
git fetch https://github.com/benchislett/vllm.git "${FIX_SAFETY}"
git cherry-pick --no-edit "${FIX_SAFETY}"
git fetch origin "pull/${FIX_LM_HEAD_PR}/head"
git cherry-pick --no-edit "${FIX_LM_HEAD}"

# ── Patch vLLM (bug main, hors PR #52816) ────────────────────────────────────
# `fused_gdn_decode_post_conv_mtp` est déclaré dans ops.h sous
# VLLM_ENABLE_FUSED_KDA_DECODE mais enregistré dans torch_bindings.cpp sous
# VLLM_ENABLE_FUSED_GDN_DECODE. Avec CUDA 13 + arch non-Blackwell (8.6/8.9),
# GDN est actif mais KDA non -> "not declared in this scope". On déplace la
# déclaration sous le bon garde.
python3 - <<'PYEOF'
from pathlib import Path
p = Path("csrc/libtorch_stable/ops.h")
s = p.read_text()
marker = "void fused_gdn_decode_post_conv_mtp(\n"
assert s.count(marker) == 1, f"attendu 1 occurrence, trouvé {s.count(marker)}"
replacement = "#endif\n\n#ifdef VLLM_ENABLE_FUSED_GDN_DECODE\n" + marker
s = s.replace(marker, replacement, 1)
p.write_text(s)
print("ops.h patché : déclaration GDN déplacée sous VLLM_ENABLE_FUSED_GDN_DECODE")
PYEOF

# ── Patch vLLM (bug main, hors PR #52816) : désactiver FA3 ───────────────────
# vllm-flash-attn compile FA3 (Hopper sm90 uniquement) dès que CUDA >= 12,
# SANS vérifier si l'arch cible inclut sm90. Sur sm_86/8.9 (RTX 3090) ces
# kernels cutlass sm90 sont inutiles ET leur compilation host (cc1plus) fait
# OOM sur 45 Go de RAM -> "gcc: internal compiler error: Segmentation fault".
# On retire _vllm_fa3_C des ext_modules : le target n'est jamais compilé
# (FA2 couvre Ampere ; l'import runtime de FA3 est protégé par try/except).
python3 - <<'PYEOF'
from pathlib import Path
p = Path("setup.py")
s = p.read_text()
old = """    if USE_PRECOMPILED_EXTENSIONS or (
        CUDA_HOME and get_nvcc_cuda_version() >= Version("12.3")
    ):
        # FA3 requires CUDA 12.3 or later
        ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))
"""
assert s.count(old) == 1, f"attendu 1 occurrence, trouvé {s.count(old)}"
new = """    if False:  # FA3 (Hopper sm90) DESACTIVE : inutile sur sm_86/8.9 + OOM cc1plus
        # FA3 requires CUDA 12.3 or later
        ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))
"""
s = s.replace(old, new, 1)
p.write_text(s)
print("setup.py patché : FA3 (Hopper) désactivé")
PYEOF

echo "==> Build de l'image ${IMAGE_TAG} (cible vllm-openai)"
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  --target vllm-openai \
  --build-arg max_jobs="${MAX_JOBS}" \
  --build-arg nvcc_threads="${NVCC_THREADS}" \
  --build-arg torch_cuda_arch_list="${TORCH_CUDA_ARCH_LIST}" \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  -f docker/Dockerfile \
  -t "${IMAGE_TAG}" .

echo "==> Terminé : ${IMAGE_TAG}"
echo "    Lancez ensuite : docker compose up -d"
