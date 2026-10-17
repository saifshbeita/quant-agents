#!/usr/bin/env bash
# Start the vLLM OpenAI-compatible server on AMD Instinct GPUs (ROCm).
# The API is published on 127.0.0.1 only; it must never be reachable from the internet.
#
# Reads from .env (or the environment): LLM_API_KEY (required), LLM_MODEL, HF_TOKEN,
# VLLM_IMAGE, TENSOR_PARALLEL_SIZE, VLLM_MAX_MODEL_LEN.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a; source ./.env; set +a
fi

: "${LLM_API_KEY:?LLM_API_KEY is not set (deploy_amd.sh generates one in .env)}"
if [[ "$LLM_API_KEY" == "EMPTY" ]]; then
  echo "Refusing to start vLLM with LLM_API_KEY=EMPTY; set a real key in .env." >&2
  exit 1
fi

# Official upstream vLLM ROCm image. v0.31.0 is the newest release tag (same digest as :latest
# on 2026-10-17); Qwen3.6 needs vLLM >= 0.17 and its model card recommends >= 0.19.
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:v0.31.0}"
MIN_VLLM_VERSION="${MIN_VLLM_VERSION:-0.19.0}"
MODEL="${LLM_MODEL:-Qwen/Qwen3.6-35B-A3B}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
NAME="vllm-server"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

echo "==> Pulling $VLLM_IMAGE"
docker pull "$VLLM_IMAGE"

version="$(docker run --rm --entrypoint python3 "$VLLM_IMAGE" \
  -c 'from importlib.metadata import version; print(version("vllm"))')"
echo "==> Image ships vLLM $version (need >= $MIN_VLLM_VERSION)"
if [[ "$(printf '%s\n%s\n' "$MIN_VLLM_VERSION" "${version%%+*}" | sort -V | head -1)" != "$MIN_VLLM_VERSION" ]]; then
  echo "vLLM $version is too old for $MODEL; set VLLM_IMAGE to a newer tag." >&2
  exit 1
fi

mkdir -p "$HF_CACHE"
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "==> Starting $NAME: $MODEL (tp=$TP, max-model-len=$MAX_LEN) on 127.0.0.1:8000"
docker run -d --name "$NAME" \
  --restart unless-stopped \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 16g \
  -p 127.0.0.1:8000:8000 \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "$VLLM_IMAGE" \
  "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "$LLM_API_KEY" \
  --reasoning-parser qwen3 \
  --max-model-len "$MAX_LEN" \
  --tensor-parallel-size "$TP"

echo "==> vLLM container started; follow with: docker logs -f $NAME"
