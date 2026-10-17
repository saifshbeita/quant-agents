#!/usr/bin/env bash
# One-shot deploy on an AMD Developer Cloud MI300X droplet (Ubuntu + ROCm + Docker).
# Safe to re-run: a healthy vLLM server is left alone; the app image is rebuilt and replaced.
#
#   ./deploy_amd.sh               full deploy
#   ./deploy_amd.sh --app-only    rebuild/restart only the app (e.g. after editing .env)
#   ./deploy_amd.sh --restart-vllm  force-recreate the vLLM container too
set -euo pipefail
cd "$(dirname "$0")"

APP_ONLY=0
RESTART_VLLM=0
for arg in "$@"; do
  case "$arg" in
    --app-only) APP_ONLY=1 ;;
    --restart-vllm) RESTART_VLLM=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

APP_NAME="quant-agents"
VLLM_NAME="vllm-server"
VLLM_URL="http://127.0.0.1:8000/v1"
READY_TIMEOUT="${VLLM_READY_TIMEOUT:-3600}"   # first start downloads ~70 GB of weights

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. preflight
step "Checking ROCm and Docker"
command -v rocm-smi >/dev/null || die "rocm-smi not found. Use a ROCm droplet image."
rocm-smi >/dev/null 2>&1 || die "rocm-smi failed; the AMD GPU driver is not working."
[[ -e /dev/kfd ]] || die "/dev/kfd is missing; the amdgpu kernel driver is not loaded."
gpu_name="$(rocm-smi --showproductname 2>/dev/null | grep -i -m1 'card series' | sed 's/.*: *//')" || true
echo "rocm-smi OK (${gpu_name:-AMD GPU})"
command -v docker >/dev/null || die "Docker is not installed (https://docs.docker.com/engine/install/ubuntu/)."
docker info >/dev/null 2>&1 || die "Docker daemon not reachable (is it running, and is $(id -un) in the docker group?)."
echo "docker OK ($(docker version --format '{{.Server.Version}}'))"

# ---------------------------------------------------------------- 2. .env + API key
step "Preparing .env"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi
chmod 600 .env
current_key="$(grep -E '^LLM_API_KEY=' .env | tail -1 | cut -d= -f2- || true)"
if [[ -z "$current_key" || "$current_key" == "EMPTY" ]]; then
  new_key="qa-$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  if grep -qE '^LLM_API_KEY=' .env; then
    sed -i "s|^LLM_API_KEY=.*|LLM_API_KEY=${new_key}|" .env
  else
    echo "LLM_API_KEY=${new_key}" >> .env
  fi
  echo "Generated a random LLM_API_KEY in .env"
else
  echo "Using existing LLM_API_KEY from .env"
fi
set -a; source ./.env; set +a
MODEL="${LLM_MODEL:-Qwen/Qwen3.6-35B-A3B}"

models_ok() {
  curl -fsS --max-time 5 -H "Authorization: Bearer ${LLM_API_KEY}" "${VLLM_URL}/models" 2>/dev/null \
    | grep -q "\"${MODEL}\""
}

# ---------------------------------------------------------------- 3. vLLM
if [[ $APP_ONLY -eq 0 ]]; then
  step "vLLM server"
  if [[ $RESTART_VLLM -eq 0 ]] && models_ok; then
    echo "vLLM is already serving ${MODEL}; leaving it running (use --restart-vllm to recreate)."
  else
    ./run_amd_vllm.sh
  fi

  step "Waiting for ${VLLM_URL}/models (timeout ${READY_TIMEOUT}s)"
  start=$(date +%s)
  until models_ok; do
    elapsed=$(( $(date +%s) - start ))
    state="$(docker inspect -f '{{.State.Status}}' "$VLLM_NAME" 2>/dev/null || echo missing)"
    if [[ "$state" != "running" ]]; then
      docker logs --tail 50 "$VLLM_NAME" 2>&1 || true
      die "vLLM container is '${state}'. See the logs above."
    fi
    if (( elapsed > READY_TIMEOUT )); then
      docker logs --tail 30 "$VLLM_NAME" 2>&1 || true
      die "vLLM not ready after ${READY_TIMEOUT}s."
    fi
    last="$(docker logs --tail 1 "$VLLM_NAME" 2>&1 | tr -d '\r' | cut -c1-110)"
    printf '[%02d:%02d] loading... %s\n' $((elapsed / 60)) $((elapsed % 60)) "$last"
    sleep 15
  done
  echo "vLLM is ready and serving ${MODEL}."
else
  models_ok || die "--app-only, but vLLM is not serving ${MODEL} at ${VLLM_URL}. Run ./deploy_amd.sh first."
fi

# ---------------------------------------------------------------- 4. app
step "Building the app image"
docker build -t "${APP_NAME}:latest" .

step "Starting the app on 0.0.0.0:8501"
mkdir -p runs
chmod a+rwx runs   # the container runs as an unprivileged user and records runs here
docker rm -f "$APP_NAME" >/dev/null 2>&1 || true
docker run -d --name "$APP_NAME" \
  --restart unless-stopped \
  --network host \
  --env-file .env \
  -e LLM_BASE_URL="$VLLM_URL" \
  -v "$PWD/runs:/app/runs" \
  "${APP_NAME}:latest" >/dev/null

for _ in $(seq 1 40); do
  curl -fsS --max-time 2 http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS --max-time 2 http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1 \
  || { docker logs --tail 40 "$APP_NAME"; die "App did not become healthy."; }
echo "App is up."

# ---------------------------------------------------------------- 5. smoke test (through the app's own client)
step "Smoke test: one chat completion via the app's LLM client"
docker exec -i "$APP_NAME" python - <<'PY'
import time
from langchain_core.messages import HumanMessage
import config
from core.llm import get_llm, strip_think
t = time.perf_counter()
resp = get_llm().invoke([HumanMessage(content="Reply with exactly: quant-agents online")])
dt = time.perf_counter() - t
usage = resp.usage_metadata or {}
out = usage.get("output_tokens", 0)
print(f"model     : {config.LLM_MODEL} @ {config.LLM_BASE_URL}")
print(f"reply     : {strip_think(resp.content)[:200]!r}")
print(f"latency   : {dt:.2f}s, {usage.get('input_tokens', 0)} prompt + {out} completion tokens"
      + (f" ({out / dt:.1f} tok/s)" if dt and out else ""))
PY

# ---------------------------------------------------------------- 6. firewall + summary
if command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q "Status: active"; then
  ufw allow 8501/tcp >/dev/null && echo "ufw: allowed 8501/tcp (port 8000 stays closed)."
fi

public_ip="$(curl -fsS --max-time 3 http://169.254.169.254/metadata/v1/interfaces/public/0/ipv4/address 2>/dev/null \
  || curl -fsS --max-time 5 https://ifconfig.me 2>/dev/null \
  || hostname -I | awk '{print $1}')"

step "GPU status"
rocm-smi || true

step "Deployed"
cat <<EOF
  App          : http://${public_ip}:8501
  vLLM (local) : ${VLLM_URL}  (bound to 127.0.0.1 only, API key in .env)
  Model        : ${MODEL}

  Logs         : docker logs -f ${APP_NAME}   |   docker logs -f ${VLLM_NAME}
  GPU proof    : ./capture_gpu_stats.sh        (run during a demo, Ctrl-C to stop)
  After .env edits: ./deploy_amd.sh --app-only
EOF
