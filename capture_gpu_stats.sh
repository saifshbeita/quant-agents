#!/usr/bin/env bash
# Record GPU utilisation during a research run as demo evidence.
#
#   ./capture_gpu_stats.sh            sample every 2s until Ctrl-C
#   ./capture_gpu_stats.sh 600 1      sample every 1s for 600s
#
# Writes gpu_stats/<timestamp>/:
#   snapshot.txt   full rocm-smi report + GPU product names at start
#   samples.log    timestamped rocm-smi readings (use %, VRAM %, power, temperature)
#   samples.csv    the same readings as CSV
#   vllm_throughput.log  vLLM's own "Avg prompt/generation throughput" log lines for the window
set -uo pipefail
cd "$(dirname "$0")"

DURATION="${1:-0}"     # 0 = until Ctrl-C
INTERVAL="${2:-2}"
VLLM_NAME="${VLLM_NAME:-vllm-server}"
command -v rocm-smi >/dev/null || { echo "rocm-smi not found" >&2; exit 1; }

out="gpu_stats/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
started_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

{
  echo "# captured on $(hostname) at ${started_iso}"
  rocm-smi --showproductname
  rocm-smi
} > "$out/snapshot.txt" 2>&1

finish() {
  trap - INT TERM EXIT
  if docker inspect "$VLLM_NAME" >/dev/null 2>&1; then
    docker logs --since "$started_iso" "$VLLM_NAME" 2>&1 | grep -E "throughput|Running:" > "$out/vllm_throughput.log" || true
  fi
  samples=$(grep -c '^=== ' "$out/samples.log" 2>/dev/null)
  echo
  echo "Saved ${samples:-0} samples to ${out}/"
  ls -1 "$out"
  exit 0
}
trap finish INT TERM EXIT

echo "Recording rocm-smi every ${INTERVAL}s into ${out}/ $([[ $DURATION -gt 0 ]] && echo "for ${DURATION}s" || echo "(Ctrl-C to stop)")"
header_done=0
end=$(( $(date +%s) + DURATION ))
while [[ $DURATION -eq 0 || $(date +%s) -lt $end ]]; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  reading="$(rocm-smi --showuse --showmemuse --showpower --showtemp 2>&1)"
  printf '=== %s\n%s\n' "$ts" "$reading" >> "$out/samples.log"

  csv="$(rocm-smi --showuse --showmemuse --showpower --showtemp --csv 2>/dev/null | grep -v '^$')"
  if [[ -n "$csv" ]]; then
    if [[ $header_done -eq 0 ]]; then
      echo "timestamp,$(head -1 <<<"$csv")" > "$out/samples.csv"
      header_done=1
    fi
    tail -n +2 <<<"$csv" | sed "s/^/${ts},/" >> "$out/samples.csv"
  fi

  # one-line live view: GPU use per card
  use="$(grep -Eo 'GPU use \(%\): *[0-9]+' <<<"$reading" | grep -Eo '[0-9]+$' | paste -sd' ' -)"
  printf '\r%s  GPU use %%: %s   ' "$ts" "${use:-n/a}"
  sleep "$INTERVAL"
done
