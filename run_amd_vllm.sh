#!/bin/bash
docker run -d --name vllm-server \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN}" \
  rocm/vllm:latest \
  vllm serve Qwen/Qwen3.6-Coder-35B-Instruct \
  --max-model-len 8192 \
  --port 8000
