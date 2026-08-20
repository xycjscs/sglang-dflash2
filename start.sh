#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the Qwen3.8-27B FP8 weights}"
: "${DFLASH_MODEL_DIR:?set DFLASH_MODEL_DIR to the DFlash2 draft weights}"

IMAGE="${IMAGE:-sglang-dflash2:latest}"
PORT="${PORT:-30000}"
GPUS="${GPUS:-all}"
TP="${TP:-2}"
API_KEY_ARGS=()
[[ -n "${API_KEY:-}" ]] && API_KEY_ARGS=(--api-key "$API_KEY")

docker run --rm --gpus "$GPUS" --shm-size 16g \
  -p "${PORT}:30000" \
  -v "$MODEL_DIR":/model \
  -v "$DFLASH_MODEL_DIR":/dflash-model \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --host 0.0.0.0 \
    --port 30000 \
    --model-path /model \
    --served-model-name Qwen \
    --tensor-parallel-size "$TP" \
    --context-length 262144 \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size 2048 \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs-prefill 2048 \
    --grammar-backend none \
    --attention-backend flashinfer \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path /dflash-model \
    --speculative-num-draft-tokens 8 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --mamba-radix-cache-strategy extra_buffer \
    --disable-fast-image-processor \
    --mm-process-config '{"image":{"max_pixels":1048576}}' \
    --limit-mm-data-per-request '{"image":8,"video":0}' \
    "${API_KEY_ARGS[@]}"
