#!/usr/bin/env bash
# 用本仓库镜像在本地端口提供 Qwen3.8-27B FP8 + DFlash2。不含 Cloudflare。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE="${BASE_IMAGE:-lmsysorg/sglang:latest}"
IMAGE_NAME="${IMAGE_NAME:-sglang-dflash2:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-27-sglang-dflash2}"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/../qwen38-27-fp8}"
DFLASH_MODEL_DIR="${DFLASH_MODEL_DIR:-$SCRIPT_DIR/../z-lab-Qwen3.8-27B-DFlash2}"
API_KEY="${API_KEY:-}"
HOST_PORT="${HOST_PORT:-5000}"
CONTAINER_PORT="${CONTAINER_PORT:-5000}"
GPUS="${GPUS:-all}"
TP_SIZE="${TP_SIZE:-2}"
SGLANG_STARTUP_TIMEOUT_SECONDS="${SGLANG_STARTUP_TIMEOUT_SECONDS:-1200}"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/logs/${CONTAINER_NAME}-$(date +%Y%m%d-%H%M%S).log}"
DOCKERFILE="${DOCKERFILE:-$SCRIPT_DIR/Dockerfile}"
REBUILD_IMAGE="${REBUILD_IMAGE:-0}"

# 可选构建代理（配合 --network=host）
BUILD_HTTP_PROXY="${BUILD_HTTP_PROXY:-}"
BUILD_HTTPS_PROXY="${BUILD_HTTPS_PROXY:-${BUILD_HTTP_PROXY:-}}"
BUILD_ALL_PROXY="${BUILD_ALL_PROXY:-}"
BUILD_NO_PROXY="${BUILD_NO_PROXY:-localhost,127.0.0.1,::1}"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "MODEL_DIR not found: $MODEL_DIR" >&2
    exit 1
fi
if [[ ! -d "$DFLASH_MODEL_DIR" ]]; then
    echo "DFLASH_MODEL_DIR not found: $DFLASH_MODEL_DIR" >&2
    exit 1
fi

if [[ "$REBUILD_IMAGE" == "1" ]] || ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    build_args=(
        --network=host
        --build-arg "BASE_IMAGE=$BASE_IMAGE"
        -t "$IMAGE_NAME"
        -f "$DOCKERFILE"
        "$SCRIPT_DIR"
    )
    if [[ -n "$BUILD_HTTP_PROXY" ]]; then
        build_args=(
            --network=host
            --build-arg "BASE_IMAGE=$BASE_IMAGE"
            --build-arg "HTTP_PROXY=$BUILD_HTTP_PROXY"
            --build-arg "HTTPS_PROXY=$BUILD_HTTPS_PROXY"
            --build-arg "ALL_PROXY=$BUILD_ALL_PROXY"
            --build-arg "NO_PROXY=$BUILD_NO_PROXY"
            --build-arg "http_proxy=$BUILD_HTTP_PROXY"
            --build-arg "https_proxy=$BUILD_HTTPS_PROXY"
            --build-arg "all_proxy=$BUILD_ALL_PROXY"
            --build-arg "no_proxy=$BUILD_NO_PROXY"
            -t "$IMAGE_NAME"
            -f "$DOCKERFILE"
            "$SCRIPT_DIR"
        )
    fi
    docker build "${build_args[@]}"
fi

mkdir -p "$(dirname "$LOG_FILE")"
FP8_CFG_DIR="${FP8_TUNED_CONFIGS:-$SCRIPT_DIR/../fp8-tuned-configs}"
mkdir -p "$FP8_CFG_DIR"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --runtime nvidia --gpus "$GPUS" \
    --add-host "host.docker.internal:host-gateway" \
    -v "$MODEL_DIR":/model \
    -v "$DFLASH_MODEL_DIR":/dflash-model \
    -v "$(dirname "$LOG_FILE")":/logs \
    -v "$FP8_CFG_DIR":/fp8-tuned-configs:ro \
    -p "$HOST_PORT:$CONTAINER_PORT" \
    -d \
    --restart unless-stopped \
    --name "$CONTAINER_NAME" \
    --shm-size=16g \
    -e API_KEY="$API_KEY" \
    -e CONTAINER_PORT="$CONTAINER_PORT" \
    -e TP_SIZE="$TP_SIZE" \
    -e SGLANG_STARTUP_TIMEOUT_SECONDS="$SGLANG_STARTUP_TIMEOUT_SECONDS" \
    -e LOG_BASENAME="$(basename "$LOG_FILE")" \
    -e SGLANG_ENABLE_SPEC_V2=1 \
    -e NCCL_P2P_DISABLE=0 \
    -e SGLANG_EAGER_INPUT_NO_COPY=1 \
    "$IMAGE_NAME" \
    bash -lc '
set -euo pipefail

log_path="/logs/${LOG_BASENAME}"

if ! python3 -c "from sglang.srt.models.dflash import DFlash2DraftModel"; then
    echo "Failed to import DFlash2DraftModel. Rebuild this image (torch 2.13 + sglang-kernel 0.4.6)." >&2
    exit 1
fi

if compgen -G "/fp8-tuned-configs/*.json" > /dev/null; then
    fp8_cfg_dir=""
    for cand in \
        /sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs \
        /sgl-workspace/sglang/python/sglang/srt/layers/quantization/configs; do
        if [[ -d "$cand" ]]; then
            fp8_cfg_dir="$cand"
            break
        fi
    done
    if [[ -n "$fp8_cfg_dir" ]]; then
        cp /fp8-tuned-configs/*.json "$fp8_cfg_dir"/
        echo "Copied FP8 tuned configs into $fp8_cfg_dir"
    fi
fi

api_key_args=()
if [[ -n "${API_KEY:-}" ]]; then
    api_key_args=(--api-key "$API_KEY")
fi

python3 -m sglang.launch_server \
    --model-path /model \
    "${api_key_args[@]}" \
    --port "$CONTAINER_PORT" \
    --host 0.0.0.0 \
    --served-model-name Qwen \
    --tensor-parallel-size "$TP_SIZE" \
    --context-length 262144 \
    --kv-cache-dtype fp8_e4m3 \
    --max-running-requests 32 \
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
    --mm-process-config "{\"image\":{\"max_pixels\":1048576}}" \
    --limit-mm-data-per-request "{\"image\":8,\"video\":0}" \
    2>&1 | tee -a "$log_path" &
sglang_pid=$!

cleanup() {
    if kill -0 "$sglang_pid" >/dev/null 2>&1; then
        kill "$sglang_pid"
    fi
}
trap cleanup INT TERM

sglang_ready=0
elapsed=0
while (( elapsed < SGLANG_STARTUP_TIMEOUT_SECONDS )); do
    if (exec 3<>"/dev/tcp/127.0.0.1/${CONTAINER_PORT}") >/dev/null 2>&1; then
        sglang_ready=1
        break
    fi
    if ! kill -0 "$sglang_pid" >/dev/null 2>&1; then
        wait "$sglang_pid"
        exit $?
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done

if [[ "$sglang_ready" != "1" ]]; then
    echo "SGLang did not become ready on port ${CONTAINER_PORT} after ${SGLANG_STARTUP_TIMEOUT_SECONDS}s." >&2
    kill "$sglang_pid" >/dev/null 2>&1 || true
    wait "$sglang_pid"
    exit 1
fi

wait "$sglang_pid"
status=$?
cleanup
exit "$status"
'
