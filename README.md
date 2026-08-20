# SGLang DFlash2 (CUDA 13)

CUDA 13 overlay on `lmsysorg/sglang:latest` for **Qwen3.8-27B FP8 + DFlash2**. Hub `latest` is still 0.5.17 and has no `DFlash2DraftModel`.

No Cloudflare. The API is published with `-p`.

[中文文档](README.zh.md)

## Verified stack

2× RTX 5090 (SM120):

| Component | Version |
| --- | --- |
| SGLang | `0.0.0.dev16855+g5f1283959` |
| PyTorch | `2.13.0+cu130` |
| sglang-kernel | `0.4.6.post1+cu130` |
| flashinfer | `0.6.17` (python / cubin / jit-cache) |
| NCCL | `nvidia-nccl-cu13 2.30.7` |

Do not upgrade `sglang-kernel` alone: the Hub image ships torch 2.11, which ABI-mismatches kernel 0.4.6 on SM120 (`undefined symbol`). Bump torch 2.13 together with the kernel.

## Build

The build downloads ~1.5GB of wheels from GitHub Releases.

```bash
docker build -t sglang-dflash2:latest .
```

Proxy:

```bash
docker build --network=host \
  --build-arg HTTP_PROXY=http://127.0.0.1:20172 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:20172 \
  -t sglang-dflash2:latest .
```

## Run

Weights:

- Target: local FP8 of `Qwen/Qwen3.8-27B` (`lm_head` not quantized)
- Draft: [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (`block_size=8`)

```bash
export MODEL_DIR=/path/to/qwen38-27-fp8
export DFLASH_MODEL_DIR=/path/to/Qwen3.8-27B-DFlash2
# optional
export API_KEY=your-key
export GPUS=all          # or '"device=0,1"'
export PORT=30000
export TP=2

./start.sh
```

```bash
curl http://127.0.0.1:30000/v1/models
```

## Launch flags (2×32GB)

Keep `--cuda-graph-max-bs-prefill` and `--chunked-prefill-size` equal. Do not raise `--mem-fraction-static` above 0.85. Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (breaks TP=2 custom all-reduce IPC).

| Flag | Value | Why |
| --- | --- | --- |
| `--mem-fraction-static` | 0.85 | ~259k KV; higher OOMs on real requests |
| `--cuda-graph-max-bs-prefill` | 2048 | 4096 graphs are too large |
| `--chunked-prefill-size` | 2048 | must match the graph cap |
| `--speculative-num-draft-tokens` | 8 | draft `block_size=8` |
| `--mamba-radix-cache-strategy` | `extra_buffer` | Qwen3.5/3.8 hybrid GDN |
| `--disable-fast-image-processor` | on | GPU image preprocess OOMs |
| `--grammar-backend` | `none` | no `json_schema` |

Mamba caps concurrency to about 3.

## Upstream DFlash2

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 \
  --speculative-num-draft-tokens 8
```
