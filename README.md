# SGLang DFlash2 (CUDA 13)

CUDA 13 overlay on `lmsysorg/sglang:latest` for **Qwen3.8-27B FP8 + DFlash2**. Hub `latest` is still 0.5.17 and has no `DFlash2DraftModel`.

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

```bash
docker build -t sglang-dflash2:latest .
```

## Run

Weights:

- Target: local FP8 of `Qwen/Qwen3.8-27B` (`lm_head` not quantized)
- Draft: [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) (`block_size=8`)

```bash
export MODEL_DIR=/path/to/qwen38-27-fp8
export DFLASH_MODEL_DIR=/path/to/z-lab-Qwen3.8-27B-DFlash2
export TP=2

./start.sh
```

```bash
curl http://127.0.0.1:30000/v1/models
```

Optional: `API_KEY`, `GPUS` (default `all`), `PORT` (default `30000`).

## Performance (2× RTX 5090, TP=2)

Measured from production logs of this image and `start.sh` flags. Prefill for large requests uses whole-request wall time.

### Prefill

| | tok/s |
| --- | --- |
| Large requests (≥2k tokens, n=93), mean | 4391 |
| Median | 3736 |
| p10 / p90 | 2257 / 7331 |
| Token-weighted | 4583 |
| Longest request (108k tokens) | 3731 (29 s) |
| Short turns (<2k tokens, median) | ~470 (includes scheduling gaps) |

The first chunk of each large request is slowed by Triton JIT (one logged chunk at 25.6 tok/s). Later chunks return to 4k–5k tok/s.

### Decode (DFlash2)

| | |
| --- | --- |
| Mean / median / p90 / peak | 172 / 156 / 280 / 542 tok/s |
| Mean accept length | 3.5 tokens / step (8 draft tokens) |
| Accept rate | ~0.40 (p10 0.20 / p90 0.59) |

Throughput barely drops with context length (hybrid GDN linear attention):

| Context | tok/s |
| --- | --- |
| 0–20k | 182 |
| 40–60k | 162 |
| 120–140k | 201 |
| 140–160k | 151 |

Samples at 100–150 tok/s line up with accept rate 0.15–0.25 (draft misses, close to raw decode).

## Launch flags (2×32GB)

Keep `--cuda-graph-max-bs-prefill` and `--chunked-prefill-size` equal. Do not raise `--mem-fraction-static` above 0.85.

| Flag | Value | Why |
| --- | --- | --- |
| `--mem-fraction-static` | 0.85 | ~259k KV; higher OOMs on real requests |
| `--cuda-graph-max-bs-prefill` | 2048 | 4096 graphs are too large |
| `--chunked-prefill-size` | 2048 | must match the graph cap |
| `--speculative-algorithm` | `DFLASH` | DFlash2 |
| `--speculative-num-draft-tokens` | 8 | draft `block_size=8` |
| `--mamba-radix-cache-strategy` | `extra_buffer` | Qwen3.5/3.8 hybrid GDN |

Mamba caps concurrency to about 3.

## Upstream DFlash2

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.8-27B-DFlash2 \
  --speculative-num-draft-tokens 8
```
