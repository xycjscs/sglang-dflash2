# SGLang DFlash2 (CUDA 13)

CUDA 13 overlay on `lmsysorg/sglang:latest` for **Qwen3.8-27B FP8 + DFlash2**. Hub `latest` is still 0.5.17 and has no `DFlash2DraftModel`.

在 `lmsysorg/sglang:latest`（0.5.17，**没有** `DFlash2DraftModel`）上叠加 CUDA 13 运行栈，用于 **Qwen3.8-27B FP8 + DFlash2**。

No Cloudflare. The API is published with `-p`.

不含 Cloudflare。HTTP API 通过 `-p` 映射到本机。

## Verified stack / 已验证版本

2× RTX 5090 (SM120):

| Component / 组件 | Version / 版本 |
| --- | --- |
| SGLang | `0.0.0.dev16855+g5f1283959` |
| PyTorch | `2.13.0+cu130` |
| sglang-kernel | `0.4.6.post1+cu130` |
| flashinfer | `0.6.17` (python / cubin / jit-cache) |
| NCCL | `nvidia-nccl-cu13 2.30.7` |

Do not upgrade `sglang-kernel` alone: the Hub image ships torch 2.11, which ABI-mismatches kernel 0.4.6 on SM120 (`undefined symbol`). Bump torch 2.13 together with the kernel.

不要只升 kernel：Hub 底包是 torch 2.11，和 0.4.6 在 SM120 上 ABI 对不上（`undefined symbol`）。必须和 torch 2.13 一起升。

## Build / 构建

The build downloads ~1.5GB of wheels from GitHub Releases.

构建期会从 GitHub Releases 拉取约 1.5GB 的 wheel。

```bash
docker build -t sglang-dflash2:latest .
```

Proxy / 代理:

```bash
docker build --network=host \
  --build-arg HTTP_PROXY=http://127.0.0.1:20172 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:20172 \
  -t sglang-dflash2:latest .
```

## Run / 运行

Weights / 权重:

- Target: local FP8 of `Qwen/Qwen3.8-27B` (`lm_head` not quantized / `lm_head` 不要量化)
- Draft: [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (`block_size=8`)

```bash
export MODEL_DIR=/path/to/qwen38-27-fp8
export DFLASH_MODEL_DIR=/path/to/Qwen3.8-27B-DFlash2
# optional / 可选
export API_KEY=your-key
export GPUS=all          # or / 或 '"device=0,1"'
export PORT=30000
export TP=2

./start.sh
```

```bash
curl http://127.0.0.1:30000/v1/models
```

## Launch flags (2×32GB) / 启动参数

Keep `--cuda-graph-max-bs-prefill` and `--chunked-prefill-size` equal. Do not raise `--mem-fraction-static` above 0.85. Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (breaks TP=2 custom all-reduce IPC).

`--cuda-graph-max-bs-prefill` 必须和 `--chunked-prefill-size` 一致。`--mem-fraction-static` 不要超过 0.85。不要开 `expandable_segments`（和 TP=2 custom all-reduce IPC 冲突）。

| Flag | Value | Why / 原因 |
| --- | --- | --- |
| `--mem-fraction-static` | 0.85 | ~259k KV; higher OOMs on real requests / 再高真实请求会 OOM |
| `--cuda-graph-max-bs-prefill` | 2048 | 4096 graphs are too large / 4096 图太吃显存 |
| `--chunked-prefill-size` | 2048 | must match the graph cap / 必须和图表对齐 |
| `--speculative-num-draft-tokens` | 8 | draft `block_size=8` |
| `--mamba-radix-cache-strategy` | `extra_buffer` | Qwen3.5/3.8 hybrid GDN |
| `--disable-fast-image-processor` | on / 开 | GPU image preprocess OOMs / 大图 GPU 预处理会 OOM |
| `--grammar-backend` | `none` | no `json_schema` / 不要发 json_schema |

Mamba caps concurrency to about 3. / Mamba 会把并发压到约 3。

## Upstream DFlash2

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 \
  --speculative-num-draft-tokens 8
```
