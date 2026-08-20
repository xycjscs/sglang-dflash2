# SGLang DFlash2 (CUDA 13)

在 `lmsysorg/sglang:latest`（0.5.17，**没有** `DFlash2DraftModel`）上叠加 CUDA 13 运行栈，用于 **Qwen3.8-27B FP8 + DFlash2**。

[English](README.md)

## 已验证版本

2× RTX 5090（SM120）：

| 组件 | 版本 |
| --- | --- |
| SGLang | `0.0.0.dev16855+g5f1283959` |
| PyTorch | `2.13.0+cu130` |
| sglang-kernel | `0.4.6.post1+cu130` |
| flashinfer | `0.6.17`（python / cubin / jit-cache） |
| NCCL | `nvidia-nccl-cu13 2.30.7` |

不要只升 kernel：Hub 底包是 torch 2.11，和 0.4.6 在 SM120 上 ABI 对不上（`undefined symbol`）。必须和 torch 2.13 一起升。

## 构建

```bash
docker build -t sglang-dflash2:latest .
```

## 运行

权重：

- Target：`Qwen/Qwen3.8-27B` 的本地 FP8（`lm_head` 不要量化）
- Draft：[`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)（`block_size=8`）

```bash
export MODEL_DIR=/path/to/qwen38-27-fp8
export DFLASH_MODEL_DIR=/path/to/Qwen3.8-27B-DFlash2
export TP=2

./start.sh
```

```bash
curl http://127.0.0.1:30000/v1/models
```

可选：`API_KEY`、`GPUS`（默认 `all`）、`PORT`（默认 `30000`）。

## 启动参数（2×32GB）

`--cuda-graph-max-bs-prefill` 必须和 `--chunked-prefill-size` 一致。`--mem-fraction-static` 不要超过 0.85。

| 参数 | 值 | 原因 |
| --- | --- | --- |
| `--mem-fraction-static` | 0.85 | KV 约 259k token；再高真实请求会 OOM |
| `--cuda-graph-max-bs-prefill` | 2048 | 4096 图太吃显存 |
| `--chunked-prefill-size` | 2048 | 必须和图表对齐 |
| `--speculative-algorithm` | `DFLASH` | DFlash2 |
| `--speculative-num-draft-tokens` | 8 | 与 draft `block_size=8` 一致 |
| `--mamba-radix-cache-strategy` | `extra_buffer` | Qwen3.5/3.8 hybrid GDN |

Mamba 会把并发压到约 3。

## 官方 DFlash2 入口

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.8-27B-DFlash2 \
  --speculative-num-draft-tokens 8
```
