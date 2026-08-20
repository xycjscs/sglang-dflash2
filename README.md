# SGLang DFlash2 (CUDA 13)

在 `lmsysorg/sglang:latest`（0.5.17，**没有** `DFlash2DraftModel`）上叠加 [SGLang](https://github.com/sgl-project/sglang) main 与匹配的 CUDA 13 运行栈，用于 **Qwen3.8-27B FP8 + DFlash2** 投机解码。

不含 Cloudflare / 隧道。HTTP API 通过 `-p` 映射到本机端口。

## 已验证版本

在 2× RTX 5090（SM120）上跑通：

| 组件 | 版本 |
| --- | --- |
| SGLang | `0.0.0.dev16855+g5f1283959` |
| PyTorch | `2.13.0+cu130` |
| sglang-kernel | `0.4.6.post1+cu130` |
| flashinfer | `0.6.17`（python / cubin / jit-cache） |
| NCCL | `nvidia-nccl-cu13 2.30.7` |

官方 Hub 镜像仍是 torch 2.11 + kernel 0.4.5。只升 kernel 会在 SM120 上出现 `undefined symbol`（ABI 不匹配），必须整套升到 torch 2.13。

## 构建

构建机会从 GitHub Releases 拉取约 1.5GB 的 flashinfer jit-cache 和 sglang-kernel wheel。需要能访问 `github.com`。走代理时：

```bash
docker build --network=host \
  --build-arg HTTP_PROXY=http://127.0.0.1:20172 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:20172 \
  -t sglang-dflash2:latest .
```

不走代理：

```bash
docker build -t sglang-dflash2:latest .
```

## 运行 Qwen3.8-27B FP8 + DFlash2

准备本地权重：

- Target：`Qwen/Qwen3.8-27B` 的 FP8 目录（`lm_head` 不要量化）
- Draft：[`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)（`block_size=8`）

```bash
export MODEL_DIR=/path/to/qwen38-27-fp8
export DFLASH_MODEL_DIR=/path/to/Qwen3.8-27B-DFlash2
export API_KEY=your-key          # 可选；不设则不启用 --api-key
export GPUS=all                  # 或 '"device=0,1"'
export HOST_PORT=5000
export TP_SIZE=2

./start-qwen38-27-fp8-dflash2.sh
```

脚本会把容器 `5000` 映射到宿主机 `$HOST_PORT`。就绪后：

```bash
curl http://127.0.0.1:5000/v1/models
```

## 启动参数说明（2× 32GB，DFlash2）

这些是 5090×2 上能稳定服务的组合，不要随意把 prefill 图或 `mem-fraction` 再往上加：

| 参数 | 值 | 原因 |
| --- | --- | --- |
| `--tensor-parallel-size` | 2 | 双卡 |
| `--context-length` | 262144 | 模型 RoPE 上限 |
| `--kv-cache-dtype` | `fp8_e4m3` | 省 KV |
| `--mem-fraction-static` | 0.85 | KV 约 259k token；再高真实请求会 OOM |
| `--cuda-graph-max-bs-prefill` | 2048 | 4096 图太吃显存 |
| `--chunked-prefill-size` | 2048 | 必须和 prefill 图表对齐，否则 eager 8192 会 OOM |
| `--speculative-algorithm` | `DFLASH` | DFlash2 |
| `--speculative-num-draft-tokens` | 8 | 与 draft `block_size=8` 一致 |
| `--mamba-radix-cache-strategy` | `extra_buffer` | Qwen3.5/3.8 hybrid GDN |
| `--disable-fast-image-processor` | 开 | Fast processor 在 GPU0 上预处理大图会 OOM |
| `--mm-process-config` | `max_pixels=1048576` | 限制视觉分辨率 |
| `--grammar-backend` | `none` | 省显存；不要发 `json_schema` |

Mamba 会把并发压到约 3。不要用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：和 TP=2 custom all-reduce IPC 冲突。

## 官方 DFlash2 入口

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 \
  --speculative-num-draft-tokens 8
```

本仓库只是把能在 CUDA 13 / RTX 5090 上跑起来的依赖叠进镜像，并给出一份内存安全的启动脚本。
