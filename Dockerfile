# SGLang + DFlash2 overlay on lmsysorg/sglang:latest (0.5.17 has no DFlash2DraftModel).
# Verified stack: sglang 0.0.0.dev16855+g5f1283959, torch 2.13.0+cu130,
# sglang-kernel 0.4.6.post1+cu130, flashinfer 0.6.17.
#
# Optional build proxy:
#   docker build --network=host \
#     --build-arg HTTP_PROXY=http://127.0.0.1:20172 \
#     --build-arg HTTPS_PROXY=http://127.0.0.1:20172 \
#     -t sglang-dflash2:latest .

ARG BASE_IMAGE=lmsysorg/sglang:latest
FROM ${BASE_IMAGE}

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG all_proxy
ARG no_proxy
ARG SGLANG_COMMIT=5f1283959

# ALL_PROXY=socks5://... 时 httpx 需要 SOCKS 支持
RUN pip install --no-cache-dir "httpx[socks]" socksio

WORKDIR /sgl-workspace/sglang
RUN git fetch origin main \
    && git checkout --force "${SGLANG_COMMIT}" \
    && pip install --no-deps --no-cache-dir -e python \
    && python3 -c "from sglang.srt.models.dflash import DFlash2DraftModel"

# SGLang main 要求 flashinfer_python>=0.6.17；底包仍是 0.6.15。cubin/jit-cache 必须同版本。
RUN pip install --no-cache-dir --no-deps -U "flashinfer-python==0.6.17"
RUN pip install --no-cache-dir --no-deps -U "flashinfer-cubin==0.6.17" \
        --index-url https://flashinfer.ai/whl

# jit-cache 在 GitHub Releases（约 1.5GB），构建期请保证能访问 github.com。
RUN curl -fL --retry 5 --retry-delay 2 -o /tmp/flashinfer_jit_cache.whl \
        "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.17/flashinfer_jit_cache-0.6.17+cu130-cp39-abi3-manylinux_2_28_x86_64.whl" \
    && pip install --no-cache-dir --no-deps /tmp/flashinfer_jit_cache.whl \
    && rm -f /tmp/flashinfer_jit_cache.whl \
    && python3 -c "from importlib.metadata import version; \
v=version('flashinfer-python'); c=version('flashinfer-cubin'); j=version('flashinfer-jit-cache'); \
assert v=='0.6.17' and c=='0.6.17' and j.startswith('0.6.17'), (v,c,j); \
import flashinfer"

# SGLang main 的 CUDA 13 栈：torch 2.13 + 配套 kernel。底包仍是 torch 2.11.0+cu130，
# 只升 sglang-kernel 会 ABI 对不上（undefined symbol in common_ops.abi3.so）。
RUN pip install --no-cache-dir --force-reinstall \
        torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu130 \
    && pip install --no-cache-dir --force-reinstall --no-deps \
        torchcodec==0.15.0 \
        --index-url https://download.pytorch.org/whl/cu130 \
    && pip install --no-cache-dir --force-reinstall --no-deps \
        nvidia-nccl-cu13==2.30.7 \
        "cuda-tile==1.6.0rc5" \
        quack-kernels==0.6.4 \
        nvidia-cutlass-dsl==4.6.2 \
        nvidia-cutlass-dsl-libs-base==4.6.2 \
        nvidia-cutlass-dsl-libs-cu13==4.6.2

RUN curl -fL --retry 5 --retry-delay 2 -o /tmp/sglang_kernel.whl \
        "https://github.com/sgl-project/whl/releases/download/v0.4.6.post1/sglang_kernel-0.4.6.post1+cu130-cp310-abi3-manylinux2014_x86_64.whl" \
    && pip install --no-cache-dir --no-deps --force-reinstall /tmp/sglang_kernel.whl \
    && rm -f /tmp/sglang_kernel.whl

RUN pip install --no-cache-dir --no-deps --force-reinstall \
        "https://github.com/sgl-project/whl/releases/download/v0.1.5.post3/sgl_deep_gemm-0.1.5.post3+cu130-py3-none-manylinux2014_x86_64.whl" \
        "https://github.com/sgl-project/whl/releases/download/v0.1.1/sgl_deep_ep-0.1.1+cu130-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" \
    && python3 -c "from importlib.metadata import version; from pathlib import Path; \
tv=version('torch'); kv=version('sglang-kernel'); \
assert tv.startswith('2.13'), tv; \
assert kv.startswith('0.4.6'), kv; \
assert 'class DFlash2DraftModel' in Path('/sgl-workspace/sglang/python/sglang/srt/models/dflash.py').read_text(); \
print('torch', tv, 'kernel', kv, 'DFlash2 source ok')"

# 清除构建期代理，避免镜像内写死本机地址
ENV HTTP_PROXY= \
    HTTPS_PROXY= \
    ALL_PROXY= \
    NO_PROXY= \
    http_proxy= \
    https_proxy= \
    all_proxy= \
    no_proxy=
