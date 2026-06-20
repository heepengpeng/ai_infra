#!/usr/bin/env bash
# TensorRT-LLM 显式三步编译流程(路线 B):convert -> build -> serve。
# 体会「编译型工具链」与编译期形状契约的硬约束。
#
# 注意:convert_checkpoint.py / quantize.py 是 TRT-LLM 官方按模型族提供的脚本,
#   位于你安装的 TRT-LLM 仓库 examples/<model>/ 目录下(不同模型族脚本不同)。
#   请从那里取对应模型的脚本,本文件只串联流程与说明参数含义。

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./Qwen2.5-1.5B}"
CKPT_DIR="${CKPT_DIR:-./ckpt_fp16}"
ENGINE_DIR="${ENGINE_DIR:-./engine_fp16}"
PORT="${PORT:-8000}"

echo "==> [1/3] 转换 HF 权重 -> TRT-LLM checkpoint (FP16)"
python convert_checkpoint.py \
    --model_dir "${MODEL_DIR}" \
    --output_dir "${CKPT_DIR}" \
    --dtype float16

echo "==> [2/3] 编译 engine (最耗时,正在做 autotuning / 算子融合)"
# max_input_len / max_seq_len 是编译期形状上界:运行期请求不能超过它
time trtllm-build \
    --checkpoint_dir "${CKPT_DIR}" \
    --output_dir "${ENGINE_DIR}" \
    --gemm_plugin float16 \
    --max_batch_size 16 \
    --max_input_len 2048 \
    --max_seq_len 4096

echo "==> 产物大小:"
du -sh "${ENGINE_DIR}"

echo "==> [3/3] 启动 OpenAI 兼容服务,端口 ${PORT}"
trtllm-serve "${ENGINE_DIR}" --tokenizer "${MODEL_DIR}" --port "${PORT}"
