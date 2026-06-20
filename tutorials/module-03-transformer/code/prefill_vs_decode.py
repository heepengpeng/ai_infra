"""实测自回归生成的 prefill 与 decode 两阶段耗时差异。

加载一个真实小模型(默认 GPT-2),分别测量:
  1. prefill:一次性处理长度为 S 的 prompt 的耗时与 shape;
  2. decode :逐 token 生成 N 个(此处刻意不用 KV Cache,Lesson 3 再加)的耗时。
对比两者「每 token 摊销耗时」,直观看到 decode 的串行/带宽代价。

依赖:torch、transformers。无 GPU 也能跑(走 CPU),数值不同但结论一致。
"""

import argparse
import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    # GPU 上 kernel 异步下发,计时前必须同步,否则测到的是「下发时间」而非真实执行时间。
    if device == "cuda":
        torch.cuda.synchronize()


def estimate_intensity(d_model: int, n_layers: int, seq_len: int) -> float:
    """粗估某阶段的计算强度 FLOP/Byte(只算 FFN+attn 投影主项,忽略常数)。

    思路:一次前向读权重字节数 ~ 2 * params(fp16);
         计算量 ~ 2 * params * seq_len(每个 token 都过一遍权重)。
    所以 FLOP/Byte ~ seq_len(约等于一次前向处理的 token 数)。

    TODO(练习):返回该阶段近似计算强度。提示:prefill 的 seq_len 是 prompt 长度,
    decode 单步的 seq_len 是 1。你会发现 decode 的计算强度约等于 1,极低。
    """
    raise NotImplementedError("请实现 estimate_intensity(见 TODO)")


@torch.no_grad()
def run_prefill(model, input_ids: torch.Tensor, device: str):
    sync(device)
    t0 = time.perf_counter()
    logits = model(input_ids).logits
    sync(device)
    elapsed = time.perf_counter() - t0
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return next_token, elapsed


@torch.no_grad()
def run_decode_no_cache(model, seq: torch.Tensor, n_gen: int, device: str):
    """不使用 KV Cache 的朴素 decode:每步把不断变长的整个序列重新前向。"""
    sync(device)
    t0 = time.perf_counter()
    for _ in range(n_gen):
        logits = model(seq).logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_token], dim=-1)
    sync(device)
    elapsed = time.perf_counter() - t0
    return seq, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--gen-len", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    logger.info("设备: %s,模型: %s", device, args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    # 构造一个长度约为 prompt_len 的输入(用重复文本凑长度,内容不重要)。
    base = "Large language model inference is fascinating. "
    text = base * (args.prompt_len // 6 + 1)
    input_ids = tokenizer(text, return_tensors="pt").input_ids[:, : args.prompt_len].to(device)
    logger.info("prompt 实际长度: %d", input_ids.shape[1])

    # warmup:首次前向含编译/缓存分配开销,先跑一次再计时(回忆 M1 L5)。
    _ = run_prefill(model, input_ids, device)

    next_token, t_prefill = run_prefill(model, input_ids, device)
    s = input_ids.shape[1]
    logger.info("[prefill] 输入 shape [1, %d] -> 耗时 %.1f ms,产出 1 个 token",
                s, t_prefill * 1e3)

    seq = torch.cat([input_ids, next_token], dim=-1)
    _, t_decode = run_decode_no_cache(model, seq, args.gen_len, device)
    per_tok_decode = t_decode / args.gen_len
    per_tok_prefill = t_prefill / s
    logger.info("[decode ] 生成 %d 个 token,总耗时 %.1f ms,平均每 token %.2f ms",
                args.gen_len, t_decode * 1e3, per_tok_decode * 1e3)
    logger.info("[对比   ] prefill %.3f ms/token vs decode %.2f ms/token -> decode 慢约 %.0f×",
                per_tok_prefill * 1e3, per_tok_decode * 1e3,
                per_tok_decode / max(per_tok_prefill, 1e-9))


if __name__ == "__main__":
    main()
