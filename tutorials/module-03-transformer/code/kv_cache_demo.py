"""KV Cache:显存计算器 + 实测提速对比。

Part A:按超参推算单请求 KV Cache 大小与最大并发数(复算 Lesson 3 的 7B 例子)。
Part B:用真实小模型(GPT-2)对比「不用 cache」与「用 cache」生成相同长度的耗时。

依赖:torch、transformers。
"""

import argparse
import logging
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    name: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    weight_gb: float       # fp16 权重占用(GB)
    bytes_per_elem: int = 2  # fp16=2, fp8=1


def kv_cache_bytes(spec: ModelSpec, seq_len: int, batch: int) -> int:
    """计算 KV Cache 总字节数。

    公式:L * 2 * S * n_kv * head_dim * P * B
    其中 2 表示 K 和 V 各一份,P 为每个元素字节数。

    TODO(练习):按公式返回字节数。注意 GQA 模型用 n_kv_heads * head_dim,
    不要用 hidden_size。完成后下方应能复算出 7B / S=2048 ≈ 1.0 GB。
    """
    raise NotImplementedError("请实现 kv_cache_bytes(见 TODO)")


def report_capacity(spec: ModelSpec, seq_len: int, total_gb: float) -> None:
    per_req = kv_cache_bytes(spec, seq_len, batch=1) / 1e9
    avail = total_gb - spec.weight_gb
    max_concurrency = int(avail / per_req) if per_req > 0 else 0
    logger.info("=== 显存计算器:%s, S=%d, B=1 ===", spec.name, seq_len)
    logger.info("单请求 KV Cache: %.2f GB", per_req)
    logger.info("%.0fGB 卡(权重 %.1fGB)最大并发 ≈ %d 条",
                total_gb, spec.weight_gb, max_concurrency)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def decode_no_cache(model, input_ids: torch.Tensor, n_gen: int, device: str) -> float:
    seq = input_ids
    sync(device)
    t0 = time.perf_counter()
    for _ in range(n_gen):
        logits = model(seq).logits
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=-1)
    sync(device)
    return time.perf_counter() - t0


@torch.no_grad()
def decode_with_cache(model, input_ids: torch.Tensor, n_gen: int, device: str) -> float:
    past = None
    cur = input_ids  # 第一次喂整个 prompt(prefill),之后只喂 1 个 token
    sync(device)
    t0 = time.perf_counter()
    for _ in range(n_gen):
        out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cur = nxt
    sync(device)
    return time.perf_counter() - t0


def benchmark(model_name: str, prompt_len: int, gen_len: int) -> None:
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    text = "Inference engines cache keys and values. " * (prompt_len // 6 + 1)
    input_ids = tokenizer(text, return_tensors="pt").input_ids[:, :prompt_len].to(device)

    # warmup
    _ = decode_with_cache(model, input_ids, 2, device)

    t_no = decode_no_cache(model, input_ids, gen_len, device)
    t_yes = decode_with_cache(model, input_ids, gen_len, device)
    logger.info("=== 实测:%s 生成 %d token(prompt=%d, 设备=%s) ===",
                model_name, gen_len, prompt_len, device)
    logger.info("[无 cache] 总耗时 %.0f ms  (每步重算整段)", t_no * 1e3)
    logger.info("[有 cache] 总耗时 %.0f ms  (每步只喂 1 token)", t_yes * 1e3)
    logger.info("加速比 ≈ %.1f×", t_no / max(t_yes, 1e-9))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--gen-len", type=int, default=128)
    args = parser.parse_args()

    # Part A:复算 Lesson 3 的 LLaMA-2-7B 例子
    llama7b = ModelSpec("LLaMA-2-7B", n_layers=32, n_kv_heads=32,
                        head_dim=128, weight_gb=14.0)
    report_capacity(llama7b, seq_len=2048, total_gb=24.0)

    # Part B:实测提速
    benchmark(args.model, args.prompt_len, args.gen_len)


if __name__ == "__main__":
    main()
