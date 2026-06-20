"""M5 L5 动手实验:KV Cache 显存计算器。

CPU 即可。运行:
    python kvcache_calc.py

直观看到:长上下文时 KV Cache 如何反超模型权重,以及量化它能省多少。
"""

import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    name: str
    num_layers: int
    num_kv_heads: int      # 注意 GQA 下 KV 头数远少于注意力头数
    head_dim: int
    weight_gb_fp16: float  # 模型权重在 FP16 下的大小(GB),用于对照


def kv_cache_bytes(spec: ModelSpec, seq_len: int, batch: int,
                   dtype_bytes: int) -> float:
    """KV Cache 字节数 = 2(K,V) * L * H_kv * d * S * B * dtype_bytes。"""
    return (2 * spec.num_layers * spec.num_kv_heads * spec.head_dim
            * seq_len * batch * dtype_bytes)


def report(spec: ModelSpec, seq_len: int, batch: int) -> None:
    kv_fp16 = kv_cache_bytes(spec, seq_len, batch, 2) / 1024 ** 3
    kv_fp8 = kv_cache_bytes(spec, seq_len, batch, 1) / 1024 ** 3
    logger.info("[%s] S=%d B=%d", spec.name, seq_len, batch)
    logger.info("  权重(FP16)      = %.1f GB", spec.weight_gb_fp16)
    logger.info("  KV Cache (FP16) = %.1f GB  (%.1fx 权重)",
                kv_fp16, kv_fp16 / spec.weight_gb_fp16)
    logger.info("  KV Cache (FP8)  = %.1f GB  (省一半)", kv_fp8)
    logger.info("")


def main() -> None:
    # Llama-2-7B 近似规格(32 层,32 KV 头,head_dim=128,权重约 13GB)
    llama7b = ModelSpec("Llama-2-7B", num_layers=32, num_kv_heads=32,
                        head_dim=128, weight_gb_fp16=13.0)

    logger.info("=== 短上下文:权重是大头 ===")
    report(llama7b, seq_len=512, batch=1)

    logger.info("=== 长上下文 + 大并发:KV Cache 反超权重 ===")
    report(llama7b, seq_len=32768, batch=8)
    report(llama7b, seq_len=131072, batch=4)


if __name__ == "__main__":
    main()
