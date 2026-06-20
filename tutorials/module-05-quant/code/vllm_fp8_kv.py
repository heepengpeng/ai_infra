"""M5 L5 实战:在 vLLM 中开启 FP8 KV Cache 并对比。

需要 N 卡(FP8 KV 的最佳硬件是 Hopper)。运行:
    python vllm_fp8_kv.py

对比 FP16 KV 与 FP8 KV 的 KV Cache 容量(可缓存 token 数)与输出质量。
"""

import logging

try:
    from vllm import LLM, SamplingParams
except ImportError as exc:
    raise SystemExit("缺少 vllm,请先 `pip install vllm`") from exc

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL = "facebook/opt-125m"   # TODO:换成你要测的模型(建议用支持长上下文的)
PROMPTS = [
    "Explain why KV cache grows with context length.",
    "In one sentence, what is FP8 E4M3?",
]


def run(kv_dtype: str | None) -> None:
    """用指定 KV dtype 启动 vLLM,打印 KV 容量与生成结果。"""
    label = kv_dtype or "fp16(default)"
    kwargs = {"model": MODEL, "gpu_memory_utilization": 0.6}
    if kv_dtype:
        kwargs["kv_cache_dtype"] = kv_dtype  # 关键开关:KV Cache 用 FP8

    llm = LLM(**kwargs)
    # num_gpu_blocks * block_size 约等于可缓存的总 token 数,反映 KV 容量
    cache_cfg = llm.llm_engine.cache_config
    blocks = getattr(cache_cfg, "num_gpu_blocks", None)
    block_size = getattr(cache_cfg, "block_size", None)
    if blocks and block_size:
        logger.info("[KV=%s] 可缓存 token 数 ≈ %d", label, blocks * block_size)

    out = llm.generate(PROMPTS, SamplingParams(max_tokens=48, temperature=0.0))
    for o in out:
        logger.info("[KV=%s] %s", label, o.outputs[0].text.strip()[:120])
    # 释放当前实例显存,便于在同进程内跑下一种配置;实际中分两次运行更稳
    del llm


def main() -> None:
    logger.info("=== 默认 FP16 KV ===")
    run(None)
    logger.info("\n=== FP8 KV(显存减半,容量约翻倍)===")
    run("fp8")


if __name__ == "__main__":
    main()
