"""用 vLLM 起张量并行服务:一个参数 tensor_parallel_size 把模型切到多卡。

需多卡实例 + pip install vllm。张量并行对调用方透明(API 不变)。
运行:python tp_serve.py --model Qwen/Qwen2.5-14B-Instruct --tp 2
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tp_serve")


def log_gpu_topology() -> None:
    """打印卡间互联拓扑,判断 TP 走的是 NVLink 还是 PCIe。"""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, check=True
        )
        logger.info("GPU 互联拓扑:\n%s", out.stdout)
        # TODO(实验 C): 解析输出矩阵,判断各卡对之间是 NV#(NVLink)还是 PHB/PXB(PCIe),
        #   在日志里明确标注「当前 TP 走的是 NVLink / PCIe」,用于解释 TP 加速比差异。
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("无法获取 GPU 拓扑: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    log_gpu_topology()

    from vllm import LLM, SamplingParams

    logger.info("加载模型 %s,tensor_parallel_size=%d", args.model, args.tp)
    # gpu_memory_utilization 留点余量给 KV cache;TP 时权重会被均摊到各卡
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.9,
    )

    params = SamplingParams(temperature=0.7, max_tokens=args.max_tokens)
    prompts = ["用一句话解释什么是张量并行,以及它为什么不是线性加速。"]
    for out in llm.generate(prompts, params):
        logger.info("生成: %s", out.outputs[0].text)


if __name__ == "__main__":
    main()
