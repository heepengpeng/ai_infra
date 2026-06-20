"""TensorRT-LLM 高阶 API 快速上手:接口和 vLLM 几乎一样,先用它跑通建立信心。

首次运行会自动为本机 GPU 构建并缓存 engine(可能要几分钟,属正常)。
需先安装:pip install tensorrt-llm -U,并准备好本地模型权重目录。

运行:python trtllm_quickstart.py --model ./Qwen2.5-1.5B
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("trtllm_quickstart")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./Qwen2.5-1.5B")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    # 延迟导入:没装 TRT-LLM 时也能看脚本结构与报错指引
    from tensorrt_llm import LLM, SamplingParams

    logger.info("加载/构建 engine: %s (首次会自动编译,请耐心等待)", args.model)
    llm = LLM(model=args.model)

    params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=args.max_tokens)
    prompts = [
        "用一句话解释什么是张量并行。",
        "推理服务里 TTFT 和 TPOT 分别指什么?",
    ]
    for out in llm.generate(prompts, params):
        logger.info("Prompt: %s", out.prompt)
        logger.info("生成: %s", out.outputs[0].text)


if __name__ == "__main__":
    main()
