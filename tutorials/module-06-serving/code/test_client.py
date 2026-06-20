"""用标准 OpenAI 客户端访问 TRT-LLM / vLLM 服务,验证两者 API 层无缝替换。

需先 pip install openai。服务需以 OpenAI 兼容模式启动(trtllm-serve / vllm serve)。
运行:python test_client.py --port 8000 --stream
"""

import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("test_client")


def run_once(base_url: str, model: str, prompt: str, stream: bool) -> None:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    messages = [{"role": "user", "content": prompt}]

    start = time.monotonic()
    if stream:
        ttft = -1.0
        text_parts: list[str] = []
        resp = client.chat.completions.create(
            model=model, messages=messages, stream=True, max_tokens=128
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content or ""
            if delta and ttft < 0:
                ttft = time.monotonic() - start
            text_parts.append(delta)
        logger.info("TTFT=%.0fms", ttft * 1000)
        logger.info("生成: %s", "".join(text_parts))
    else:
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=128
        )
        logger.info("E2E=%.0fms", (time.monotonic() - start) * 1000)
        logger.info("生成: %s", resp.choices[0].message.content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen2.5-1.5B")
    parser.add_argument("--prompt", default="什么是张量并行?一句话。")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}/v1"
    run_once(base_url, args.model, args.prompt, args.stream)

    # TODO(选做 A/B 对比): 接收两个 base_url(vLLM 与 TRT-LLM),对同一批 prompt 各压一轮,
    #   统计两者 TTFT p99 与 token 吞吐,打印对比表,用数据决策选型。


if __name__ == "__main__":
    main()
