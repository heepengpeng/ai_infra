"""迷你压测客户端:同时发起 N 个流式请求,统计每请求 TTFT 与整体 token 吞吐。

配合 mini_serving.py 使用,用来观察「并发增大时吞吐与 TTFT 的反向变化」。
运行:python load_client.py --concurrency 8
"""

import argparse
import asyncio
import json
import logging
import time

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("load_client")


async def one_request(
    client: httpx.AsyncClient, url: str, prompt: str, max_tokens: int
) -> tuple[float, int]:
    """发一个流式请求,返回 (TTFT 秒, 收到的 token 数)。"""
    start = time.monotonic()
    ttft = -1.0
    tokens = 0
    payload = {"prompt": prompt, "max_tokens": max_tokens}
    async with client.stream("POST", url, json=payload) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            json.loads(data)  # 校验是合法 SSE 数据块
            if ttft < 0:
                ttft = time.monotonic() - start
            tokens += 1
    return ttft, tokens


async def run(concurrency: int, max_tokens: int, url: str) -> None:
    async with httpx.AsyncClient(timeout=120.0) as client:
        wall_start = time.monotonic()
        tasks = [
            one_request(client, url, f"prompt-{i}", max_tokens)
            for i in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
        wall = time.monotonic() - wall_start

    ttfts = sorted(r[0] for r in results)
    total_tokens = sum(r[1] for r in results)
    p50 = ttfts[len(ttfts) // 2]
    p99 = ttfts[min(len(ttfts) - 1, int(len(ttfts) * 0.99))]
    logger.info("concurrency=%d", concurrency)
    logger.info("  TTFT p50=%.0fms p99=%.0fms", p50 * 1000, p99 * 1000)
    logger.info("  total_tokens=%d wall=%.2fs", total_tokens, wall)
    logger.info("  throughput=%.1f tokens/s", total_tokens / wall)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--url", default="http://localhost:8000/generate")
    args = parser.parse_args()
    asyncio.run(run(args.concurrency, args.max_tokens, args.url))


if __name__ == "__main__":
    main()
