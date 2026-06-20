"""LLM 服务压测器:逐并发档位压测,采集 TTFT/E2E 的 P50/P99 与 token 吞吐,输出 CSV。

被测服务需为 OpenAI 兼容(vllm serve / trtllm-serve)。配合 plot_curve.py 画吞吐-延迟曲线找拐点。
运行:python benchmark.py --url http://localhost:8000/v1 --model <name> \
        --concurrency 1,2,4,8,16,32 --input-len 256 --output-len 200 --out result.csv
"""

import argparse
import asyncio
import csv
import logging
import time

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("benchmark")


def make_prompt(input_len: int) -> str:
    """构造大致 input_len 个 token 的 prompt(用重复词近似,真实压测应回放线上分布)。"""
    return "请基于以下内容继续写作。" + "数据 " * max(1, input_len - 6)


async def one_request(
    client: httpx.AsyncClient, url: str, model: str, prompt: str, output_len: int
) -> tuple[float, float, int]:
    """发一个流式请求,返回 (TTFT 秒, E2E 秒, 输出 token 数)。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len,
        "stream": True,
        "temperature": 0.7,
    }
    start = time.monotonic()
    ttft = -1.0
    tokens = 0
    async with client.stream("POST", f"{url}/chat/completions", json=payload) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            if ttft < 0:
                ttft = time.monotonic() - start
            tokens += 1
    e2e = time.monotonic() - start
    return ttft, e2e, tokens


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[idx]


async def run_level(
    url: str, model: str, concurrency: int, input_len: int, output_len: int
) -> dict:
    """对单个并发档位压测一轮,返回该档位的聚合指标。"""
    prompt = make_prompt(input_len)
    async with httpx.AsyncClient(timeout=300.0) as client:
        wall_start = time.monotonic()
        tasks = [
            one_request(client, url, model, prompt, output_len)
            for _ in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
        wall = time.monotonic() - wall_start

    ttfts = [r[0] for r in results if r[0] >= 0]
    e2es = [r[1] for r in results]
    total_tokens = sum(r[2] for r in results)
    return {
        "concurrency": concurrency,
        "ttft_p50": round(percentile(ttfts, 0.5), 3),
        "ttft_p99": round(percentile(ttfts, 0.99), 3),
        "e2e_p50": round(percentile(e2es, 0.5), 3),
        "e2e_p99": round(percentile(e2es, 0.99), 3),
        "throughput_tok_s": round(total_tokens / wall, 1),
    }


async def main_async(args: argparse.Namespace) -> None:
    levels = [int(c) for c in args.concurrency.split(",")]
    rows: list[dict] = []
    for c in levels:
        row = await run_level(args.url, args.model, c, args.input_len, args.output_len)
        rows.append(row)
        logger.info(
            "并发=%d  TTFT p99=%.2fs  E2E p99=%.2fs  吞吐=%.1f tok/s",
            c, row["ttft_p99"], row["e2e_p99"], row["throughput_tok_s"],
        )
        # 档位之间稍歇,让服务回到稳态再压下一档
        await asyncio.sleep(2)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("结果已写入 %s", args.out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", default="1,2,4,8,16,32,64")
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=200)
    parser.add_argument("--out", default="result.csv")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
