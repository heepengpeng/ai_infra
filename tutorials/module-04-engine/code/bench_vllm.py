"""M4 Lesson 6:vLLM OpenAI 兼容 server 压测脚本。

测量推理服务的四个关键指标,并支持不同并发(concurrency)对比:
- TTFT (Time To First Token):从发请求到收到第一个 token 的时间,衡量"响应快不快"。
- TPOT (Time Per Output Token):生成阶段平均每个 token 的间隔,衡量"吐字快不快"。
- 吞吐 (throughput):整个压测期间每秒生成的 token 总数,衡量"机器服务能力"。
- P99 延迟:端到端延迟的 99 分位,衡量"长尾体验"(分布式系统里你最熟的尾延迟)。

依赖(在能访问 server 的机器上):
    pip install aiohttp numpy

用法(先按 lesson 正文起好 vLLM server,默认 8000 端口):
    python bench_vllm.py --model Qwen/Qwen2.5-7B-Instruct --concurrency 1 8 32 64

注意:本脚本用流式(stream=True)接口,才能精确测到 TTFT。
"""

import argparse
import asyncio
import logging
import time

import aiohttp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROMPT = "请用通俗的语言解释一下什么是大语言模型的推理,并举一个生活中的类比。"


class RequestMetric:
    """单条请求的测量结果。"""

    def __init__(self):
        self.ttft = 0.0          # 首 token 延迟(秒)
        self.latency = 0.0       # 端到端延迟(秒)
        self.output_tokens = 0   # 生成的 token 数(用 chunk 数近似)
        self.success = False

    @property
    def tpot(self) -> float:
        # TPOT = (端到端 - 首 token) / (生成 token 数 - 1),即除首 token 外的平均间隔。
        if self.output_tokens <= 1:
            return 0.0
        return (self.latency - self.ttft) / (self.output_tokens - 1)


async def one_request(session, url, model, max_tokens) -> RequestMetric:
    """发一条流式请求,记录 TTFT 和端到端延迟。"""
    metric = RequestMetric()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    start = time.perf_counter()
    first_token_time = None
    try:
        async with session.post(url, json=payload) as resp:
            # 逐行读 SSE 流;第一行有效数据到达的时刻就是首 token 时间。
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                metric.output_tokens += 1
        end = time.perf_counter()
        metric.ttft = (first_token_time or end) - start
        metric.latency = end - start
        metric.success = True
    except Exception as exc:  # 压测脚本对单条失败要容错,不能整批崩
        logger.warning("请求失败: %s", exc)
    return metric


async def worker(name, session, url, model, max_tokens, deadline, results) -> None:
    """一个并发 worker:在 deadline 之前持续发请求(闭环压测)。"""
    while time.perf_counter() < deadline:
        results.append(await one_request(session, url, model, max_tokens))


async def run_level(url, model, concurrency, duration, max_tokens) -> dict:
    """以固定并发数压测 duration 秒,汇总指标。"""
    results: list = []
    deadline = time.perf_counter() + duration
    timeout = aiohttp.ClientTimeout(total=duration + 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        workers = [
            worker(f"w{i}", session, url, model, max_tokens, deadline, results)
            for i in range(concurrency)
        ]
        wall_start = time.perf_counter()
        await asyncio.gather(*workers)
        wall = time.perf_counter() - wall_start

    ok = [r for r in results if r.success and r.output_tokens > 0]
    if not ok:
        logger.error("并发=%d 没有成功请求,检查 server 是否在跑", concurrency)
        return {}

    ttfts = np.array([r.ttft for r in ok])
    latencies = np.array([r.latency for r in ok])
    tpots = np.array([r.tpot for r in ok if r.tpot > 0])
    total_out_tokens = sum(r.output_tokens for r in ok)

    return {
        "concurrency": concurrency,
        "requests": len(ok),
        "throughput_tok_s": total_out_tokens / wall,
        "ttft_mean_ms": ttfts.mean() * 1000,
        "ttft_p99_ms": np.percentile(ttfts, 99) * 1000,
        "tpot_mean_ms": (tpots.mean() * 1000) if len(tpots) else 0.0,
        "latency_p99_ms": np.percentile(latencies, 99) * 1000,
    }


def report(rows: list) -> None:
    header = ("并发", "请求数", "吞吐(tok/s)", "TTFT均值(ms)",
              "TTFT_P99(ms)", "TPOT均值(ms)", "延迟_P99(ms)")
    logger.info("%-6s %-8s %-14s %-14s %-14s %-14s %-14s", *header)
    for r in rows:
        logger.info("%-6d %-8d %-14.1f %-14.1f %-14.1f %-14.1f %-14.1f",
                    r["concurrency"], r["requests"], r["throughput_tok_s"],
                    r["ttft_mean_ms"], r["ttft_p99_ms"], r["tpot_mean_ms"],
                    r["latency_p99_ms"])


def parse_args():
    p = argparse.ArgumentParser(description="vLLM benchmark")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", required=True, help="server 上 --served-model-name 对应的名字")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32],
                   help="要对比的并发档位")
    p.add_argument("--duration", type=int, default=20, help="每档压测秒数")
    p.add_argument("--max-tokens", type=int, default=256)
    return p.parse_args()


async def main_async():
    args = parse_args()
    url = f"{args.base_url}/v1/chat/completions"
    rows = []
    for c in args.concurrency:
        logger.info("开始压测:并发=%d,持续 %d 秒 ...", c, args.duration)
        row = await run_level(url, args.model, c, args.duration, args.max_tokens)
        if row:
            rows.append(row)
    logger.info("==== 压测汇总 ====")
    report(rows)


if __name__ == "__main__":
    asyncio.run(main_async())
