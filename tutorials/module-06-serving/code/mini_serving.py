"""极简推理服务框架:演示 API server / engine 分离、异步队列、continuous batching、SSE 流式。

为了不依赖 GPU,engine 用「假模型」替代真实 forward:每一步给 running batch 里的每条序列
吐一个字符。架构是真的,只有算 token 那一步是假的——把 _step 换成 vLLM 的 step() 即生产骨架。

运行:python mini_serving.py
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mini_serving")

# 假模型每生成一个 token 的耗时,用来模拟 GPU forward 的 TPOT
FAKE_STEP_LATENCY_S = 0.03
# 模拟显存对并发的约束:running batch 里最多同时跑这么多条序列
MAX_BATCH_SIZE = 16
# waiting 队列堆积超过此阈值时应触发过载保护(实验 C 的 TODO)
WAITING_OVERLOAD_THRESHOLD = 256


@dataclass
class Request:
    """一条在系统里流动的请求。queue 是它和 API 层之间的单向 token 通道。"""

    prompt: str
    max_tokens: int
    queue: "asyncio.Queue[Optional[str]]"
    arrival_time: float = field(default_factory=time.monotonic)
    first_token_time: Optional[float] = None
    generated: int = 0


class FakeEngine:
    """模拟 continuous batching 的推理引擎:一个永不停歇的 step 循环。"""

    def __init__(self, max_batch_size: int = MAX_BATCH_SIZE) -> None:
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.max_batch_size = max_batch_size

    def submit(self, req: Request) -> None:
        self.waiting.append(req)

    def _admit_new_requests(self) -> None:
        """把 waiting 队列里的请求招进 running,直到 batch 满(模拟显存上限)。"""
        # TODO(实验 C): 当 running 已满且 len(self.waiting) > WAITING_OVERLOAD_THRESHOLD,
        #   对最老的 waiting 请求做快速失败(往其 queue 推一个错误哨兵),实现过载保护。
        while self.waiting and len(self.running) < self.max_batch_size:
            self.running.append(self.waiting.pop(0))

    def _step(self) -> list[Request]:
        """对整个 running batch 走一步「假 forward」,返回本步完成的请求。"""
        now = time.monotonic()
        finished: list[Request] = []
        for req in self.running:
            if req.first_token_time is None:
                req.first_token_time = now
            req.generated += 1
            # 假 token:用循环字母代替真实采样结果
            token = chr(ord("a") + (req.generated % 26))
            req.queue.put_nowait(token)
            if req.generated >= req.max_tokens:
                finished.append(req)
        for req in finished:
            self.running.remove(req)
        return finished

    async def engine_loop(self) -> None:
        """引擎主循环:招新 → forward → 分发,周而复始。对应生产引擎的 step() 循环。"""
        logger.info("engine loop started, max_batch_size=%d", self.max_batch_size)
        while True:
            self._admit_new_requests()
            if not self.running:
                await asyncio.sleep(0.005)
                continue
            finished = self._step()
            for req in finished:
                ttft_ms = (req.first_token_time - req.arrival_time) * 1000
                logger.info(
                    "request finished: tokens=%d ttft=%.1fms", req.generated, ttft_ms
                )
                # None 作为「生成结束」哨兵,通知 API 层关闭这条 SSE 流
                req.queue.put_nowait(None)
            # 让出控制权,让事件循环去处理新进来的 HTTP 连接
            await asyncio.sleep(FAKE_STEP_LATENCY_S)

    async def generate_stream(
        self, prompt: str, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        """API 层调用:入队后立刻 await,绝不阻塞事件循环。"""
        req = Request(prompt=prompt, max_tokens=max_tokens, queue=asyncio.Queue())
        self.submit(req)
        while True:
            token = await req.queue.get()
            if token is None:
                break
            yield token


engine = FakeEngine()
app = FastAPI()


@app.on_event("startup")
async def _start_engine() -> None:
    # 引擎循环和 HTTP server 跑在同一个事件循环里,互不阻塞
    asyncio.create_task(engine.engine_loop())


@app.post("/generate")
async def generate(body: dict) -> StreamingResponse:
    prompt = body.get("prompt", "")
    max_tokens = int(body.get("max_tokens", 32))

    async def event_source() -> AsyncGenerator[str, None]:
        async for token in engine.generate_stream(prompt, max_tokens):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.get("/health")
async def health() -> dict:
    return {"running": len(engine.running), "waiting": len(engine.waiting)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
