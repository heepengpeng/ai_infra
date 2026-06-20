"""M4 Lesson 2:Continuous Batching(迭代级调度)纯 Python 模拟。

这是一个**不依赖 GPU、不依赖任何深度学习框架**的离散事件模拟器,
用来对比"静态 batching"和"continuous batching"两种调度策略的吞吐与利用率差异。

核心简化(刻意为之,聚焦调度而非真实计算):
- 不做真实的矩阵运算,用 token 数量代表"工作量"。
- 假设每个 decode step 对 batch 里的每个请求各产出 1 个 token,耗时恒定。
- prefill 也折算成若干个 step 的等价开销(便于统一计时)。

运行:
    python continuous_batching_sim.py
"""

import logging
import random
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 8  # 一个 batch 同时容纳的最大请求数(显存上限的抽象)


@dataclass
class Request:
    """一个推理请求。output_len 模拟"它最终会生成多少 token",事先未知。"""

    rid: int
    prompt_len: int
    output_len: int
    arrival: int = 0  # 到达时刻(step 编号)
    generated: int = 0  # 已生成的 token 数
    start_step: int = -1  # 进入运行态的 step
    finish_step: int = -1  # 结束的 step

    @property
    def done(self) -> bool:
        return self.generated >= self.output_len


@dataclass
class SimResult:
    """一次模拟的统计结果。"""

    strategy: str
    total_steps: int
    finished: int
    # 利用率 = 有效 token 槽位 / (总 step × 最大 batch)
    useful_slots: int = 0
    capacity_slots: int = 0
    latencies: list = field(default_factory=list)

    @property
    def utilization(self) -> float:
        if self.capacity_slots == 0:
            return 0.0
        return self.useful_slots / self.capacity_slots

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0


def make_workload(n: int, seed: int = 42) -> list:
    """造一批长度方差很大的请求:大多数短,少数话痨。这是 LLM 流量的典型特征。"""
    rng = random.Random(seed)
    reqs = []
    for rid in range(n):
        prompt_len = rng.randint(4, 40)
        # 90% 的请求生成 4~20 个 token,10% 是话痨生成 150~400 个。
        if rng.random() < 0.10:
            output_len = rng.randint(150, 400)
        else:
            output_len = rng.randint(4, 20)
        reqs.append(Request(rid=rid, prompt_len=prompt_len, output_len=output_len))
    return reqs


def run_static(requests: list) -> SimResult:
    """静态 batching:攒满一批,一起从头跑到尾,等最慢的结束才整批释放。"""
    result = SimResult(strategy="static", total_steps=0, finished=0)
    pending = list(requests)
    step = 0

    while pending:
        # 攒一个 batch(取队首 MAX_BATCH_SIZE 个),这一批同生共死。
        batch = pending[:MAX_BATCH_SIZE]
        pending = pending[MAX_BATCH_SIZE:]
        for r in batch:
            r.start_step = step
        # 木桶效应:整批要跑到最长的那个请求结束。
        batch_steps = max(r.output_len for r in batch)
        for _ in range(batch_steps):
            for r in batch:
                # 即使某请求已 done,它的槽位仍被占着空跑(静态 batching 的病根)。
                if not r.done:
                    r.generated += 1
                    result.useful_slots += 1
                    if r.done:
                        r.finish_step = step + 1
            result.capacity_slots += MAX_BATCH_SIZE  # 整批容量都被占着,无法他用
            step += 1
        for r in batch:
            result.finished += 1
            # 端到端延迟:从请求到达到它彻底完成,含排队等待。
            result.latencies.append(r.finish_step - r.arrival)

    result.total_steps = step
    return result


def run_continuous(requests: list) -> SimResult:
    """Continuous batching:每个 step 重组 batch,done 的立刻退出、腾位给等待队列里的新请求。"""
    result = SimResult(strategy="continuous", total_steps=0, finished=0)
    waiting = list(requests)
    running: list = []
    step = 0

    while waiting or running:
        # 调度:只要还有空槽且等待队列非空,就把新请求拉进 running。
        while len(running) < MAX_BATCH_SIZE and waiting:
            r = waiting.pop(0)
            r.start_step = step
            running.append(r)

        # 这一步:给 running 里每个请求各生成 1 个 token。
        for r in running:
            r.generated += 1
            result.useful_slots += 1
            if r.done:
                r.finish_step = step + 1
        result.capacity_slots += MAX_BATCH_SIZE
        step += 1

        # 退出:done 的请求立刻离开,腾出的槽位下一轮就能给别人。
        finished_now = [r for r in running if r.done]
        for r in finished_now:
            result.finished += 1
            result.latencies.append(r.finish_step - r.arrival)
        running = [r for r in running if not r.done]

    result.total_steps = step
    return result


def report(result: SimResult) -> None:
    logger.info("策略=%s", result.strategy)
    logger.info("  总 step 数        : %d", result.total_steps)
    logger.info("  完成请求数        : %d", result.finished)
    logger.info("  GPU 槽位利用率    : %.1f%%", result.utilization * 100)
    logger.info("  平均端到端延迟(step): %.1f", result.avg_latency)


def main() -> None:
    n = 200
    # 两种策略用各自独立的工作负载副本,避免请求对象状态被复用污染。
    static_result = run_static(make_workload(n))
    continuous_result = run_continuous(make_workload(n))

    report(static_result)
    logger.info("")
    report(continuous_result)
    logger.info("")

    speedup = static_result.total_steps / continuous_result.total_steps
    logger.info("吞吐提升(总 step 之比): %.1fx", speedup)
    logger.info(
        "平均端到端延迟: 静态 %.1f -> 连续 %.1f step",
        static_result.avg_latency,
        continuous_result.avg_latency,
    )


if __name__ == "__main__":
    main()
