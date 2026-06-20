"""M4 Lesson 5:投机解码(Speculative Decoding)加速比模拟。

不做真实模型推理,用概率模型刻画"小模型 draft + 大模型 verify"的核心权衡:
- 每轮:小模型连猜 k 个 token(便宜),大模型一次并行 verify 这 k 个(一次大模型前向)。
- 按接受率 alpha 决定接受前多少个,第一个被拒的位置用大模型的分布重采样纠正。
- 统计:平均每次"大模型前向"能产出多少个被接受的 token => 等效加速比。

关键结论(脚本会算给你看):加速比既取决于接受率 alpha(小模型多像大模型),
也取决于猜测长度 k 和"小模型相对大模型的成本比"。alpha 太低反而可能更慢。

运行:
    python speculative_decoding_sim.py
"""

import logging
import random

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def accepted_count(k: int, alpha: float, rng: random.Random) -> int:
    """模拟一轮里 draft 的 k 个 token 中,从头连续被接受的数量。

    投机采样的接受是"逐位置、前缀连续"的:一旦某位置被拒,后面全部作废。
    所以这里按几何过程数:每位置以 alpha 概率被接受,直到第一次失败。
    """
    accepted = 0
    for _ in range(k):
        if rng.random() < alpha:
            accepted += 1
        else:
            break
    return accepted


def simulate(k: int, alpha: float, target_tokens: int, draft_cost: float,
             seed: int = 0) -> dict:
    """模拟生成 target_tokens 个 token,统计大模型前向次数和等效成本。

    成本单位:1 次大模型前向 = 1.0;1 次小模型前向 = draft_cost(如 0.1)。
    基线(纯自回归):每个 token 1 次大模型前向,总成本 = target_tokens。
    """
    rng = random.Random(seed)
    produced = 0
    target_forwards = 0  # 大模型前向次数(每轮 verify 算 1 次)
    draft_forwards = 0   # 小模型前向次数(每轮猜 k 个 = k 次小模型前向)

    while produced < target_tokens:
        acc = accepted_count(k, alpha, rng)
        # 一轮产出 = 被接受的 acc 个 + 1 个(被拒位置由大模型重采样补 1 个,
        # 或 k 个全接受时大模型顺手多吐 1 个 bonus token)。这是投机解码的标准账。
        produced += acc + 1
        target_forwards += 1
        draft_forwards += k

    baseline_cost = target_tokens  # 纯自回归基线
    spec_cost = target_forwards * 1.0 + draft_forwards * draft_cost
    return {
        "k": k,
        "alpha": alpha,
        "produced": produced,
        "target_forwards": target_forwards,
        "tokens_per_forward": produced / target_forwards,
        "speedup_vs_baseline": baseline_cost / spec_cost,
    }


def main() -> None:
    target_tokens = 5000
    draft_cost = 0.15  # 小模型约为大模型 15% 的单步成本

    logger.info("基线:纯自回归,每 token 1 次大模型前向。")
    logger.info("小模型单步成本 = 大模型的 %.0f%%", draft_cost * 100)
    logger.info("")
    logger.info("%-6s %-8s %-18s %-12s", "k", "alpha", "tokens/大模型前向", "等效加速比")
    for alpha in (0.5, 0.7, 0.9):
        for k in (1, 4, 8):
            r = simulate(k, alpha, target_tokens, draft_cost)
            logger.info("%-6d %-8.1f %-18.2f %-12.2fx",
                        r["k"], r["alpha"], r["tokens_per_forward"], r["speedup_vs_baseline"])
        logger.info("")

    logger.info("观察:alpha 越高(小模型越像大模型)加速越明显;")
    logger.info("k 不是越大越好——alpha 低时猜太多会白费小模型算力,反而拉低加速比。")


if __name__ == "__main__":
    main()
