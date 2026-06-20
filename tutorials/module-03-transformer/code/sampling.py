"""从零实现各采样策略,并组装成工业级采样管线。

实现:softmax_with_temperature、top_k_filter、top_p_filter、apply_repetition_penalty,
组装成 sample(),并用固定 logits 反复采样统计多样性(唯一 token 数 / 熵)。

管线顺序(与 vLLM / HF 一致):
  penalty -> temperature -> top_k -> top_p -> softmax -> multinomial

纯 CPU 即可运行,不依赖大模型。
"""

import logging
import math
from collections import Counter

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

NEG_INF = float("-inf")


def apply_repetition_penalty(logits: torch.Tensor, prev_tokens, penalty: float) -> torch.Tensor:
    if penalty == 1.0 or not prev_tokens:
        return logits
    logits = logits.clone()
    idx = torch.tensor(sorted(set(prev_tokens)), dtype=torch.long)
    selected = logits[idx]
    # 正 logit 除以 penalty(变小),负 logit 乘以 penalty(更负),都是往下压。
    selected = torch.where(selected > 0, selected / penalty, selected * penalty)
    logits[idx] = selected
    return logits


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / temperature, dim=-1)


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    kth = torch.topk(logits, top_k).values[-1]  # 第 k 大的值
    return torch.where(logits < kth, torch.full_like(logits, NEG_INF), logits)


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """保留累积概率达到 top_p 的最小 token 集合(nucleus),其余置 -inf。

    TODO(练习):实现核采样过滤。步骤提示:
      1. 对 logits 降序排序,得到 sorted_logits 和 sorted_idx;
      2. 对 sorted_logits 求 softmax 得到概率,再 cumsum 得累积概率;
      3. 标记「累积概率已超过 top_p」的位置为移除(注意:至少保留第 1 个 token);
      4. 把这些位置在原始 logits 中置为 NEG_INF 并返回。
    完成后下方 _self_check 的断言应通过。
    """
    raise NotImplementedError("请实现 top_p_filter(见 TODO)")


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
           top_p: float = 1.0, penalty: float = 1.0, prev_tokens=None) -> int:
    prev_tokens = prev_tokens or []
    logits = apply_repetition_penalty(logits, prev_tokens, penalty)  # ①
    if temperature == 0.0:
        return int(logits.argmax())                                  # greedy
    logits = logits / temperature                                    # ②
    logits = top_k_filter(logits, top_k)                             # ③
    if top_p < 1.0:
        logits = top_p_filter(logits, top_p)                         # ④
    probs = torch.softmax(logits, dim=-1)                            # ⑤
    return int(torch.multinomial(probs, num_samples=1))              # ⑥


def entropy_of(counter: Counter, total: int) -> float:
    h = 0.0
    for c in counter.values():
        p = c / total
        h -= p * math.log(p + 1e-12)
    return h


def _self_check() -> None:
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -3.0])
    filtered = top_p_filter(logits.clone(), top_p=0.9)
    # 至少保留最大值;被移除的应为 -inf
    assert filtered.argmax() == logits.argmax(), "top_p 不应移除最大概率 token"
    assert torch.isinf(filtered).any(), "top_p=0.9 应至少移除部分长尾 token"
    logger.info("[self-check] top_p_filter 正确")


def demo() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([3.0, 2.5, 2.0, 1.0, 0.5, 0.0, -1.0, -2.0, -3.0, -5.0])
    trials = 1000
    configs = [
        ("greedy      ", dict(temperature=0.0)),
        ("T=0.7       ", dict(temperature=0.7)),
        ("T=1.0       ", dict(temperature=1.0)),
        ("top_k=3     ", dict(temperature=1.0, top_k=3)),
        ("top_p=0.9   ", dict(temperature=1.0, top_p=0.9)),
        ("T=1.3+top_p ", dict(temperature=1.3, top_p=0.9)),
    ]
    for name, cfg in configs:
        counter = Counter(sample(logits.clone(), **cfg) for _ in range(trials))
        h = entropy_of(counter, trials)
        logger.info("[%s] %d 次采样 -> 唯一 token 数 %d   熵 %.2f",
                    name, trials, len(counter), h)


if __name__ == "__main__":
    try:
        _self_check()
    except NotImplementedError as exc:
        logger.warning("top_p_filter 尚未实现:%s。请先完成 TODO 再运行 demo()。", exc)
    else:
        demo()
