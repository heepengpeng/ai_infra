"""Online softmax:用一遍递推求出 max 和 sum,验证与标准 softmax 等价。

这是 FlashAttention 的数学心脏。本文件不依赖 GPU/Triton,纯 PyTorch CPU 即可跑,
目的是先把算法和递推公式吃透。
运行: python online_softmax.py
"""

import logging

import torch

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def softmax_reference(x: torch.Tensor) -> torch.Tensor:
    """标准 safe softmax(减 max),作为正确性基准。"""
    m = x.max()
    e = torch.exp(x - m)
    return e / e.sum()


def online_softmax_elementwise(x: torch.Tensor) -> torch.Tensor:
    """逐元素 online 递推求 (m, d),再算输出。

    递推核心:
        m_new = max(m_old, x_i)
        d_new = d_old * exp(m_old - m_new) + exp(x_i - m_new)
    """
    m = torch.tensor(float("-inf"))
    d = torch.tensor(0.0)
    for xi in x:
        m_old = m
        m = torch.maximum(m_old, xi)
        # max 变大时,把之前累积的 d 缩放到新基准 m 上
        d = d * torch.exp(m_old - m) + torch.exp(xi - m)
    return torch.exp(x - m) / d


def online_softmax_blockwise(x: torch.Tensor, block: int) -> torch.Tensor:
    """逐块 online 递推:每块算局部 (m_b, d_b),再用块合并公式增量合并。

    块合并:
        m = max(m_A, m_B)
        d = d_A * exp(m_A - m) + d_B * exp(m_B - m)
    这正是 FlashAttention 沿 K/V 序列分块时合并各块统计量的方式。
    """
    m = torch.tensor(float("-inf"))
    d = torch.tensor(0.0)
    for start in range(0, x.numel(), block):
        xb = x[start:start + block]
        m_b = xb.max()
        d_b = torch.exp(xb - m_b).sum()
        m_new = torch.maximum(m, m_b)
        d = d * torch.exp(m - m_new) + d_b * torch.exp(m_b - m_new)
        m = m_new
    return torch.exp(x - m) / d


def main() -> None:
    torch.manual_seed(0)
    # 故意混入大值,检验数值稳定性(不减 max 会溢出)。
    x = torch.cat([torch.randn(60), torch.tensor([120.0, 95.0])])

    ref = softmax_reference(x)
    out_elem = online_softmax_elementwise(x)
    out_block = online_softmax_blockwise(x, block=8)

    err_elem = (out_elem - ref).abs().max().item()
    err_block = (out_block - ref).abs().max().item()
    logger.info("online (elementwise) vs reference: max abs err = %.2e", err_elem)
    logger.info("online (blockwise)   vs reference: max abs err = %.2e", err_block)
    assert err_elem < 1e-6 and err_block < 1e-6, "online 递推与标准 softmax 不一致"
    logger.info("两种 online 递推都与标准 softmax 一致 -> 公式正确")

    # TODO: 把 blockwise 版扩展到二维(每行独立),并让它同时累积一个加权 V,
    #       就得到了 FlashAttention 的雏形(见 Lesson 7)。


if __name__ == "__main__":
    main()
