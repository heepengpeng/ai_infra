"""M5 L4 动手实验:AWQ"放大显著权重"提高量化信噪比。

CPU 即可。运行:
    python awq_scale_toy.py

演示:量化噪声大小与权重量级无关,因此放大显著权重能降低其相对误差。
"""

import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def quant_dequant(w: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """对称 INT4 量化->反量化(整列共享一个 scale)。"""
    qmax = 2 ** (num_bits - 1) - 1
    scale = w.abs().max() / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q * scale


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.norm(a - b) / torch.norm(a)).item()


def main() -> None:
    torch.manual_seed(0)
    in_f, out_f = 32, 16
    w = torch.randn(in_f, out_f) * 0.05
    # 设通道 3 是"显著通道"(对应大激活),其权重对输出影响大
    salient = 3
    act = torch.ones(in_f)
    act[salient] = 8.0   # 该通道激活幅度大 -> AWQ 认定为显著

    # 不保护:整体量化
    wq_plain = quant_dequant(w)
    logger.info("不保护    显著列相对误差 = %.4e",
                rel_err(w[salient], wq_plain[salient]))

    # AWQ 保护:放大显著列再量化(激活同步缩小以保持等价,这里只看权重侧误差)
    scale_factor = 4.0
    w_scaled = w.clone()
    w_scaled[salient] *= scale_factor
    wq_scaled = quant_dequant(w_scaled)
    wq_scaled[salient] /= scale_factor   # 还原以比较真实误差
    logger.info("AWQ 放大%dx 显著列相对误差 = %.4e  (信噪比提高)",
                int(scale_factor), rel_err(w[salient], wq_scaled[salient]))


if __name__ == "__main__":
    main()
