"""M5 L2 动手实验:用 autograd.Function 实现 STE(直通估计器)。

CPU 即可。运行:
    python ste_demo.py

演示:前向是真量化(离散),反向梯度直通(≈1),对比朴素 round 梯度恒为 0。
"""

import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class FakeQuantSTE(torch.autograd.Function):
    """伪量化 + STE:前向做量化->反量化,反向让梯度直通。"""

    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        q = torch.clamp(torch.round(x / scale), qmin, qmax)
        # 记录哪些元素落在量化范围内,范围外的梯度要置 0(带 clip 的 STE)
        in_range = (q > qmin) & (q < qmax)
        ctx.save_for_backward(in_range)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output):
        (in_range,) = ctx.saved_tensors
        # STE 核心:假装 round/clamp 的导数为 1,梯度直接穿过;
        # 仅对被 clip 的元素截断梯度。后三个返回 None 对应非张量参数。
        grad_x = grad_output * in_range.to(grad_output.dtype)
        return grad_x, None, None, None


def main() -> None:
    x = torch.tensor([0.13, 0.47, -0.31, 0.92, 1.50], requires_grad=True)
    scale = torch.tensor(0.05)
    qmin, qmax = -127, 127

    y = FakeQuantSTE.apply(x, scale, qmin, qmax)
    logger.info("输入   x = %s", x.detach().tolist())
    logger.info("量化后 y = %s  (离散到 scale 的整数倍)",
                [round(v, 4) for v in y.detach().tolist()])

    y.sum().backward()
    logger.info("STE 反向梯度 = %s  (直通,≈1)", x.grad.tolist())

    # 对照组:朴素 round,梯度恒为 0,训练根本无法进行
    x2 = torch.tensor([0.13, 0.47, -0.31], requires_grad=True)
    y2 = torch.round(x2 / scale) * scale
    y2.sum().backward()
    logger.info("朴素 round 梯度 = %s  (恒为 0,梯度断链)", x2.grad.tolist())


if __name__ == "__main__":
    main()
