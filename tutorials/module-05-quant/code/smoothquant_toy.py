"""M5 L4 动手实验:SmoothQuant 的"难度迁移"。

CPU 即可。运行:
    python smoothquant_toy.py

演示:把激活的 outlier 难度通过逐通道缩放迁移给权重,使两边都好量化。
"""

import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def quant_dequant_per_tensor(t: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """对称 per-tensor INT8 量化->反量化。"""
    qmax = 2 ** (num_bits - 1) - 1
    scale = t.abs().max() / qmax
    q = torch.clamp(torch.round(t / scale), -qmax, qmax)
    return q * scale


def compute_smooth_scale(x: torch.Tensor, w: torch.Tensor,
                         alpha: float = 0.5) -> torch.Tensor:
    """计算 SmoothQuant 逐通道缩放因子 s。

    x: (n, in) 激活;w: (in, out) 权重。返回形状 (in,) 的 s。

    TODO(实验 A):按公式补全
        s_j = (max_i|X_ij|)^alpha / (max_j|W_ij|)^(1-alpha)
      提示:
        act_max = x.abs().amax(dim=0)         # (in,) 每个输入通道激活最大值
        wgt_max = w.abs().amax(dim=1)         # (in,) 每个输入通道对应权重最大值
        s = act_max.pow(alpha) / wgt_max.pow(1 - alpha)
        s = s.clamp(min=1e-5)                 # 防止除 0
      返回 s。
    """
    raise NotImplementedError("请补全 compute_smooth_scale(实验 A)")


def output_error(x, w, xq, wq) -> float:
    """对比量化前后输出 Y=XW 的相对误差。"""
    y = x @ w
    yq = xq @ wq
    return (torch.norm(y - yq) / torch.norm(y)).item()


def main() -> None:
    torch.manual_seed(0)
    n, in_f, out_f = 64, 32, 48
    # 激活:大部分通道正常,通道 5、17 是 outlier(放大 30 倍)
    x = torch.randn(n, in_f) * 0.5
    x[:, 5] *= 30.0
    x[:, 17] *= 30.0
    w = torch.randn(in_f, out_f) * 0.1   # 权重平滑、好量化

    # ① 朴素 W8A8:直接量化,激活 outlier 毁精度
    xq0 = quant_dequant_per_tensor(x)
    wq0 = quant_dequant_per_tensor(w)
    logger.info("朴素 W8A8        输出相对误差 = %.4e", output_error(x, w, xq0, wq0))

    for alpha in (0.0, 0.5, 1.0):
        try:
            s = compute_smooth_scale(x, w, alpha)
        except NotImplementedError as exc:
            logger.warning("缩放未实现:%s", exc)
            return
        # 等价变换:激活 ÷ s,权重 × s(逐通道,沿 in 维)
        x_s = x / s
        w_s = w * s.unsqueeze(1)
        xq = quant_dequant_per_tensor(x_s)
        wq = quant_dequant_per_tensor(w_s)
        logger.info("SmoothQuant α=%.1f 输出相对误差 = %.4e", alpha,
                    output_error(x_s, w_s, xq, wq))


if __name__ == "__main__":
    main()
