"""M5 L2 动手实验:PTQ 校准方法对比(Min-Max vs MSE vs Percentile)。

CPU 即可。运行:
    python ptq_calibration.py

核心问题:激活里有 outlier 时,如何选量化范围,使量化误差最小?
"""

import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def fake_quant_symmetric(x: torch.Tensor, amax: torch.Tensor, num_bits: int = 8):
    """给定范围 amax 做对称量化->反量化,返回还原值。

    amax 即截断阈值:超过 ±amax 的值会被 clip。
    """
    qmax = 2 ** (num_bits - 1) - 1
    scale = amax / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q * scale


def calibrate_minmax(x: torch.Tensor) -> torch.Tensor:
    """Min-Max 校准:范围取绝对值最大值。对 outlier 极敏感。"""
    return x.abs().max()


def calibrate_mse(x: torch.Tensor, num_bits: int = 8, steps: int = 100):
    """MSE 校准:在 [0, amax] 间搜索一个截断阈值,使量化 MSE 最小。"""
    amax = x.abs().max()
    best_thr, best_err = amax, float("inf")
    # 从 amax 往下扫,逐步收紧范围(主动 clip 一点尾部换取更小的 round 误差)
    for i in range(1, steps + 1):
        thr = amax * i / steps
        err = torch.mean((x - fake_quant_symmetric(x, thr, num_bits)) ** 2)
        if err < best_err:
            best_err, best_thr = err, thr
    return best_thr


def calibrate_percentile(x: torch.Tensor, pct: float = 99.9) -> torch.Tensor:
    """Percentile 校准:取绝对值的 pct 分位数作为范围,砍掉极端尾部。

    TODO(实验 B):补全本函数。
      1. 对 x.abs() 取 pct 分位数作为阈值 thr。
         提示:torch.quantile(x.abs(), pct / 100.0)
      2. 返回 thr。
    """
    raise NotImplementedError("请补全 calibrate_percentile(实验 B)")


def quant_error(x: torch.Tensor, amax: torch.Tensor) -> float:
    return torch.mean((x - fake_quant_symmetric(x, amax)) ** 2).item()


def main() -> None:
    torch.manual_seed(0)
    # 模拟带 outlier 的激活:正常值约 [-0.4,0.4],塞几个幅度 ~8 的离群点
    x = torch.randn(4096) * 0.1
    x[torch.randint(0, 4096, (3,))] = 8.0
    logger.info("数据:正常值 ~N(0,0.1),含 3 个幅度=8 的 outlier")
    logger.info("绝对值最大 = %.2f", x.abs().max().item())

    thr_mm = calibrate_minmax(x)
    logger.info("\nMin-Max    范围=%.3f  MSE=%.3e", thr_mm.item(),
                quant_error(x, thr_mm))

    thr_mse = calibrate_mse(x)
    logger.info("MSE        范围=%.3f  MSE=%.3e  (主动 clip outlier)",
                thr_mse.item(), quant_error(x, thr_mse))

    try:
        thr_pct = calibrate_percentile(x, 99.9)
        logger.info("Percentile 范围=%.3f  MSE=%.3e", thr_pct.item(),
                    quant_error(x, thr_pct))
    except NotImplementedError as exc:
        logger.warning("Percentile 尚未实现:%s", exc)


if __name__ == "__main__":
    main()
