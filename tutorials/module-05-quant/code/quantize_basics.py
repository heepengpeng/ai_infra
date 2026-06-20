"""M5 L1 动手实验:手写 INT8 量化 / 反量化,并测量误差。

只依赖 CPU 与 PyTorch,无需 GPU。运行:
    python quantize_basics.py

实验内容:
    A. 对称 / 非对称 per-tensor 量化对比
    B. per-channel 量化(留有 TODO 待补全)
    C. 离群值(outlier)如何摧毁 per-tensor 量化
"""

import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def quantize_symmetric(x: torch.Tensor, num_bits: int = 8):
    """对称量化:zero-point 恒为 0,适合分布以 0 为中心的权重。"""
    qmax = 2 ** (num_bits - 1) - 1  # INT8 -> 127
    amax = x.abs().max()
    scale = amax / qmax
    # round 后必须 clamp,防止极端 round 越过 qmax 边界
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q, scale


def dequantize_symmetric(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q * scale


def quantize_asymmetric(x: torch.Tensor, num_bits: int = 8):
    """非对称量化:用 zero-point 做平移,适合偏向一侧的激活。"""
    qmin = -(2 ** (num_bits - 1))      # -128
    qmax = 2 ** (num_bits - 1) - 1     # 127
    xmin, xmax = x.min(), x.max()
    scale = (xmax - xmin) / (qmax - qmin)
    # zero-point 取整,保证浮点 0 能精确映射到某个整数格
    zero_point = qmin - torch.round(xmin / scale)
    q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    return q, scale, zero_point


def dequantize_asymmetric(q, scale, zero_point) -> torch.Tensor:
    return scale * (q - zero_point)


def quantize_per_channel(x: torch.Tensor, num_bits: int = 8):
    """per-channel 对称量化:对二维权重 (out, in) 的每一行用独立 scale。

    TODO(实验 B):补全本函数。
      1. 沿 dim=1 求每行的绝对值最大值,得到形状 (out, 1) 的 amax。
         提示:x.abs().amax(dim=1, keepdim=True)
      2. 用 qmax = 2**(num_bits-1) - 1 计算每行的 scale = amax / qmax。
      3. q = clamp(round(x / scale), -qmax, qmax),注意 scale 会广播。
      4. 返回 (q, scale)。
    """
    raise NotImplementedError("请补全 quantize_per_channel(实验 B)")


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean((a - b) ** 2).item()


def demo_symmetric_vs_asymmetric() -> None:
    logger.info("=== 实验 A:对称 vs 非对称(全正数据,模拟 ReLU 输出)===")
    # 全正、偏向一侧的数据:非对称应明显更优
    x = torch.rand(4096) * 0.9 + 0.1  # 范围约 [0.1, 1.0]

    q_s, s_s = quantize_symmetric(x)
    x_s = dequantize_symmetric(q_s, s_s)

    q_a, s_a, z_a = quantize_asymmetric(x)
    x_a = dequantize_asymmetric(q_a, s_a, z_a)

    logger.info("对称   MSE = %.3e  (一半整数格被浪费)", mse(x, x_s))
    logger.info("非对称 MSE = %.3e  (zero_point=%d 把范围贴合到正区间)",
                mse(x, x_a), int(z_a))


def demo_per_channel() -> None:
    logger.info("\n=== 实验 B:per-tensor vs per-channel ===")
    torch.manual_seed(0)
    # 构造各行范围差异极大的矩阵:第 0 行放大 30 倍,模拟"难量化的通道"
    w = torch.randn(8, 1024) * 0.05
    w[0] *= 30.0

    q_t, s_t = quantize_symmetric(w)
    w_t = dequantize_symmetric(q_t, s_t)
    logger.info("per-tensor  MSE = %.3e  (被第 0 行大值撑大 scale,小行精度崩)",
                mse(w, w_t))

    try:
        q_c, s_c = quantize_per_channel(w)
        w_c = dequantize_symmetric(q_c, s_c)
        logger.info("per-channel MSE = %.3e  (每行独立 scale,互不拖累)",
                    mse(w, w_c))
    except NotImplementedError as exc:
        logger.warning("per-channel 尚未实现:%s", exc)


def outlier_experiment(outlier_value: float = 10.0) -> None:
    logger.info("\n=== 实验 C:离群值对 per-tensor 的破坏(outlier=%.0f)===",
                outlier_value)
    torch.manual_seed(1)
    x = torch.randn(2048) * 0.1  # 正常分布,范围约 [-0.4, 0.4]

    q0, s0 = quantize_symmetric(x)
    logger.info("无离群值   MSE = %.3e", mse(x, dequantize_symmetric(q0, s0)))

    x_out = x.clone()
    x_out[0] = outlier_value  # 塞一个极端离群值
    q1, s1 = quantize_symmetric(x_out)
    # 只看正常值部分的误差(排除离群点本身),才能看清"小值被毁"的程度
    deq = dequantize_symmetric(q1, s1)
    logger.info("含离群值后 正常值 MSE = %.3e  (scale 被撑大 %.0f 倍)",
                mse(x[1:], deq[1:]), (s1 / s0).item())


def main() -> None:
    demo_symmetric_vs_asymmetric()
    demo_per_channel()
    outlier_experiment(10.0)
    # TODO(选做):把上一行改成 outlier_experiment(100.0),观察误差进一步爆炸


if __name__ == "__main__":
    main()
