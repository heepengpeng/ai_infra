"""M5 L5 动手实验:FP8(E4M3)vs INT8 在带 outlier 数据上的量化误差。

CPU 即可(FP8 用软件模拟编解码,无需 H 卡)。运行:
    python fp8_vs_int8.py

结论预期:对"小值多 + 有 outlier"的张量,FP8 误差明显低于 INT8。
"""

import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# E4M3:4 指数位、3 尾数位。指数偏置 7,最大规格化值约 448。
E4M3_MANTISSA_BITS = 3
E4M3_MAX = 448.0


def fake_int8(x: torch.Tensor) -> torch.Tensor:
    """对称 INT8 均匀量化->反量化(per-tensor)。"""
    qmax = 127
    scale = x.abs().max() / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q * scale


def fake_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """软件模拟 FP8 E4M3:对每个数按其指数,把尾数舍入到 3 bit。

    思路:浮点的精度是"相对"的——每个数在它自己的 2 的幂区间内,
    被均匀量化成 2^mantissa_bits 个台阶。
    """
    sign = torch.sign(x)
    ax = x.abs().clamp(min=1e-12, max=E4M3_MAX)  # 超出范围饱和到 ±448
    # 取每个数的二进制指数 e,使 ax 落在 [2^e, 2^(e+1))
    e = torch.floor(torch.log2(ax))
    # TODO(实验 A):把尾数舍入到 E4M3_MANTISSA_BITS 位。
    #   step = 2^e / 2^mantissa_bits    # 该指数区间内每个台阶的大小
    #   q = round(ax / step) * step     # 把 ax 对齐到最近台阶
    #   返回 sign * q
    #   提示:step = torch.pow(2.0, e) / (2 ** E4M3_MANTISSA_BITS)
    raise NotImplementedError("请补全 fake_fp8_e4m3 的尾数舍入(实验 A)")


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean((a - b) ** 2).item()


def main() -> None:
    torch.manual_seed(0)
    # "小值多 + 有 outlier"的典型张量:正常 ~N(0,0.1),塞几个幅度 ~6 的离群点
    x = torch.randn(4096) * 0.1
    x[torch.randint(0, 4096, (5,))] = 6.0
    logger.info("数据:正常值 ~N(0,0.1) + 5 个幅度=6 的 outlier")

    x_int8 = fake_int8(x)
    logger.info("INT8(均匀)     MSE = %.4e", mse(x, x_int8))

    try:
        x_fp8 = fake_fp8_e4m3(x)
        logger.info("FP8 E4M3(非均匀)MSE = %.4e  (近 0 密,扛 outlier)",
                    mse(x, x_fp8))
    except NotImplementedError as exc:
        logger.warning("FP8 模拟未实现:%s", exc)


if __name__ == "__main__":
    main()
