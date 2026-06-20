"""Triton fused LayerNorm:一个 program 处理一行,均值/方差/标准化/仿射一气呵成。

对比 torch.layer_norm,展示融合(减少显存往返)的收益。
假设最后一维 N 能装进一个 BLOCK_SIZE。
运行: python triton_layernorm.py
"""

import logging

import torch
import triton
import triton.language as tl

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@triton.jit
def layernorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, stride, N, eps,
                     BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)  # 越界位置置 0,不参与方差
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask)
    b = tl.load(b_ptr + cols, mask=mask)
    y = xc * rstd * w + b
    tl.store(out_ptr + row * stride + cols, y, mask=mask)


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
              eps: float = 1e-5) -> torch.Tensor:
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block_size = triton.next_power_of_2(n_cols)
    layernorm_kernel[(n_rows,)](
        x, weight, bias, out, x.stride(0), n_cols, eps, BLOCK_SIZE=block_size,
    )
    return out


def benchmark(fn, *args, runs: int = 100) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    fn(*args)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(runs):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end))
    return best


def main() -> None:
    if not torch.cuda.is_available():
        logger.error("需要 CUDA GPU 才能运行 Triton")
        return

    n_rows, n_cols = 8192, 4096
    x = torch.randn(n_rows, n_cols, device="cuda")
    weight = torch.randn(n_cols, device="cuda")
    bias = torch.randn(n_cols, device="cuda")
    eps = 1e-5

    out_triton = layernorm(x, weight, bias, eps)
    out_torch = torch.nn.functional.layer_norm(x, (n_cols,), weight, bias, eps)
    max_err = (out_triton - out_torch).abs().max().item()
    logger.info("correctness vs torch.layer_norm: max abs err = %.2e", max_err)
    assert max_err < 1e-3, "结果与 torch 不一致"

    t_torch = benchmark(
        lambda t: torch.nn.functional.layer_norm(t, (n_cols,), weight, bias, eps), x)
    t_triton = benchmark(layernorm, x, weight, bias, eps)
    moved = 2 * n_rows * n_cols * 4  # 读一遍 + 写一遍
    logger.info("shape = (%d, %d)", n_rows, n_cols)
    logger.info("torch.layer_norm : %.3f ms, %.0f GB/s", t_torch, moved / (t_torch / 1e3) / 1e9)
    logger.info("triton fused     : %.3f ms, %.0f GB/s", t_triton, moved / (t_triton / 1e3) / 1e9)
    logger.info("speedup = %.1fx", t_torch / t_triton)

    # TODO 1: 实现 RMSNorm(去掉减均值,仅用 sqrt(mean(x^2)+eps) 归一化),对比速度。
    # TODO 2: 用更数值稳定的方式求 mean/var(或 Welford),验证在极端输入下更稳。


if __name__ == "__main__":
    main()
