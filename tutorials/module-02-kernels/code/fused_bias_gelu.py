"""融合 bias + GELU:中间结果留在寄存器,不落 HBM。

对比三种实现:
  1) torch eager(逐算子,中间结果落 HBM)
  2) Triton 融合(本课主角)
  3) torch.compile(编译器自动融合,作参照)
运行: python fused_bias_gelu.py
"""

import logging

import torch
import triton
import triton.language as tl

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2/pi)


@triton.jit
def bias_gelu_kernel(x_ptr, bias_ptr, out_ptr, n_rows, n_cols,
                     BLOCK_SIZE: tl.constexpr):
    n_col_blocks = tl.cdiv(n_cols, BLOCK_SIZE)
    pid = tl.program_id(0)
    row = pid // n_col_blocks
    col_block = pid % n_col_blocks
    col_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (row < n_rows) & (col_offsets < n_cols)
    offs = row * n_cols + col_offsets

    x = tl.load(x_ptr + offs, mask=mask)
    bias = tl.load(bias_ptr + col_offsets, mask=col_offsets < n_cols)
    z = x + bias  # 加 bias,结果留寄存器
    # GELU tanh 近似,全程不写回 HBM
    inner = SQRT_2_OVER_PI * (z + 0.044715 * z * z * z)
    gelu = 0.5 * z * (1.0 + tl.math.tanh(inner))
    tl.store(out_ptr + offs, gelu, mask=mask)


def fused_bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block_size = 1024
    grid = (n_rows * triton.cdiv(n_cols, block_size),)
    bias_gelu_kernel[grid](x, bias, out, n_rows, n_cols, BLOCK_SIZE=block_size)
    return out


def eager_bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x + bias, approximate="tanh")


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

    n_rows, n_cols = 8192, 8192
    x = torch.randn(n_rows, n_cols, device="cuda")
    bias = torch.randn(n_cols, device="cuda")

    out_triton = fused_bias_gelu(x, bias)
    out_torch = eager_bias_gelu(x, bias)
    max_err = (out_triton - out_torch).abs().max().item()
    logger.info("correctness vs torch: max abs err = %.2e", max_err)
    assert max_err < 1e-4, "结果与 torch 不一致"

    t_eager = benchmark(eager_bias_gelu, x, bias)
    t_fused = benchmark(fused_bias_gelu, x, bias)
    # 融合理想:读 x + 写 out = 2 * 元素 * 4 字节(bias 很小忽略)。
    moved = 2 * n_rows * n_cols * 4
    logger.info("shape = (%d, %d)", n_rows, n_cols)
    logger.info("torch (eager)   : %.3f ms, %.0f GB/s", t_eager, moved / (t_eager / 1e3) / 1e9)
    logger.info("triton fused    : %.3f ms, %.0f GB/s", t_fused, moved / (t_fused / 1e3) / 1e9)

    try:
        compiled = torch.compile(eager_bias_gelu)
        t_comp = benchmark(compiled, x, bias)
        logger.info("torch.compile   : %.3f ms, %.0f GB/s", t_comp, moved / (t_comp / 1e3) / 1e9)
    except Exception as exc:  # 老版本 torch 可能没有 compile
        logger.warning("torch.compile 不可用: %s", exc)

    logger.info("speedup fused/eager = %.1fx", t_eager / t_fused)

    # TODO 1: 把 GELU 换成 SiLU(x * sigmoid(x)),验证融合同样有效。
    # TODO 2: 写一个未融合的 Triton 两-kernel 版(bias 一个、gelu 一个),
    #         实测它比融合版慢,体会中间结果落 HBM 的代价。


if __name__ == "__main__":
    main()
