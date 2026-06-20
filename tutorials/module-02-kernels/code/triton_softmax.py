"""Triton fused-softmax:一个 program 处理一行,在片上完成 max/exp/sum/div。

对比 torch.softmax(会拆成多个 kernel、多趟显存往返),展示"融合减少访存"的收益。
假设一行能装进一个 BLOCK_SIZE(列数不太大);超大列见 Lesson 6 的 online 策略。
运行: python triton_softmax.py
"""

import logging

import torch
import triton
import triton.language as tl

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)  # 一个 program 负责一整行
    col_offsets = tl.arange(0, BLOCK_SIZE)
    in_ptrs = in_ptr + row * in_row_stride + col_offsets
    mask = col_offsets < n_cols
    # 越界处补 -inf:exp 后为 0,既不影响 max 也不影响 sum。
    x = tl.load(in_ptrs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)  # 数值稳定:减每行 max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom
    out_ptrs = out_ptr + row * out_row_stride + col_offsets
    tl.store(out_ptrs, y, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    # BLOCK_SIZE 取 >= n_cols 的最近 2 的幂,保证一行装得下。
    block_size = triton.next_power_of_2(n_cols)
    softmax_kernel[(n_rows,)](
        out, x, x.stride(0), out.stride(0), n_cols, BLOCK_SIZE=block_size,
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

    n_rows, n_cols = 4096, 4096
    x = torch.randn(n_rows, n_cols, device="cuda")

    out_triton = softmax(x)
    out_torch = torch.softmax(x, dim=1)
    max_err = (out_triton - out_torch).abs().max().item()
    logger.info("correctness vs torch.softmax: max abs err = %.2e", max_err)
    assert max_err < 1e-5, "结果与 torch 不一致"

    t_torch = benchmark(lambda t: torch.softmax(t, dim=1), x)
    t_triton = benchmark(softmax, x)
    # softmax 主要成本是读写矩阵:理想下读一遍 + 写一遍 = 2 * 元素数 * 4 字节。
    moved = 2 * n_rows * n_cols * 4
    logger.info("shape = (%d, %d)", n_rows, n_cols)
    logger.info("torch.softmax : %.3f ms, %.0f GB/s", t_torch, moved / (t_torch / 1e3) / 1e9)
    logger.info("triton fused  : %.3f ms, %.0f GB/s", t_triton, moved / (t_triton / 1e3) / 1e9)
    logger.info("speedup = %.1fx (来源:融合减少了显存往返,见 Lesson 5)", t_torch / t_triton)


if __name__ == "__main__":
    main()
