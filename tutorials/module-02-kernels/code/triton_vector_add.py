"""Triton 入门:vector-add。

带宽受限算子,用于建立 Triton 的最小心智模型,并验证与 PyTorch 结果一致。
运行: python triton_vector_add.py
依赖: pip install triton (Linux + N 卡)
"""

import logging

import torch
import triton
import triton.language as tl

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # 防止最后一个 program 越界
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out


def benchmark(fn, *args, runs: int = 100) -> float:
    """返回最优毫秒数,用 CUDA event 计时。"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    fn(*args)  # warm-up
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

    n = 1 << 24  # 16M 元素
    x = torch.rand(n, device="cuda")
    y = torch.rand(n, device="cuda")

    out_triton = vector_add(x, y)
    out_torch = x + y
    max_err = (out_triton - out_torch).abs().max().item()
    logger.info("correctness vs torch: max abs err = %.2e", max_err)
    assert max_err < 1e-5, "结果与 torch 不一致"

    t_triton = benchmark(vector_add, x, y)
    t_torch = benchmark(lambda a, b: a + b, x, y)
    # 每元素搬运 = 读 x + 读 y + 写 out = 12 字节(带宽受限,见 Lesson 1)。
    moved = 3 * n * 4
    logger.info("n = %d (%.0f M elements)", n, n / 1e6)
    logger.info("triton add : %.3f ms, %.0f GB/s", t_triton, moved / (t_triton / 1e3) / 1e9)
    logger.info("torch  add : %.3f ms, %.0f GB/s", t_torch, moved / (t_torch / 1e3) / 1e9)
    logger.info("两者都贴着显存带宽跑,说明 Triton 无性能损失")


if __name__ == "__main__":
    main()
