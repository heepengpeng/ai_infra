"""简化版 FlashAttention(前向、非因果)的 Triton 实现 + 与朴素实现对比。

教学目标:把 Lesson 7 的 online softmax 推导一比一翻译成 kernel,并验证:
  - 正确性:与朴素 attention 在数值容差内一致(无损)
  - 速度:数倍加速
  - 显存:不实例化 N×N 中间矩阵(O(N) vs O(N^2))
要求 N 为 BLOCK_M/BLOCK_N 的倍数,head_dim 为 2 的幂(为聚焦核心思想)。
运行: python flash_attention.py
"""

import logging

import torch
import triton
import triton.language as tl

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@triton.jit
def flash_kernel(Q, K, V, O, sm_scale, N,
                 stride_b, stride_m, stride_d,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
    start_m = tl.program_id(0)   # Q 的第几个行块
    off_b = tl.program_id(1)     # 第几个 (batch*head)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    base = off_b * stride_b

    # 加载本块的 Q,留在 SRAM 复用(整个内层循环都用它)。
    q_ptrs = Q + base + offs_m[:, None] * stride_m + offs_d[None, :] * stride_d
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)

    # online softmax 的三个状态:行最大 m、指数和 d、未归一化加权 V 累积 acc。
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    d_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # 内层:遍历 K/V 的所有块,增量合并(永不实例化整行分数)。
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + base + offs_n[:, None] * stride_m + offs_d[None, :] * stride_d
        v_ptrs = V + base + offs_n[:, None] * stride_m + offs_d[None, :] * stride_d
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale            # 本块分数 (BLOCK_M, BLOCK_N)
        s = tl.where(offs_n[None, :] < N, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))        # 更新全局行最大
        alpha = tl.exp(m_i - m_new)                       # 旧累积的缩放因子
        p = tl.exp(s - m_new[:, None])                    # 本块新基准下的指数
        d_i = d_i * alpha + tl.sum(p, axis=1)             # 更新分母
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)  # 更新加权 V(同步缩放)
        m_i = m_new

    acc = acc / d_i[:, None]                               # 最后归一化
    o_ptrs = O + base + offs_m[:, None] * stride_m + offs_d[None, :] * stride_d
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N)


def flash_attention(q, k, v):
    """q,k,v: (B, H, N, D)。返回同形状输出。"""
    b, h, n, d = q.shape
    scale = 1.0 / (d ** 0.5)
    qf = q.reshape(b * h, n, d).contiguous()
    kf = k.reshape(b * h, n, d).contiguous()
    vf = v.reshape(b * h, n, d).contiguous()
    out = torch.empty_like(qf)

    block_m, block_n = 64, 64
    grid = (triton.cdiv(n, block_m), b * h)
    flash_kernel[grid](
        qf, kf, vf, out, scale, n,
        qf.stride(0), qf.stride(1), qf.stride(2),
        BLOCK_M=block_m, BLOCK_N=block_n, D=d,
    )
    return out.reshape(b, h, n, d)


def naive_attention(q, k, v):
    """朴素实现:显式实例化 N×N 的 S 和 P(O(N^2) 显存)。"""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    s = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B,H,N,N)
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)


def benchmark(fn, *args, runs: int = 50) -> float:
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


def peak_extra_mem_mb(fn, *args) -> float:
    """测一次调用的显存峰值增量(MB),粗略反映中间矩阵开销。"""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn(*args)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 1e6


def main() -> None:
    if not torch.cuda.is_available():
        logger.error("需要 CUDA GPU 才能运行 Triton")
        return

    b, h, n, d = 1, 8, 4096, 64
    torch.manual_seed(0)
    q = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)

    out_flash = flash_attention(q, k, v)
    out_naive = naive_attention(q, k, v)
    max_err = (out_flash - out_naive).abs().max().item()
    logger.info("N=%d, d=%d", n, d)
    logger.info("correctness vs naive: max abs err = %.2e (无损,仅浮点舍入级)", max_err)

    t_naive = benchmark(naive_attention, q, k, v)
    t_flash = benchmark(flash_attention, q, k, v)
    mem_naive = peak_extra_mem_mb(naive_attention, q, k, v)
    mem_flash = peak_extra_mem_mb(flash_attention, q, k, v)

    logger.info("naive attention : %.2f ms, peak extra mem = %.1f MB (S + P)", t_naive, mem_naive)
    logger.info("flash attention : %.2f ms, peak extra mem = %.1f MB (无 NxN 矩阵)", t_flash, mem_flash)
    logger.info("speedup = %.1fx; 显存:flash 不实例化 N×N 中间矩阵", t_naive / t_flash)

    # TODO 1: 加 causal mask(自回归必需):内层跳过 key>query 的块,块内对角用 mask。
    # TODO 2: 把 N 加到 8192/16384,观察朴素版 OOM、flash 版稳健,见证 O(N^2)->O(N)。


if __name__ == "__main__":
    main()
