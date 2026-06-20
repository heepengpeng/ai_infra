"""最小但结构正确的 decoder-only Transformer block。

目标:用 PyTorch 手搭 RMSNorm + RoPE + (GQA-ready) 多头自注意力 + SwiGLU FFN,
并打印各子模块参数量与一次前向的 shape 流动,验证 Lesson 1 的计算量/参数量结论。

本脚本纯 CPU 即可运行,不做训练,只看结构、shape 与 FLOPs 直觉。
"""

import logging
import math
from dataclasses import dataclass

import torch
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: int = 8  # 设成 < n_heads 即为 GQA
    d_ff: int = 688      # 约 (8/3)*d_model,SwiGLU 常用比例
    vocab_size: int = 1000
    max_seq_len: int = 512

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    """预计算 RoPE 所需的 cos/sin 表,形状均为 [seq_len, head_dim]。"""
    half = head_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half).float() / half))
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, theta)  # [seq_len, half]
    # 复制成 [..., x0,x1,...,x0,x1,...] 以匹配下面的成对旋转实现
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把后半维放到前面并取负,用于旋转的虚部组合。"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """给 Q/K 施加旋转位置编码。

    q, k: [B, n_heads, S, head_dim];cos, sin: [S, head_dim]。
    旋转公式:x_rot = x * cos + rotate_half(x) * sin。

    TODO(练习):用上面的 rotate_half 和广播,实现 q_rot、k_rot。
    完成后下方 _self_check 的断言应当通过。
    提示:cos/sin 需要 reshape 成 [1, 1, S, head_dim] 才能和 [B, n_heads, S, head_dim] 广播。
    """
    raise NotImplementedError("请实现 apply_rope(见 TODO)")


class Attention(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_rep = cfg.n_heads // cfg.n_kv_heads  # 每组 KV 被几个 Q 头共享
        hd = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * hd, bias=False)
        self.wo = nn.Linear(cfg.n_heads * hd, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        hd, nh, nkv = self.cfg.head_dim, self.cfg.n_heads, self.cfg.n_kv_heads
        q = self.wq(x).view(b, s, nh, hd).transpose(1, 2)
        k = self.wk(x).view(b, s, nkv, hd).transpose(1, 2)
        v = self.wv(x).view(b, s, nkv, hd).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        # GQA:把 KV 头复制 n_rep 份,对齐 Q 头数
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)  # [B, nh, S, S]
        causal = torch.triu(torch.full((s, s), float("-inf")), diagonal=1)
        scores = scores + causal
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v                                       # [B, nh, S, hd]
        out = out.transpose(1, 2).reshape(b, s, nh * hd)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(torch.nn.functional.silu(self.w_gate(x)) * self.w_up(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Pre-Norm + residual:先 norm 再进子层,输出加回原始 x
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _self_check(cfg: Config) -> None:
    """验证 apply_rope 正确性:旋转应保持向量范数不变(纯旋转,不缩放)。"""
    torch.manual_seed(0)
    cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim)
    q = torch.randn(1, cfg.n_heads, 8, cfg.head_dim)
    k = torch.randn(1, cfg.n_kv_heads, 8, cfg.head_dim)
    q_rot, k_rot = apply_rope(q, k, cos[:8], sin[:8])
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4), "RoPE 应保持范数不变"
    logger.info("[self-check] apply_rope 正确:旋转保持向量范数不变")


def main() -> None:
    cfg = Config()
    block = DecoderBlock(cfg)

    logger.info("=== 子模块参数量 ===")
    p_attn = count_params(block.attn)
    p_ffn = count_params(block.ffn)
    logger.info("Attention 投影参数量: %d  (理论 ~4*d^2 = %d, 非 GQA 时)",
                p_attn, 4 * cfg.d_model ** 2)
    logger.info("FFN(SwiGLU)参数量 : %d  (理论 3*d*d_ff = %d)",
                p_ffn, 3 * cfg.d_model * cfg.d_ff)
    logger.info("FFN / Attention 比值: %.2f", p_ffn / p_attn)

    logger.info("=== 一次前向的 shape 流动 ===")
    b, s = 2, 16
    cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim)
    x = torch.randn(b, s, cfg.d_model)
    logger.info("输入 hidden          : %s", tuple(x.shape))
    out = block(x, cos[:s], sin[:s])
    logger.info("输出 hidden          : %s", tuple(out.shape))

    logger.info("=== S^2 膨胀观察(attention 分数矩阵内存) ===")
    for seq in (128, 256, 512):
        # 分数矩阵 [B, n_heads, S, S],float32 占 4 字节
        mem_mb = b * cfg.n_heads * seq * seq * 4 / 1e6
        logger.info("S=%4d -> 分数矩阵 [%d,%d,%d,%d] 占用 %.2f MB",
                    seq, b, cfg.n_heads, seq, seq, mem_mb)


if __name__ == "__main__":
    try:
        _self_check(Config())
    except NotImplementedError as exc:
        logger.warning("apply_rope 尚未实现:%s。请先完成 TODO 再运行 main()。", exc)
    else:
        main()
