"""M5 L3 动手实验:在小矩阵上对比 RTN 与简化版 GPTQ(误差补偿)。

CPU 即可。运行:
    python gptq_toy.py

目的:把"逐列量化 + Hessian 误差补偿"看见,理解它为何优于 RTN。
本实现是教学简化版(对称量化、单层、忽略 Cholesky/分块等工程细节)。
"""

import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def quantize_column(w: torch.Tensor, scale: torch.Tensor, qmax: int = 7):
    """对称量化一列权重到 INT4(qmax=7),返回反量化后的浮点值。"""
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q * scale


def rtn_quantize(w: torch.Tensor, qmax: int = 7) -> torch.Tensor:
    """RTN:每列独立按自身范围量化,互不补偿。"""
    out = torch.empty_like(w)
    for j in range(w.shape[1]):
        scale = w[:, j].abs().max() / qmax
        out[:, j] = quantize_column(w[:, j], scale, qmax)
    return out


def gptq_compensate(w: torch.Tensor, x: torch.Tensor, qmax: int = 7,
                    damp: float = 1e-2) -> torch.Tensor:
    """简化版 GPTQ:逐列量化,用 H^{-1} 把误差补偿到右侧未量化列。

    w: (out, in) 权重;x: (in, n) 校准激活。
    """
    w = w.clone()
    n_in = w.shape[1]
    # Hessian = X Xᵀ,加阻尼防奇异
    h = x @ x.t()
    h += damp * torch.mean(torch.diag(h)) * torch.eye(n_in)
    hinv = torch.linalg.inv(h)

    for q in range(n_in):
        scale = w[:, q].abs().max() / qmax
        w_q = quantize_column(w[:, q], scale, qmax)
        err = w[:, q] - w_q          # 本列量化误差 (out,)
        w[:, q] = w_q
        if q + 1 < n_in:
            # TODO(实验 A):把 err 按 H^{-1} 补偿到右侧未量化列 j>q。
            #   对每个 j>q:w[:, j] -= err * (hinv[q, j] / hinv[q, q])
            #   提示:用向量化写法,err 形状 (out,),系数是标量。
            #   coef = hinv[q, q+1:] / hinv[q, q]            # (剩余列数,)
            #   w[:, q+1:] -= torch.outer(err, coef)
            raise NotImplementedError("请补全 GPTQ 误差补偿步骤(实验 A)")
    return w


def recon_error(w_orig: torch.Tensor, w_quant: torch.Tensor,
                x: torch.Tensor) -> float:
    """逐层输出重构误差 ||WX - Ŵ X||。"""
    return torch.norm(w_orig @ x - w_quant @ x).item()


def main() -> None:
    torch.manual_seed(0)
    out_f, in_f, n = 16, 24, 512
    w = torch.randn(out_f, in_f) * 0.1
    # 制造若干"强激活"输入维度,让列之间重要性差异明显
    x = torch.randn(in_f, n)
    x[0] *= 5.0
    x[3] *= 4.0

    w_rtn = rtn_quantize(w)
    logger.info("RTN          重构误差 = %.4f", recon_error(w, w_rtn, x))

    try:
        w_gptq = gptq_compensate(w, x)
        logger.info("简化 GPTQ    重构误差 = %.4f  (应明显更低)",
                    recon_error(w, w_gptq, x))
    except NotImplementedError as exc:
        logger.warning("补偿步骤未实现:%s", exc)


if __name__ == "__main__":
    main()
