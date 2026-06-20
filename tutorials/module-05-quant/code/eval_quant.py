"""M5 L3 实战:评测量化前后的 PPL / 显存 / decode 速度。

需要 N 卡。运行:
    python eval_quant.py --model facebook/opt-125m
    python eval_quant.py --model ./opt-125m-gptq-int4 --gptq

对比两次输出,看 INT4 相对 FP16 的精度/显存/速度变化。
"""

import argparse
import logging
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def eval_ppl(model, tokenizer, n_tokens: int = 4096) -> float:
    """在 wikitext-2 上计算困惑度(滑窗简化版)。"""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(r["text"] for r in ds)
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :n_tokens]
    ids = ids.to(model.device)
    out = model(ids, labels=ids)
    # HF 的 loss 已是 token 平均交叉熵,PPL = exp(loss)
    return torch.exp(out.loss).item()


@torch.no_grad()
def eval_decode_speed(model, tokenizer, n_new: int = 128) -> float:
    """测纯 decode 速度(tokens/s),用短 prompt 放大 decode 占比。"""
    ids = tokenizer("The history of computing", return_tensors="pt").input_ids
    ids = ids.to(model.device)
    torch.cuda.synchronize()
    t0 = time.time()
    model.generate(ids, max_new_tokens=n_new, do_sample=False)
    torch.cuda.synchronize()
    return n_new / (time.time() - t0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gptq", action="store_true",
                        help="加载 AutoGPTQ 量化模型")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    torch.cuda.reset_peak_memory_stats()

    if args.gptq:
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(args.model, device="cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="cuda:0"
        )
    model.eval()

    ppl = eval_ppl(model, tokenizer)
    speed = eval_decode_speed(model, tokenizer)
    mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

    logger.info("模型      : %s", args.model)
    logger.info("PPL       : %.3f", ppl)
    logger.info("显存峰值  : %.3f GB", mem_gb)
    logger.info("decode 速度: %.1f tokens/s", speed)


if __name__ == "__main__":
    main()
