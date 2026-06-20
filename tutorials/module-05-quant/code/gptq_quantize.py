"""M5 L3 实战:用 AutoGPTQ 把一个小模型量化到 INT4(W4A16)。

需要 N 卡。运行:
    python gptq_quantize.py

先用小模型跑通流程(opt-125m / Qwen2.5-0.5B),再换更大的。
AutoGPTQ 接口随版本演进,如报错可改用 gptqmodel(API 类似)。
"""

import logging

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
except ImportError as exc:  # 给出明确的安装指引而非裸崩
    raise SystemExit("缺少 auto-gptq,请先 `pip install auto-gptq`") from exc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/opt-125m"      # TODO:换成你要量化的模型
QUANT_OUT = "./opt-125m-gptq-int4"
N_CALIB = 128                          # 校准样本数,够估 Hessian 即可


def build_calibration(tokenizer, n_samples: int):
    """从 wikitext 取若干段文本作为校准数据(无需标签)。"""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    samples = []
    for row in ds:
        text = row["text"].strip()
        if len(text) < 200:           # 跳过太短的行,保证激活统计有代表性
            continue
        enc = tokenizer(text, return_tensors="pt", max_length=512,
                        truncation=True)
        samples.append({"input_ids": enc.input_ids,
                        "attention_mask": enc.attention_mask})
        if len(samples) >= n_samples:
            break
    return samples


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    quant_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,               # per-group,精度/元数据开销的经典平衡
        desc_act=True,                # act-order:先量化重要列,精度更好
        sym=True,
    )

    logger.info("加载 FP16 模型:%s", MODEL_NAME)
    model = AutoGPTQForCausalLM.from_pretrained(
        MODEL_NAME, quant_config, torch_dtype=torch.float16
    )

    logger.info("构建校准数据(%d 条)", N_CALIB)
    calib = build_calibration(tokenizer, N_CALIB)

    logger.info("开始 GPTQ 量化(逐层估 Hessian + 误差补偿)……")
    model.quantize(calib)

    logger.info("保存量化模型到 %s", QUANT_OUT)
    model.save_quantized(QUANT_OUT, use_safetensors=True)
    tokenizer.save_pretrained(QUANT_OUT)
    logger.info("完成。可用 eval_quant.py 评测 PPL/显存/速度。")


if __name__ == "__main__":
    main()
