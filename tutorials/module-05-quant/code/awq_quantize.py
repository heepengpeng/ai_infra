"""M5 L4 实战:用 AutoAWQ 把小模型量化到 INT4(W4A16)。

需要 N 卡。运行:
    python awq_quantize.py

量化完成后可复用 eval_quant.py(去掉 --gptq,用 HF 接口加载)对比精度/显存/速度。
AutoAWQ 接口随版本演进,如报错请对照其官方 README 调整。
"""

import logging

try:
    from awq import AutoAWQForCausalLM
except ImportError as exc:
    raise SystemExit("缺少 autoawq,请先 `pip install autoawq`") from exc

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/opt-125m"      # TODO:换成你要量化的模型
QUANT_OUT = "./opt-125m-awq-int4"

# AWQ 量化配置:zero_point 非对称、group=128、INT4、GEMM kernel
QUANT_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}


def main() -> None:
    logger.info("加载 FP16 模型:%s", MODEL_NAME)
    model = AutoAWQForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    logger.info("开始 AWQ 量化(激活感知缩放保护显著权重)……")
    # AutoAWQ 内部用内置校准集统计激活幅度;也可传入自定义校准数据
    model.quantize(tokenizer, quant_config=QUANT_CONFIG)

    logger.info("保存量化模型到 %s", QUANT_OUT)
    model.save_quantized(QUANT_OUT)
    tokenizer.save_pretrained(QUANT_OUT)
    logger.info("完成。可用 eval_quant.py 评测,或交给 vLLM 推理。")


if __name__ == "__main__":
    main()
