"""迷你推理引擎:prefill + 带 KV Cache 的 decode + 采样,并 benchmark。

把 Module 3 前四课的零件焊成一台能跑的单请求引擎:
  - 加载真实小模型(默认 Qwen2-0.5B,可 --model gpt2);
  - generate():prefill 一次 + 逐 token decode(复用 KV Cache);
  - 内置简化采样(greedy / temperature / top-k / top-p);
  - benchmark():正确同步计时,报告 TTFT / TPOT / 吞吐,作为 Module 4 基线。

依赖:torch、transformers、accelerate。
"""

import argparse
import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

NEG_INF = float("-inf")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def sample_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
    """在单个位置的 logits [V] 上采样。temperature=0 即 greedy。"""
    if temperature == 0.0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, NEG_INF), logits)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()  # 右移一位,保证至少留第一个
        remove[..., 0] = False
        sorted_logits[remove] = NEG_INF
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))


class MiniEngine:
    def __init__(self, model_name: str) -> None:
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        logger.info("加载完成: %s (设备 %s)", model_name, self.device)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 128,
                 temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0):
        """返回 (生成文本, 计时字典)。计时字典含 ttft / decode_time / n_gen。"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # ---- prefill:一次处理整个 prompt,建立 KV Cache ----
        sync(self.device)
        t0 = time.perf_counter()
        out = self.model(input_ids, use_cache=True)
        sync(self.device)
        ttft = time.perf_counter() - t0
        past = out.past_key_values
        logits = out.logits[:, -1, :]  # 只取最后一个位置(它预测下一个 token)

        # ---- decode:逐 token 生成,每步只喂 1 个新 token + 历史 cache ----
        generated = []
        eos = self.tokenizer.eos_token_id
        sync(self.device)
        t1 = time.perf_counter()
        for _ in range(max_new_tokens):
            next_id = sample_token(logits[0], temperature, top_k, top_p)
            if eos is not None and next_id == eos:
                break
            generated.append(next_id)
            # TODO(练习):完成 decode 的一步。
            #   1) 把 next_id 包成 [[next_id]] 的张量 cur(放到 self.device);
            #   2) out = self.model(cur, past_key_values=past, use_cache=True);
            #   3) 更新 past = out.past_key_values;
            #   4) 更新 logits = out.logits[:, -1, :]。
            # 注意:必须只喂 1 个 token 并传入 past,否则 KV Cache 失效、退化成 O(N^2)。
            raise NotImplementedError("请完成 decode 单步(见 TODO)")
        sync(self.device)
        decode_time = time.perf_counter() - t1

        text = self.tokenizer.decode(generated)
        timing = {"ttft": ttft, "decode_time": decode_time, "n_gen": len(generated),
                  "prompt_len": input_ids.shape[1]}
        return text, timing


def report(timing: dict) -> None:
    n = max(timing["n_gen"], 1)
    tpot = timing["decode_time"] / n
    throughput = n / timing["decode_time"] if timing["decode_time"] > 0 else 0.0
    logger.info("=== benchmark(单请求基线) ===")
    logger.info("TTFT(首 token / prefill): %.1f ms", timing["ttft"] * 1e3)
    logger.info("TPOT(每输出 token)      : %.2f ms/token", tpot * 1e3)
    logger.info("吞吐(decode)            : %.1f token/s", throughput)
    logger.info("prompt 长度 %d, 生成 %d token", timing["prompt_len"], timing["n_gen"])
    logger.info("→ 记下这组数字,作为 Module 4 的优化前基线")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--prompt", default="用一句话解释什么是KV Cache:")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    engine = MiniEngine(args.model)
    # warmup:首次前向含编译/分配开销,先短跑一次
    engine.generate(args.prompt, max_new_tokens=4)

    text, timing = engine.generate(
        args.prompt, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    logger.info("[生成结果] %s", text)
    report(timing)


if __name__ == "__main__":
    main()
