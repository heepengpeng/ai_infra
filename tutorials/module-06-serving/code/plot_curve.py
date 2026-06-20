"""读取 benchmark.py 产出的 CSV,画吞吐-延迟曲线,辅助肉眼找拐点(knee)。

横轴吞吐(tokens/s),纵轴 P99 E2E 延迟,点上标注并发数。
运行:python plot_curve.py --csv result.csv --out curve.png
"""

import argparse
import csv
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("plot_curve")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="result.csv")
    parser.add_argument("--out", default="curve.png")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    concurrency: list[int] = []
    throughput: list[float] = []
    latency_p99: list[float] = []
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            concurrency.append(int(row["concurrency"]))
            throughput.append(float(row["throughput_tok_s"]))
            latency_p99.append(float(row["e2e_p99"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(throughput, latency_p99, marker="o")
    for c, x, y in zip(concurrency, throughput, latency_p99):
        ax.annotate(f"c={c}", (x, y), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Throughput (output tokens/s)")
    ax.set_ylabel("P99 E2E latency (s)")
    ax.set_title("Throughput vs Latency (find the knee)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    logger.info("曲线已保存到 %s,请肉眼找延迟开始上翘的拐点", args.out)


if __name__ == "__main__":
    main()
