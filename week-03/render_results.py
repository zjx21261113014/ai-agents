r"""读取真实的 results.csv，生成报告所需的表格、参数与图片。

直接运行 ``uv run --locked python render_results.py``。本步不重新投针、不编造结果，
只根据当前工作目录中已有的 results.csv 生成：
- results-table.tex：可直接 \input 的 LaTeX 表格（n、crossings、pi_estimate）。
- results.png：不同 n 对应的估计值，附真实 pi 参考线。
- results-params.tex：实际使用的 seed、各组 n、针长与线间距。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("results.csv 中没有实验结果，请先运行 experiment.py")
    return rows


def write_table(rows: list[dict[str, str]], path: Path) -> None:
    lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"seed & $N$ & 相交数 $K$ & $\widehat{\pi}$ & $|\widehat{\pi}-\pi|$ \\",
        r"\midrule",
    ]
    for row in rows:
        estimate = float(row["pi_estimate"])
        lines.append(
            f"{int(row['seed'])} & {int(row['n'])} & {int(row['crossings'])} & "
            f"{estimate:.6f} & {abs(estimate - math.pi):.6f} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_params(rows: list[dict[str, str]], path: Path) -> None:
    seeds = "、".join(str(value) for value in sorted({int(row["seed"]) for row in rows}))
    sizes = "、".join(str(int(row["n"])) for row in rows)
    lengths = "、".join(sorted({row["needle_length"] for row in rows}))
    spacings = "、".join(sorted({row["line_spacing"] for row in rows}))
    text = (
        f"本次实际参数：seed = {seeds}；$N$ = {sizes}；"
        f"针长 $L$ = {lengths}；平行线间距 $d$ = {spacings}。\n"
    )
    path.write_text(text, encoding="utf-8")


def write_figure(rows: list[dict[str, str]], path: Path) -> None:
    ns = [int(row["n"]) for row in rows]
    estimates = [float(row["pi_estimate"]) for row in rows]

    fig, ax = plt.subplots(figsize=(6.8, 3.5), constrained_layout=True)
    ax.plot(ns, estimates, marker="o", label="Buffon estimate")
    ax.axhline(math.pi, color="black", linestyle="--", label=r"$\pi$")
    ax.set_xscale("log")
    ax.set_xlabel("Needles (N)")
    ax.set_ylabel(r"Estimated $\pi$")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    root = Path.cwd()
    rows = load_rows(root / "results.csv")
    write_table(rows, root / "results-table.tex")
    write_params(rows, root / "results-params.tex")
    write_figure(rows, root / "results.png")
    print("已写入 results-table.tex、results-params.tex 和 results.png")


if __name__ == "__main__":
    main()
