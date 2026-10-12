"""蒲丰投针实验：用随机投针的相交比例估计圆周率。

本模块可被 import；import 时不会启动实验、也不会写文件。
直接运行 ``uv run --locked python experiment.py`` 时，才按命令行参数运行实验并写出
``results.csv``（写入当前工作目录）。
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

# results.csv 的表头；顺序即字段顺序，六列一组的实验写成一行。
FIELDS = ("seed", "n", "crossings", "pi_estimate", "needle_length", "line_spacing")

# 起始配置：三组投针数，固定 seed=42。
DEFAULT_SEED = 42
DEFAULT_SIZES = (1000, 10000, 100000)
DEFAULT_LENGTH = 0.5
DEFAULT_SPACING = 1.0


def _check_geometry(length: float, spacing: float) -> None:
    """几何参数必须满足 0 < length <= spacing，且为有限数。"""
    if not (math.isfinite(length) and math.isfinite(spacing)):
        raise ValueError("length 和 spacing 必须是有限数")
    if not 0 < length <= spacing:
        raise ValueError("必须满足 0 < length <= spacing")


def crosses_line(
    distance: float,
    angle: float,
    length: float = DEFAULT_LENGTH,
    spacing: float = DEFAULT_SPACING,
) -> bool:
    """判断一次投针是否与平行线相交（接触到线也算相交）。

    distance 是针中心到最近一条线的垂直距离，取值 [0, spacing/2]；
    angle 是针与平行线的夹角（弧度），取值 [0, pi/2]。
    半根针垂直于线的伸出距离为 (length/2) * sin(angle)；
    distance <= 该值时相交，包括恰好触线。
    非法几何参数或越界的输入抛 ValueError，不静默修改。
    """
    _check_geometry(length, spacing)
    if not math.isfinite(distance) or not 0 <= distance <= spacing / 2:
        raise ValueError("distance 必须位于 [0, spacing/2]")
    if not math.isfinite(angle) or not 0 <= angle <= math.pi / 2:
        raise ValueError("angle 必须位于 [0, pi/2]")
    return distance <= (length / 2) * math.sin(angle)


def estimate_pi(
    n: int,
    crossings: int,
    length: float = DEFAULT_LENGTH,
    spacing: float = DEFAULT_SPACING,
) -> float:
    """由投针数 n 与相交数 crossings 估计圆周率：2 * length * n / (spacing * crossings)。

    n 为正整数，crossings 为整数且 0 <= crossings <= n；
    当 crossings = 0 时该式无意义，抛 ValueError。
    """
    _check_geometry(length, spacing)
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n 必须是正整数")
    if isinstance(crossings, bool) or not isinstance(crossings, int) or not 0 <= crossings <= n:
        raise ValueError("crossings 必须是 0 到 n 之间的整数")
    if crossings == 0:
        raise ValueError("crossings=0 时无法用该公式估计圆周率")
    return 2 * length * n / (spacing * crossings)


def simulate(
    n: int,
    seed: int = DEFAULT_SEED,
    length: float = DEFAULT_LENGTH,
    spacing: float = DEFAULT_SPACING,
) -> dict:
    """模拟 n 次投针并返回一次实验的六个字段。

    每次调用只在取样循环开始前初始化一次随机数生成器；
    循环内持续取样，不在每次投针时重设 seed。
    """
    _check_geometry(length, spacing)
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n 必须是正整数")

    rng = random.Random(seed)
    crossings = 0
    for _ in range(n):
        distance = rng.uniform(0.0, spacing / 2)
        angle = rng.uniform(0.0, math.pi / 2)
        if crosses_line(distance, angle, length, spacing):
            crossings += 1

    if crossings == 0:
        raise ValueError("本次实验未出现相交，无法给出有效的圆周率估计")

    return {
        "seed": seed,
        "n": n,
        "crossings": crossings,
        "pi_estimate": estimate_pi(n, crossings, length, spacing),
        "needle_length": length,
        "line_spacing": spacing,
    }


def run_experiment(
    sizes: list[int],
    seed: int = DEFAULT_SEED,
    length: float = DEFAULT_LENGTH,
    spacing: float = DEFAULT_SPACING,
    output: Path | None = None,
) -> Path:
    """按给定的各组投针数依次运行实验，写出 results.csv，返回输出路径。"""
    target = Path.cwd() / "results.csv" if output is None else output
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for n in sizes:
            row = simulate(n, seed, length, spacing)
            writer.writerow(row)
            print(
                f"seed={seed}, n={n}, crossings={row['crossings']}, "
                f"pi_estimate={row['pi_estimate']:.8f}"
            )
    print(f"已写入 {target.name}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="蒲丰投针实验：由相交比例估计圆周率")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子（默认 42）")
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="各组投针数（默认 1000 10000 100000）",
    )
    parser.add_argument("--length", type=float, default=DEFAULT_LENGTH, help="针长（默认 0.5）")
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING, help="平行线间距（默认 1.0）")
    args = parser.parse_args()
    run_experiment(args.n, args.seed, args.length, args.spacing)


if __name__ == "__main__":
    main()
