"""一键复现全部实验：exp1 -- exp7，并汇总耗时。

用法
----
    uv run python run_all.py            # 全部跑一遍
    uv run python run_all.py exp6 exp7  # 只跑指定模块

每个脚本都会把结果写到 results/*.json 与 figures/*.png，
报告（REPORT.md）里的所有数字都取自这些 JSON。
"""

from __future__ import annotations

import os

# 必须在 import numpy 之前：固定 BLAS 线程数，否则 n = 1000 的 CPU 计时会抖动
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import importlib
import sys
import time

STEPS = [
    ("exp1_problem", "Q1 问题构造与特征值/条件数"),
    ("exp2_stepsizes", "Q2/Q3 步长对收敛速度的影响与收敛区间"),
    ("exp3_order", "Q4 收敛阶、步长比与超线性收敛"),
    ("exp4_momentum", "Q5 Heavy Ball 与 Nesterov 加速"),
    ("exp5_bb", "Q6 Barzilai-Borwein 步长"),
    ("exp6_contour", "Q7 n = 2 时四种方法的迭代点与等值线"),
    ("exp7_cg", "第 2 节 共轭梯度法与一阶方法的对比"),
]


def main(argv: list[str]) -> int:
    wanted = [a for a in argv[1:] if not a.startswith("-")]
    steps = [s for s in STEPS if not wanted or any(w in s[0] for w in wanted)]
    if not steps:
        print(f"没有匹配的模块：{wanted}")
        return 1

    print("=" * 78)
    print("复现全部实验（工作目录：%s）" % os.getcwd())
    print("=" * 78)

    timings: list[tuple[str, str, float]] = []
    t_all = time.perf_counter()
    for name, desc in steps:
        print()
        print("#" * 78)
        print(f"### {name}  —  {desc}")
        print("#" * 78)
        mod = importlib.import_module(name)
        t0 = time.perf_counter()
        mod.main()
        dt = time.perf_counter() - t0
        timings.append((name, desc, dt))
        print(f"\n>>> {name} 用时 {dt:.1f} s")

    total = time.perf_counter() - t_all
    print()
    print("=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"{'模块':<18}{'用时 (s)':>10}  说明")
    for name, desc, dt in timings:
        print(f"{name:<18}{dt:>10.1f}  {desc}")
    print(f"{'合计':<18}{total:>10.1f}")
    print()
    print("产物：figures/fig1..fig7*.png，results/*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
