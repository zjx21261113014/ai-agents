"""第 2 节：共轭梯度法，并与最速下降 / 固定步长 GD / Nesterov 对比。

题目要求（第 2 节）
------------------
编写共轭梯度法的程序求解问题 (5)，并与最速下降法、固定步长为 alpha = L/2 的
梯度下降法和 Nesterov 加速方法进行对比。针对 n = 50 和 n = 1000 两种规模，
画出

    (a) 迭代步数 VS ln f(x_k)
    (b) 迭代步数 VS ln ||x_k - x*||_2
    (c) CPU 运行时间 VS ln ||x_k - x*||_2

的曲线，数值感受不同方法的优劣。

设定
----
* 问题 (5)：min f(x) = 1/2 ||Ax - b||^2，A 为 n x n 标准正态矩阵，seed = 0，
  x0 = 0（与第 1 节的实验保持一致）。
* 四种方法
    - CG          : 共轭梯度法（经典三项递推，精确算术下 n 步收敛）
    - SD          : 最速下降法（每步精确线搜索 alpha_k = g^T g / g^T Q g）
    - GD (L/2)    : 固定步长 alpha = L/2 = 1/lambda_max
    - Nesterov    : alpha = 1/lambda_max（Nesterov 格式的稳定前提），
                    beta = (sqrt(k)-1)/(sqrt(k)+1)，理论谱半径 1 - 1/sqrt(k)
* 停止准则统一为 ||x_k - x*||_2 <= 1e-10 * ||x_0 - x*||_2；
  为控制 CPU 开销，一阶方法的迭代上限按规模设定（见 CAPS），
  未达精度的方法在表和图里都明确标成"未收敛"，这本身就是一个结论。
* 计时用 ``time.process_time``；BLAS 线程数固定为 1，以保证 CPU 时间可重复。

输出
----
figures/fig7_cg.png    2 x 3 面板（上行 n = 50，下行 n = 1000）
results/sec2_cg.json   数值记录
"""

from __future__ import annotations

import os

# 固定 BLAS 线程数：n = 1000 的矩阵乘法用多线程反而使 CPU 计时剧烈抖动
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

import matplotlib.pyplot as plt

import numopt as no
from numopt import style
from numopt.analysis import iterations_to, momentum_spectral_rate

from common import print_table, save_json, sci

SEED = 0
RTOL = 1e-10
CASES = (50, 1000)
# 每个方法的迭代上限（按规模）：只为控制 CPU 开销，未跑满精度的方法在图/表里标成未收敛
CAPS = {
    50: {"cg": None, "sd": 300_000, "gd_half": 700_000, "nes": 300_000},
    1000: {"cg": None, "sd": 20_000, "gd_half": 20_000, "nes": 90_000},
}

ORDER = ("cg", "sd", "gd_half", "nes")
LABEL = {
    "cg": "CG（共轭梯度）",
    "sd": "最速下降（精确线搜索）",
    "gd_half": "GD（$\\alpha=L/2$）",
    "nes": "Nesterov（$\\alpha=1/\\lambda_{\\max}$，常 $\\beta$）",
}
LABEL_PLAIN = {
    "cg": "CG",
    "sd": "最速下降",
    "gd_half": "GD (alpha = L/2)",
    "nes": "Nesterov",
}
COLOR = {
    "cg": style.COLORS["CG"],
    "sd": style.COLORS["SD"],
    "gd_half": style.COLORS["GD"],
    "nes": style.COLORS["Nesterov"],
}
MARKER = {"cg": "o", "sd": "s", "gd_half": "^", "nes": "D"}
LINESTYLE = {"cg": "-", "sd": "-", "gd_half": "-", "nes": "-"}


def run_case(n: int) -> tuple[object, dict, dict]:
    """在 n 维问题上跑四种方法。"""
    prob = no.make_problem(n, seed=SEED)
    x0 = np.zeros(n)
    caps = CAPS[n]
    sig = np.sqrt(prob.kappa)
    beta_std = (sig - 1.0) / (sig + 1.0)          # 强凸问题的经典动量系数
    alpha_nes = 1.0 / prob.lam_max                # Nesterov 格式要求 alpha <= 1/L
    rho_nes = momentum_spectral_rate(prob.eigvals, alpha_nes, beta_std,
                                     scheme="nesterov")

    hists = {
        "cg": no.conjugate_gradient(prob, x0, rtol=RTOL),
        "sd": no.steepest_descent(prob, x0, max_iter=caps["sd"], rtol=RTOL),
        "gd_half": no.gradient_descent(prob, x0, 0.5 * prob.L,
                                       max_iter=caps["gd_half"], rtol=RTOL),
        "nes": no.nesterov(prob, x0, alpha_nes, max_iter=caps["nes"],
                           rtol=RTOL, beta_const=beta_std),
    }
    meta = {
        "n": n,
        "lambda_min": float(prob.lam_min),
        "lambda_max": float(prob.lam_max),
        "kappa": float(prob.kappa),
        "sqrt_kappa": float(np.sqrt(prob.kappa)),
        "L": float(prob.L),
        "alpha_gd": 0.5 * float(prob.L),
        "alpha_nesterov": float(alpha_nes),
        "beta_nesterov": float(beta_std),
        "rho_gd_theory": float(prob.predicted_rate(0.5 * prob.L)),
        "rho_sd_theory": float(prob.predicted_rate(1.0 / prob.lam_max) ** 2),
        "rho_nes_theory": float(rho_nes),
        "rho_nes_analytic": float(1.0 - 1.0 / sig),
        "f_star": float(prob.f_star),
        "err0": float(np.linalg.norm(x0 - prob.x_star)),
        "max_iter": caps,
        "rtol": RTOL,
    }
    return prob, hists, meta


def main() -> None:
    style.setup()

    data: dict[int, dict] = {}
    records: dict[str, dict] = {}
    rows = []

    for n in CASES:
        prob, hists, meta = run_case(n)
        data[n] = {"prob": prob, "hists": hists, "meta": meta}
        print(f"n = {n}: lambda_min = {meta['lambda_min']:.4e}, "
              f"lambda_max = {meta['lambda_max']:.4e}, kappa = {meta['kappa']:.4e}, "
              f"sqrt(kappa) = {meta['sqrt_kappa']:.2f}")
        print(f"       理论谱半径：GD(L/2) {meta['rho_gd_theory']:.8f}, "
              f"SD {meta['rho_sd_theory']:.8f}, "
              f"Nesterov {meta['rho_nes_theory']:.8f} "
              f"(解析值 1 - 1/sqrt(k) = {meta['rho_nes_analytic']:.8f})")

        for key in ORDER:
            h = hists[key]
            rec = {
                "iters": int(h.iters),
                "converged": bool(h.converged),
                "iters_to_rel": {f"{t:.0e}": iterations_to(h.err, t)
                                 for t in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)},
                "final_f": float(h.f[-1]) if np.isfinite(h.f[-1]) else None,
                "final_err": float(h.err[-1]),
                "final_rel_err": float(h.err[-1] / meta["err0"]),
                "cpu_time": float(h.time[-1]),
                "cpu_per_iter": float(h.time[-1] / max(h.iters, 1)),
            }
            if key == "nes":
                rec["beta_const"] = meta["beta_nesterov"]
                rec["alpha"] = meta["alpha_nesterov"]
            elif key == "gd_half":
                rec["alpha"] = meta["alpha_gd"]
            records[f"n{n}_{key}"] = rec
        print()

    # ------------------------------------------------------------ 表 8
    header = ["规模", "方法", "步数(→相对误差 1e-10)", "收敛", "末步 $\\ln\\|x_k-x^*\\|$",
              "CPU (s)", "相对 CG 的时间倍数"]
    cpu_cg = {n: data[n]["hists"]["cg"].time[-1] for n in CASES}
    for n in CASES:
        for key in ORDER:
            h = data[n]["hists"][key]
            rows.append([
                f"n = {n}",
                LABEL_PLAIN[key],
                str(h.iters) if h.converged else f">{h.iters}（到上限）",
                "是" if h.converged else "否",
                f"{np.log(max(h.err[-1], 1e-300)):8.3f}",
                sci(h.time[-1], 3),
                "1.0" if key == "cg" else f"{h.time[-1] / cpu_cg[n]:.1f}",
            ])
    print_table(header, rows,
                "表 8  CG 与三种一阶方法的代价对比（停止准则：相对误差 1e-10）")

    # ------------------------------------------------------------ 图 7
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.6), constrained_layout=True)

    panels = [
        ("step", "f", "$\\ln f(x_k)$", "迭代步数 $k$"),
        ("step", "err", "$\\ln\\|x_k-x^*\\|_2$", "迭代步数 $k$"),
        ("cpu", "err", "$\\ln\\|x_k-x^*\\|_2$", "CPU 时间 (s)"),
    ]
    pname = ["步数 — $\\ln f$", "步数 — $\\ln\\|x_k-x^*\\|$", "CPU — $\\ln\\|x_k-x^*\\|$"]

    # 每个面板一句结论（数字全部取自实际运行结果）
    def note_of(n: int, j: int, meta: dict, hists: dict) -> str:
        cg, sd = hists["cg"], hists["sd"]
        gd, nes = hists["gd_half"], hists["nes"]
        lnf_cg = np.log(max(cg.f[-1], 1e-300))
        if n == 50 and j == 0:
            return (f"CG {cg.iters} 步就把 $f$ 压到舍入误差量级"
                    f"（$\\ln f\\approx{lnf_cg:.0f}$）；其余三条要 1e3–1e5 步")
        if n == 50 and j == 1:
            r = (1 - meta["rho_sd_theory"]) / (1 - meta["rho_gd_theory"])
            return (f"下降斜率 $\\ln\\rho$：SD 的 $|\\ln\\rho|$ 是 GD 的 {r:.1f} 倍，"
                    f"Nesterov 再大 {(1 - meta['rho_nes_theory']) / (1 - meta['rho_sd_theory']):.0f} 倍")
        if n == 50 and j == 2:
            return (f"换成 CPU 时间轴结论不变：CG 只花 {cg.time[-1]:.3f} s，"
                    f"SD 要 {sd.time[-1]:.2f} s（{sd.time[-1] / cg.time[-1]:.0f} 倍）")
        if n == 1000 and j == 0:
            return (f"$\\kappa={meta['kappa']:.1e}$：SD/GD 的 {sd.iters} 步里 $\\ln f$ 几乎不动，"
                    f"CG 仍在 {cg.iters} 步内收敛")
        if n == 1000 and j == 1:
            if nes.converged:
                return (f"Nesterov 把 $\\kappa$ 换成 $\\sqrt{{\\kappa}}$：{nes.iters} 步达 1e-10；"
                        f"GD($L/2$) 在 {gd.iters} 步内几乎没动")
            return (f"Nesterov 在 {nes.iters} 步内把 $\\ln\\|e\\|$ 降了 "
                    f"{np.log(meta['err0']) - np.log(max(nes.err[-1], 1e-300)):.1f} 个量级；"
                    f"GD($L/2$) 仍在 0 附近")
        return (f"同样的精度：CG {cg.time[-1]:.2f} s，"
                f"SD {sd.time[-1]:.0f} s / GD {gd.time[-1]:.0f} s 都未跑到 1e-10")

    handles, labels = None, None
    for i, xn in enumerate(CASES):
        meta = data[xn]["meta"]
        hists = data[xn]["hists"]
        for j, (mx, my, ylabel, xlabel) in enumerate(panels):
            ax = axes[i, j]
            for key in ORDER:
                h = hists[key]
                if mx == "step":
                    x = np.arange(h.err.size, dtype=float)
                    keep = x >= 1
                else:
                    x = h.time
                    keep = x > 1e-9
                y = np.log(np.maximum(h.f if my == "f" else h.err, 1e-300))
                ax.plot(x[keep], y[keep], LINESTYLE[key], color=COLOR[key],
                        lw=1.5, label=LABEL[key])
            ax.set_xscale("log")
            style.log_axis(ax, "x")
            ax.set_xlabel(xlabel + "（对数轴）")
            ax.set_ylabel(ylabel)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.30 * (hi - lo))
            ax.set_title(f"({chr(ord('a') + 3 * i + j)}) n = {xn}：{pname[j]}", fontsize=10.5)
            ax.annotate(note_of(xn, j, meta, hists), xy=(0.5, 0.975),
                        xycoords="axes fraction", ha="center", va="top",
                        fontsize=7.4,
                        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.7", alpha=0.94))
            if i == 0 and j == 0:
                handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9.5,
               frameon=False)

    fig.suptitle(
        "第 2 节：共轭梯度法与最速下降 / GD($\\alpha=L/2$) / Nesterov 的对比"
        f"（seed = {SEED}，$x_0=0$，停止准则：相对误差 1e-10）",
        fontsize=13,
    )
    style.save(fig, "fig7_cg")

    save_json("sec2_cg", {"seed": SEED, "cases": {f"n{n}": data[n]["meta"] for n in CASES},
                          "methods": records})

    print()
    print("判读要点：")
    for n in CASES:
        h_cg = data[n]["hists"]["cg"]
        h_sd = data[n]["hists"]["sd"]
        h_gd = data[n]["hists"]["gd_half"]
        h_nes = data[n]["hists"]["nes"]
        print(f"  n = {n}：CG {h_cg.iters} 步 / {h_cg.time[-1]:.3f} s"
              f"（收敛：{h_cg.converged}）；"
              f"SD {'未收敛' if not h_sd.converged else str(h_sd.iters) + ' 步'}"
              f" / {h_sd.time[-1]:.2f} s；"
              f"GD(L/2) {'未收敛' if not h_gd.converged else str(h_gd.iters) + ' 步'}"
              f" / {h_gd.time[-1]:.2f} s；"
              f"Nesterov {h_nes.iters} 步 / {h_nes.time[-1]:.2f} s"
              f"（收敛：{h_nes.converged}）")
    print()
    print("已写出 figures/fig7_cg.png 与 results/sec2_cg.json")


if __name__ == "__main__":
    main()
