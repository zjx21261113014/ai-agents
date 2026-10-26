"""问题 6：实现两种 Barzilai-Borwein（BB）方法，并与梯度下降法对比。

对应作业第 "1." 节第 6 小问。沿用第 3、4 小问的技术：

  * "迭代步数 — f(x_k)"、"迭代步数 — ‖x_k - x*‖₂" 曲线（第 3 小问的技术）；
  * rho_k = e_{k+1}/e_k 的演化（第 4 小问的技术），用来暴露 BB 的**非单调性**。

BB 方法的步长由最近两步的信息给出（s_k = x_k - x_{k-1}, y_k = g_k - g_{k-1}）：

    BB1: alpha_k = (s_k^T s_k) / (s_k^T y_k) = 1 / R_Q(s_k)
    BB2: alpha_k = (s_k^T y_k) / (y_k^T y_k) = R_Q(s_k) / R_{Q^2}(s_k)

对二次函数 s^T y = s^T Q s > 0，两种步长都自动落在 [1/λmax, 1/λmin] 内（由瑞利商
的界可证），**远大于** GD 的稳定上界 2/λmax —— 这正是 BB 快、但 f 不单调的原因。
首步取 alpha_0 = 1/λmax（= L/2），与固定步长 GD 的第一步相同，保证起点公平。

运行：uv run python exp5_bb.py
"""

from __future__ import annotations

import numpy as np

import numopt as no
from numopt import style
from numopt.analysis import iterations_to, semilog_slope

from common import print_table, save_json, sci

N = 50
SEED = 0
RTOL = 1e-10
TOLS = (1e-4, 1e-6, 1e-8, 1e-10)


def tail_median(values: np.ndarray, tail_frac: float = 0.1) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    n_tail = max(1, int(np.ceil(tail_frac * v.size)))
    return float(np.median(v[-n_tail:]))


def ratios_of(values: np.ndarray) -> np.ndarray:
    """返回 r_k = v_k / v_{k-1}，并丢弃模量过小（下溢、有效位数不足）的比值。"""
    v = np.asarray(values, dtype=float)
    scale = v[np.isfinite(v) & (v > 0)]
    if scale.size == 0:
        return np.array([])
    floor = 1e-14 * scale[0]          # f - f* 在收敛末期会下溢，比值不再可信
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        r = np.where(v[:-1] > floor, v[1:] / np.maximum(v[:-1], 1e-300), np.nan)
    return r[np.isfinite(r)]


def describe(key: str, label: str, h: no.History, prob: no.LeastSquaresProblem) -> dict:
    """把一个 History 归纳成一行表格 + 一份 JSON 记录。

    同时统计两种口径的收缩率：2-范数的 e_k，以及 f_k - f*（它恰好等于
    1/2 (x_k-x*)^T Q (x_k-x*)，即 Q-范数的平方）。BB 在这两种口径下表现差别极大。
    """
    e = h.relative_err()
    k = np.arange(e.size, dtype=float)
    fit = semilog_slope(k, e, start_frac=0.30, end_frac=0.95)
    n_step = max(h.iters, 1)

    rho_e = ratios_of(e)
    energy = h.f - prob.f_star
    rho_f = ratios_of(energy)
    # 只统计“能量尚未下溢”的区间，否则末期 $10^{-26}$ 量级的舍入噪声会被当成“f 上升”
    e0 = energy[0]
    valid = energy[:-1] > 1e-14 * e0
    n_f_increase = int(np.sum((np.diff(energy) > 0) & valid))
    return {
        "key": key,
        "label": label,
        "rho_fit": fit["rate"],
        "semilog_r2": fit["r2"],
        "rho_tail_median": tail_median(rho_e),
        "frac_rho_gt1": float(np.mean(rho_e > 1.0)),
        "rho_f_tail_median": tail_median(rho_f),
        "frac_rho_f_gt1": float(np.mean(rho_f > 1.0)),
        "geomean_rate": float((e[-1] / e[0]) ** (1.0 / n_step)) if e[-1] > 0 else 0.0,
        "energy_geomean_rate": float(((h.f[-1] - prob.f_star) / (h.f[0] - prob.f_star)) ** (1.0 / n_step)),
        "n_f_increase": n_f_increase,
        "n_valid": int(valid.sum()),
        "iters": h.iters,
        "converged": h.converged,
        "final_f": float(h.f[-1]),
        "final_err": float(h.err[-1]),
        "cpu_time": float(h.time[-1]),
        "iters_to": {f"{t:.0e}": iterations_to(h.err, t) for t in TOLS},
    }


def row_of(rec: dict, ref_iters: int) -> list[str]:
    def num(x: float, digits: int = 6) -> str:
        return f"{x:.{digits}f}" if np.isfinite(x) else "—"

    return [
        rec["label"],
        num(rec["rho_fit"]),
        num(rec["semilog_r2"]),
        num(rec["rho_tail_median"]),
        num(rec["rho_f_tail_median"]),
        f"{rec['frac_rho_f_gt1'] * 100:.0f}%",
        num(rec["geomean_rate"], 5),
        *[str(rec["iters_to"][f"{t:.0e}"]) if rec["iters_to"][f"{t:.0e}"] is not None else "未达到"
          for t in TOLS],
        str(rec["iters"]) if rec["converged"] else f">{rec['iters']}",
        f"{ref_iters / max(rec['iters'], 1):.1f}×" if rec["converged"] else "—",
        sci(rec["cpu_time"], 2) + " s",
    ]


TABLE_HEADER = [
    "方法", "ρ 拟合", "R²", "ρ 尾部中位数", "$\\rho^f$ 中位数", "$\\rho^f>1$",
    "几何平均速率", "→1e-4", "→1e-6", "→1e-8", "→1e-10", "步数", "对 GD(L/2)", "CPU",
]


def main() -> None:
    style.setup()
    prob = no.make_problem(N, seed=SEED)
    x0 = np.zeros(N)

    L = prob.L
    alpha_gd = L / 2.0          # = 1/λmax，与第 5 小问的主对比步长一致
    alpha_star = prob.alpha_opt  # = 2/(λmin+λmax)，GD 的最优固定步长

    print(f"n = {N}, seed = {SEED}, κ(Q) = {prob.kappa:.4e}, √κ = {np.sqrt(prob.kappa):.4f}")
    print(f"λmin = {prob.lam_min:.6e}, λmax = {prob.lam_max:.6e}")
    print(f"GD 步长：α = L/2 = 1/λmax = {alpha_gd:.6e}（ρ 理论 = {prob.predicted_rate(alpha_gd):.8f}）；"
          f"α* = 2/(λmin+λmax) = {alpha_star:.6e}（ρ 理论 = {prob.predicted_rate(alpha_star):.8f}）")
    print(f"BB 步长的理论区间：[1/λmax, 1/λmin] = [{1 / prob.lam_max:.6e}, {1 / prob.lam_min:.6e}]，"
          f"其右端是 GD 稳定上界 2/λmax = {L:.6e} 的 {prob.kappa / 2:.1f} 倍\n")

    runs = {
        "gd_half": ("GD(α=L/2)", no.gradient_descent(prob, x0, alpha_gd, rtol=RTOL, max_iter=800_000)),
        "gd_opt": ("GD(α=α*)", no.gradient_descent(prob, x0, alpha_star, rtol=RTOL, max_iter=800_000)),
        "bb1": ("BB1", no.bb(prob, x0, variant=1, rtol=RTOL, max_iter=400_000)),
        "bb2": ("BB2", no.bb(prob, x0, variant=2, rtol=RTOL, max_iter=400_000)),
    }
    recs = {k: describe(k, label, h, prob) for k, (label, h) in runs.items()}
    ref_iters = recs["gd_half"]["iters"]

    # ---- 表 6 ---------------------------------------------------------------
    rows = [row_of(recs[k], ref_iters) for k in ("gd_half", "gd_opt", "bb1", "bb2")]
    print_table(TABLE_HEADER, rows, "表 6  BB1 / BB2 与梯度下降法的数值对比（n=50, seed=0, 同一 x0=0）")

    print("判读要点：")
    print(f"  1. BB 的误差并不单调：按 f（即 Q-范数）口径有 "
          f"{recs['bb1']['frac_rho_f_gt1'] * 100:.1f}%（BB1）/"
          f"{recs['bb2']['frac_rho_f_gt1'] * 100:.1f}%（BB2）的步子是**变大**的，"
          f"f 分别上升过 {recs['bb1']['n_f_increase']} / {recs['bb2']['n_f_increase']} 次；"
          "GD 严格单调。\n"
          "     所以对 BB 不能用单步 rho_k 判断好坏，必须看几何平均速率 (e_N/e_0)^(1/N)。")
    print(f"  2. 几何平均速率：GD(L/2) {recs['gd_half']['geomean_rate']:.6f}，"
          f"GD(α*) {recs['gd_opt']['geomean_rate']:.6f}，"
          f"BB1 {recs['bb1']['geomean_rate']:.6f}，BB2 {recs['bb2']['geomean_rate']:.6f}。\n"
          f"     BB 的平均速率明显更小 → 步数少 {ref_iters / max(recs['bb1']['iters'], 1):.1f}×（BB1）/ "
          f"{ref_iters / max(recs['bb2']['iters'], 1):.1f}×（BB2）。"
          f"按 f 口径 BB1 的几何平均速率为 {recs['bb1']['energy_geomean_rate']:.6f}，"
          f"也是全部方法中最低的。")
    print(f"  3. semilog 拟合的 R² 只有 {recs['bb1']['semilog_r2']:.3f} / "
          f"{recs['bb2']['semilog_r2']:.3f}（GD 为 "
          f"{recs['gd_half']['semilog_r2']:.3f}）：BB 不是几何收敛，"
          "ln e_k 呈**阶梯下台阶**，每次下台阶对应步长被 \"重新校准\" 到 λmin 方向。")
    print(f"  4. 两种口径的反差：BB2 的 2-范数几乎从不上升（仅 {recs['bb2']['frac_rho_gt1'] * 100:.2f}% 的步），"
          f"但 f（Q-范数）上升了 {recs['bb2']['frac_rho_f_gt1'] * 100:.0f}% 的步。\n"
          "     说明 BB 的过冲主要发生在高曲率（λmax）方向：Q-范数被这一步放大，"
          "而 2-范数不受影响。选用哪个量度“单调”必须以结论所需为准。")
    print(f"  5. BB1 / BB2 的步数比为 {recs['bb1']['iters']}:{recs['bb2']['iters']} = "
          f"{recs['bb2']['iters'] / max(recs['bb1']['iters'], 1):.2f}。"
          "两者的步长中位数其实很接近（~1/λmax 量级），差异来自少数\"大步\"：\n"
          "     BB1 的 α_k = ||s||²/(sᵀQs) 直接是 1/R_Q(s)，当 s 主要含 λmin 方向分量时 α_k 可以"
          "很大，一次性消掉慢方向，故台阶宽、总步数少；\n"
          "     BB2 的 α_k = sᵀQs/(sᵀQ²s) 对高曲率方向更敏感，步长偏保守，收敛稳但需要更多步。\n")

    # ---- 图 5 ---------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), layout="constrained")
    C = style.COLORS
    pal = {
        "gd_half": (C["GD"], "-"),
        "gd_opt": (C["HB"], "-"),
        "bb1": (C["BB1"], "-"),
        "bb2": (C["BB2"], "-"),
    }
    XMAX = 600_000
    f0 = prob.f(x0)

    # (a) 步数 — f(x_k)
    ax = axes[0, 0]
    for k, (label, h) in runs.items():
        c, ls = pal[k]
        kk = np.arange(h.f.size)
        m = (kk >= 1) & (kk <= XMAX)
        ax.semilogy(kk[m], h.f[m], color=c, ls=ls, lw=1.2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, XMAX)
    ax.set_ylim(1e-28, 3 * f0)
    ax.set_xlabel("迭代步数 $k$（对数轴）")
    ax.set_ylabel("$f(x_k)$")
    ax.set_title("(a) 步数 — $f(x_k)$：BB 非单调、整体更快")
    ax.legend(loc="lower left", fontsize=7.8)
    style.log_axis(ax, "x")
    style.log_axis(ax, "y")
    ax.annotate(
        "BB 的曲线是锯齿状的：步长 α_k 常超过 GD 的稳定上界 $2/\\lambda_{\\max}$，\n"
        f"使 $f$ 上升过 {recs['bb1']['n_f_increase']} 次（BB1）/{recs['bb2']['n_f_increase']} 次（BB2），\n"
        "但平均下降速率仍远快于任何固定步长 GD",
        xy=(0.02, 0.30), xycoords="axes fraction", va="bottom", ha="left", fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75"),
    )

    # (b) 步数 — ‖x_k - x*‖
    ax = axes[0, 1]
    for k, (label, h) in runs.items():
        c, ls = pal[k]
        kk = np.arange(h.err.size)
        m = (kk >= 1) & (kk <= XMAX)
        ax.loglog(kk[m], h.err[m], color=c, ls=ls, lw=1.2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, XMAX)
    ax.set_ylim(1e-12, 1e2)
    ax.set_xlabel("迭代步数 $k$（对数轴）")
    ax.set_ylabel("$\\|x_k-x^*\\|_2$")
    ax.set_title("(b) 步数 — $\\|x_k-x^*\\|_2$：阶梯式下降")
    ax.legend(loc="lower left", fontsize=7.8)
    style.log_axis(ax, "x")
    style.log_axis(ax, "y")
    ax.annotate(
        "在 log-log 图上 BB 呈\"阶梯\"：每级台阶对应步长\n"
        "被校准到当前最慢的 λ 方向；台阶宽度逐步变大，\n"
        "因此 semilog 直线拟合的 $R^2$ 反而很低（表 6）",
        xy=(0.30, 0.85), xycoords="axes fraction", va="top", ha="left", fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75"),
    )

    # (c) BB 的步长序列
    ax = axes[1, 0]
    lo, hi = 1.0 / prob.lam_max, 1.0 / prob.lam_min
    ax.axhline(hi, color="0.45", ls="--", lw=1.0, label=r"$1/\lambda_{\min}$（理论区间上界）")
    ax.axhline(L, color=C["GD"], ls="--", lw=1.0, label=r"$2/\lambda_{\max}=L$（GD 稳定上界）")
    ax.axhline(alpha_star, color=C["HB"], ls=":", lw=1.2,
               label=r"$\alpha_\star=2/(\lambda_{\min}+\lambda_{\max})$")
    for k in ("bb1", "bb2"):
        h = runs[k][1]
        a = np.asarray(h.info["alphas"], dtype=float)
        c, _ = pal[k]
        ax.plot(np.arange(1, a.size + 1), a, color=c, lw=0.7, alpha=0.85, label=runs[k][0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, XMAX)
    ax.set_ylim(lo / 2, hi * 1.45)
    ax.set_xlabel("迭代步数 $k$（对数轴）")
    ax.set_ylabel("步长 $\\alpha_k$")
    ax.set_title("(c) BB 的自适应步长：剧烈摆动，但从不超出 $[1/\\lambda_{\\max},\,1/\\lambda_{\\min}]$")
    ax.legend(loc="upper left", fontsize=7.0, ncol=2, framealpha=0.92)
    style.log_axis(ax, "x")
    style.log_axis(ax, "y")
    ax.annotate(
        f"上界 $1/\\lambda_{{\\min}}={hi:.2e}$，是 GD 上界的 $\\kappa/2\\approx{prob.kappa / 2:.0f}$ 倍",
        xy=(0.99, 0.985), xycoords="axes fraction", va="top", ha="right", fontsize=7.0,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75"),
    )

    # (d) 判据 2 的正确用法：ρ^f_k 的分布
    ax = axes[1, 1]
    for k in ("gd_opt", "bb1", "bb2"):
        h = runs[k][1]
        energy = h.f - prob.f_star
        rho = ratios_of(energy)
        rho = rho[rho > 0]
        w = np.full(rho.size, 1.0 / rho.size)
        c, ls = pal[k]
        ax.hist(np.log10(rho), bins=60, range=(-3.0, 3.0), weights=w, histtype="step",
                color=c, ls=ls, lw=1.5, label=runs[k][0])
    ax.axvline(0.0, color="0.35", ls=":", lw=1.1)
    ax.set_yscale("log")
    ax.set_ylim(2e-3, 2.0)
    ax.set_xlim(-3.0, 3.0)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xlabel("$\\log_{10}\\rho^f_k$，其中 $\\rho^f_k=(f_k-f^*)/(f_{k-1}-f^*)$")
    ax.set_ylabel("落在该区间内的步数占比")
    ax.set_title("(d) 判据 2：单步收缩率的分布（$x>0$ 即该步 $f$ 上升）")
    ax.legend(loc="upper left", fontsize=7.6)
    style.log_axis(ax, "y")
    ax.annotate(
        f"GD($\\alpha_\\star$) 是 $\\log_{{10}}\\!\\approx\\!0$ 处的单根尖柱；\n"
        f"BB1 有 {recs['bb1']['frac_rho_f_gt1'] * 100:.0f}%、BB2 有 "
        f"{recs['bb2']['frac_rho_f_gt1'] * 100:.0f}% 的步落在 $\\rho^f_k>1$ 一侧，\n"
        f"也有大量 $\\rho^f_k\\sim10^{{-2}}$ 的“大步” $\\Rightarrow$ 只能比几何平均",
        xy=(0.98, 0.45), xycoords="axes fraction", va="top", ha="right", fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75"),
    )

    fig.suptitle("第 6 题：两种 Barzilai-Borwein 方法 vs 梯度下降法（n=50, seed=0, 同一初值）",
                 fontsize=13)
    style.save(fig, "fig5_bb")

    save_json("q6_bb", {
        "n": N, "seed": SEED, "rtol": RTOL,
        "lam_min": prob.lam_min, "lam_max": prob.lam_max, "kappa": prob.kappa,
        "alpha_gd_half": alpha_gd, "alpha_star": alpha_star,
        "bb_step_band": [lo, hi],
        "methods": recs,
        "speedup": {k: ref_iters / max(recs[k]["iters"], 1) for k in recs},
    })
    print("已写出 figures/fig5_bb.png 与 results/q6_bb.json")


if __name__ == "__main__":
    main()
