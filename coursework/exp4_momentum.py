"""问题 5：在**相同步长 α 和相同初值 x0** 下对比梯度下降 / Heavy Ball / Nesterov。

对应作业第 "1." 节第 5 小问。为了与第 3、4 小问的技术衔接，这里同时给出

  * "迭代步数 — f(x_k)"、"迭代步数 — ‖x_k - x*‖₂" 两组曲线（第 3 小问的技术）；
  * rho_k = e_{k+1}/e_k 的演化与理论谱半径的对照（第 4 小问的技术）；
  * 两种步长：
      - 主对比取 α = L/2 = 1/λmax（三种方法都稳定，且正好是 Nesterov 常数动量的
        理论最优步长）；
      - 极端对比取 α = L = 2/λmax（GD 的收敛上界）：GD 停滞，而 Heavy Ball 仍收敛，
        用来说明动量法扩大了稳定步长区间。

公平性的处理：题目要求三种方法用同一个 α。Heavy Ball 的经典 β = ((√κ-1)/(√κ+1))^2
是针对它自己的最优步长 α = 4/(√λmax+√λmin)^2 推导的，直接搬到 α = L/2 上并不是最优。
因此每种动量法都在给定 α 下用一维搜索取使谱半径最小的 β，另附"经典 β"作为对照。

运行：uv run python exp4_momentum.py
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


def describe(name: str, h: no.History, beta_label: str, rho_theory: float) -> dict:
    """把一个 History 归纳成一行表格 + 一份 JSON 记录。"""
    e = h.relative_err()
    k = np.arange(e.size, dtype=float)
    fit = semilog_slope(k, e, start_frac=0.30, end_frac=0.95)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(e[:-1] > 0, e[1:] / np.maximum(e[:-1], 1e-300), np.nan)
    return {
        "name": name,
        "beta_label": beta_label,
        "rho_theory": rho_theory,
        "rho_fit": fit["rate"],
        "semilog_r2": fit["r2"],
        "rho_tail_median": tail_median(ratios),
        "iters": h.iters,
        "converged": h.converged,
        "final_f": float(h.f[-1]),
        "final_err": float(h.err[-1]),
        "cpu_time": float(h.time[-1]),
        "iters_to": {f"{t:.0e}": iterations_to(h.err, t) for t in TOLS},
    }


def row_of(rec: dict, gd_iters: int) -> list[str]:
    return [
        rec["name"],
        rec["beta_label"],
        f"{rec['rho_theory']:.8f}" if np.isfinite(rec["rho_theory"]) else "—",
        f"{rec['rho_fit']:.6f}" if np.isfinite(rec["rho_fit"]) else "—",
        f"{rec['semilog_r2']:.6f}" if np.isfinite(rec["semilog_r2"]) else "—",
        *[str(rec["iters_to"][f"{t:.0e}"]) if rec["iters_to"][f"{t:.0e}"] is not None else "未达到"
          for t in TOLS],
        str(rec["iters"]) if rec["converged"] else f">{rec['iters']}",
        f"{gd_iters / max(rec['iters'], 1):.1f}×" if rec["converged"] else "—",
        sci(rec["cpu_time"], 2) + " s",
    ]


TABLE_HEADER = [
    "方法", "β", "ρ 理论", "ρ 实测", "R²",
    "→1e-4", "→1e-6", "→1e-8", "→1e-10", "步数", "对 GD 加速", "CPU",
]


def main() -> None:
    style.setup()
    prob = no.make_problem(N, seed=SEED)
    x0 = np.zeros(N)

    L = prob.L
    lam_min, lam_max = prob.lam_min, prob.lam_max
    kappa = prob.kappa
    sq = np.sqrt(kappa)

    alpha_main = L / 2.0            # = 1/λmax，三种方法都稳定的"共同步长"
    alpha_edge = L                  # = 2/λmax，GD 的收敛上界

    # ---- 三种动量系数的取法 --------------------------------------------------
    beta_hb_classic = ((sq - 1.0) / (sq + 1.0)) ** 2   # 配 α_HB = 4/(√λmax+√λmin)^2
    beta_nes_classic = (sq - 1.0) / (sq + 1.0)         # 配 α = 1/λmax

    hb_tune = no.optimal_momentum_beta(prob.eigvals, alpha_main, "heavy_ball")
    nes_tune = no.optimal_momentum_beta(prob.eigvals, alpha_main, "nesterov")
    hb_tune_edge = no.optimal_momentum_beta(prob.eigvals, alpha_edge, "heavy_ball")
    nes_tune_edge = no.optimal_momentum_beta(prob.eigvals, alpha_edge, "nesterov")

    rho_gd_main = prob.predicted_rate(alpha_main)      # |1 - α·λ| 的最大值 = (κ-1)/(κ+1)
    rho_gd_edge = prob.predicted_rate(alpha_edge)      # |1 - 2| = 1

    print(f"n = {N}, seed = {SEED}, κ(Q) = {kappa:.4e}, √κ = {sq:.4f}")
    print(f"L = 2/λmax = {L:.6e}")
    print(f"主对比步长 α = L/2 = 1/λmax = {alpha_main:.6e}；"
          f"极端步长 α = L = 2/λmax = {alpha_edge:.6e}")
    print(f"GD 的 ρ：α=L/2 时 1-1/κ = {rho_gd_main:.8f}；α=L 时 |1-2| = {rho_gd_edge:.8f}")
    print(f"动量系数（α=L/2）：HB 经典 {beta_hb_classic:.6f} → ρ="
          f"{no.momentum_spectral_rate(prob.eigvals, alpha_main, beta_hb_classic, 'heavy_ball'):.8f}；"
          f"HB 最优 {hb_tune['beta']:.6f} → ρ={hb_tune['rho']:.8f}")
    print(f"                    Nesterov 经典 {beta_nes_classic:.6f} → ρ="
          f"{no.momentum_spectral_rate(prob.eigvals, alpha_main, beta_nes_classic, 'nesterov'):.8f}；"
          f"最优 {nes_tune['beta']:.6f} → ρ={nes_tune['rho']:.8f}")
    print(f"动量系数（α=L）  ：HB 最优 {hb_tune_edge['beta']:.6f} → ρ={hb_tune_edge['rho']:.8f}；"
          f"Nesterov 最优 {nes_tune_edge['beta']:.6f} → ρ={nes_tune_edge['rho']:.8f}"
          f"（=0 即退化为 GD）；Nesterov 经典 {beta_nes_classic:.6f} → ρ="
          f"{no.momentum_spectral_rate(prob.eigvals, alpha_edge, beta_nes_classic, 'nesterov'):.8f} > 1（发散）\n")
    print(f"理论加速比：GD 需要 ~1/κ 型收缩，动量法 ~1-1/√κ 型，步数比 ≈ √κ = {sq:.1f}\n")

    # =============================================== 第一部分：α = L/2 的主对比
    specs_main = [
        ("GD", "—", rho_gd_main,
         lambda: no.gradient_descent(prob, x0, alpha_main, max_iter=800_000, rtol=RTOL)),
        ("Heavy Ball（经典 β）", f"{beta_hb_classic:.6f}",
         no.momentum_spectral_rate(prob.eigvals, alpha_main, beta_hb_classic, "heavy_ball"),
         lambda: no.heavy_ball(prob, x0, alpha_main, beta_hb_classic, max_iter=100_000, rtol=RTOL)),
        ("Heavy Ball（最优 β）", f"{hb_tune['beta']:.6f}", hb_tune["rho"],
         lambda: no.heavy_ball(prob, x0, alpha_main, hb_tune["beta"], max_iter=100_000, rtol=RTOL)),
        ("Nesterov（常数 β）", f"{beta_nes_classic:.6f}",
         no.momentum_spectral_rate(prob.eigvals, alpha_main, beta_nes_classic, "nesterov"),
         lambda: no.nesterov(prob, x0, alpha_main, max_iter=100_000, rtol=RTOL,
                             beta_const=beta_nes_classic)),
        ("Nesterov（默认 β_k）", "(k-1)/(k+2)", float("nan"),
         lambda: no.nesterov(prob, x0, alpha_main, max_iter=800_000, rtol=RTOL)),
    ]

    hists_main, recs_main = {}, {}
    for name, beta_label, rho_th, fn in specs_main:
        h = fn()
        hists_main[name] = h
        recs_main[name] = describe(name, h, beta_label, rho_th)

    gd_iters = recs_main["GD"]["iters"]
    print_table(
        TABLE_HEADER,
        [row_of(recs_main[name], gd_iters) for name, *_ in specs_main],
        f"表 4  同一 α = L/2 = 1/λmax 下三种方法的数值对比"
        f"（n={N}, seed={SEED}, 相对容差 {RTOL:.0e}）",
    )
    best = min((r for r in recs_main.values() if r["converged"]), key=lambda r: r["iters"])
    print(f"最快：{best['name']}，{best['iters']} 步，是 GD 的 {gd_iters / best['iters']:.1f} 倍；"
          f"其 ρ 实测 {best['rho_fit']:.6f} ≈ 理论 1-1/√κ = {1 - 1 / sq:.6f}。")
    print("注：'对 GD 加速' = GD 步数 / 该方法步数；ρ 实测为 semilogy 尾段（30%~95%）的拟合因子。")
    print("注：α = L/2 = 1/λmax 恰是 Nesterov 常数动量的理论最优步长，所以它的经典 β 已是"
          "该步长下的最优 β；\n    Heavy Ball 的经典 β 是配 α = 4/(√λmax+√λmin)^2 ≈ 2L 的，"
          "搬到 α = L/2 上并非最优，需重新取 β 才能达到同样的 ρ。\n")
    print(f"注：Nesterov 默认 β_k=(k-1)/(k+2) 是给一般凸问题求 O(1/k^2) 用的，β_k→1 使有效动量"
          f"长期偏大，\n    在强凸问题上只比 GD 快 {gd_iters / recs_main['Nesterov（默认 β_k）']['iters']:.1f} 倍，"
          "且误差不是严格几何衰减（R² 仅 "
          f"{recs_main['Nesterov（默认 β_k）']['semilog_r2']:.2f}）。")
    # =============================================== 第二部分：α = L 的极端对比
    specs_edge = [
        ("GD", "—", rho_gd_edge,
         lambda: no.gradient_descent(prob, x0, alpha_edge, max_iter=60_000, rtol=RTOL)),
        ("Heavy Ball（最优 β）", f"{hb_tune_edge['beta']:.6f}", hb_tune_edge["rho"],
         lambda: no.heavy_ball(prob, x0, alpha_edge, hb_tune_edge["beta"], max_iter=100_000, rtol=RTOL)),
        ("Heavy Ball（经典 β）", f"{beta_hb_classic:.6f}",
         no.momentum_spectral_rate(prob.eigvals, alpha_edge, beta_hb_classic, "heavy_ball"),
         lambda: no.heavy_ball(prob, x0, alpha_edge, beta_hb_classic, max_iter=100_000, rtol=RTOL)),
        ("Nesterov（常数 β）", f"{beta_nes_classic:.6f}",
         no.momentum_spectral_rate(prob.eigvals, alpha_edge, beta_nes_classic, "nesterov"),
         lambda: no.nesterov(prob, x0, alpha_edge, max_iter=150, rtol=RTOL,
                             beta_const=beta_nes_classic)),
        ("Nesterov（默认 β_k）", "(k-1)/(k+2)", float("nan"),
         lambda: no.nesterov(prob, x0, alpha_edge, max_iter=150, rtol=RTOL)),
    ]

    hists_edge, recs_edge = {}, {}
    for name, beta_label, rho_th, fn in specs_edge:
        with np.errstate(over="ignore", invalid="ignore"):  # 发散算例会溢出，属预期
            h = fn()
        hists_edge[name] = h
        recs_edge[name] = describe(name, h, beta_label, rho_th)

    print_table(
        TABLE_HEADER,
        [row_of(recs_edge[name], recs_edge["GD"]["iters"]) for name, *_ in specs_edge],
        f"表 5  极端步长 α = L = 2/λmax 下三种方法的数值对比"
        f"（n={N}, seed={SEED}, 相对容差 {RTOL:.0e}）",
    )
    print("注：α = L 时 GD 的 ρ = |1 - 2| = 1（λmax 方向分量永不衰减），只能靠其他分量变慢而"
          "令误差代数式缓慢下降，\n    60 000 步后相对误差仍有 "
          f"{recs_edge['GD']['final_err'] / (np.linalg.norm(x0 - prob.x_star)):.3f}；"
          "Heavy Ball 的稳定区间是 0 < α < 4/λmax = 2L，故仍收敛。")
    print("注：Nesterov 的梯度在*外推点* y_k = x_k + β(x_k - x_{k-1}) 处取值，外推放大了高频分量，\n"
          "    其稳定步长上限退回 α < 1/λmax；α = L 时任何 β>0 都发散（ρ = "
          f"{recs_edge['Nesterov（常数 β）']['rho_theory']:.4f}），最优 β 只能是 0（退化为 GD）。\n")

    # ------------------------------------------------------------------ 图 4
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), layout="constrained")
    C = style.COLORS
    palette = {
        "GD": (C["GD"], "-"),
        "Heavy Ball（经典 β）": (C["HB"], "--"),
        "Heavy Ball（最优 β）": (C["HB"], "-"),
        "Nesterov（常数 β）": (C["Nesterov"], "-"),
        "Nesterov（默认 β_k）": (C["Nesterov"], "--"),
    }
    XMAX = 600_000
    f0 = prob.f(x0)
    OSC = "Nesterov（默认 β_k）"

    def draw(ax, name: str, values: np.ndarray, *, force: str = "") -> None:
        """画一条方法曲线；默认 β_k 的序列每步剧烈振荡，用上下包络表示。"""
        c, ls = palette[name]
        k = np.arange(values.size)
        m = (k >= 1) & (k <= XMAX)
        if name != OSC:
            ax.plot(k[m], values[m], color=c, ls=ls, lw=1.5, label=name)
            return
        ax.plot(k[m], np.maximum.accumulate(values)[m], color=c, ls="--", lw=1.1,
                label=f"{name}（振荡上包络）")
        ax.plot(k[m], np.minimum.accumulate(values)[m], color=c, ls=":", lw=1.1,
                label=f"{name}（振荡下包络）")

    # (a) 步数 — f(x_k)
    ax = axes[0, 0]
    for name, h in hists_main.items():
        draw(ax, name, h.f)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-28, 3 * f0)
    ax.set_xlim(1, XMAX)
    ax.set_xlabel("迭代步数 $k$（对数轴）")
    ax.set_ylabel("$f(x_k)$")
    ax.set_title("(a) 同一 $\\alpha=L/2$：步数 — $f(x_k)$")
    ax.legend(loc="lower left", fontsize=7.6)
    style.log_axis(ax, "x")
    style.log_axis(ax, "y")
    ax.annotate(
        "默认 $\\beta_k=(k-1)/(k+2)\\to 1$ 使迭代剧烈振荡：\n"
        "同一轨迹上 $f$ 每步在上下包络之间跳动约 8 个数量级，\n"
        "因此它没有稳定的收缩率（表 4 中 $R^2$ 仅 "
        f"{recs_main[OSC]['semilog_r2']:.2f}）",
        xy=(0.02, 0.37), xycoords="axes fraction", va="bottom", ha="left", fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75"),
    )

    # (b) 步数 — ‖x_k - x*‖
    ax = axes[0, 1]
    for name, h in hists_main.items():
        draw(ax, name, h.err)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, XMAX)
    ax.set_ylim(1e-12, 1e2)
    ax.set_xlabel("迭代步数 $k$（对数轴）")
    ax.set_ylabel("$\\|x_k-x^*\\|_2$")
    ax.set_title("(b) 同一 $\\alpha=L/2$：步数 — $\\|x_k-x^*\\|_2$")
    ax.legend(loc="lower left", fontsize=7.6)
    style.log_axis(ax, "x")
    style.log_axis(ax, "y")
    for name in ("Nesterov（常数 β）", "Heavy Ball（最优 β）"):
        n_it = recs_main[name]["iters_to"]["1e-10"]
        if n_it:
            ax.plot([n_it], [1e-10], marker="o", ms=5.5, color=palette[name][0], mfc="white")
    ax.annotate(
        "Heavy Ball（最优 β）与 Nesterov 两条曲线几乎重合：\n"
        "$\\alpha=1/\\lambda_{\\max}$ 时两者的谱半径相同（见 (c)），\n"
        f"分别 {recs_main['Heavy Ball（最优 β）']['iters']} 步与 "
        f"{recs_main['Nesterov（常数 β）']['iters']} 步达到 $10^{{-10}}$",
        xy=(0.02, 0.66), xycoords="axes fraction", va="top", ha="left", fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75"),
    )

    # (c) rho_k 演化（判据 2）
    ax = axes[1, 0]
    XR = 4000
    for name in ("GD", "Heavy Ball（最优 β）", "Nesterov（常数 β）"):
        h = hists_main[name]
        e = h.relative_err()[: XR + 1]
        rk = e[1:] / np.maximum(e[:-1], 1e-300)
        m = np.isfinite(rk) & (rk > 0.985) & (rk < 1.012)
        c, ls = palette[name]
        ax.plot(np.arange(1, e.size)[m], rk[m], color=c, ls=ls, lw=1.1, alpha=0.9, label=name)
    for name, rho in (("GD", rho_gd_main), ("Heavy Ball（最优 β）", hb_tune["rho"]),
                      ("Nesterov（常数 β）", nes_tune["rho"])):
        ax.axhline(rho, color=palette[name][0], ls=":", lw=1.2,
                   label=f"$\\rho$ 理论 = {rho:.6f}")
    ax.set_xlim(0, XR)
    ax.set_ylim(0.988, 1.004)
    ax.set_xlabel("迭代步数 $k$")
    ax.set_ylabel("$\\rho_k = e_{k+1}/e_k$")
    ax.set_title("(c) 判据 2：$\\rho_k$ 与精确谱半径（$\\alpha=L/2$）")
    ax.legend(loc="center right", fontsize=7.8, ncol=1)
    ax.annotate(
        f"GD 的 $\\rho_k\\equiv 1-1/\\kappa$ = {rho_gd_main:.6f} 是常数；\n"
        f"动量法 $\\rho_k$ 有阻尼振荡，但始终低于 GD 的 $\\rho_k$，\n"
        f"并最终落在谱半径 $1-1/\\sqrt{{\\kappa}}$ = {1 - 1 / sq:.6f} 附近",
        xy=(0.02, 0.03), xycoords="axes fraction", va="bottom", fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75"),
    )

    # (d) 极端情形 α = L
    ax = axes[1, 1]
    XE = 6000
    for name, h in hists_edge.items():
        c, ls = palette[name]
        k = np.arange(h.err.size)
        m = (k >= 1) & (k <= XE)
        ax.semilogy(k[m], np.clip(h.err[m], 1e-16, 1e12), color=c, ls=ls, lw=1.5, label=name)
    ax.set_xlim(0, XE)
    ax.set_ylim(1e-12, 1e3)
    ax.set_xlabel("迭代步数 $k$")
    ax.set_ylabel("$\\|x_k-x^*\\|_2$")
    ax.set_title("(d) 极端情形 $\\alpha=L=2/\\lambda_{\\max}$：GD 停滞，动量法仍收敛")
    ax.legend(loc="center left", fontsize=8.0)
    style.log_axis(ax, "y")
    ax.annotate(
        "GD：$\\lambda_{\\max}$ 方向的分量永不衰减，误差停在停滞水平\n"
        f"Nesterov：梯度在外推点取值，$\\alpha=L$ 时任何 $\\beta>0$ 都发散"
        f"（$\\rho$={recs_edge['Nesterov（常数 β）']['rho_theory']:.2f}）\n"
        f"Heavy Ball：稳定区间 $0<\\alpha<4/\\lambda_{{\\max}}=2L$，"
        f"取 $\\beta$={hb_tune_edge['beta']:.3f} 时 $\\rho$={hb_tune_edge['rho']:.4f}",
        xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75"),
    )

    fig.suptitle("第 5 题：相同步长与初值下梯度下降 / Heavy Ball / Nesterov 的对比",
                 fontsize=13.5)
    out = style.save(fig, "fig4_momentum")
    print(f"\n图 4 已保存: {out}")

    save_json("q5_momentum", {
        "n": N, "seed": SEED, "rtol": RTOL,
        "kappa": kappa, "L": L,
        "alpha_main": alpha_main, "alpha_edge": alpha_edge,
        "beta_hb_classic": beta_hb_classic, "beta_nes_classic": beta_nes_classic,
        "hb_tuned_main": hb_tune, "nes_tuned_main": nes_tune,
        "hb_tuned_edge": hb_tune_edge, "nes_tuned_edge": nes_tune_edge,
        "rho_gd_main": rho_gd_main, "rho_gd_edge": rho_gd_edge,
        "rho_1_minus_inv_sqrt_kappa": 1.0 - 1.0 / sq,
        "speedup_sqrt_kappa": sq,
        "main": recs_main,
        "edge": recs_edge,
    })


if __name__ == "__main__":
    main()
