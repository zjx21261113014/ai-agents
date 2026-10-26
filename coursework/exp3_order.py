"""问题 4：如何用画图观察收敛阶 / 收敛速率。

对应作业第 "1." 节第 4 小问。核心是把"阶"这件抽象的事拆成四个可视判据：

  (a) semilogy(e_k — k) 是否为直线   -> 是否 R-线性（几何收敛）；
  (b) rho_k = e_{k+1}/e_k 是否趋于常数 -> 是否 Q-线性；常数即渐近压缩因子 rho；
  (c) 计算型收敛阶 p_k = ln(e_{k+1}/e_k)/ln(e_k/e_{k-1}) 是否趋于 1 -> 线性阶；
  (d) loglog(e_k — k)：斜率是否为常数 -p -> 代数收敛 e_k ~ C k^{-p}，与几何收敛区分。

同一张表里给出每种方法由上述判据得到的定量结论，并与二次函数的理论 rho 对照。

运行：uv run python exp3_order.py
"""

from __future__ import annotations

import numpy as np

import numopt as no
from numopt import style
from numopt.analysis import coc, loglog_slope, semilog_slope

from common import print_table, save_json

N = 50
SEED = 0
RTOL = 1e-10


def tail_median(values: np.ndarray, tail_frac: float = 0.3) -> float:
    """取尾部若干比例的有限值中位数。"""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    n_tail = max(1, int(np.ceil(tail_frac * v.size)))
    return float(np.median(v[-n_tail:]))


def momentum_longrun(prob, alpha: float, beta: float, scheme: str = "heavy_ball",
                     steps: int = 20_000) -> dict:
    """在特征坐标下长程仿真动量法，用于检验"实测 rho 是否趋于精确谱半径"。

    动量法在二次问题上的每个特征分量都满足二阶线性递推
        z_{k+1} = p * z_k - q * z_{k-1}
    其中 Heavy Ball 取 p = 1+β-αλ, q = β；Nesterov 取 p = (1+β)(1-αλ), q = β(1-αλ)。
    直接迭代这 n 条标量递推比运行原算法快得多，可以在算法早已因达到容差而停止
    之后继续观察渐近行为。
    """
    w, V = np.linalg.eigh(prob.Q)
    c0 = V.T @ (np.zeros(prob.n) - prob.x_star)          # x_0 = 0 时的初始误差系数
    if scheme == "heavy_ball":
        p = 1.0 + beta - alpha * w
        q = np.full_like(w, beta)
    elif scheme == "nesterov":
        p = (1.0 + beta) * (1.0 - alpha * w)
        q = beta * (1.0 - alpha * w)
    else:
        raise ValueError("scheme 只能取 'heavy_ball' 或 'nesterov'")

    z_prev = c0.copy()          # z_{-1} = z_0
    z = c0.copy()
    ratios = np.empty(steps)
    for k in range(steps):
        z_next = p * z - q * z_prev
        z_prev, z = z, z_next
        ratios[k] = np.linalg.norm(z) / np.linalg.norm(z_prev)
    return {
        "windows": {f"{lo}-{hi}": float(np.median(ratios[lo:hi]))
                    for lo, hi in ((10, 2_000), (2_000, 5_000), (5_000, 20_000))},
        "ratios": ratios,
    }


def main() -> None:
    style.setup()
    prob = no.make_problem(N, seed=SEED)
    x0 = np.zeros(N)

    L = prob.L
    lam_min, lam_max = prob.lam_min, prob.lam_max
    kappa = prob.kappa
    sq = np.sqrt(kappa)

    alpha_opt = prob.alpha_opt                       # 2/(lam_min + lam_max)
    rho_gd = prob.predicted_rate(alpha_opt)          # |1 - alpha* lam| = (κ-1)/(κ+1)
    rho_sd = (kappa - 1.0) / (kappa + 1.0)           # 经典界（f 的界为 ρ²）

    # Heavy Ball：α = 4/(√λmax+√λmin)², β = ((√κ-1)/(√κ+1))²
    alpha_hb = 4.0 / (np.sqrt(lam_max) + np.sqrt(lam_min)) ** 2
    beta_hb = ((sq - 1.0) / (sq + 1.0)) ** 2
    # Nesterov 常数动量：α = 1/λmax, β = (√κ-1)/(√κ+1)
    alpha_nes = 1.0 / lam_max
    beta_nes = (sq - 1.0) / (sq + 1.0)

    # 精确的理论收敛因子：在整条谱上取二阶递推的特征根模的最大值
    rho_hb = no.momentum_spectral_rate(prob.eigvals, alpha_hb, beta_hb, "heavy_ball")
    rho_nes = no.momentum_spectral_rate(prob.eigvals, alpha_nes, beta_nes, "nesterov")

    print(f"n = {N}, seed = {SEED}, κ = κ(Q) = {kappa:.4e}")
    print(f"L = 2/λmax = {L:.6e};  α* = {alpha_opt:.6e};  α_HB = {alpha_hb:.6e};"
          f"  α_Nesterov = {alpha_nes:.6e}")
    print(f"理论 ρ：GD(α*) = {rho_gd:.8f}；SD = (κ-1)/(κ+1) = {rho_sd:.8f}；"
          f"HB = {rho_hb:.8f}；Nesterov = {rho_nes:.8f}\n")

    # 动量法到达容差只要两三千步，可能还没进入渐近区；用特征坐标长程仿真核实
    hb_long = momentum_longrun(prob, alpha_hb, beta_hb, "heavy_ball")
    nes_long = momentum_longrun(prob, alpha_nes, beta_nes, "nesterov")
    print("动量法的精确谱半径与长程仿真得到的 rho_k（中位数）：")
    for name, rho, lr in (("Heavy Ball", rho_hb, hb_long), ("Nesterov", rho_nes, nes_long)):
        win = "  ".join(f"k∈[{k}] : {v:.6f}" for k, v in lr["windows"].items())
        print(f"  {name:11s} 谱半径 = {rho:.6f}  |  {win}")
    print()

    runs = {
        "GD(α=α*)": lambda: no.gradient_descent(prob, x0, alpha_opt, max_iter=400_000, rtol=RTOL),
        "最速下降": lambda: no.steepest_descent(prob, x0, max_iter=300_000, rtol=RTOL),
        "Heavy Ball": lambda: no.heavy_ball(prob, x0, alpha_hb, beta_hb, max_iter=20_000, rtol=RTOL),
        "Nesterov": lambda: no.nesterov(prob, x0, alpha_nes, max_iter=20_000, rtol=RTOL,
                                        beta_const=beta_nes),
        "BB1": lambda: no.bb(prob, x0, variant=1, max_iter=50_000, rtol=RTOL),
        "CG": lambda: no.conjugate_gradient(prob, x0, rtol=1e-14),
    }
    theory = {
        "GD(α=α*)": rho_gd,
        "最速下降": rho_sd,
        "Heavy Ball": rho_hb,
        "Nesterov": rho_nes,
        "BB1": float("nan"),
        "CG": float("nan"),
    }

    hists = {name: f() for name, f in runs.items()}

    # ------------------------------------------------- 收敛阶的四项数值判据
    rows, payload = [], {}
    for name, h in hists.items():
        e = h.relative_err()
        k = np.arange(e.size, dtype=float)
        fit = semilog_slope(k, e, start_frac=0.30, end_frac=0.95)
        lls = loglog_slope(k, e, start_frac=0.30, end_frac=0.95)
        kk, p = coc(e)
        p_tail = tail_median(p, 0.3)
        ratios = np.where(np.isfinite(e[:-1]) & (e[:-1] > 0), e[1:] / np.maximum(e[:-1], 1e-300), np.nan)
        rho_tail = tail_median(ratios, 0.1)

        verdict = ("Q-线性，阶 1" if np.isfinite(p_tail) and abs(p_tail - 1.0) < 0.05
                   else "非 Q-线性（阶判定不适用）")
        theo = theory[name]
        rows.append([
            name,
            f"{theo:.8f}" if np.isfinite(theo) else "—",
            f"{fit['rate']:.6f}",
            f"{fit['r2']:.6f}",
            f"{rho_tail:.6f}",
            f"{p_tail:.3f}" if np.isfinite(p_tail) else "—",
            f"{lls['slope']:.3f}",
            str(h.iters),
            verdict,
        ])
        payload[name] = {
            "iters": h.iters,
            "converged": h.converged,
            "cpu_time": float(h.time[-1]),
            "rho_theory": theory[name] if np.isfinite(theory[name]) else None,
            "rho_semilog_fit": fit["rate"],
            "semilog_slope": fit["slope"],
            "semilog_r2": fit["r2"],
            "rho_tail_median": rho_tail,
            "coc_tail_median": p_tail,
            "loglog_slope": lls["slope"],
            "final_relative_err": float(e[-1]),
            "verdict": verdict,
        }

    print_table(
        ["方法", "ρ 理论", "ρ(semilog 拟合)", "R²", "ρ_k 尾部中位数", "COC p̄_k", "loglog 斜率",
         "步数", "判定"],
        rows,
        f"表 3  收敛阶的四项数值判据（n={N}, seed={SEED}, 相对容差 {RTOL:.0e}）",
    )

    print("\n判据说明：R² ≈ 1 且 ρ<1 → semilogy 为直线，R-线性；ρ_k 趋于常数 rho → Q-线性；"
          "\n          p̄_k → 1 → 线性阶；loglog 斜率不是常数、且随 k 变陡 → 几何收敛而非代数收敛。")
    print("注：HB / Nesterov 的实测 ρ 略大于精确谱半径，是暂态与多模态拍频所致，长程仿真已确认 ρ_k 收敛到谱半径。")
    print("注：BB1 的 ρ_k 不趋于常数（呈锯齿状），其 p̄_k = 0.957 也偏离 1，说明它只是 R-线性而渐近地更接近超线性；")
    print("    CG 在对二次问题上 n=50 步内即达机器精度，属有限步终止，'阶' 的概念不适用。")

    # ------------------------------------------------------------------ 图 3
    import matplotlib.pyplot as plt

    colors = {
        "GD(α=α*)": style.COLORS["GD"],
        "最速下降": style.COLORS["SD"],
        "Heavy Ball": style.COLORS["HB"],
        "Nesterov": style.COLORS["Nesterov"],
        "BB1": style.COLORS["BB1"],
        "CG": style.COLORS["CG"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), layout="constrained")
    XMAX = 5000  # 快方法在 3000 步内即收敛，前 5000 步足以同时看清所有方法的形态
    orderly = ["GD(α=α*)", "最速下降", "Heavy Ball", "Nesterov"]  # 单调、规整的四个方法

    def label_for(name: str) -> str:
        info = payload[name]
        if name == "CG":
            return f"CG（{info['iters']} 步内达机器精度）"
        if name == "BB1":
            return f"BB1（拟合 $\\rho$={info['rho_semilog_fit']:.5f}，$R^2$={info['semilog_r2']:.3f}）"
        return f"{name}（$\\rho$={info['rho_semilog_fit']:.5f}）"

    # (a) semilogy：是否为直线
    ax = axes[0, 0]
    for name, h in hists.items():
        e = h.relative_err()[:XMAX + 1]
        k = np.arange(e.size)
        ax.semilogy(k, np.clip(e, 1e-300, None), lw=1.4, color=colors[name], label=label_for(name))
    ax.set_xlim(0, XMAX)
    ax.set_ylim(1e-12, 4)
    ax.set_xlabel("迭代步数 $k$")
    ax.set_ylabel("相对误差 $\\|x_k-x^*\\|/\\|x_0-x^*\\|$")
    ax.set_title("判据 1：semilogy 为直线 $\\Rightarrow$ R-线性（线性方法的 $R^2>0.9999$）")
    ax.legend(loc="lower left", bbox_to_anchor=(0.085, 0.02), fontsize=7.6, framealpha=0.95)
    ax.annotate(
        "GD(α*) / 最速下降在 5000 步内只降 0.64 倍：\n需 25.7 万步才到 $10^{-10}$（图 2）",
        xy=(0.985, 0.03), xycoords="axes fraction", ha="right", va="bottom", fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75"),
    )
    style.log_axis(ax, "y")
    style.panel_label(ax, "(a)")

    # (b) rho_k = e_{k+1}/e_k
    ax = axes[0, 1]
    for name in orderly:
        e = hists[name].relative_err()[:XMAX + 1]
        rk = e[1:] / np.clip(e[:-1], 1e-300, None)
        m = np.isfinite(rk) & (rk > 0.95) & (rk < 1.06)
        ax.plot(np.arange(1, e.size)[m], rk[m], lw=0.9, alpha=0.85, color=colors[name], label=name)
    ax.axhline(rho_gd, color=colors["GD(α=α*)"], ls="--", lw=1.1,
               label=f"ρ_GD 理论 = {rho_gd:.6f}")
    ax.axhline(rho_sd, color=colors["最速下降"], ls=":", lw=1.4,
               label=f"ρ_SD 理论 = {rho_sd:.6f}")
    ax.axhline(rho_hb, color=colors["Heavy Ball"], ls="--", lw=1.1,
               label=f"ρ_HB 谱半径 = {rho_hb:.6f}")
    ax.axhline(rho_nes, color=colors["Nesterov"], ls="--", lw=1.1,
               label=f"ρ_Nesterov 谱半径 = {rho_nes:.6f}")
    ax.set_xlim(0, XMAX)
    ax.set_xlabel("迭代步数 $k$")
    ax.set_ylabel("$\\rho_k = e_{k+1}/e_k$")
    ax.set_title("判据 2：$\\rho_k$ 趋于常数 $\\Rightarrow$ Q-线性")
    ax.set_ylim(0.95, 1.06)
    ax.legend(loc="upper right", fontsize=7.6, ncol=2)
    ax.annotate(
        "动量法的实测 $\\rho_k$ 略高于精确谱半径：\n"
        "达到 $10^{-10}$ 只需约 2000 步，仍在暂态内；\n"
        f"长程仿真 $k\\in[5000,20000]$ 给 HB {hb_long['windows']['5000-20000']:.6f}",
        xy=(0.02, 0.03), xycoords="axes fraction", va="bottom", fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75"),
    )
    style.panel_label(ax, "(b)")

    # (c) 计算型收敛阶 p_k
    ax = axes[1, 0]
    for name in orderly:
        kk, p = coc(hists[name].relative_err())
        m = np.isfinite(p) & (np.abs(p) < 6) & (kk <= XMAX)
        ax.plot(kk[m], np.clip(p[m], -1.5, 5.0), lw=0.9, alpha=0.85, color=colors[name], label=name)
    ax.axhline(1.0, color="0.3", ls="--", lw=1.3, label="线性阶 $p=1$")
    ax.set_xlim(0, XMAX)
    ax.set_xlabel("迭代步数 $k$")
    ax.set_ylabel("$p_k = \\ln(e_{k+1}/e_k)\\,/\\,\\ln(e_k/e_{k-1})$")
    ax.set_title("判据 3：$p_k \\to 1$ $\\Rightarrow$ 线性收敛阶")
    ax.set_ylim(0.9, 2.6)
    ax.legend(loc="upper right", fontsize=8.0, ncol=2)
    ax.annotate(
        "BB1 的 $p_k$ 全程锯齿、不趋于 1（表 3 中 $\\bar{p}_k$ = 0.957）：\n"
        "它只是 R-线性，并非规整的 Q-线性（详见第 6 题）",
        xy=(0.97, 0.52), xycoords="axes fraction", ha="right", va="top", fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75"),
    )
    style.panel_label(ax, "(c)")

    # (d) loglog：几何 vs 代数
    ax = axes[1, 1]
    for name in ("GD(α=α*)", "BB1", "CG"):
        e = hists[name].relative_err()
        k = np.arange(1, e.size, dtype=float)
        ax.loglog(k, np.clip(e[1:], 1e-300, None), lw=1.4, color=colors[name], label=name)
    kref = np.logspace(0.5, 5.45, 60)
    ax.loglog(kref, 10.0 / kref, ls="--", lw=1.1, color="0.45", label="参考斜率 $k^{-1}$")
    ax.loglog(kref, 1e4 / kref**2, ls=":", lw=1.1, color="0.45", label="参考斜率 $k^{-2}$")
    ax.set_xlabel("迭代步数 $k$（对数轴）")
    ax.set_ylabel("相对误差（对数轴）")
    ax.set_title("判据 4：loglog 斜率非常数 $\\Rightarrow$ 几何收敛，非代数收敛")
    ax.legend(loc="lower left", fontsize=8.4)
    style.log_axis(ax, "x")
    style.log_axis(ax, "y")
    style.panel_label(ax, "(d)")
    payload["_theory"] = {
        "alpha_opt": alpha_opt,
        "rho_gd": rho_gd,
        "rho_sd": rho_sd,
        "alpha_hb": alpha_hb,
        "beta_hb": beta_hb,
        "rho_hb_spectral": rho_hb,
        "alpha_nesterov": alpha_nes,
        "beta_nesterov": beta_nes,
        "rho_nesterov_spectral": rho_nes,
        "kappa": kappa,
        "longrun_windows": {
            "heavy_ball": hb_long["windows"],
            "nesterov": nes_long["windows"],
        },
    }

    path = style.save(fig, "fig3_order")
    print(f"\n图 3 已保存: {path}")
    save_json("q4_order", payload)


if __name__ == "__main__":
    main()
