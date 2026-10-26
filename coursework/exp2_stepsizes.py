"""问题 2、3：固定步长梯度下降的收敛区间与不同步长的实验。

对应作业第 "1." 节的第 2、3 小问：

  Q2  理论推导步长收敛区间 (0, 2/λ_max)，并数值求解该区间；
  Q3  在 (0, L]（L = 2/λ_max）内取 α = L/3, L/2, 3L/4, L，
      分别画出"迭代步数 — f(x_k)"与"迭代步数 — ‖x_k - x*‖₂"。

运行：uv run python exp2_stepsizes.py
"""

from __future__ import annotations

import numpy as np

import numopt as no
from numopt import style
from numopt.analysis import asymptotic_ratio, iterations_to

from common import HIST_HEADER, history_row, print_table, save_json, sci

N = 50
SEED = 0
RTOL = 1e-10


def main() -> None:
    style.setup()
    prob = no.make_problem(N, seed=SEED)
    x0 = np.zeros(N)

    L = prob.L
    cases = {
        "L/3": L / 3.0,
        "L/2": L / 2.0,
        "3L/4": 3.0 * L / 4.0,
        "L": L,
    }
    caps = {"L/3": 2_000_000, "L/2": 2_000_000, "3L/4": 2_000_000, "L": 60_000}

    hists = {}
    rows, payload = [], {}
    for label, alpha in cases.items():
        h = no.gradient_descent(prob, x0, alpha, max_iter=caps[label], rtol=RTOL)
        hists[label] = h
        rho_th = prob.predicted_rate(alpha)
        rho_obs = asymptotic_ratio(h.relative_err(), tail_frac=0.05)
        row = [
            label,
            sci(alpha),
            f"{rho_th:.8f}",
            f"{rho_obs:.6f}",
            *[str(iterations_to(h.err, t)) if iterations_to(h.err, t) is not None else "未达到"
              for t in (1e-4, 1e-6, 1e-8, 1e-10)],
            str(h.iters),
            "是" if h.converged else "否",
            sci(h.f[-1]),
            sci(h.err[-1]),
            sci(h.time[-1], 2) + " s",
        ]
        rows.append(row)
        payload[label] = {
            "alpha": alpha,
            "rho_theory": rho_th,
            "rho_observed": rho_obs,
            "iters": h.iters,
            "converged": h.converged,
            "final_f": float(h.f[-1]),
            "final_err": float(h.err[-1]),
            "cpu_time": float(h.time[-1]),
            "iters_to": {f"{t:.0e}": iterations_to(h.err, t) for t in (1e-4, 1e-6, 1e-8, 1e-10)},
        }

    print_table(
        ["α", "α 数值", "ρ 理论", "ρ 实测", "步数→1e-4", "步数→1e-6",
         "步数→1e-8", "步数→1e-10", "实际步数", "达到容差", "f(x_K)", "‖x_K−x*‖₂", "CPU"],
        rows,
        f"表 2  固定步长梯度下降（n={N}, seed={SEED}, 相对容差 {RTOL:.0e}）",
    )

    # 理论收敛区间：收敛 <=> 0 < α < 2/λ_max；最优步长 α* = 2/(λ_min+λ_max)
    payload["_theory"] = {
        "L": L,
        "alpha_opt": prob.alpha_opt,
        "rho_opt": prob.rho_opt,
        "lambda_min": prob.lam_min,
        "lambda_max": prob.lam_max,
        "upper_bound_2_over_lambda_max": 2.0 / prob.lam_max,
    }

    # ------------------------------------------------------------------ 图 2
    import matplotlib.pyplot as plt

    colors = {
        "L/3": style.COLORS["HB"],
        "L/2": style.COLORS["GD"],
        "3L/4": style.COLORS["Nesterov"],
        "L": style.COLORS["BB1"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2), layout="constrained")

    ax = axes[0, 0]
    for label, h in hists.items():
        k = np.arange(h.f.size)
        ax.semilogy(k, style.positive(h.f), lw=1.5, color=colors[label],
                    label=f"α = {label} = {cases[label]:.3e}")
    ax.set_xlabel("迭代步数 $k$")
    ax.set_ylabel("$f(x_k)$")
    ax.set_title("目标函数值（几何衰减，故为直线）")
    ax.legend(loc="lower left", fontsize=9)
    style.log_axis(ax, "y")
    style.panel_label(ax, "(a)")

    ax = axes[0, 1]
    for label, h in hists.items():
        k = np.arange(h.err.size)
        ax.semilogy(k, style.positive(h.err), lw=1.5, color=colors[label],
                    label=f"α = {label}")
        rho = asymptotic_ratio(h.relative_err(), tail_frac=0.05)
        if 0 < rho < 1:
            kk = np.linspace(0.35 * h.err.size, h.err.size - 1, 20)
            ax.semilogy(kk, h.err[0] * rho**kk, ":", lw=1.1, color=colors[label], alpha=0.9)
    ax.set_xlabel("迭代步数 $k$")
    ax.set_ylabel("$\\|x_k - x^*\\|_2$")
    ax.set_title("误差范数（点线：实测 $\\rho^k$ 参考斜率）")
    ax.legend(loc="lower left", fontsize=9)
    style.log_axis(ax, "y")
    style.panel_label(ax, "(b)")

    # (c) 达到 1e-10 相对误差所需步数的理论预测与实测
    ax = axes[1, 0]
    aa = np.linspace(0.02 * L, (1 - 1e-7) * L, 1500)
    rho_a = np.maximum(np.abs(1 - aa * prob.lam_min), np.abs(1 - aa * prob.lam_max))
    n_pred = np.log(RTOL) / np.log(rho_a)
    ax.semilogy(aa / L, n_pred, "-", color="0.35", lw=1.8,
                label="理论步数 $\\ln(\\varepsilon)/\\ln\\rho(\\alpha)$")
    ax.plot(prob.alpha_opt / L, np.log(RTOL) / np.log(prob.rho_opt), "*", ms=15,
            color=style.COLORS["CG"], label=f"$\\alpha^*$（理论最优）")
    for label, alpha in cases.items():
        rho = prob.predicted_rate(alpha)
        if rho < 1:
            ax.plot(alpha / L, np.log(RTOL) / np.log(rho), "o", ms=8,
                    color=colors[label], label=f"α = {label}")
            ax.plot(alpha / L, payload[label]["iters"], "x", ms=9, mew=1.8,
                    color=colors[label])
        else:
            ax.annotate("α = L：不收敛\n（步数发散，未绘）", xy=(alpha / L, 1.6e5),
                        xytext=(-6, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=8.6, color=colors[label],
                        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=colors[label], lw=0.9))
    ax.set_xlabel("α / L")
    ax.set_ylabel("达到 $\\|x_k-x^*\\|\\leq 10^{-10}\\|x_0-x^*\\|$ 的步数")
    ax.set_title("理论步数曲线与实测值（× 为实测）")
    ax.set_ylim(1e5, 3e7)
    ax.legend(loc="upper center", fontsize=8, ncol=2)
    style.log_axis(ax, "y")
    style.panel_label(ax, "(c)")

    # (d) alpha = L：|1 - alpha*lam_max| = 1，误差在 lam_max 方向永不衰减 -> 停在停滞水平
    ax = axes[1, 1]
    hL = hists["L"]
    kk = np.arange(hL.f.size)
    ax.semilogy(kk, style.positive(hL.f), lw=1.4, color=colors["L"], label="α = L 时的 $f(x_k)$")
    w, V = np.linalg.eigh(prob.Q)
    c_max = float(V[:, -1] @ (x0 - prob.x_star))
    floor_f = 0.5 * prob.lam_max * c_max**2
    ax.axhline(floor_f, color="0.4", ls="--", lw=1.3,
               label=f"理论停滞水平 $\\frac{{1}}{{2}}\\lambda_{{\\max}}c_{{\\max}}^2$ = {floor_f:.4f}")
    ax.set_xlabel("迭代步数 $k$")
    ax.set_ylabel("$f(x_k)$")
    ax.set_title("α = L：收敛到停滞水平后不再下降")
    ax.legend(loc="upper right", fontsize=8.6)
    ax.annotate(
        f"$c_{{\\max}}=v_{{\\max}}^{{T}}(x_0-x^*)$ = {c_max:.3e}\n"
        f"$f$ 最终停在 {floor_f:.4f}，不趋于 $f^*\\approx 0$",
        xy=(0.03, 0.90), xycoords="axes fraction", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff5f5", ec="0.75"),
    )
    style.log_axis(ax, "y")
    style.panel_label(ax, "(d)")
    payload["L"]["stall_floor_f"] = floor_f
    payload["L"]["c_max"] = c_max

    path = style.save(fig, "fig2_stepsizes")
    print(f"图 2 已保存: {path}")

    save_json("q2_q3_stepsizes", payload)

    print_table(HIST_HEADER, [history_row(h) for h in hists.values()], "各步长运行汇总")


if __name__ == "__main__":
    main()
