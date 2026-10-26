"""问题 1：构造问题、数值验证其二次性与正定性、求解析最优解。

对应作业第 "1." 节的第 1 小问：

    A = randn(n, n), b = randn(n), min f(x) = 1/2 ||Ax - b||_2^2, n = 50

需要数值验证 (a) 这是正定的二次规划；(b) 给出最优解 x*。

运行：uv run python exp1_problem.py
"""

from __future__ import annotations

import numpy as np

import numopt as no
from numopt import style

from common import md_table, print_table, save_json, sci

N = 50
SEED = 0


def quadratic_checks(prob: no.LeastSquaresProblem, rng: np.random.Generator) -> dict:
    """用有限差分独立验证"目标函数是二次的、Hessian 恒为 Q"。"""
    x = rng.standard_normal(prob.n)

    # (1) 二次函数的三点恒等式: f(x+d) - 2f(x) + f(x-d) = d^T Q d
    d = rng.standard_normal(prob.n)
    d = d / np.linalg.norm(d)
    t = 0.5
    lhs = prob.f(x + t * d) - 2.0 * prob.f(x) + prob.f(x - t * d)
    rhs = (t**2) * float(d @ prob.Q @ d)
    cubic_residual = abs(lhs - rhs) / max(abs(rhs), 1e-300)

    # (2) 中心差分近似 Hessian，与 Q 比较
    h = 1e-5
    H_hat = np.empty((prob.n, prob.n))
    g0 = prob.grad(x)
    for j in range(prob.n):
        e = np.zeros(prob.n)
        e[j] = h
        H_hat[:, j] = (prob.grad(x + e) - g0) / h
    H_err = float(np.linalg.norm(H_hat - prob.Q) / np.linalg.norm(prob.Q))

    # (3) 凸性: f((x+y)/2) <= (f(x)+f(y))/2 对随机点成立
    y = rng.standard_normal(prob.n)
    mid = prob.f(0.5 * (x + y)) - 0.5 * (prob.f(x) + prob.f(y))
    mid_normalized = mid / max(0.5 * (prob.f(x) + prob.f(y)), 1e-300)

    return {
        "cubic_term_relative_residual": cubic_residual,
        "hessian_fd_relative_error": H_err,
        "convexity_gap_normalized": float(mid_normalized),
    }


def main() -> None:
    style.setup()
    rng = np.random.default_rng(2024)

    prob = no.make_problem(N, seed=SEED)
    info = prob.verify_spd()
    info.update(quadratic_checks(prob, rng))
    info.update(prob.rayleigh_range())
    info["seed"] = SEED
    info["objective"] = "f(x) = 1/2 * ||A x - b||_2^2"
    info["grad_norm_at_x_star_relative"] = info["grad_norm_at_x_star"] / float(
        np.linalg.norm(prob.c)
    )
    # x* 残差：A x* - b 是 b 在 range(A)^perp 上的投影，一般非零
    r = prob.A @ prob.x_star - prob.b
    info["residual_norm_at_x_star"] = float(np.linalg.norm(r))
    info["b_norm"] = float(np.linalg.norm(prob.b))
    save_json("q1_problem", info)

    rows = [
        ["Q = AᵀA 对称性相对误差", sci(info["Q_symmetry_rel_error"], 3)],
        ["Cholesky 分解成功", "是" if info["cholesky_success"] else "否"],
        ["Q 的特征值全为正", "是" if info["all_eigenvalues_positive"] else "否"],
        ["λ_min(Q)", sci(prob.lam_min)],
        ["λ_max(Q)", sci(prob.lam_max)],
        ["条件数 κ(Q) = λ_max/λ_min", sci(prob.kappa)],
        ["条件数 κ(A)", sci(info["condition_number_A"])],
        ["二次性检验：三阶项相对残差", sci(info["cubic_term_relative_residual"], 3)],
        ["Hessian 有限差分相对误差", sci(info["hessian_fd_relative_error"], 3)],
        ["凸性检验：中点间隙（归一化）", sci(info["convexity_gap_normalized"], 3)],
        ["‖x*‖₂", sci(info["x_star_norm"])],
        ["f(x*)", sci(info["f_star"], 6)],
        ["‖∇f(x*)‖₂", sci(info["grad_norm_at_x_star"], 3)],
        ["‖∇f(x*)‖₂ / ‖c‖₂", sci(info["grad_norm_at_x_star_relative"], 3)],
        ["‖A x* - b‖₂", sci(info["residual_norm_at_x_star"])],
        ["三种解法求 x* 的最大相对差异", sci(
            max(info["rel_diff_solve_vs_lstsq"], info["rel_diff_solve_vs_inv"]), 3)],
        ["Rayleigh 商探测范围", f'[{sci(info["rayleigh_min_probe"], 3)}, '
                              f'{sci(info["rayleigh_max_probe"], 3)}]'],
        ["固定步长区间上端点 L = 2/λ_max", sci(prob.L)],
        ["最优固定步长 α* = 2/(λ_min+λ_max)", sci(prob.alpha_opt)],
        ["α* 对应的收敛因子 ρ*", f"{prob.rho_opt:.8f}"],
    ]
    print_table(["检验项", "数值结果"], rows, f"表 1  问题正定性检验与最优解（n={N}, seed={SEED}）")

    # ------------------------------------------------------------------ 图 1
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    ax = axes[0]
    ax.semilogy(np.arange(1, N + 1), prob.eigvals, "o-", ms=3.2, lw=1.2,
                color=style.COLORS["GD"], label=f"n = {N}（seed = {SEED}）")
    p1000 = no.make_problem(1000, seed=SEED)
    ax.semilogy(np.arange(1, 1001), p1000.eigvals, "-", lw=1.4, alpha=0.85,
                color=style.COLORS["Nesterov"], label="n = 1000（seed = 0）")
    ax.set_xlabel("特征值序号 $i$")
    ax.set_ylabel("特征值 $\\lambda_i(Q)$")
    ax.set_title("$Q=A^{T}A$ 的特征值谱")
    ax.set_ylim(1e-4, 1e5)
    ax.legend(loc="lower right")
    style.log_axis(ax, "y")
    style.panel_label(ax, "(a)")

    # (b) 数值验证"二次性"：真值 f(x_0 + t d) 与二阶 Taylor 模型必须完全重合
    rng2 = np.random.default_rng(7)
    x0 = rng2.standard_normal(prob.n)
    d = rng2.standard_normal(prob.n)
    d /= np.linalg.norm(d)
    g = prob.grad(x0)
    curv = float(d @ prob.Q @ d)          # 二阶方向导数 dᵀQd
    t = np.linspace(-3.0, 3.0, 61)
    f_exact = np.array([prob.f(x0 + s * d) for s in t])
    f_model = prob.f(x0) + t * float(g @ d) + 0.5 * t**2 * curv
    gap = np.abs(f_exact - f_model)
    info["quadratic_model_max_abs_gap"] = float(gap.max())
    info["quadratic_model_rel_gap"] = float(gap.max() / max(np.abs(f_exact).max(), 1e-300))
    save_json("q1_problem", info)

    ax = axes[1]
    ax.plot(t, f_exact, "-", color=style.COLORS["GD"], lw=2.2,
            label="真实 $f(x_0+td)$")
    ax.plot(t[::5], f_model[::5], "o", ms=5.0, mfc="none", mew=1.3,
            color=style.COLORS["Nesterov"], label="二次模型 $f_0+t g^{T}d+\\frac{t^2}{2}d^{T}Qd$")
    ax.set_xlabel("步长 $t$")
    ax.set_ylabel("$f(x_0 + t d)$")
    ax.set_title("沿任意射线与二次模型处处重合")
    ax.legend(loc="upper center")
    ax.annotate(
        f"两条曲线最大偏差 = {gap.max():.2e}\n"
        f"相对误差 = {info['quadratic_model_rel_gap']:.2e}",
        xy=(0.03, 0.14), xycoords="axes fraction", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fffdf5", ec="0.75"),
    )
    style.panel_label(ax, "(b)")

    path = style.save(fig, "fig1_problem")
    print(f"\n图 1 已保存: {path}")
    print(
        "说明：图 (a) 中 n=50 的特征值跨越 4 个数量级（条件数约 2.2e4）；\n"
        "图 (b) 中真实函数值与二阶 Taylor 模型的差异仅 1e-15 量级，"
        "两条曲线在视觉上完全重合，\n这直接验证了 f 的二次性与 Hessian 恒等于 Q。"
    )


if __name__ == "__main__":
    main()
