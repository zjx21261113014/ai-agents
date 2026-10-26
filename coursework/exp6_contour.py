"""第 7 题：n = 2 时四种方法的迭代点画在 f 的等值线上。

题目要求（第 7 题）
------------------
取 n = 2，在同样的初值条件下，把 BB 方法（取一种，这里用 BB1）、梯度下降法、
Heavy Ball 方法与 Nesterov 方法的 {x_k} 分别描述在 f(x) 的等值线上，
画出四种方法的结果并陈述现象。

统一设定
--------
* 问题仍是 min f(x) = 1/2 ||Ax - b||^2 = 1/2 x^T Q x - c^T x + const，n = 2；
  取 seed = 2，lambda = (0.0721, 6.3683)，kappa = 88.4（等高线主轴比 sqrt(kappa) ≈ 9.4，
  拉长的"香蕉谷"足够明显，又不至于像 seed = 3 那样窄到看不清）。
* 四种方法用**同一个初值** x0；步长取各自的“标准”值：梯度下降与 Heavy Ball
  用二次问题的最优固定步长 alpha = alpha* = 2/(lambda_min + lambda_max)，
  Nesterov 用 alpha = 1/lambda_max（= 1 / 梯度 Lipschitz 常数），
  BB1 用 alpha_0 = 1/lambda_max 起步。
* Heavy Ball 的 beta 用 `optimal_momentum_beta` 在 alpha = alpha* 下数值搜索得到，
  Nesterov 用经典 beta = ((sqrt(kappa)-1)/(sqrt(kappa)+1))^2。
  注：alpha = alpha* 时 Nesterov 对任何常数 beta > 0.02 都不稳定（前瞻点过冲），
  所以只能取 alpha = 1/lambda_max，这也是 Nesterov 基本格式的标准前提。

输出
----
figures/fig6_contour.png    四联图（每张一个方法，同一等高线、同一坐标范围）
results/q7_contour.json     数值记录
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from common import print_table, save_json
from numopt import (make_problem, momentum_spectral_rate, optimal_momentum_beta,
                    style)

N = 2
SEED = 2
X0 = np.array([-1.0, 2.0])
RTOL = 1e-10
N_MAX = 20_000
# 每张图最多画多少个点（None = 全部画出）
N_SHOW = {"bb1": None, "gd": 40, "hb": None, "nes": None}

PANEL_LABEL = {"bb1": "(a) BB1", "gd": "(b) 梯度下降",
               "hb": "(c) Heavy Ball", "nes": "(d) Nesterov"}
LABEL_TEX = {"bb1": "BB1",
             "gd": "梯度下降（$\\alpha=\\alpha_\\star$）",
             "hb": "Heavy Ball（$\\alpha=\\alpha_\\star$）",
             "nes": "Nesterov（$\\alpha=1/\\lambda_{\\max}$）"}
LABEL_PLAIN = {"bb1": "BB1", "gd": "梯度下降（alpha = alpha*）",
               "hb": "Heavy Ball（alpha = alpha*）",
               "nes": "Nesterov（alpha = 1/lambda_max）"}


def label_of(key: str) -> str:
    return LABEL_TEX[key]


# --------------------------------------------------------------------- 轨迹
def trajectory(prob, x0, kind, *, alpha=None, beta=None, n_max=N_MAX, rtol=RTOL):
    """按方法递推并记录所有迭代点（含 x0），同时返回 CPU 时间。

    这里为了画图需要保留 {x_k} 本身，而 numopt.methods 里的 History 只存
    f / err / 时间，所以对 n = 2 的小问题单独写一遍递推（与库中实现逐项一致）。
    """
    x0 = np.asarray(x0, dtype=float)
    e0 = np.linalg.norm(x0 - prob.x_star)
    tol = max(0.0, rtol * e0)

    xs = [x0.copy()]
    x_prev = x0.copy()
    x = x0.copy()
    g_prev = prob.grad(x0)
    a = 1.0 / prob.lam_max if alpha is None else float(alpha)

    t0 = time.process_time()
    for k in range(n_max):
        if not np.all(np.isfinite(x)):
            break
        g = prob.grad(x)
        if np.linalg.norm(x - prob.x_star) <= tol:
            break
        if kind == "gd":
            x_new = x - a * g
        elif kind == "hb":
            x_new = x - a * g + beta * (x - x_prev)
        elif kind == "nes":
            y = x + beta * (x - x_prev)
            x_new = y - a * prob.grad(y)
        elif kind == "bb1":
            s = -a * g
            x_try = x + s
            y_g = prob.grad(x_try) - g
            sy = float(s @ y_g)
            a = float(s @ s) / sy if sy > 0 else a
            x_new = x_try
        else:  # pragma: no cover
            raise ValueError(kind)
        x_prev, x = x, x_new
        xs.append(x.copy())
        if not np.all(np.isfinite(x)):
            break
    cpu = time.process_time() - t0
    return np.array(xs), cpu


def main() -> None:
    style.setup()
    prob = make_problem(N, seed=SEED)
    lmin, lmax = float(prob.eigvals[0]), float(prob.eigvals[-1])
    kappa = prob.kappa
    alpha = prob.alpha_opt  # = 2/(lambda_min + lambda_max)：GD 的最优固定步长
    # Nesterov 取 alpha = 1 / lambda_max = 1 / (梯度 Lipschitz 常数)
    alpha_nes = 1.0 / lmax
    beta_hb = optimal_momentum_beta(prob.eigvals, alpha, "heavy_ball")
    beta_val = float(beta_hb["beta"])
    rho_hb = float(beta_hb["rho"])
    beta_nes = ((np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0)) ** 2

    print(f"n = {N}, seed = {SEED}")
    print(f"lambda_min = {lmin:.6e}, lambda_max = {lmax:.6e}, kappa = {kappa:.4f}")
    print(f"x0 = {X0}, x* = {prob.x_star}, ||x0 - x*|| = {np.linalg.norm(X0 - prob.x_star):.4f}")
    print(f"GD 与 HB：alpha = alpha* = 2/(lambda_min+lambda_max) = {alpha:.6e}，"
          f"HB 的 beta = {beta_val:.6f}（rho = {rho_hb:.6f}）")
    print(f"Nesterov：alpha = 1/lambda_max = {alpha_nes:.6e}，beta = "
          f"(sqrt(k)-1)^2/(sqrt(k)+1)^2 = {beta_nes:.6f}")
    print()

    runs = {
        "bb1": ({"alpha": 1.0 / lmax}, r"BB1"),
        "gd": ({"alpha": alpha}, r"GD($\alpha=\alpha_\star$)"),
        "hb": ({"alpha": alpha, "beta": beta_val}, r"Heavy Ball"),
        "nes": ({"alpha": alpha_nes, "beta": beta_nes}, r"Nesterov($\alpha=1/\lambda_{\max}$)"),
    }

    recs: dict[str, dict] = {}
    trajs: dict[str, np.ndarray] = {}
    for key, (kw, label) in runs.items():
        xs, cpu = trajectory(prob, X0, key, **kw)
        trajs[key] = xs
        err = float(np.linalg.norm(xs[-1] - prob.x_star))
        recs[key] = {
            "label": label,
            "params": {k: float(v) for k, v in kw.items()},
            "iters": int(xs.shape[0] - 1),
            "final_err": err,
            "converged": bool(err <= max(0.0, RTOL * np.linalg.norm(X0 - prob.x_star))),
            "cpu_time": cpu,
        }

    # 理论谱半径：GD 由 1-alpha*lambda 决定，HB / Nesterov 用各自 (alpha, beta) 的谱半径
    rho_gd = float(prob.predicted_rate(alpha))
    rho_nes = float(momentum_spectral_rate(prob.eigvals, alpha_nes, beta_nes,
                                           scheme="nesterov"))
    recs["gd"]["rho_theory"] = rho_gd
    recs["hb"]["rho_theory"] = rho_hb
    recs["nes"]["rho_theory"] = rho_nes

    # ---------------------------------------------------------------- 打印表
    header = ["方法", "步数(→1e-10)", "理论 $\\rho$", "末步误差", "CPU (ms)", "轨迹形状"]
    shape = {
        "bb1": "一步下到谷底，再沿谷底大步命中 $x^*$",
        "gd": "沿陡方向来回锯齿，同时沿谷底缓慢漂移",
        "hb": "螺旋：冲过谷底再折回（过冲）",
        "nes": "沿谷底几乎是直线（陡方向分量一步清零）",
    }
    rows = []
    for key in ("bb1", "gd", "hb", "nes"):
        r = recs[key]
        rows.append([
            LABEL_PLAIN[key],
            str(r["iters"]),
            "—（自适应）" if r.get("rho_theory") is None else f"{r['rho_theory']:.4f}",
            f"{r['final_err']:.2e}",
            f"{r['cpu_time'] * 1e3:.2f}",
            shape[key],
        ])
    print_table(header, rows,
                "表 7  n = 2 时四种方法（各取标准参数，同一初值 x0）的迭代步数")

    # ---------------------------------------------------------------- 等值线
    allx = np.vstack([trajs[k] for k in trajs] + [prob.x_star[None, :], X0[None, :]])
    cx = 0.5 * (allx[:, 0].min() + allx[:, 0].max())
    cy = 0.5 * (allx[:, 1].min() + allx[:, 1].max())
    side = 1.30 * max(np.ptp(allx[:, 0]), np.ptp(allx[:, 1]), 4.0)
    xlim = (cx - 0.5 * side, cx + 0.5 * side)
    ylim = (cy - 0.5 * side, cy + 0.5 * side)

    gx = np.linspace(*xlim, 420)
    gy = np.linspace(*ylim, 420)
    GX, GY = np.meshgrid(gx, gy)
    G = np.stack([GX, GY], axis=-1)
    energy = 0.5 * np.einsum("...i,ij,...j->...", G, prob.Q, G) - G @ prob.c
    energy = np.maximum(energy - prob.f_star, 1e-30)
    Z = np.log10(energy)
    levels = np.linspace(Z.min() + 0.30, Z.max(), 16)

    # 主方向（等值线主轴）：过 x* 的两条特征方向
    _, V = np.linalg.eigh(prob.Q)
    dirs = {0: "$q_{\\min}$（$\\lambda_{\\min}$，谷底方向）",
            1: "$q_{\\max}$（$\\lambda_{\\max}$，陡方向）"}

    palette = {
        "bb1": style.COLORS["BB1"],
        "gd": style.COLORS["GD"],
        "hb": style.COLORS["HB"],
        "nes": style.COLORS["Nesterov"],
    }
    rho_gd = float(prob.predicted_rate(alpha))
    notes = {
        "bb1": (
            "首步 $\\alpha_0=1/\\lambda_{\\max}$ 与 GD 相同，先落到谷底；\n"
            "之后 $\\alpha_k=s^\\top s/s^\\top y$ 自动放大到 $1/\\lambda_{\\min}$ 量级，\n"
            "于是能沿谷底迈大步 —— 折线极少，几乎直接命中 $x^*$"
        ),
        "gd": (
            "在 $\\alpha=\\alpha_\\star$ 下快方向每步乘上 $1-\\alpha\\lambda_{\\max}$、慢方向乘上\n"
            f"$1-\\alpha\\lambda_{{\\min}}$，两者绝对值都等于 $(\\kappa-1)/(\\kappa+1)\\approx{rho_gd:.4f}$，\n"
            "于是沿陡方向来回锯齿，同时沿谷底慢慢漂移"
        ),
        "hb": (
            "动量把相邻两步位移累加，一步就跳到了谷底另一侧（过冲），\n"
            f"之后沿谷底螺旋进入 $x^*$；理论谱半径 $\\rho\\approx{rho_hb:.4f}$，\n"
            f"比 GD 小一个量级，所以只需 {recs['hb']['iters']} 步"
        ),
        "nes": (
            "$\\alpha=1/\\lambda_{\\max}$ 使 $1-\\alpha\\lambda_{\\max}=0$：陡方向分量一步就被消掉，\n"
            "所以看不到锯齿，点几乎沿着谷底推进；慢方向仍靠前瞻动量加速，\n"
            f"理论谱半径 $\\rho\\approx{rho_nes:.4f}$（GD 为 {rho_gd:.4f}），共 {recs['nes']['iters']} 步"
        ),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 11.8), constrained_layout=True)

    for ax, key in zip(axes.ravel(), ("bb1", "gd", "hb", "nes")):
        ax.contourf(GX, GY, Z, levels=levels, cmap="Blues_r", alpha=0.45, extend="both")
        ax.contour(GX, GY, Z, levels=levels, colors="0.45", linewidths=0.5, alpha=0.8)
        # 两条主轴
        for j, name in dirs.items():
            v = V[:, j] / np.linalg.norm(V[:, j])
            t = np.array([-2.0, 2.0]) * side
            ax.plot(prob.x_star[0] + t * v[0], prob.x_star[1] + t * v[1],
                    color="0.45", ls="--", lw=0.65, zorder=1)
            p = prob.x_star + (0.40 if j else -0.40) * side * v
            p = np.clip(p, [xlim[0] + 0.03 * side, ylim[0] + 0.03 * side],
                        [xlim[1] - 0.03 * side, ylim[1] - 0.03 * side])
            ax.annotate(name, xy=p, fontsize=7.4, color="0.30", ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65))

        r = recs[key]
        limit = r["iters"] if N_SHOW[key] is None else N_SHOW[key]
        shown = min(limit, r["iters"])
        xs = trajs[key][: shown + 1]
        c = palette[key]
        ax.plot(xs[:, 0], xs[:, 1], "-", color=c, lw=1.4, zorder=3)
        ax.plot(xs[:, 0], xs[:, 1], "o", color=c, ms=2.8, zorder=3)
        ax.plot(*X0, marker="s", ms=7, mfc="white", mec="black", mew=1.3, zorder=4)
        ax.plot(*prob.x_star, marker="*", ms=14, color="crimson", mec="white",
                mew=0.6, zorder=5)
        ax.annotate("$x_0$", xy=X0, xytext=(8, -10), textcoords="offset points",
                    fontsize=10, zorder=6)
        ax.annotate("$x^*$", xy=prob.x_star, xytext=(9, 6), textcoords="offset points",
                    fontsize=10, color="crimson", zorder=6)

        r = recs[key]
        shown = limit
        title_params = {
            "bb1": "$\\alpha_0=1/\\lambda_{\\max}$",
            "gd": "$\\alpha=\\alpha_\\star$，无动量",
            "hb": f"$\\alpha=\\alpha_\\star$，$\\beta={beta_val:.3f}$",
            "nes": f"$\\alpha=1/\\lambda_{{\\max}}$，$\\beta={beta_nes:.3f}$",
        }
        ax.set_title(
            f"{PANEL_LABEL[key]}  {title_params[key]}\n"
            f"{r['iters']} 步达到 $\\|x_k-x^*\\|_2\\leq10^{{-10}}\\|x_0-x^*\\|_2$",
            fontsize=10.0,
        )
        ax.set_xlabel("$x^{(1)}$")
        ax.set_ylabel("$x^{(2)}$")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.annotate(
            notes[key] + (f"\n（图中只画前 {shown} 步）" if shown < r["iters"] else ""),
            xy=(0.02, 0.02), xycoords="axes fraction", va="bottom", ha="left",
            fontsize=7.2, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.94),
        )

    fig.suptitle(
        f"第 7 题：四种方法的迭代点与 $f$ 的等值线（$n=2$, seed = {SEED}, "
        f"$\\kappa={kappa:.1f}$, $\\sqrt{{\\kappa}}\\approx{np.sqrt(kappa):.1f}$, 同一 $x_0$）",
        fontsize=13,
    )
    style.save(fig, "fig6_contour")

    save_json("q7_contour", {
        "n": N, "seed": SEED, "x0": X0.tolist(), "x_star": prob.x_star.tolist(),
        "lambda_min": lmin, "lambda_max": lmax, "kappa": kappa,
        "sqrt_kappa": float(np.sqrt(kappa)),
        "alpha_gd_hb": alpha, "beta_heavy_ball": beta_val, "rho_heavy_ball": rho_hb,
        "alpha_nesterov": alpha_nes, "beta_nesterov": beta_nes,
        "rtol": RTOL,
        "methods": recs,
    })

    print()
    print("判读要点：")
    print("  1. 等高线是长轴比 sqrt(kappa)≈9.4 的椭圆族：沿 q_max（陡方向）f 变得极快，"
          "沿 q_min（谷底方向）几乎不变——这就是条件数造成的困难。")
    print("  2. GD(α=α*) 的轨迹沿陡方向来回锯齿（因为 1-αλmax = -0.978 < 0 交替变号），"
          "同时沿谷底缓慢漂移；锯齿的幅度衰减率正好是 (kappa-1)/(kappa+1) = 0.9776。")
    print("  3. Heavy Ball 用动量把位移累加，第一步就冲到谷底另一侧（过冲），之后螺旋进入 x*；"
          "谱半径从 %.4f 降到 %.4f，步数相应从 %d 降到 %d。"
          % (rho_gd, rho_hb, recs["gd"]["iters"], recs["hb"]["iters"]))
    print("  4. Nesterov 取 alpha=1/lambda_max（梯度 Lipschitz 常数的倒数）：1-alpha*lambda_max=0，"
          "陡方向分量一步清零，所以轨迹几乎贴着谷底、看不出锯齿；"
          "慢方向仍需 %d 步，比 GD(alpha=alpha*) 的 %d 步少，但不如 HB。"
          % (recs["nes"]["iters"], recs["gd"]["iters"]))
    print("  5. BB1 用 s^T s/s^T y 自适应放大步长，n=2 时几乎“一眼看穿”："
          "%d 步就到 1e-10，轨迹是一条折线，只在谷底拐一下。" % recs["bb1"]["iters"])
    print()
    print("已写出 figures/fig6_contour.png 与 results/q7_contour.json")


if __name__ == "__main__":
    main()
