"""收敛速率与收敛阶的数值估计工具。

术语
----
R-线性收敛：||e_k|| <= C rho^k，在 semilogy 图上呈直线，斜率为 ln(rho)。
Q-线性收敛：||e_{k+1}|| / ||e_k|| -> rho in (0, 1)。
Q-阶 p    ：||e_{k+1}|| / ||e_k||^p -> C in (0, inf)。

因此"用画图观察收敛阶"可以这样做：

1. semilogy(e_k vs k) 是否为直线  -> 判别是否 R-线性；
2. 直线斜率 exp(slope)            -> 渐近收敛因子 rho；
3. 比值 rho_k = e_{k+1}/e_k 是否趋于常数 -> 判别是否 Q-线性；
4. 计算阶 p_k = ln(e_{k+1}/e_k) / ln(e_k/e_{k-1}) 是否趋于 1 -> 线性阶；
5. loglog(e_k vs k) 的斜率        -> 若为有限负数 -p，则误差按 k^{-p} 代数衰减，
                                     即 f 的 O(1/k^2) 型加速收敛。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "iterations_to",
    "asymptotic_ratio",
    "semilog_slope",
    "loglog_slope",
    "coc",
    "exponential_rate",
    "fit_line",
    "momentum_spectral_rate",
    "optimal_momentum_beta",
]


def _clean(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return np.where(v > 0, v, np.nan)


def iterations_to(seq: np.ndarray, tol: float, *, relative: bool = True) -> int | None:
    """返回首次满足误差阈值所需的迭代步数（下标），未达到则返回 None。"""
    v = _clean(seq)
    target = tol * v[0] if relative else tol
    idx = np.flatnonzero(np.nan_to_num(v, nan=np.inf) <= target)
    return int(idx[0]) if idx.size else None


def fit_line(x: np.ndarray, y: np.ndarray, start_frac: float = 0.0,
             end_frac: float = 1.0) -> dict:
    """在给定窗口内做最小二乘直线拟合，返回斜率、截距、R^2。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"),
                "n_points": int(x.size)}

    lo = int(np.floor(start_frac * (x.size - 1)))
    hi = int(np.ceil(end_frac * (x.size - 1)))
    xs, ys = x[lo:hi + 1], y[lo:hi + 1]
    if xs.size < 2:
        xs, ys = x, y

    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "n_points": int(xs.size),
    }


def semilog_slope(k: np.ndarray, seq: np.ndarray, start_frac: float = 0.0,
                  end_frac: float = 1.0) -> dict:
    """ln(e_k) 关于 k 的拟合，斜率即 ln(rho)，同时给出 rho = exp(斜率)。"""
    v = _clean(seq)
    res = fit_line(k, np.log(v), start_frac, end_frac)
    res["rate"] = float(np.exp(res["slope"])) if np.isfinite(res["slope"]) else float("nan")
    return res


def loglog_slope(k: np.ndarray, seq: np.ndarray, start_frac: float = 0.0,
                 end_frac: float = 1.0) -> dict:
    """ln(e_k) 关于 ln(k) 的拟合；斜率 -p 表示 e_k ~ C k^{-p}。"""
    k = np.asarray(k, dtype=float)
    v = _clean(seq)
    mask = k > 0
    res = fit_line(np.log(k[mask]), np.log(v[mask]), start_frac, end_frac)
    res["algebraic_order"] = float(-res["slope"]) if np.isfinite(res["slope"]) else float("nan")
    return res


def asymptotic_ratio(seq: np.ndarray, tail_frac: float = 0.1) -> float:
    """尾部 e_{k+1}/e_k 的中位数，用于估计渐近收缩因子。"""
    v = _clean(seq)
    ratios = v[1:] / v[:-1]
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return float("nan")
    n_tail = max(1, int(np.ceil(tail_frac * ratios.size)))
    return float(np.median(ratios[-n_tail:]))


def coc(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算型收敛阶 (computational order of convergence)。

        p_k = ln(e_{k+1}/e_k) / ln(e_k/e_{k-1}),   k = 1, ..., K-2

    线性收敛时 p_k -> 1；超线性时 p_k -> 0 或随 k 增大而增大。
    仅对相邻三步都下降的片段给出有限值，其余为 NaN。
    """
    v = _clean(seq)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_curr = v[1:-1] / v[:-2]   # e_k / e_{k-1}
        r_next = v[2:] / v[1:-1]    # e_{k+1} / e_k
        p = np.log(r_next) / np.log(r_curr)
    p = np.where(np.isfinite(p), p, np.nan)
    k = np.arange(1, p.size + 1, dtype=float)
    return k, p


def exponential_rate(seq: np.ndarray, start_frac: float = 0.2,
                     end_frac: float = 0.9) -> float:
    """用尾部 semilog 斜率估计收敛因子 rho。"""
    v = np.clip(_clean(seq), 1e-300, None)
    k = np.arange(v.size, dtype=float)
    return semilog_slope(k, v, start_frac, end_frac)["rate"]


def momentum_spectral_rate(eigvals: np.ndarray, alpha: float, beta: float,
                           scheme: str = "heavy_ball") -> float:
    """动量类方法在二次问题上的精确渐近收敛因子（谱半径）。

    二次问题上两种动量法的误差分量都满足二阶线性递推

        Heavy Ball:   x_{k+1} = x_k - α Q(x_k - x*) + β(x_k - x_{k-1})
                      => t^2 - (1 + β - αλ) t + β = 0
        Nesterov:     y_k = x_k + β(x_k - x_{k-1}),  x_{k+1} = y_k - α Q(y_k - x*)
                      => t^2 - (1 + β)(1 - αλ) t + β(1 - αλ) = 0

    对每个特征值 λ 取两个根的模的最大值，再在整条谱上取最大，即得
    误差的二范数每步渐近收缩因子。它比"两极端特征值"的界更精确
    （当 λ 不等距分布时会略大于 (√κ-1)/(√κ+1)），可与实测 rho 直接对照。
    """
    lam = np.asarray(eigvals, dtype=float)
    if scheme == "heavy_ball":
        b = 1.0 + beta - alpha * lam
        const = np.full_like(lam, beta)
    elif scheme == "nesterov":
        s = 1.0 - alpha * lam
        b = (1.0 + beta) * s
        const = beta * s
    else:
        raise ValueError("scheme 只能取 'heavy_ball' 或 'nesterov'")

    disc = b * b - 4.0 * const
    # 复根：两共轭复根的模都是 sqrt(const)；实根：取绝对值较大的那个
    with np.errstate(invalid="ignore"):
        real_branch = 0.5 * (np.abs(b) + np.sqrt(np.maximum(disc, 0.0)))
    rho = np.where(disc >= 0.0, real_branch, np.sqrt(np.maximum(const, 0.0)))
    return float(np.max(rho))


def optimal_momentum_beta(eigvals: np.ndarray, alpha: float,
                          scheme: str = "heavy_ball", *,
                          n_grid: int = 2001, n_refine: int = 60) -> dict:
    """固定步长 alpha 时，数值搜索使谱半径最小的动量系数 beta。

    为什么需要它：经典公式 β = ((√κ-1)/(√κ+1))^2 是针对 Heavy Ball 的
    *最优* 步长 α = 4/(√λmax+√λmin)^2 推导的；而第 5 题要求三种方法
    用 **同一个** α，此时最优 β 会随 α 改变。为了做公平对比，就在
    给定 α 下直接对 β 做一维搜索（先网格、再逐级细化）。

    返回 ``{"beta", "rho", "alpha", "scheme"}``；``rho >= 1`` 表示该步长下
    无论怎么选 β 都不收敛。
    """
    def rho_of(beta: float) -> float:
        return momentum_spectral_rate(eigvals, alpha, beta, scheme=scheme)

    hi = 1.0 - 1e-9
    grid = np.linspace(0.0, hi, n_grid)
    vals = np.array([rho_of(b) for b in grid])
    j = int(np.argmin(vals))
    best_beta, best_rho = float(grid[j]), float(vals[j])

    lo = float(grid[max(j - 1, 0)])
    hi = float(grid[min(j + 1, grid.size - 1)])
    for _ in range(n_refine):  # 逐级缩小区间，精度提高到 ~1e-15
        cand = np.linspace(lo, hi, 9)
        vals = np.array([rho_of(b) for b in cand])
        j = int(np.argmin(vals))
        if float(vals[j]) < best_rho:
            best_beta, best_rho = float(cand[j]), float(vals[j])
        lo, hi = float(cand[max(j - 1, 0)]), float(cand[min(j + 1, cand.size - 1)])
    return {"beta": best_beta, "rho": best_rho, "alpha": float(alpha), "scheme": scheme}

