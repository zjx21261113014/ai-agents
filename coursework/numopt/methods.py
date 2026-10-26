"""求解 min 1/2||Ax-b||^2 的六类迭代法。

统一接口
--------
每个算法返回 :class:`History`，其中按迭代步记录

* ``f``    : 目标函数值 f(x_k)（用 f = f* + 1/2 g^T (x - x*) 精确计算，
             不额外做矩阵乘法）
* ``err``  : 误差 2-范数 ||x_k - x*||_2
* ``time`` : 从开始计时到该步的 CPU 时间（``time.process_time``）

约定：``x_k`` 一律指算法对外输出的迭代点。对 Nesterov 加速法，
梯度在查询点 y_k 处计算，但记录的仍是输出点 x_k 的 f 与误差。

停止准则：||x_k - x*|| <= max(atol, rtol * ||x_0 - x*||)，或达到 ``max_iter``。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .problems import LeastSquaresProblem

__all__ = [
    "History",
    "gradient_descent",
    "steepest_descent",
    "heavy_ball",
    "nesterov",
    "bb",
    "conjugate_gradient",
]


@dataclass
class History:
    """一次算法运行的完整轨迹。"""

    name: str
    f: np.ndarray
    err: np.ndarray
    time: np.ndarray
    converged: bool
    info: dict = field(default_factory=dict)

    @property
    def iters(self) -> int:
        """实际迭代步数（不含初始点）。"""
        return int(self.err.size - 1)

    def relative_err(self) -> np.ndarray:
        return self.err / self.err[0]

    def __repr__(self) -> str:  # pragma: no cover - 便于交互调试
        return (
            f"History({self.name!r}, iters={self.iters}, "
            f"final_err={self.err[-1]:.3e}, converged={self.converged})"
        )


class _Trace:
    """逐步记录 f / err / CPU 时间，并负责停止判定。"""

    __slots__ = ("prob", "e0", "f", "err", "t", "t0")

    def __init__(self, prob: LeastSquaresProblem, x0: np.ndarray) -> None:
        self.prob = prob
        self.e0 = max(prob.err(x0), np.finfo(float).tiny)
        self.f: list[float] = []
        self.err: list[float] = []
        self.t: list[float] = []
        self.t0 = time.process_time()

    def push(self, x: np.ndarray, g: np.ndarray) -> None:
        self.f.append(self.prob.f_from_grad(x, g))
        self.err.append(float(np.linalg.norm(x - self.prob.x_star)))
        self.t.append(time.process_time() - self.t0)

    def done(self, rtol: float, atol: float) -> bool:
        return self.err[-1] <= max(atol, rtol * self.e0)

    def finish(self, name: str, converged: bool, **info) -> History:
        return History(
            name=name,
            f=np.asarray(self.f),
            err=np.asarray(self.err),
            time=np.asarray(self.t),
            converged=converged,
            info=info,
        )


# ====================================================================== 固定步长法
def gradient_descent(
    prob: LeastSquaresProblem,
    x0: np.ndarray,
    alpha: float,
    *,
    max_iter: int = 1_000_000,
    rtol: float = 1e-10,
    atol: float = 0.0,
) -> History:
    """固定步长梯度下降  x_{k+1} = x_k - alpha * grad f(x_k)。"""
    x = np.asarray(x0, dtype=float).copy()
    tr = _Trace(prob, x0)
    converged = False
    for _ in range(max_iter + 1):
        g = prob.grad(x)
        tr.push(x, g)
        if tr.done(rtol, atol):
            converged = True
            break
        x = x - alpha * g
    return tr.finish(f"GD(α={alpha:.4e})", converged, alpha=alpha, max_iter=max_iter)


def steepest_descent(
    prob: LeastSquaresProblem,
    x0: np.ndarray,
    *,
    max_iter: int = 1_000_000,
    rtol: float = 1e-10,
    atol: float = 0.0,
) -> History:
    """最速下降法（精确线搜索）。

    对二次函数，精确线搜索步长为
        alpha_k = (g_k^T g_k) / (g_k^T Q g_k)。
    """
    x = np.asarray(x0, dtype=float).copy()
    tr = _Trace(prob, x0)
    converged = False
    for _ in range(max_iter + 1):
        g = prob.grad(x)
        tr.push(x, g)
        if tr.done(rtol, atol):
            converged = True
            break
        Qg = prob.Q @ g
        alpha = float(g @ g) / float(g @ Qg)
        x = x - alpha * g
    return tr.finish("最速下降（精确线搜索）", converged, max_iter=max_iter)


# ====================================================================== 动量类方法
def heavy_ball(
    prob: LeastSquaresProblem,
    x0: np.ndarray,
    alpha: float,
    beta: float,
    *,
    max_iter: int = 1_000_000,
    rtol: float = 1e-10,
    atol: float = 0.0,
) -> History:
    """Heavy Ball（Polyak 重球法）

        x_{k+1} = x_k - alpha * grad f(x_k) + beta * (x_k - x_{k-1})。

    初始取 x_{-1} = x_0，即第一步退化为普通梯度下降。
    """
    x_prev = np.asarray(x0, dtype=float).copy()
    x = x_prev.copy()
    tr = _Trace(prob, x0)
    converged = False
    for _ in range(max_iter + 1):
        g = prob.grad(x)
        tr.push(x, g)
        if tr.done(rtol, atol):
            converged = True
            break
        x_new = x - alpha * g + beta * (x - x_prev)
        x_prev, x = x, x_new
    return tr.finish(
        f"Heavy Ball(α={alpha:.4e}, β={beta:.4f})",
        converged,
        alpha=alpha,
        beta=beta,
        max_iter=max_iter,
    )


def nesterov(
    prob: LeastSquaresProblem,
    x0: np.ndarray,
    alpha: float,
    *,
    max_iter: int = 1_000_000,
    rtol: float = 1e-10,
    atol: float = 0.0,
    beta_const: float | None = None,
) -> History:
    """Nesterov 加速梯度法

        y_k     = x_k + beta_k (x_k - x_{k-1})
        x_{k+1} = y_k - alpha * grad f(y_k)

    momentum
        * ``beta_const is None``（默认）：``beta_k = (k - 1) / (k + 2)``，
          即 Nesterov(1983)/FISTA 形式，对一般光滑凸问题达到最优的 O(1/k^2)；
        * 给定 ``beta_const``：使用常数动量。对 mu-强凸、L-光滑问题，
          取 ``beta = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)``（kappa = L/mu）
          可恢复 O((1 - 1/sqrt(kappa))^k) 的线性收敛。
    """
    x_prev = np.asarray(x0, dtype=float).copy()
    x = x_prev.copy()
    tr = _Trace(prob, x0)
    converged = False
    for k in range(max_iter + 1):
        if beta_const is None:
            beta = max((k - 1.0) / (k + 2.0), 0.0)
        else:
            beta = float(beta_const)
        y = x + beta * (x - x_prev)
        gy = prob.grad(y)
        x_new = y - alpha * gy
        g_new = prob.grad(x_new)  # 只为记录 f(x_new)，保持输出点与梯度一致
        tr.push(x_new, g_new)
        if tr.done(rtol, atol):
            converged = True
            break
        x_prev, x = x, x_new
    return tr.finish(
        f"Nesterov(α={alpha:.4e})",
        converged,
        alpha=alpha,
        beta_const=beta_const,
        max_iter=max_iter,
    )


# ====================================================================== Barzilai-Borwein
def bb(
    prob: LeastSquaresProblem,
    x0: np.ndarray,
    variant: int = 1,
    *,
    alpha0: float | None = None,
    max_iter: int = 1_000_000,
    rtol: float = 1e-10,
    atol: float = 0.0,
) -> History:
    """Barzilai-Borwein 步长法。

    记 s_k = x_k - x_{k-1}, y_k = grad f(x_k) - grad f(x_{k-1})，则

        BB1: alpha_k = (s_k^T s_k) / (s_k^T y_k)
        BB2: alpha_k = (s_k^T y_k) / (y_k^T y_k)

    初始步长取 ``alpha0 = 1 / lam_max``（即 L/2），首步与固定步长 GD 相同。
    对二次函数 s^T y = s^T Q s > 0，故不会出现非正步长；仍保留保护分支。
    """
    if variant not in (1, 2):
        raise ValueError("variant 只能取 1 或 2")
    if alpha0 is None:
        alpha0 = 1.0 / prob.lam_max

    x = np.asarray(x0, dtype=float).copy()
    g = prob.grad(x)
    alpha = float(alpha0)

    tr = _Trace(prob, x0)
    tr.push(x, g)
    converged = tr.done(rtol, atol)
    alphas: list[float] = [alpha]

    for _ in range(max_iter):
        if converged:
            break
        x_prev, g_prev = x, g
        x = x - alpha * g
        g = prob.grad(x)
        tr.push(x, g)
        if tr.done(rtol, atol):
            converged = True
            break

        s = x - x_prev
        y = g - g_prev
        sy = float(s @ y)
        if sy > 0.0:
            if variant == 1:
                alpha = float(s @ s) / sy
            else:
                alpha = sy / float(y @ y)
            alphas.append(alpha)
        # sy <= 0 时保持上一步步长（理论上不会发生，留作数值保护）

    return tr.finish(
        f"BB{variant}(α0={alpha0:.4e})",
        converged,
        variant=variant,
        alpha0=float(alpha0),
        alphas=np.asarray(alphas),
        max_iter=max_iter,
    )


# ====================================================================== 共轭梯度
def conjugate_gradient(
    prob: LeastSquaresProblem,
    x0: np.ndarray,
    *,
    max_iter: int | None = None,
    rtol: float = 1e-10,
    atol: float = 0.0,
) -> History:
    """求解正规方程 Q x = c 的共轭梯度法（Hestenes-Stiefel / Fletcher-Reeves 形式）。

        r_0 = c - Q x_0 = -grad f(x_0),  p_0 = r_0
        alpha_k = (r_k^T r_k) / (p_k^T Q p_k)
        x_{k+1} = x_k + alpha_k p_k
        r_{k+1} = r_k - alpha_k Q p_k
        beta_k  = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
        p_{k+1} = r_{k+1} + beta_k p_k

    无舍入误差时最多 n 步得到精确解；``max_iter`` 默认取 ``3n`` 以观察舍入影响。
    """
    if max_iter is None:
        max_iter = 3 * prob.n

    x = np.asarray(x0, dtype=float).copy()
    g = prob.grad(x)
    r = -g.copy()  # r = c - Qx
    p = r.copy()
    rr = float(r @ r)

    tr = _Trace(prob, x0)
    tr.push(x, g)
    converged = tr.done(rtol, atol)

    for _ in range(max_iter):
        if converged:
            break
        Qp = prob.Q @ p
        alpha = rr / float(p @ Qp)
        x = x + alpha * p
        g = g + alpha * Qp   # 因为 g = Qx - c
        r = r - alpha * Qp
        tr.push(x, g)
        if tr.done(rtol, atol):
            converged = True
            break
        rr_new = float(r @ r)
        beta = rr_new / rr
        rr = rr_new
        p = r + beta * p

    return tr.finish("共轭梯度 (CG)", converged, max_iter=max_iter)
