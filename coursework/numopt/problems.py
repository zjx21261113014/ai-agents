"""最小二乘二次优化问题的构造与解析信息。

问题
----
    min f(x) = 1/2 * ||A x - b||_2^2 ,  A in R^{n x n},  b in R^n

其梯度与 Hessian 为

    grad f(x) = A^T (A x - b) = Q x - c ,
    Hessian   = Q = A^T A ,   c = A^T b ,

因此 f 是二次函数，并且（几乎必然）Q 正定，问题为无约束严格凸二次规划。

数值约定
--------
* A、b 由 ``numpy.random.default_rng(seed).standard_normal`` 生成，
  与 Matlab 的 ``randn(n, n)`` / ``randn(n)`` 同分布（标准正态），
  固定 seed 保证报告中的每个数字都可复现。
* 记 ``lam_max`` / ``lam_min`` 为 Q 的最大 / 最小特征值。
  ``L = 2 / lam_max`` 是固定步长梯度下降收敛区间 (0, 2/lam_max) 的上端点，
  与题目中 "收敛区间 (0, L)" 的记号一致；注意它与梯度 Lipschitz 常数
  ``lam_max`` 本身不同，报告正文对此做了区分。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LeastSquaresProblem", "make_problem"]


@dataclass
class LeastSquaresProblem:
    """保存 A、b 以及所有派生量（谱、最优解、最优值）。"""

    A: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        A = np.asarray(self.A, dtype=float)
        b = np.asarray(self.b, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A 必须是方阵")
        if b.shape != (A.shape[0],):
            raise ValueError("b 的长度必须与 A 的阶数一致")

        self.A = A
        self.b = b
        self.n = A.shape[0]

        # 问题数据（只依赖 A、b）
        self.Q = A.T @ A
        self.c = A.T @ b
        self.half_b2 = 0.5 * float(b @ b)

        # 解析最优解：解正规方程 Q x = c
        self.x_star = np.linalg.solve(self.Q, self.c)
        self.f_star = 0.5 * float(np.sum((A @ self.x_star - b) ** 2))

        # 谱信息（Q 对称，用 eigvalsh 更稳更快）
        eig = np.linalg.eigvalsh(self.Q)
        self.eigvals = eig
        self.lam_min = float(eig[0])
        self.lam_max = float(eig[-1])
        self.kappa = self.lam_max / self.lam_min

        # 步长相关的理论量
        self.L = 2.0 / self.lam_max                 # 收敛区间 (0, L) 的上端点
        self.alpha_opt = 2.0 / (self.lam_min + self.lam_max)  # 最优固定步长
        self.rho_opt = (self.kappa - 1.0) / (self.kappa + 1.0)

    # ------------------------------------------------------------------ 基本运算
    def f(self, x: np.ndarray) -> float:
        """目标函数值 f(x)。"""
        r = self.A @ x - self.b
        return 0.5 * float(r @ r)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """梯度 grad f(x) = Q x - c。"""
        return self.Q @ x - self.c

    def hess(self, x: np.ndarray | None = None) -> np.ndarray:
        """Hessian 为常数矩阵 Q。"""
        return self.Q

    def f_from_grad(self, x: np.ndarray, g: np.ndarray) -> float:
        """用已知梯度计算 f(x)。

        利用恒等式 f(x) - f* = 1/2 * (x - x*)^T Q (x - x*)
                            = 1/2 * g(x)^T (x - x*)，
        可以在已有梯度时 O(n) 得到精确的目标值，省去一次矩阵乘法。
        """
        return self.f_star + 0.5 * float(g @ (x - self.x_star))

    def err(self, x: np.ndarray) -> float:
        """误差的 2-范数 ||x - x*||_2。"""
        return float(np.linalg.norm(x - self.x_star))

    # ------------------------------------------------------------------ 检验
    def verify_spd(self) -> dict:
        """数值检验 Q 是否对称正定，并交叉验证最优解。"""
        Q = self.Q
        asym = float(np.linalg.norm(Q - Q.T) / np.linalg.norm(Q))
        try:
            np.linalg.cholesky(Q)
            cholesky_ok = True
        except np.linalg.LinAlgError:
            cholesky_ok = False

        # 三种解正规方程的方式互相校验
        x_solve = np.linalg.solve(Q, self.c)
        x_lstsq = np.linalg.lstsq(Q, self.c, rcond=None)[0]
        x_inv = np.linalg.inv(Q) @ self.c

        scale = max(np.linalg.norm(self.x_star), 1.0)
        return {
            "n": self.n,
            "Q_symmetry_rel_error": asym,
            "cholesky_success": cholesky_ok,
            "all_eigenvalues_positive": bool(np.all(self.eigvals > 0)),
            "lambda_min": self.lam_min,
            "lambda_max": self.lam_max,
            "condition_number_Q": self.kappa,
            "condition_number_A": float(np.linalg.cond(self.A)),
            "A_frobenius": float(np.linalg.norm(self.A)),
            "x_star_norm": float(np.linalg.norm(self.x_star)),
            "f_star": self.f_star,
            "grad_norm_at_x_star": float(np.linalg.norm(self.grad(self.x_star))),
            "rel_diff_solve_vs_lstsq": float(np.linalg.norm(x_solve - x_lstsq) / scale),
            "rel_diff_solve_vs_inv": float(np.linalg.norm(x_solve - x_inv) / scale),
            # 二次型恒等式的最优性验证：x* 处梯度为 0，且 f* 取到最小
            "f_star_identity_rel_error": abs(
                self.f_star - (self.half_b2 - 0.5 * float(self.c @ self.x_star))
            )
            / max(abs(self.f_star), 1.0),
        }

    def rayleigh_range(self, n_probe: int = 512, seed: int = 12345) -> dict:
        """用随机方向上的 Rayleigh 商数值估计 Q 的谱范围，作为独立佐证。"""
        rng = np.random.default_rng(seed)
        V = rng.standard_normal((self.n, n_probe))
        V /= np.linalg.norm(V, axis=0, keepdims=True)
        rq = np.einsum("ij,ij->j", V, self.Q @ V)
        return {
            "rayleigh_min_probe": float(rq.min()),
            "rayleigh_max_probe": float(rq.max()),
            "lambda_min_true": self.lam_min,
            "lambda_max_true": self.lam_max,
        }

    # ------------------------------------------------------------------ 便捷步长
    def step_sizes(self) -> dict:
        """题目第 1.3 问要求的四个步长以及最优步长。"""
        L = self.L
        return {
            "L/3": L / 3.0,
            "L/2": L / 2.0,
            "3L/4": 3.0 * L / 4.0,
            "L": L,
            "alpha_opt": self.alpha_opt,
        }

    def predicted_rate(self, alpha: float) -> float:
        """固定步长 alpha 下 GD 的渐近收敛因子

            rho(alpha) = max_i |1 - alpha * lam_i|
                       = max(|1 - alpha*lam_min|, |1 - alpha*lam_max|)。
        """
        return max(abs(1.0 - alpha * self.lam_min), abs(1.0 - alpha * self.lam_max))


def make_problem(n: int, seed: int = 0) -> LeastSquaresProblem:
    """按题目要求生成 A = randn(n, n)、b = randn(n)。"""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    b = rng.standard_normal(n)
    return LeastSquaresProblem(A, b)
