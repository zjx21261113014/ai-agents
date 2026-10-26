"""数值优化课程实践：最小二乘二次问题的算法实现与分析工具。

模块划分
--------
problems : 问题构造、谱信息与解析最优解
methods  : 梯度下降、最速下降、Heavy Ball、Nesterov、BB1/BB2、共轭梯度
analysis : 收敛速率、收敛阶的数值估计
style    : 统一的中文绘图样式

所有随机量均由固定种子生成，脚本可重复还原报告中的全部数值结果。
"""

from .problems import LeastSquaresProblem, make_problem
from .methods import (
    History,
    bb,
    conjugate_gradient,
    gradient_descent,
    heavy_ball,
    nesterov,
    steepest_descent,
)
from .analysis import (
    asymptotic_ratio,
    coc,
    iterations_to,
    loglog_slope,
    momentum_spectral_rate,
    optimal_momentum_beta,
    semilog_slope,
)

__all__ = [
    "LeastSquaresProblem",
    "make_problem",
    "History",
    "gradient_descent",
    "steepest_descent",
    "heavy_ball",
    "nesterov",
    "bb",
    "conjugate_gradient",
    "asymptotic_ratio",
    "semilog_slope",
    "loglog_slope",
    "coc",
    "iterations_to",
    "momentum_spectral_rate",
    "optimal_momentum_beta",
]

