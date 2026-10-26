"""统一绘图样式：中文标签、配色、保存路径。

注意：本机只安装了 "AR PL UMing CN" 一款中文字体（Noto/文泉驿均缺失），
且图形在无显示环境下生成，因此强制使用 Agg 后端并把字体写成该字体族。
用 ``axes.unicode_minus = False`` 让负号以 ASCII 连字符渲染，
避免部分中文字体缺少 U+2212 造成方框。
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 必须在 import pyplot 之前设置

import matplotlib.pyplot as plt  # noqa: E402

# 该字体的 weight=normal 缺失，matplotlib 会打印一条无害提示，这里静音
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

CJK_FONT = "AR PL UMing CN"
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

# 各算法固定配色，保证全文一致
COLORS = {
    "GD": "#1f77b4",
    "SD": "#ff7f0e",
    "HB": "#2ca02c",
    "Nesterov": "#d62728",
    "BB1": "#9467bd",
    "BB2": "#8c564b",
    "CG": "#17becf",
}

MARKERS = {
    "GD": "o",
    "SD": "s",
    "HB": "^",
    "Nesterov": "v",
    "BB1": "D",
    "BB2": "P",
    "CG": "*",
}


def setup(scale: float = 1.0) -> None:
    """设置全局 rcParams。"""
    plt.rcParams.update(
        {
            "font.family": CJK_FONT,
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
            "font.size": 10.5 * scale,
            "axes.titlesize": 12.0 * scale,
            "axes.labelsize": 10.5 * scale,
            "legend.fontsize": 9.2 * scale,
            "xtick.labelsize": 9.2 * scale,
            "ytick.labelsize": 9.2 * scale,
            "figure.dpi": 120,
            "savefig.dpi": 170,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.7,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "0.75",
        }
    )


def save(fig, name: str, *, subdir: str = "") -> Path:
    """保存到 figures/ 下，返回路径。"""
    out = FIG_DIR / subdir if subdir else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def panel_label(ax, text: str) -> None:
    """在子图左上角标注 (a)/(b) 之类的编号。"""
    ax.annotate(
        text,
        xy=(0.015, 0.975),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.8", alpha=0.9),
    )


def positive(seq):
    """把序列裁剪到正数，便于 semilogy；非正值替换为极小值。"""
    import numpy as np

    v = np.asarray(seq, dtype=float)
    return np.clip(v, 1e-300, None)


def _fmt_pow10(value, _pos=None) -> str:
    """把 10^e 渲染成纯 ASCII 的 "1e-8"。"""
    import numpy as np

    if value <= 0:
        return ""
    e = float(np.log10(value))
    e_round = int(round(e))
    if abs(e - e_round) > 1e-8:
        return f"{value:g}"
    if e_round == 0:
        return "1"
    return f"1e{e_round}"


def log_axis(ax, which: str = "y") -> None:
    """对数轴刻度改为纯 ASCII 标签。

    matplotlib 默认用 mathtext 渲染 10^{-8}，其中指数负号是 U+2212，
    而 "AR PL UMing CN" 没有该字形，会打印大量缺字形警告并显示空框。
    这里换成 "1e-8" 形式的普通文本，彻底避开 mathtext。
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_locator(LogLocator(base=10.0, numticks=14))
    axis.set_major_formatter(FuncFormatter(_fmt_pow10))
    axis.set_minor_formatter(NullFormatter())
