"""实验脚本共用的输出工具：JSON 存档 + 控制台 Markdown 表格。

这样报告里的每一个数字都能由脚本重新生成，避免手工抄写误差。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


def _default(obj):
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    raise TypeError(f"无法序列化 {type(obj)!r}")


def save_json(name: str, payload: dict) -> Path:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    path = RES_DIR / f"{name}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_default),
        encoding="utf-8",
    )
    return path


def md_table(header: list[str], rows: list[list[str]], title: str = "") -> str:
    lines = []
    if title:
        lines.append(f"**{title}**\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def print_table(header, rows, title=""):
    print()
    print(md_table(header, rows, title))
    print()


def sci(x, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{digits}e}"


def history_row(h) -> list[str]:
    return [
        h.name,
        str(h.iters),
        "是" if h.converged else "否",
        sci(h.f[-1]),
        sci(h.err[-1]),
        sci(h.time[-1], 3) + " s",
    ]


HIST_HEADER = ["方法", "迭代步数", "达到容差", "f(x_K)", "‖x_K − x*‖₂", "CPU 时间"]
