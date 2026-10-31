#!/usr/bin/env bash
# 把 REPORT.md 编译为 PDF（pandoc + xelatex，中文用 AR PL UMing CN）
# 用法：bash build_pdf.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PANDOC="$(uv run --quiet --with pypandoc-binary python -c \
  'import pypandoc; print(pypandoc.get_pandoc_path())')"
echo "pandoc: $PANDOC ($("$PANDOC" --version | head -1))"

# 预处理：把 6 面板宽图（图 7）改为横向整页。版心宽度下 6 个面板的字会小到读不清，
# 所以给它单独一页；位置留在 7.3 的表之后、图 7 结论之前。
uv run --quiet python - <<'PY'
import pathlib, re
src = pathlib.Path("REPORT.md").read_text(encoding="utf-8")
pat = re.compile(r"^!\[([^\]]+)\]\(figures/fig7_cg\.png\)\n", re.M)

# 7.3 的第一张表（5 行）正好压在页底时，LaTeX 会把表头单独重复到下一页，
# 变成一个只有表头的孤儿行。单独给它留够 14 行，让它整张一起走。
guard = "```{=latex}\n\\needspace{14\\baselineskip}\n```\n\n"
assert src.count("**n = 50**：") == 1, "cannot locate 7.3 first table"
src = src.replace("**n = 50**：", guard + "**n = 50**：", 1)

plate = (
    "```{=latex}\n"
    "\\begin{sidewaysfigure}\n"
    "\\centering\n"
    "\\includegraphics[width=\\linewidth]{figures/fig7_cg.png}\n"
    "\\caption{%s}\n"
    "\\end{sidewaysfigure}\n"
    "\\clearpage\n"
    "```\n"
)
out, n = pat.subn(lambda m: plate % m.group(1), src)
assert n == 1, f"expect exactly 1 wide figure, got {n}"
print("preprocessed: fig 7 -> full-page landscape plate, kept before its notes")

# 其余图片一律改为"非浮动"（center + \captionof），保证图与其结论顺序不被 LaTeX 打乱
fig = re.compile(r"^!\[([^\]]+)\]\((figures/[^)]+)\)$", re.M)


def _fig(m):
    cap, path = m.group(1), m.group(2)
    # 图 6 是 2×2 等值线，且接近正方形：全宽时高 15.9cm，整块顶走会留下半页空白
    w = "0.86\\linewidth" if "fig6" in path else "\\linewidth"
    return (
        "```{=latex}\n"
        "\\begin{center}\n"
        "\\begin{minipage}{\\linewidth}\n"
        "\\centering\n"
        f"\\includegraphics[width={w}]{{{path}}}\n"
        f"\\captionof{{figure}}{{{cap}}}\n"
        "\\end{minipage}\n"
        "\\end{center}\n"
        "```"
    )


out, k = fig.subn(_fig, out)
print(f"preprocessed: 1 wide figure -> sidewaysfigure, {k} figures -> non-float")

# 附录从新页开始，避免源码清单表被拆成"表头 + 3 行"
out = out.replace(
    "\n## 附录 A 源码清单",
    "\n```{=latex}\n\\needspace{24\\baselineskip}\n```\n\n## 附录 A 源码清单",
    1,
)

# 注：表格不做全局 needspace（会让多页出现大片留白），只在必要时逐处处理

# 表 6a 的题注正好压在页底时，表体整块会跑到下一页，留下一个只有题注的孤儿行
out = out.replace(
    "\n**表 6a：收敛速率**",
    "\n```{=latex}\n\\needspace{9\\baselineskip}\n```\n\n**表 6a：收敛速率**",
    1,
)

# 表标题居中：正文有 \parindent，表题不该跟着缩进两字
# 用 {\centering ...} 而不是 center 环境，避免额外上下梢
cap = re.compile(r"^\*\*(表 [^*]+)\*\*\s*$", re.M)
out, t = cap.subn(
    lambda x: (
        "```{=latex}\n"
        "\\begin{center}\n"
        f"\\textbf{{{x.group(1)}}}\n"
        "\\end{center}\n"
        "```"
    ),
    out,
)
print(f"preprocessed: {t} table captions centred")

pathlib.Path("pdf/_REPORT_pdf.md").write_text(out, encoding="utf-8")
PY

"$PANDOC" pdf/_REPORT_pdf.md \
  -o REPORT.pdf \
  --pdf-engine=xelatex \
  --resource-path=. \
  --toc --toc-depth=2 \
  -V documentclass=article \
  -V fontsize=10pt \
  -V toc-title=目录 \
  -V mainfont="DejaVu Serif" \
  -V monofont="DejaVu Sans Mono" \
  -V CJKmainfont="AR PL UMing CN" \
  -V CJKsansfont="Droid Sans Fallback" \
  -V geometry:a4paper \
  -V geometry:margin=2.1cm \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolor=blue \
  -V toccolor=black \
  -H pdf/header.tex

echo "OK -> $(pwd)/REPORT.pdf ($(du -h REPORT.pdf | cut -f1))"
