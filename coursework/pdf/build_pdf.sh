#!/usr/bin/env bash
# 把 REPORT.md 编译为 PDF（pandoc + xelatex，中文用 AR PL UMing CN）
# 用法：bash build_pdf.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PANDOC="$(uv run --quiet --with pypandoc-binary python -c \
  'import pypandoc; print(pypandoc.get_pandoc_path())')"
echo "pandoc: $PANDOC ($("$PANDOC" --version | head -1))"

# 预处理：把 6 面板宽图（图 7）改为横向整页，否则缩到版心宽度后字太小读不清
uv run --quiet python - <<'PY'
import pathlib, re
src = pathlib.Path("REPORT.md").read_text(encoding="utf-8")
pat = re.compile(r"^!\[([^\]]+)\]\((figures/fig7_cg\.png)\)$", re.M)
new = (
    "```{=latex}\n"
    "\\begin{sidewaysfigure}\n"
    "\\centering\n"
    "\\includegraphics[width=\\linewidth]{figures/fig7_cg.png}\n"
    "\\caption{图 7}\n"
    "\\end{sidewaysfigure}\n"
    "```"
)
out, n = pat.subn(lambda m: new, src)
assert n == 1, f"expect exactly 1 wide figure, got {n}"

# 其余图片一律改为"非浮动"（center + \captionof），保证图与其结论顺序不被 LaTeX 打乱
fig = re.compile(r"^!\[([^\]]+)\]\((figures/[^)]+)\)$", re.M)


def _fig(m):
    cap, path = m.group(1), m.group(2)
    return (
        "```{=latex}\n"
        "\\begin{center}\n"
        "\\begin{minipage}{\\linewidth}\n"
        "\\centering\n"
        f"\\includegraphics[width=\\linewidth]{{{path}}}\n"
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
    "\n```{=latex}\n\\clearpage\n```\n\n## 附录 A 源码清单",
    1,
)

# "图 N 结论" 标签段落前留够行数，避免标签孤立在页底
lab = re.compile(r"^(\*\*图 \d+ 结论\*\*)", re.M)
out, m = lab.subn(
    lambda x: "```{=latex}\n\\needspace{8\\baselineskip}\n```\n\n" + x.group(1),
    out,
)
print(f"preprocessed: wide figure -> sidewaysfigure, {m} conclusion labels guarded")

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
