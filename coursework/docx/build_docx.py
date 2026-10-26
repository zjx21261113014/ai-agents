"""把 REPORT.md 编译为可提交的 Word 版报告（REPORT.docx）。

流程：
  1. 预处理 Markdown —— 给 7 张图统一加 {width=16cm}（pandoc 默认按 96 DPI 换算会溢出页面）
  2. pandoc 转换为 docx（公式转成 Word 原生 OMML，不是图片也不是 LaTeX 源码）
  3. python-docx 后处理 —— 中文字体、页边距、图与题注同页、表格跨页重复表头

用法：uv run --with pypandoc-binary --with python-docx python docx/build_docx.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pypandoc
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "REPORT.md"
TMP = ROOT / "docx" / "_REPORT_docx.md"
OUT = ROOT / "REPORT.docx"
REF = ROOT / "docx" / "reference.docx"

# A4 宽 21cm − 两侧 2.1cm 边距 = 16.8cm。按各图宽高比定宽：
# 16cm 宽时 fig2–fig5 高约 10.5cm、fig7 高 9.5cm，都放得下；
# fig6 是 1:1 方图，16cm 宽会得到 16cm 高的巨块，压到 12.5cm。
FIG_WIDTHS = {
    1: "16cm",
    2: "15.5cm",
    3: "15.5cm",
    4: "15.5cm",
    5: "15.5cm",
    6: "12.5cm",
    7: "16cm",
}
BODY_CN, BODY_EN = "宋体", "Times New Roman"
HEAD_CN, HEAD_EN = "黑体", "Times New Roman"
MONO = "Consolas"


# ---------------------------------------------------------------- 1. 预处理
def normalize_figures(text: str) -> str:
    """`![图 N](path)` -> `![图 N](path){width=...}`，幂等。"""
    pat = re.compile(r"!\[图 (\d)\]\((figures/[^)]+\.png)\)(?!\{)")

    def repl(m: re.Match[str]) -> str:
        width = FIG_WIDTHS.get(int(m.group(1)), "16cm")
        return f"![图 {m.group(1)}]({m.group(2)}){{width={width}}}"

    text, n = pat.subn(repl, text)
    print(f"preprocessed: {n} figures given explicit widths")
    return text


def split_blockquote_lines(text: str) -> str:
    """Markdown 中连续的 `>` 行是同一段落，Word 里会挤成一大段。

    给连续的 `>` 行（除最后一行）补行尾硬换行，使"题目/问题/实现/代码"
    各自成行。
    """
    lines = text.split("\n")
    out: list[str] = []
    n = 0
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.startswith(">") and nxt.startswith(">"):
            stripped = line.rstrip()
            if not stripped.endswith("\\"):
                line = stripped + "  "
                n += 1
        out.append(line)
    print(f"preprocessed: {n} blockquote lines given hard breaks")
    return "\n".join(out)


# ---------------------------------------------------------------- 2. pandoc
def ensure_reference() -> None:
    """reference.docx 只用来继承 pandoc 的样式表；字体等由后处理覆盖。
    不提交到版本库，缺失时从 pandoc 内置模板导出。"""
    if REF.exists():
        return
    pandoc = pypandoc.get_pandoc_path()
    subprocess.run(
        [pandoc, "-o", str(REF), "--print-default-data-file=reference.docx"],
        check=True,
    )
    print(f"created {REF.relative_to(ROOT)} from pandoc defaults")


def run_pandoc(verbose: bool = False) -> None:
    pandoc = pypandoc.get_pandoc_path()
    cmd = [
        pandoc,
        str(TMP),
        "-o",
        str(OUT),
        "--from=markdown+tex_math_dollars+pipe_tables+raw_tex",
        f"--reference-doc={REF}",
        "--resource-path=.",
        "--toc",
        "--toc-depth=2",
        "--columns=120",
        "-V", "toc-title=目录",
    ]
    if verbose:
        cmd.append("--verbose")
    print("pandoc:", " ".join(cmd[1:]))
    subprocess.run(cmd, cwd=ROOT, check=True)


# ---------------------------------------------------------------- 3. 后处理
def set_run_fonts(rpr_owner, cn: str, en: str) -> None:
    """同时设置 ascii / hAnsi / eastAsia / cs 四种字体槽。"""
    fonts = rpr_owner.get_or_add_rPr().get_or_add_rFonts()
    fonts.set(qn("w:ascii"), en)
    fonts.set(qn("w:hAnsi"), en)
    fonts.set(qn("w:eastAsia"), cn)
    fonts.set(qn("w:cs"), en)


def style_fonts(doc: Document) -> None:
    # 文档默认字体
    defaults = doc.styles.element.find(qn("w:docDefaults"))
    if defaults is not None:
        rpr_default = defaults.find(qn("w:rPrDefault"))
        if rpr_default is not None:
            rpr = rpr_default.find(qn("w:rPr"))
            if rpr is not None:
                fonts = rpr.find(qn("w:rFonts"))
                if fonts is None:
                    fonts = rpr.makeelement(qn("w:rFonts"), {})
                    rpr.insert(0, fonts)
                fonts.set(qn("w:ascii"), BODY_EN)
                fonts.set(qn("w:hAnsi"), BODY_EN)
                fonts.set(qn("w:eastAsia"), BODY_CN)
                fonts.set(qn("w:cs"), BODY_EN)

    plan = {
        "Normal": (BODY_CN, BODY_EN, 10.5, None),
        "Body Text": (BODY_CN, BODY_EN, 10.5, None),
        "First Paragraph": (BODY_CN, BODY_EN, 10.5, None),
        "Compact": (BODY_CN, BODY_EN, 10.5, None),
        "Title": (HEAD_CN, HEAD_EN, 20, True),
        "Subtitle": (BODY_CN, BODY_EN, 12, None),
        "Heading 1": (HEAD_CN, HEAD_EN, 15, True),
        "Heading 2": (HEAD_CN, HEAD_EN, 12.5, True),
        "Heading 3": (HEAD_CN, HEAD_EN, 11, True),
        "Heading 4": (HEAD_CN, HEAD_EN, 10.5, True),
        "Block Text": (BODY_CN, BODY_EN, 9.5, None),
        "Image Caption": (BODY_CN, BODY_EN, 9, None),
        "Table Caption": (BODY_CN, BODY_EN, 9, None),
        "Captioned Figure": (BODY_CN, BODY_EN, 10.5, None),
        "Figure": (BODY_CN, BODY_EN, 10.5, None),
        "TOC Heading": (HEAD_CN, HEAD_EN, 14, True),
        "Source Code": (MONO, MONO, 8.5, None),
        "Verbatim Char": (MONO, MONO, 9, None),
    }
    for name, (cn, en, size, bold) in plan.items():
        try:
            st = doc.styles[name]
        except KeyError:
            continue
        set_run_fonts(st.element, cn, en)
        st.font.size = Pt(size)
        if bold is not None:
            st.font.bold = bold
        if name.startswith("Heading") or name == "Title":
            st.font.color.rgb = RGBColor(0, 0, 0)

    # 中文字体下"斜体"几乎不可读，标题一律不斜
    for name in ("Heading 1", "Heading 2", "Heading 3", "Heading 4", "Title"):
        try:
            doc.styles[name].font.italic = False
        except KeyError:
            pass


def setup_page(doc: Document) -> None:
    for sec in doc.sections:
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        sec.top_margin = sec.bottom_margin = Cm(2.1)
        sec.left_margin = sec.right_margin = Cm(2.1)


def _has_drawing(p) -> bool:
    return p._p.find(".//" + qn("w:drawing")) is not None


def fix_figures(doc: Document) -> None:
    """图片段落：居中 + 与下一段（题注）同页。题注：居中、小字、不缩进。"""
    paras = doc.paragraphs
    n_img = 0
    for i, p in enumerate(paras):
        if _has_drawing(p):
            n_img += 1
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
        elif p.style.name in ("Image Caption", "Table Caption", "Caption"):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(10)
            if i > 0:
                paras[i - 1].paragraph_format.keep_with_next = True
    print(f"postprocess: {n_img} figure paragraphs centred")


def resize_figures(doc: Document) -> None:
    """pandoc 会把图片宽度夹到它自己算出的可用宽度（约 14.8cm）。

    这里按图号重设 `wp:extent` 与 `a:ext`：前者决定排版占位，后者决定
    图片本体缩放，两者必须同步，否则图会显示不清或被裁。
    """
    target = {
        1: Cm(16.0), 2: Cm(15.5), 3: Cm(15.5), 4: Cm(15.5),
        5: Cm(15.5), 6: Cm(12.5), 7: Cm(16.0),
    }
    idx = 0
    for p in doc.paragraphs:
        if not _has_drawing(p):
            continue
        idx += 1
        width = target.get(idx)
        if width is None:
            continue
        for tag in ("wp:extent", "a:ext"):
            for ext in p._p.findall(".//" + qn(tag)):
                cx, cy = int(ext.get("cx")), int(ext.get("cy"))
                if cx == 0:
                    continue
                ratio = cy / cx
                ext.set("cx", str(int(width)))
                ext.set("cy", str(int(width * ratio)))
    print(f"postprocess: {idx} figures resized")


def widen_tables(doc: Document) -> None:
    """pandoc 把每张表都压到 13.97cm。等比放大到 16.5cm 用满文本宽，
    列宽相对比例不变，避免 6–7 列的表挤在一起。"""
    twips_per_cm = 567.0
    target = 16.5 * twips_per_cm
    for t in doc.tables:
        grid = t._tbl.find(qn("w:tblGrid"))
        if grid is None:
            continue
        cols = grid.findall(qn("w:gridCol"))
        widths = [int(c.get(qn("w:w"))) for c in cols]
        total = sum(widths)
        if total <= 0:
            continue
        scale = target / total
        for c, w in zip(cols, widths):
            c.set(qn("w:w"), str(int(w * scale)))
        tbl_pr = t._tbl.tblPr
        tbl_w = tbl_pr.find(qn("w:tblW"))
        if tbl_w is None:
            tbl_w = tbl_pr.makeelement(qn("w:tblW"), {})
            tbl_pr.insert_element_before(
                tbl_w, "w:jc", "w:tblCellSpacing", "w:tblInd", "w:tblBorders",
                "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook",
                "w:tblCaption", "w:tblDescription", "w:tblPrChange",
            )
        tbl_w.set(qn("w:type"), "dxa")
        tbl_w.set(qn("w:w"), str(int(target)))
    print(f"postprocess: {len(doc.tables)} tables widened to 16.5cm")


def drop_empty_paragraphs(doc: Document) -> None:
    """删除 Markdown `---` 分隔线产生的空段落。

    注意：公式在 `m:` 命名空间，图片在 `w:` 命名空间——凡是含公式或图片的
    段落都不能删，否则展示公式会被整段清空（曾因此丢失 15 个公式块）。
    """
    removed = 0
    for p in list(doc.paragraphs):
        if p.text.strip():
            continue
        el = p._p
        if el.find(".//" + qn("m:oMath")) is not None:
            continue
        if el.find(".//" + qn("w:drawing")) is not None:
            continue
        if el.find(".//" + qn("w:sectPr")) is not None:
            continue
        el.getparent().remove(el)
        removed += 1
    if removed:
        print(f"postprocess: dropped {removed} empty paragraphs")


def center_display_math(doc: Document) -> None:
    """Markdown 的 `$$...$$` 转成"整段只有一个公式"的段落，居中并留出上下间距。"""
    n = 0
    for p in doc.paragraphs:
        el = p._p
        if p.text.strip():
            continue
        if el.find(".//" + qn("m:oMath")) is None:
            continue
        n += 1
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format
        pf.first_line_indent = Cm(0)
        pf.space_before = Pt(6)
        pf.space_after = Pt(6)
        pf.line_spacing = 1.0
    print(f"postprocess: {n} display equations centred")


def keep_captions_with_object(doc: Document) -> None:
    """表标题、图注所在段落与紧随的对象同页，避免标题孤零零留在页底。

    遍历 body 的块序列：若某段落后紧跟 w:tbl，则对该段落设 keep_with_next。
    """
    body = doc.element.body
    from docx.text.paragraph import Paragraph

    def ptext(p_el) -> str:
        return "".join(
            (el.text or "")
            for el in p_el.iter()
            if el.tag in (qn("w:t"), qn("m:t"))
        )

    pending = None          # 上一个段落
    bound = 0
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            pending = child
        elif child.tag == qn("w:tbl"):
            if pending is not None:
                para = Paragraph(pending, doc)
                name = para.style.name
                text = ptext(pending).strip()
                # 只绑定短引导句、表标题与标题；长正文段落绑定会挤出大片留白
                if (
                    len(text) <= 120
                    or name.startswith("Heading")
                    or text.startswith("表 ")
                ):
                    para.paragraph_format.keep_with_next = True
                    bound += 1
            pending = None
        else:
            pending = None
    print(f"postprocess: {bound} captions bound to their table")


def style_table_captions(doc: Document) -> None:
    """`表 Na：…` 这类表标题：居中、9.5pt、加粗，并与表格贴紧。"""
    n = 0
    for para in doc.paragraphs:
        text = para.text.strip()
        if not (text.startswith("表 ") and len(text) <= 60):
            continue
        if para.style.name.startswith("Heading"):
            continue
        fmt = para.paragraph_format
        fmt.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fmt.left_indent = Cm(0)
        fmt.right_indent = Cm(0)
        fmt.first_line_indent = Cm(0)
        fmt.space_before = Pt(8)
        fmt.space_after = Pt(2)
        fmt.keep_with_next = True
        for run in para.runs:
            run.font.size = Pt(9.5)
            run.font.bold = True
            set_run_fonts(run._element, HEAD_CN, HEAD_EN)
        n += 1
    print(f"postprocess: {n} table captions styled")


def localize_toc(doc: Document) -> None:
    """pandoc 的 docx 目录是"域"，其中文标题它固定输出成 `Table of Contents`。

    目录位于 w:sdt 内容控件内，python-docx 的 doc.paragraphs 看不到，只能在
    原始 XML 上改。
    """
    body = doc.element.body
    for sdt in body.iter(qn("w:sdt")):
        gallery = sdt.find(".//" + qn("w:docPartGallery"))
        if gallery is None or "Table of Contents" not in (gallery.get(qn("w:val")) or ""):
            continue
        for t in sdt.iter(qn("w:t")):
            if t.text and "Table of Contents" in t.text:
                t.text = "目录"
                print("postprocess: TOC heading -> 目录")
        # 目录标题本身也居中加粗
        for p in sdt.findall(".//" + qn("w:p")):
            ppr = p.find(qn("w:pPr"))
            if ppr is None:
                continue
            style = ppr.find(qn("w:pStyle"))
            if style is None or style.get(qn("w:val")) != "TOCHeading":
                continue
            jc = ppr.find(qn("w:jc"))
            if jc is None:
                jc = ppr.makeelement(qn("w:jc"), {})
                ppr.append(jc)
            jc.set(qn("w:val"), "center")
        return


def normalize_toc_field(doc: Document) -> None:
    """pandoc 把 begin / instrText / separate / end 四个元素塞进同一个 run，
    这在 schema 上不违规，但部分阅读器（WPS、旧版 Word）不会刷新。

    这里把它们拆成各自独立的 run，并在 separate 与 end 之间放一段占位文字：
    不刷新域时看到占位提示，刷新后占位文字被真实目录替换。
    """
    body = doc.element.body
    for sdt in body.iter(qn("w:sdt")):
        gallery = sdt.find(".//" + qn("w:docPartGallery"))
        if gallery is None or "Table of Contents" not in (gallery.get(qn("w:val")) or ""):
            continue
        for p in sdt.findall(".//" + qn("w:p")):
            if p.find(".//" + qn("w:instrText")) is None:
                continue
            for r in p.findall(qn("w:r")):
                p.remove(r)

            def add_run(children) -> None:
                r = p.makeelement(qn("w:r"), {})
                for c in children:
                    r.append(c)
                p.append(r)

            def fld(kind: str):
                attrs = {qn("w:fldCharType"): kind}
                if kind == "begin":
                    attrs[qn("w:dirty")] = "true"
                return p.makeelement(qn("w:fldChar"), attrs)

            add_run([fld("begin")])

            instr = p.makeelement(qn("w:instrText"), {qn("xml:space"): "preserve"})
            instr.text = 'TOC \\o "1-2" \\h \\z \\u'
            add_run([instr])
            add_run([fld("separate")])

            t = p.makeelement(qn("w:t"), {qn("xml:space"): "preserve"})
            t.text = "（打开文档时自动生成目录；若无内容，请右键选择「更新域」）"
            add_run([t])
            add_run([fld("end")])

            print("postprocess: TOC field normalized (begin/instr/separate/end)")
            return


def autoupdate_fields(doc: Document) -> None:
    """让 Word 打开文档时刷新目录与页码域（否则目录是空白/过期页码）。

    settings.xml 的子元素同样有顺序要求，updateFields 必须排在
    compat / rsids / mathPr 等之前。
    """
    settings = doc.settings.element
    tag = qn("w:updateFields")
    if settings.find(tag) is None:
        el = settings.makeelement(tag, {qn("w:val"): "true"})
        settings.insert_element_before(
            el,
            "w:hdrShapeDefaults", "w:footnotePr", "w:endnotePr", "w:compat",
            "w:docVars", "w:rsids", "m:mathPr", "w:attachedSchema",
            "w:themeFontLang", "w:clrSchemeMapping",
            "w:doNotIncludeSubdocsInStats", "w:doNotAutoCompressPictures",
            "w:forceUpgrade", "w:captions", "w:readModeInkLockDown",
            "w:smartTagType", "sl:schemaLibrary", "w:shapeDefaults",
            "w:doNotEmbedSmartTags", "w:decimalSymbol", "w:listSeparator",
        )


def _cell_borders(tbl_pr) -> None:
    """加细边框。OOXML 的 tblPr 有严格子元素顺序，必须插在 tblLayout/tblLook
    之前——直接 append 会让 Word 认为文件损坏并提示"修复"。"""
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = tbl_pr.makeelement(qn("w:tblBorders"), {})
        tbl_pr.insert_element_before(
            borders,
            "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook",
            "w:tblCaption", "w:tblDescription", "w:tblPrChange",
        )
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = borders.find(qn(f"w:{edge}"))
        if el is None:
            el = borders.makeelement(qn(f"w:{edge}"), {})
            borders.append(el)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), "808080")


def fix_tables(doc: Document) -> None:
    """表格：居中、单行不跨页拆开、表头行跨页重复、正文小一号字。"""
    for t in doc.tables:
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        _cell_borders(t._tbl.tblPr)

        if t.rows:
            tr_pr = t.rows[0]._tr.get_or_add_trPr()
            if tr_pr.find(qn("w:tblHeader")) is None:
                tr_pr.insert_element_before(
                    tr_pr.makeelement(qn("w:tblHeader"), {}),
                    "w:tblCellSpacing", "w:jc", "w:hidden",
                    "w:ins", "w:del", "w:trPrChange",
                )

        for row in t.rows:
            tr_pr = row._tr.get_or_add_trPr()
            if tr_pr.find(qn("w:cantSplit")) is None:
                tr_pr.insert_element_before(
                    tr_pr.makeelement(qn("w:cantSplit"), {}),
                    "w:trHeight", "w:tblHeader", "w:tblCellSpacing", "w:jc",
                    "w:hidden", "w:ins", "w:del", "w:trPrChange",
                )
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(1)
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.first_line_indent = Cm(0)
                    for r in p.runs:
                        r.font.size = Pt(8.5)
    print(f"postprocess: {len(doc.tables)} tables styled")


def fix_body(doc: Document) -> None:
    """正文：中文首行缩进 2 字，行距 1.25，段前段后 0。"""
    body_styles = {"Normal", "Body Text", "First Paragraph", "Compact"}
    heading_like = re.compile(r"^(Heading \d|Title|Subtitle|Caption|Image|Table|Source Code)")
    for p in doc.paragraphs:
        name = p.style.name
        if name in body_styles:
            pf = p.paragraph_format
            # pandoc 的 First Paragraph 已经是首段，不必缩进
            pf.first_line_indent = Cm(0)
            pf.line_spacing = 1.25
            pf.space_before = Pt(0)
            pf.space_after = Pt(4)
        elif heading_like.match(name):
            pf = p.paragraph_format
            lo = {"Heading 1": 12, "Heading 2": 10, "Heading 3": 8}.get(name, 0)
            pf.space_before = Pt(lo)
            pf.space_after = Pt(4)
            pf.keep_with_next = True


def style_document_title(doc: Document) -> None:
    """第一处 Heading 1 是报告标题，单独放大居中；其余 Heading 1 保持节标题样式。"""
    for p in doc.paragraphs:
        if p.style.name != "Heading 1":
            continue
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(14)
        for r in p.runs:
            r.font.size = Pt(18)
            r.font.bold = True
        print(f"postprocess: title paragraph styled -> {p.text[:30]}")
        break


def add_page_numbers(doc: Document) -> None:
    """页脚居中页码。"""
    from docx.oxml import OxmlElement

    for sec in doc.sections:
        footer_p = sec.footer.paragraphs[0]
        footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer_p.add_run()
        for tag, attr in (("w:fldChar", {"w:fldCharType": "begin"}),
                          ("w:instrText", None),
                          ("w:fldChar", {"w:fldCharType": "end"})):
            el = OxmlElement(tag)
            if attr:
                for k, v in attr.items():
                    el.set(qn(k), v)
            else:
                el.set(qn("xml:space"), "preserve")
                el.text = " PAGE "
            run._r.append(el)
        run.font.size = Pt(9)


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}", file=sys.stderr)
        return 1

    text = normalize_figures(SRC.read_text(encoding="utf-8"))
    text = split_blockquote_lines(text)
    TMP.write_text(text, encoding="utf-8")

    ensure_reference()
    run_pandoc()
    if not OUT.exists():
        print("pandoc produced no output", file=sys.stderr)
        return 1

    doc = Document(str(OUT))
    style_fonts(doc)
    setup_page(doc)
    fix_body(doc)
    fix_figures(doc)
    fix_tables(doc)
    resize_figures(doc)
    widen_tables(doc)
    center_display_math(doc)
    drop_empty_paragraphs(doc)
    keep_captions_with_object(doc)
    style_table_captions(doc)
    style_document_title(doc)
    localize_toc(doc)
    normalize_toc_field(doc)
    add_page_numbers(doc)
    autoupdate_fields(doc)
    doc.save(str(OUT))

    size = OUT.stat().st_size / 1024
    print(f"OK -> {OUT} ({size:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
