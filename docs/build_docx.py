"""
Convert docs/PROJECT_REPORT.md into docs/PROJECT_REPORT.docx.

A small, dependency-light converter (python-docx only) tailored to the
Markdown features used in the report: headings, paragraphs, bold/italic/code,
inline LaTeX math, bullet and numbered lists, tables, code blocks, and images.
The Mermaid diagram is replaced by docs/pipeline_diagram.png
(render it with docs/make_pipeline_diagram.py).

Usage (from the repository root):
    pip install python-docx
    python docs/build_docx.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

DOCS = Path(__file__).resolve().parent
SRC = DOCS / "PROJECT_REPORT.md"
OUT = DOCS / "PROJECT_REPORT.docx"
DIAGRAM = DOCS / "pipeline_diagram.png"

BODY_FONT = "Calibri"
HEAD_FONT = "Cambria"
CODE_FONT = "Consolas"
ACCENT = RGBColor(0x1F, 0x3A, 0x5F)

# ---------------------------------------------------------------- LaTeX math
SYMBOLS = {
    r"\alpha": "α", r"\rho": "ρ", r"\tau": "τ", r"\ell": "ℓ", r"\in": "∈",
    r"\ge": "≥", r"\le": "≤", r"\cdot": "·", r"\times": "×", r"\log": "log",
    r"\exp": "exp", r"\big": "", r"\,": "\u2009", r"\mathcal{X}": "𝒳",
    r"\mathbb{E}": "𝔼",
}
HATS = {"y": "ŷ", "z": "ẑ", "u": "û", r"\ell": "ℓ̂"}


def latex_to_runs(expr: str) -> list[tuple[str, str]]:
    """Return [(text, kind)] where kind is 'n' (normal), 'sup', 'sub', or 'b' (bold)."""
    s = expr
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\hat\{(\\ell|[a-z])\}", lambda m: HATS.get(m.group(1), m.group(1) + "\u0302"), s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", "\x01\\1\x02", s)  # bold markers
    for k in sorted(SYMBOLS, key=len, reverse=True):
        s = s.replace(k, SYMBOLS[k])
    runs: list[tuple[str, str]] = []
    i, buf = 0, ""
    bold = False
    while i < len(s):
        c = s[i]
        if c in "^_":
            if buf:
                runs.append((buf, "b" if bold else "n"))
                buf = ""
            kind = "sup" if c == "^" else "sub"
            if i + 1 < len(s) and s[i + 1] == "{":
                j = s.index("}", i + 2)
                runs.append((s[i + 2 : j], kind))
                i = j + 1
            else:
                runs.append((s[i + 1], kind))
                i += 2
            continue
        if c in "\x01\x02":
            if buf:
                runs.append((buf, "b" if bold else "n"))
                buf = ""
            bold = c == "\x01"
            i += 1
            continue
        buf += c
        i += 1
    if buf:
        runs.append((buf, "b" if bold else "n"))
    return runs


# ---------------------------------------------------------------- inline markdown
TOKEN = re.compile(
    r"(\*\*(?P<b>.+?)\*\*)"
    r"|(?<![\w*])\*(?P<i>[^*\s`](?:`[^`]*`|[^*`])*?)\*(?!\w)"
    r"|`(?P<c>[^`]+)`"
    r"|\$(?P<m>[^$]+)\$"
    r"|\[(?P<lt>[^\]]+)\]\((?P<lu>[^)]+)\)"
    r"|\\(?P<e>[*_$\\])"
)


def add_inline(par, text: str, bold=False, italic=False, size=None) -> None:
    pos = 0
    for m in TOKEN.finditer(text):
        if m.start() > pos:
            _run(par, text[pos : m.start()], bold, italic, size=size)
        if m.group("b") is not None:
            add_inline(par, m.group("b"), True, italic, size)
        elif m.group("i") is not None:
            add_inline(par, m.group("i"), bold, True, size)
        elif m.group("c") is not None:
            _run(par, m.group("c"), bold, italic, code=True, size=size)
        elif m.group("m") is not None:
            for t, kind in latex_to_runs(m.group("m")):
                r = _run(par, t, bold or kind == "b", True, size=size, font="Cambria Math")
                if kind == "sup":
                    r.font.superscript = True
                elif kind == "sub":
                    r.font.subscript = True
        elif m.group("lt") is not None:
            _run(par, m.group("lt"), bold, italic, size=size)
        elif m.group("e") is not None:
            _run(par, m.group("e"), bold, italic, size=size)
        pos = m.end()
    if pos < len(text):
        _run(par, text[pos:], bold, italic, size=size)


def _run(par, text, bold=False, italic=False, code=False, size=None, font=None):
    r = par.add_run(text)
    r.bold = bold or None
    r.italic = italic or None
    if code:
        r.font.name = CODE_FONT
        r._element.rPr.rFonts.set(qn("w:eastAsia"), CODE_FONT)
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0x8B, 0x1E, 0x3F)
    elif font:
        r.font.name = font
    if size and not code:
        r.font.size = Pt(size)
    return r


# ---------------------------------------------------------------- block helpers
def shade(cell_or_par, hex_fill: str) -> None:
    el = cell_or_par._tc if hasattr(cell_or_par, "_tc") else cell_or_par._p
    pr = el.get_or_add_tcPr() if hasattr(cell_or_par, "_tc") else el.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    pr.append(shd)


def add_table(doc, rows: list[str]) -> None:
    def cells(line):
        line = line.strip().strip("|")
        return [c.strip() for c in re.split(r"(?<!\\)\|", line)]

    header, align_row, body = cells(rows[0]), cells(rows[1]), [cells(r) for r in rows[2:]]
    aligns = []
    for a in align_row:
        if a.startswith(":") and a.endswith(":"):
            aligns.append(WD_ALIGN_PARAGRAPH.CENTER)
        elif a.endswith(":"):
            aligns.append(WD_ALIGN_PARAGRAPH.RIGHT)
        else:
            aligns.append(WD_ALIGN_PARAGRAPH.LEFT)
    ncols = len(header)
    table = doc.add_table(rows=1 + len(body), cols=ncols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    size = 8 if ncols >= 7 else 9
    for r_i, row in enumerate([header] + body):
        for c_i in range(ncols):
            text = row[c_i] if c_i < len(row) else ""
            cell = table.rows[r_i].cells[c_i]
            p = cell.paragraphs[0]
            p.alignment = aligns[c_i] if r_i else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            add_inline(p, text.replace("\\|", "|"), bold=(r_i == 0), size=size)
            if r_i == 0:
                shade(cell, "DCE6F1")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_code(doc, lines: list[str]) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.space_after = Pt(8)
    shade(p, "F3F4F6")
    for i, line in enumerate(lines):
        r = p.add_run(line)
        r.font.name = CODE_FONT
        r._element.rPr.rFonts.set(qn("w:eastAsia"), CODE_FONT)
        r.font.size = Pt(8.5)
        if i < len(lines) - 1:
            r.add_break()


def add_image(doc, path: Path, width_in: float = 6.3) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width_in))


def setup_styles(doc) -> None:
    st = doc.styles["Normal"]
    st.font.name = BODY_FONT
    st.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    st.font.size = Pt(11)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.1
    for name, size in [("Title", 20), ("Heading 1", 15), ("Heading 2", 13), ("Heading 3", 11.5), ("Heading 4", 11)]:
        h = doc.styles[name]
        h.font.name = HEAD_FONT
        h.element.rPr.rFonts.set(qn("w:eastAsia"), HEAD_FONT)
        h.font.size = Pt(size)
        h.font.color.rgb = ACCENT
        h.font.bold = True
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Inches(1)
        sec.top_margin = sec.bottom_margin = Inches(0.9)


def add_page_numbers(doc) -> None:
    p = doc.sections[0].footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for tag, text in [("begin", None), (None, "PAGE"), ("end", None)]:
        if tag:
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), tag)
            run._r.append(fc)
        else:
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = text
            run._r.append(it)


# ---------------------------------------------------------------- main conversion
def convert() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    doc = Document()
    setup_styles(doc)
    add_page_numbers(doc)

    i, seen_title, para_buf = 0, False, []

    def flush():
        if para_buf:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_inline(p, " ".join(s.strip() for s in para_buf))
            para_buf.clear()

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        if line.startswith("```"):
            flush()
            lang = line[3:].strip()
            j = i + 1
            while not lines[j].strip().startswith("```"):
                j += 1
            block = lines[i + 1 : j]
            indent = len(raw) - len(raw.lstrip())
            block = [b[indent:] if b[:indent].isspace() else b for b in block]
            if lang == "mermaid":
                if DIAGRAM.exists():
                    add_image(doc, DIAGRAM, 6.5)
                    cap = doc.add_paragraph()
                    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    add_inline(cap, "*Pipeline overview: data flow, grouped splits, model training, and interval calibration.*", size=9.5)
            else:
                add_code(doc, block)
            i = j + 1
            continue

        if not line:
            flush()
            i += 1
            continue

        if line == "---":
            flush()
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            flush()
            level, text = len(m.group(1)), m.group(2)
            if level == 1 and not seen_title:
                p = doc.add_paragraph(style="Title")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_inline(p, text)
                seen_title = True
            else:
                if level == 2 and text.lower().startswith(("1. intro", "bibliography", "reproducing")):
                    pass
                h = doc.add_heading(level=max(1, level - 1))
                add_inline(h, text)
            i += 1
            continue

        if line.startswith("|"):
            flush()
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            add_table(doc, [l.strip() for l in lines[i:j]])
            i = j
            continue

        img = re.match(r"^!\[[^\]]*\]\(([^)]+)\)$", line)
        if img:
            flush()
            path = (DOCS / img.group(1)).resolve()
            if path.exists():
                add_image(doc, path)
            i += 1
            continue

        lm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", raw)
        if lm:
            flush()
            indent = len(lm.group(1))
            numbered = lm.group(2)[0].isdigit()
            text = lm.group(3)
            # gather continuation lines (indented, non-list, non-table, non-fence)
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                ns = nxt.strip()
                if not ns or ns.startswith(("|", "```")) or re.match(r"^\s*([-*]|\d+\.)\s+", nxt):
                    break
                text += " " + ns
                j += 1
            style = "List Number" if numbered else "List Bullet"
            if indent >= 2:
                style += " 2"
            p = doc.add_paragraph(style=style)
            p.paragraph_format.space_after = Pt(3)
            if numbered:
                # keep the author's numbering explicitly (Word restarts lists unpredictably)
                p.style = doc.styles["List Bullet 2" if indent >= 2 else "Normal"]
                p.paragraph_format.left_indent = Inches(0.3 + 0.25 * (indent >= 2))
                p.paragraph_format.first_line_indent = Inches(-0.25)
                _run(p, lm.group(2) + "\t")
            add_inline(p, text)
            i = j
            continue

        # Centered byline/caption lines directly under the title
        if seen_title and len(doc.paragraphs) <= 3 and (line.startswith("*") or line.startswith("**")):
            flush()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(p, line)
            i += 1
            continue

        # Figure captions: a paragraph that is entirely italic and starts with "Figure"
        if line.startswith("*Figure") or line.startswith("\\*"):
            flush()
            p = doc.add_paragraph()
            if line.startswith("*Figure"):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_inline(p, line, size=9.5)
            else:
                add_inline(p, line, size=9.5)
            i += 1
            continue

        para_buf.append(line)
        i += 1

    flush()
    doc.save(OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    convert()
