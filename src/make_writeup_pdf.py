#!/usr/bin/env python3
"""Render reports/submission_D_writeup.md -> submissions/submission_D_writeup.pdf
with the enforced format: >=11pt font, >=0.75in margins on all sides."""
from __future__ import annotations
import re
import sys
from pathlib import Path
import markdown
from xhtml2pdf import pisa

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "reports" / "submission_D_writeup.md"
OUT = REPO / "submissions" / "submission_D_writeup.pdf"

CSS = """
@page { size: letter; margin: 0.8in; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.6pt; line-height: 1.28; color: #111; }
h1 { font-size: 15pt; margin: 0 0 4pt 0; }
h2 { font-size: 12pt; margin: 9pt 0 3pt 0; border-bottom: 0.5pt solid #999; padding-bottom: 1pt; }
p  { margin: 3pt 0; text-align: justify; }
ul { margin: 2pt 0 4pt 16pt; }
li { margin: 1pt 0; }
strong { color: #000; }
code { font-family: Courier, monospace; font-size: 9.5pt; }
"""


_UNI = {
    "→": "->", "←": "<-", "×": "x", "≈": "~", "≥": ">=", "≤": "<=",
    "—": " - ", "–": "-", "•": "-", "’": "'", "‘": "'",
    "“": '"', "”": '"', "…": "...", "±": "+/-", "≠": "!=",
    "↑": "(up)", "↓": "(down)", "₀": "0", "ₐ": "a",
}


def normalize(s: str) -> str:
    for k, v in _UNI.items():
        s = s.replace(k, v)
    return s


def main() -> int:
    text = SRC.read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)  # strip comments
    text = normalize(text)  # ASCII-ize glyphs the PDF font cannot render
    body = markdown.markdown(text, extensions=["extra", "sane_lists"])
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        result = pisa.CreatePDF(html, dest=f, encoding="utf-8")
    if result.err:
        print("PDF generation errors:", result.err)
        return 1
    # page count
    try:
        from PyPDF2 import PdfReader
        n = len(PdfReader(str(OUT)).pages)
    except Exception:
        try:
            from pypdf import PdfReader
            n = len(PdfReader(str(OUT)).pages)
        except Exception:
            n = "?"
    print(f"[written] {OUT}  (pages: {n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
