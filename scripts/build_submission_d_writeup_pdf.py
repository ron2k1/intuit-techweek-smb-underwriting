#!/usr/bin/env python3
"""Render the Deliverable D markdown draft to a compact PDF.

This intentionally avoids external PDF dependencies such as pandoc. It uses
matplotlib's PDF backend, which is already available in the local environment.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "outputs" / "submission"
SOURCE = SUBMISSION_DIR / "submission_D_writeup.md"
TARGET = SUBMISSION_DIR / "submission_D_writeup.pdf"

PAGE_W, PAGE_H = 8.5, 11.0
LEFT, RIGHT, TOP, BOTTOM = 0.75, 0.75, 0.72, 0.72
BODY_SIZE = 11.0
HEADER_SIZE = 12.5
TITLE_SIZE = 14.5
LINE_H = 0.205


def clean_inline(text: str) -> str:
    text = text.replace("`", "")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    return text


def wrapped_lines(text: str, width: int) -> list[str]:
    if not text.strip():
        return [""]
    return textwrap.wrap(clean_inline(text), width=width, break_long_words=False) or [""]


def draw_page(pdf: PdfPages, page_lines: list[tuple[str, str]]) -> None:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    y = PAGE_H - TOP
    for kind, line in page_lines:
        if kind == "title":
            size, weight, gap = TITLE_SIZE, "bold", LINE_H * 1.35
        elif kind == "h2":
            size, weight, gap = HEADER_SIZE, "bold", LINE_H * 1.25
        else:
            size, weight, gap = BODY_SIZE, "normal", LINE_H
        ax.text(
            LEFT / PAGE_W,
            y / PAGE_H,
            line,
            transform=ax.transAxes,
            fontsize=size,
            fontweight=weight,
            va="top",
            ha="left",
            family="DejaVu Sans",
        )
        y -= gap
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    text = SOURCE.read_text()
    logical_lines: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("# "):
            logical_lines.extend(("title", part) for part in wrapped_lines(line[2:], 58))
            logical_lines.append(("body", ""))
        elif line.startswith("## "):
            logical_lines.append(("body", ""))
            logical_lines.extend(("h2", part) for part in wrapped_lines(line[3:], 66))
        else:
            for part in wrapped_lines(line, 82):
                logical_lines.append(("body", part))

    max_lines = int((PAGE_H - TOP - BOTTOM) / LINE_H)
    pages: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    for item in logical_lines:
        if len(current) >= max_lines:
            pages.append(current)
            current = []
        current.append(item)
    if current:
        pages.append(current)

    # The challenge truncates beyond page 4; fail loudly if the body is too long.
    if len(pages) > 4:
        raise SystemExit(f"writeup rendered to {len(pages)} pages; shorten to <= 4 pages")

    with PdfPages(TARGET) as pdf:
        for page in pages:
            draw_page(pdf, page)

    print(f"Wrote {TARGET} ({len(pages)} page(s))")


if __name__ == "__main__":
    main()
