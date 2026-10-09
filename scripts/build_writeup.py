#!/usr/bin/env python3
"""Render writeup/report.html to writeup/report.pdf.

Requires the optional docs dependency: pip install -r requirements-docs.txt
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "writeup" / "report.html"
OUT = ROOT / "writeup" / "report.pdf"


def main():
    from weasyprint import HTML

    HTML(filename=str(SRC)).write_pdf(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
