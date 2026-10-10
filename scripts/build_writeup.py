#!/usr/bin/env python3
"""Render the writeup HTML files to PDF.

- ``writeup/progression.html`` -> ``writeup/progression.pdf`` (a self-contained
  walkthrough for a general audience)
- ``writeup/submission.html``  -> ``writeup/submission.pdf``  (the assignment
  writeup, organised by problem)

Requires the optional docs dependency: pip install -r requirements-docs.txt
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRITEUP = ROOT / "writeup"
DOCS = ["progression", "submission"]


def main():
    from weasyprint import HTML

    for name in DOCS:
        src = WRITEUP / f"{name}.html"
        out = WRITEUP / f"{name}.pdf"
        if not src.exists():
            print(f"skip missing {src}")
            continue
        HTML(filename=str(src)).write_pdf(str(out))
        print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
