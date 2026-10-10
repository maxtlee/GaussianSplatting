#!/usr/bin/env python3
"""Download a set of Eiffel Tower photographs from Wikimedia Commons.

Uses the public MediaWiki API (search + imageinfo), keeps JPEG photographs
above a minimum size, downloads 1280 px thumbnails, and records the license and
author of every file for attribution. Dry-run by default; pass --execute to
download.

    python eiffel/fetch_images.py --query "Eiffel Tower" --limit 240
    python eiffel/fetch_images.py --query "Eiffel Tower" --limit 240 --execute
"""

import argparse
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = DATA / "raw"
API = "https://commons.wikimedia.org/w/api.php"
UA = "NeuralGraphics-Eiffel/1.0 (educational use; maxwe@cmu.edu)"

# Titles that are almost never useful for multi-view reconstruction.
BAD_WORDS = re.compile(
    r"(map|plan|diagram|logo|coat of arms|drawing|sketch|chart|graph|"
    r"stamp|banknote|coin|signature|poster|painting|illustration|"
    r"1900|exposition universelle)",
    re.IGNORECASE,
)


def api(params):
    params = {**params, "format": "json"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def search_titles(query, total):
    titles, offset = [], 0
    while len(titles) < total:
        d = api(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srnamespace": 6,
                "srlimit": 50,
                "sroffset": offset,
            }
        )
        hits = d.get("query", {}).get("search", [])
        if not hits:
            break
        titles.extend(h["title"] for h in hits)
        if "continue" not in d:
            break
        offset = d["continue"]["sroffset"]
        time.sleep(0.2)
    return titles[:total]


def category_titles(cat, limit):
    titles, cont = [], None
    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": cat,
            "cmtype": "file",
            "cmlimit": 100,
        }
        if cont:
            params["cmcontinue"] = cont
        d = api(params)
        titles.extend(m["title"] for m in d.get("query", {}).get("categorymembers", []))
        cont = d.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        time.sleep(0.2)
    return titles[:limit]


def clean(value):
    if value is None:
        return ""
    return re.sub(r"<[^>]+>", "", html.unescape(str(value))).strip()


def image_info(titles):
    records = []
    for i in range(0, len(titles), 50):
        chunk = titles[i : i + 50]
        d = api(
            {
                "action": "query",
                "titles": "|".join(chunk),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "iiurlwidth": 1280,
            }
        )
        for page in d.get("query", {}).get("pages", {}).values():
            info = page.get("imageinfo")
            if not info:
                continue
            info = info[0]
            em = info.get("extmetadata", {})
            records.append(
                {
                    "title": page["title"],
                    "pageid": page.get("pageid"),
                    "download": info.get("thumburl") or info["url"],
                    "url": info.get("descriptionurl", info["url"]),
                    "width": info.get("width"),
                    "height": info.get("height"),
                    "mime": info.get("mime"),
                    "license": clean(em.get("LicenseShortName", {}).get("value")),
                    "artist": clean(em.get("Artist", {}).get("value")),
                    "date": clean(em.get("DateTimeOriginal", {}).get("value")),
                }
            )
        time.sleep(0.2)
    return records


def keep(rec, min_side=800, min_long=1200):
    if rec["mime"] not in ("image/jpeg", "image/png"):
        return False
    w, h = rec["width"] or 0, rec["height"] or 0
    if min(w, h) < min_side or max(w, h) < min_long:
        return False
    if BAD_WORDS.search(rec["title"]):
        return False
    return True


def download(rec, index):
    RAW.mkdir(parents=True, exist_ok=True)
    suffix = ".jpg" if rec["mime"] == "image/jpeg" else ".png"
    path = RAW / f"{index:04d}{suffix}"
    if path.exists() and path.stat().st_size > 0:
        return path, True
    req = urllib.request.Request(rec["download"], headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    path.write_bytes(data)
    return path, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="Eiffel Tower")
    ap.add_argument("--category", action="append", default=[])
    ap.add_argument("--per-category", type=int, default=40)
    ap.add_argument("--limit", type=int, default=240)
    ap.add_argument("--min-side", type=int, default=800)
    ap.add_argument("--min-long", type=int, default=1200)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    titles = []
    if args.category:
        for cat in args.category:
            got = category_titles(cat, args.per_category)
            print(f"  {cat}: {len(got)} files")
            titles.extend(got)
    else:
        print(f"searching Commons for {args.limit} files: {args.query!r}")
        titles = search_titles(args.query, args.limit * 2)
    # de-duplicate, preserving order
    seen, ordered = set(), []
    for t in titles:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    titles = ordered
    records = [r for r in image_info(titles) if keep(r, args.min_side, args.min_long)]
    records = records[: args.limit]
    print(f"kept {len(records)} photographs")

    if not args.execute:
        print("dry run; pass --execute to download")
        for r in records[:12]:
            print(f"  {r['title']}  {r['width']}x{r['height']}  {r['license']}")
        return

    RAW.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, rec in enumerate(records):
        try:
            path, cached = download(rec, i)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i:3d}] FAILED {rec['title']}: {exc}")
            continue
        rec["file"] = str(path.relative_to(DATA))
        rec["bytes"] = path.stat().st_size
        manifest.append(rec)
        if i % 20 == 0:
            print(f"  [{i:3d}] {'cache' if cached else 'new  '} {path.name}")

    (DATA / "metadata.json").write_text(json.dumps(manifest, indent=2))

    lines = ["# Image credits", "",
             "Photographs downloaded from Wikimedia Commons via the MediaWiki API.",
             "Licenses and authors as recorded in the file metadata.", "",
             "| file | title | author | license | source |",
             "|---|---|---|---|---|"]
    for r in manifest:
        lines.append(
            f"| {r['file']} | {clean(r['title'])} | {r['artist'] or 'n/a'} | "
            f"{r['license'] or 'n/a'} | [link]({r['url']}) |"
        )
    (HERE / "CREDITS.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(manifest)} images to {RAW}")
    print(f"wrote {DATA / 'metadata.json'} and {HERE / 'CREDITS.md'}")


if __name__ == "__main__":
    main()
