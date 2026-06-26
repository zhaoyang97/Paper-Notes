#!/usr/bin/env python3
"""
Post-process the search_index.json produced by `mkdocs build` to drastically
shrink the search index and reduce browser memory footprint.

Problem: 12K+ paper pages x body text => 200+ MB JSON => lunr in-memory index ~450 MB.
Strategy v5:
  1. Replace body text with the page's **keywords** (read from the note's
     `**关键词**` / `**Keywords**` line) so keyword search works, instead of
     dropping text entirely. This mirrors hooks/slim_search_index.py; this
     standalone script runs again in CI after `mkdocs build`, so it must keep
     the keywords the hook injected rather than wiping them.
  2. Keep only non-conference tags (conference name is already in the URL path,
     and lunr indexes the location field).
  3. Do not inject bilingual synonyms (handled client-side via a Worker shim,
     see overrides/main.html).
  4. Drop low-frequency tags (paper-specific noise tags) to shrink the lunr
     token table.
  5. Goal: lunr memory stays small so the desktop search no longer freezes on
     "Initializing search engine".

Usage: python scripts/trim_search_index.py [site_dir]
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Conference name pattern. These tags are removed because the conference is
# already encoded in the URL path, so indexing them again is redundant.
_CONF_PATTERN = re.compile(
    r'^(ICLR|CVPR|ACL|NeurIPS|AAAI|ECCV|ICCV|ICML|EMNLP|NAACL)\s*\d{4}$',
    re.IGNORECASE,
)

# Tags appearing fewer than this many times across the whole site are
# discarded. In a 13K-paper corpus the long tail of single-paper tags adds
# zero discoverability value but balloons the lunr inverted index from ~2K
# tokens to 34K, which freezes the desktop search worker on init.
MIN_TAG_FREQ = 5

# Body metadata line listing a paper's keywords, e.g.
#   **关键词**: diffusion model, image editing      (zh notes)
#   **Keywords**: diffusion model, image editing    (en notes)
_KW_PATTERN = re.compile(r"\*\*(?:关键词|Keywords)\*\*\s*[:：]\s*(.+)")


def _read_keywords(docs_dir: Path, base: str) -> str:
    """Return the comma-joined keywords for a page, read from its source note.

    ``base`` is the directory-URL location (e.g. ``CVPR2026/3d_vision/slug/``);
    the source markdown is ``<docs_dir>/CVPR2026/3d_vision/slug.md``. Only the
    file head is scanned since the keyword line lives in the metadata block.
    """
    rel = base.strip("/")
    if not rel:
        return ""
    src = docs_dir / (rel + ".md")
    try:
        head = src.read_text(encoding="utf-8")[:2000]
    except OSError:
        return ""
    m = _KW_PATTERN.search(head)
    if not m:
        return ""
    kws = [k.strip() for k in m.group(1).split(",") if k.strip()]
    return ", ".join(kws)


def main():
    site_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("site")
    docs_dir = site_dir.parent / "docs"
    index_path = site_dir / "search" / "search_index.json"

    if not index_path.exists():
        print(f"ERROR: {index_path} not found")
        sys.exit(1)

    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_size = index_path.stat().st_size
    original_count = len(data["docs"])

    # Merge multiple section docs of the same page into a single doc.
    pages = {}  # location_base -> merged doc
    for doc in data["docs"]:
        loc = doc.get("location", "")
        base = loc.split("#")[0] if "#" in loc else loc
        title = doc.get("title", "").strip()
        tags = doc.get("tags", [])

        # Skip index/home pages.
        base_clean = base.rstrip("/")
        if base_clean == "" or base_clean.count("/") < 2 or base_clean.endswith("index"):
            continue

        if base not in pages:
            pages[base] = {
                "location": base if base else loc,
                "title": title,
                "text": _read_keywords(docs_dir, base),
                "_tags": set(),
            }
        else:
            merged = pages[base]
            if not merged["title"] and title:
                merged["title"] = title

        if tags:
            pages[base]["_tags"].update(tags)

    # text field: keep the page's keywords (set above from the source note) so
    # keyword search works; the full body is still dropped. Tags stay on the doc
    # for the MkDocs UI to render them, but are NOT pushed into the text field
    # (30K+ unique tags would blow up the inverted index). Bilingual search is
    # handled client-side.

    # Compute global tag frequency first, used to drop rare (paper-specific)
    # noise tags.
    tag_freq: Counter[str] = Counter()
    for doc in pages.values():
        for tag in doc["_tags"]:
            if not _CONF_PATTERN.match(tag.strip()):
                tag_freq[tag] += 1
    keep_tags = {t for t, c in tag_freq.items() if c >= MIN_TAG_FREQ}

    conf_removed = 0
    rare_removed = 0
    for doc in pages.values():
        all_tags = list(doc.pop("_tags"))
        # Filter out conference-name tags and rare tags.
        filtered_tags = []
        for tag in all_tags:
            if _CONF_PATTERN.match(tag.strip()):
                conf_removed += 1
            elif tag not in keep_tags:
                rare_removed += 1
            else:
                filtered_tags.append(tag)

        if filtered_tags:
            doc["tags"] = filtered_tags
        # text already holds the page's keywords (see _read_keywords above).

    data["docs"] = list(pages.values())

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    new_size = index_path.stat().st_size
    print(f"  ✓ search_index.json: {original_size/1024/1024:.2f} MB → {new_size/1024/1024:.2f} MB")
    print(f"    merged sections: {original_count} → {len(data['docs'])} docs")
    print(f"    removed {conf_removed} conference name tags (redundant with URL)")
    print(f"    removed {rare_removed} rare tag occurrences (freq < {MIN_TAG_FREQ})")
    print(f"    unique tags kept: {len(keep_tags)} (was {len(tag_freq)})")
    print(f"    bilingual search: handled client-side (no synonyms in index)")


if __name__ == "__main__":
    main()
