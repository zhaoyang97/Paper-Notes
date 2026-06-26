"""Post-build hook: slim down search_index.json for fast client-side search.

Why this matters
----------------
MkDocs' search plugin emits one doc per <h2> heading (~230k entries / 262MB on
this repo). Worse, every paper note shares the same Chinese section headings
("一句话总结", "研究背景与动机", "方法详解", ...), so if we naively concatenate
those into each paper's title field, those generic phrases end up in ~14k
documents. Lunr.js then spends forever building the inverted index and the
desktop UI sits on "正在初始化搜索引擎" forever.

Strategy
--------
  1. Collapse all section-level entries (#anchor) into a single doc per page.
  2. Keep **only the base page's title** — do NOT merge section headings into
     the title field. This is the single biggest win for lunr performance.
  3. Set each page's 'text' field to its **keywords only** (read from the
     note's `**关键词**:` / `**Keywords**:` line) — never the full body. This
     makes keyword search work (e.g. a paper whose title omits "diffusion" but
     is tagged so) while costing only ~1.8 MB over title-only, vs the ~250 MB
     a full-body index would weigh.
  4. Preserve the first doc's `tags` so tag-based filtering still works.
  5. Filter low-frequency tags (freq < MIN_TAG_FREQ) — long-tail paper-specific
     tags explode the lunr token table without aiding discoverability.
  6. Drop conference-name tags (the conference is already in the URL path and
     the location field is indexed separately).
  7. Skip conference-root / section-index stubs (they have huge nav-derived
     titles that bloat the index without helping search).
"""

import json
import os
import re
from collections import Counter, OrderedDict

_CONF_PATTERN = re.compile(
    r"^(ICLR|CVPR|ACL|NeurIPS|AAAI|ECCV|ICCV|ICML|EMNLP|NAACL)\s*\d{4}$",
    re.IGNORECASE,
)

MIN_TAG_FREQ = 5

# Body metadata line listing a paper's keywords, e.g.
#   **关键词**: diffusion model, image editing      (zh notes)
#   **Keywords**: diffusion model, image editing    (en notes)
_KW_PATTERN = re.compile(r"\*\*(?:关键词|Keywords)\*\*\s*[:：]\s*(.+)")


def _is_index_page(base: str) -> bool:
    """Conference root / section index pages: e.g. 'AAAI2026/', 'AAAI2026/3d_vision/'.

    Paper pages have the form '<CONF>/<AREA>/<slug>/' (>=3 path segments).
    """
    clean = base.strip("/")
    if not clean:
        return True
    return clean.count("/") < 2


def _read_keywords(docs_dir: str, base: str) -> str:
    """Return the space-joined keywords for a page, read from its source note.

    ``base`` is the directory-URL location (e.g. ``CVPR2026/3d_vision/slug/``);
    the source markdown is ``<docs_dir>/CVPR2026/3d_vision/slug.md``. Only the
    file head is scanned since the keyword line lives in the metadata block.
    """
    rel = base.strip("/")
    if not rel:
        return ""
    src = os.path.join(docs_dir, rel + ".md")
    try:
        with open(src, encoding="utf-8") as f:
            head = f.read(2000)
    except OSError:
        return ""
    m = _KW_PATTERN.search(head)
    if not m:
        return ""
    kws = [k.strip() for k in m.group(1).split(",") if k.strip()]
    # Comma-join (not space-join) so multi-word keywords keep their boundaries
    # for chip rendering on /search/. Search is unaffected: the tokenizers on
    # both lunr and the /search/ page split on punctuation + whitespace alike.
    return ", ".join(kws)


def on_post_build(config, **kwargs):
    index_path = os.path.join(config["site_dir"], "search", "search_index.json")
    if not os.path.exists(index_path):
        return

    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = data.get("docs", [])
    if not docs:
        return

    original_count = len(docs)
    original_size = os.path.getsize(index_path)

    docs_dir = config["docs_dir"]

    # ── Pass 1: collapse section entries, keep only the base page title ──
    pages: "OrderedDict[str, dict]" = OrderedDict()
    for doc in docs:
        loc = doc.get("location", "")
        base = loc.split("#")[0]

        if _is_index_page(base):
            continue

        if base not in pages:
            pages[base] = {
                "location": base,
                "title": doc.get("title", "").strip(),
                "text": _read_keywords(docs_dir, base),
                "_tags": set(doc.get("tags") or []),
            }
        else:
            # Subsequent section entry: only collect tags, never merge titles.
            tags = doc.get("tags") or []
            if tags:
                pages[base]["_tags"].update(tags)

    # ── Pass 2: tag frequency filtering ──
    tag_freq: "Counter[str]" = Counter()
    for page in pages.values():
        for tag in page["_tags"]:
            if _CONF_PATTERN.match(tag.strip()):
                continue
            tag_freq[tag] += 1
    keep_tags = {t for t, c in tag_freq.items() if c >= MIN_TAG_FREQ}

    conf_removed = 0
    rare_removed = 0
    for page in pages.values():
        raw_tags = page.pop("_tags")
        kept = []
        for tag in raw_tags:
            if _CONF_PATTERN.match(tag.strip()):
                conf_removed += 1
            elif tag not in keep_tags:
                rare_removed += 1
            else:
                kept.append(tag)
        if kept:
            page["tags"] = kept

    slim_docs = list(pages.values())
    data["docs"] = slim_docs

    kw_pages = sum(1 for p in slim_docs if p.get("text"))

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    new_size = os.path.getsize(index_path)
    print(
        f"[slim_search_index] {original_count} -> {len(slim_docs)} docs "
        f"({kw_pages} with keywords), "
        f"{original_size / 1024 / 1024:.1f} MB -> {new_size / 1024 / 1024:.1f} MB; "
        f"tags kept={len(keep_tags)} (dropped {rare_removed} rare, {conf_removed} conf)"
    )
