#!/usr/bin/env python3
"""
后处理 mkdocs build 生成的 search_index.json，截断过长的文本以优化手机端搜索性能。
用法: python scripts/trim_search_index.py [site_dir]
"""
import json
import sys
from pathlib import Path

MAX_TEXT_LEN = 800  # 每个文档最多保留 800 字符的搜索文本

def main():
    site_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("site")
    index_path = site_dir / "search" / "search_index.json"

    if not index_path.exists():
        print(f"ERROR: {index_path} not found")
        sys.exit(1)

    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_size = index_path.stat().st_size
    original_count = len(data["docs"])

    # 将同一页面的多个 section 合并为一个文档
    # mkdocs 会把每个 <h2> 拆成独立 doc，导致 32K+ 条目
    pages = {}  # location_base -> merged doc
    for doc in data["docs"]:
        loc = doc.get("location", "")
        # 提取页面基础路径（去掉 #anchor）
        base = loc.split("#")[0] if "#" in loc else loc
        title = doc.get("title", "").strip()
        text = doc.get("text", "").strip()

        if base not in pages:
            # 第一次见到这个页面（通常是页面级条目）
            pages[base] = {
                "location": base if base else loc,
                "title": title,
                "text": text,
            }
        else:
            # 合并 section 文本到页面级条目
            merged = pages[base]
            # 如果页面级 title 为空但 section 有 title，用 section 的
            if not merged["title"] and title:
                merged["title"] = title
            # 追加 section 标题和文本
            extra = ""
            if title:
                extra += " " + title
            if text:
                extra += " " + text
            merged["text"] += extra

    # 截断过长文本
    truncated = 0
    for doc in pages.values():
        if len(doc["text"]) > MAX_TEXT_LEN:
            doc["text"] = doc["text"][:MAX_TEXT_LEN]
            truncated += 1

    data["docs"] = list(pages.values())

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    new_size = index_path.stat().st_size
    print(f"  ✓ search_index.json: {original_size/1024/1024:.2f} MB → {new_size/1024/1024:.2f} MB")
    print(f"    merged sections: {original_count} → {len(data['docs'])} docs")
    print(f"    truncated {truncated} docs (max {MAX_TEXT_LEN} chars)")


if __name__ == "__main__":
    main()
