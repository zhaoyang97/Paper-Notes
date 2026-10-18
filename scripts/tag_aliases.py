"""Load the manually reviewed tag/keyword correspondence table."""

import json
import re
from pathlib import Path


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("‐", "-").replace("‑", "-").strip().casefold())


def load_rules(docs_dir):
    path = Path(docs_dir) / "tag_aliases.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    seen = {}
    groups = []
    for rule in data.get("rules", []):
        canonical = str(rule["canonical"]).strip()
        terms = list(dict.fromkeys([canonical] + [str(alias).strip() for alias in rule.get("aliases", [])]))
        for term in terms:
            key = normalize(term)
            if key in seen and seen[key] != canonical:
                raise ValueError(f"Alias belongs to multiple rules: {term}")
            seen[key] = canonical
        modes = rule.get("modes", ["rough", "exact"])
        if not modes or any(mode not in {"rough", "exact"} for mode in modes):
            raise ValueError(f"Invalid search modes for alias rule: {canonical}")
        groups.append({
            "canonical": canonical,
            "terms": terms,
            "keys": {normalize(term) for term in terms},
            "relation": rule.get("relation", "translation"),
            "modes": modes,
        })
    return groups


def load_registry(docs_dir):
    path = Path(docs_dir) / "tag_registry.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 2:
        raise ValueError("tag_registry.json must use schema version 2")
    terms = {normalize(item["term"]) for item in data.get("tags", [])}
    if len(terms) != len(data.get("tags", [])):
        # Multiple spelling variants may normalize to the same lookup key; this
        # is expected only when the registry explicitly records a format rule.
        format_keys = {
            normalize(term)
            for rule in data.get("rules", [])
            if rule.get("relation") == "format"
            for term in [rule["canonical"], *rule.get("aliases", [])]
        }
        duplicates = []
        seen = set()
        for item in data.get("tags", []):
            key = normalize(item["term"])
            if key in seen and key not in format_keys:
                duplicates.append(item["term"])
            seen.add(key)
        if duplicates:
            raise ValueError(f"Unexplained normalized tag collisions: {duplicates[:5]}")
    return data, terms
