#!/usr/bin/env python3
"""Build the complete, mode-aware tag normalization registry.

Every observed non-conference tag is recorded. Only high-confidence relations
are emitted as searchable aliases:
  * format variants and bilingual translations: rough + exact search
  * abbreviations and their expanded forms: exact search only
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import yaml


CONF_RE = re.compile(
    r"^(?:ICLR|CVPR|ACL|NeurIPS|AAAI|ECCV|ICCV|ICML|EMNLP|NAACL)\s*\d{4}"
    r"(?:\s*(?:\([^)]*\)|\*{0,2}(?:oral|spotlight|highlight|findings)\*{0,2}))?$",
    re.IGNORECASE,
)
PAREN_RE = re.compile(r"^\s*(.*?)\s*[（(]\s*([^()（）]+)\s*[）)]\s*$")

# Manually reviewed full-phrase translations found in bilingual composite tags.
# Terms not listed here remain covered identity tags; they are not guessed into
# a relation merely because a generated tag contains parentheses.
APPROVED_INLINE_TRANSLATIONS = {
    "三元角色", "中训练", "人类不确定性", "余预层", "保形推断", "光子堆积",
    "公共补救", "共形预测", "内在维度", "协同", "可交换性", "可被利用性",
    "回归规划", "固定预算", "场景元像", "场景图", "多重校准", "学习优化器",
    "对手塑形", "应急规划", "弃答", "微缩放", "总相关性", "拒答", "拒绝",
    "提升式规划", "早退出", "时态越狱", "暗黑模式", "来源追踪", "泛函图",
    "流匹配", "流映射", "溯因", "点图", "空间推理", "简洁性", "箭图变异",
    "簇代数", "组合性", "自提示", "规范理论", "评分手册", "调制", "跨形态",
    "适配器", "逆向缩放", "锐界", "镶嵌", "频率带",
}

# Manually reviewed abbreviation/full-name pairs. The whitelist is deliberately
# conservative: ambiguous short forms such as CV, VQA, ERP, ERM and GPR remain
# identity tags instead of creating symmetric false-positive expansions.
APPROVED_INLINE_ABBREVIATIONS = {
    "IRM": "不变风险最小化", "CLT": "中心极限定理", "SBI": "仿真推断",
    "CSI": "信道状态信息", "WSI": "全切片图像",
    "FHE": "全同态加密", "EQA": "具身问答", "CCA": "典型相关分析",
    "IGB": "初始猜测偏见", "DOA": "到达方向",
    "DMFT": "动力学平均场理论", "DTR": "动态治疗策略", "BSH": "包围球层级",
    "SDP": "半定规划", "SPML": "单正标签多标签学习", "PEFT": "参数高效微调",
    "R2V": "参考图到视频", "RLVR": "可验证奖励强化学习",
    "GIB": "图信息瓶颈", "MIL": "多示例学习", "LTSM": "大时序模型",
    "LVLM": "大视觉语言模型", "LSH": "局部敏感哈希", "UDC": "屏下相机",
    "RPT": "强化后训练", "RFT": "强化微调",
    "ToM": "心智理论", "IMU": "惯性测量单元", "AST": "抽象语法树",
    "CTTA": "持续测试时适应", "CBF": "控制障碍函数",
    "MLM": "掩码语言建模", "DDE": "时滞微分方程", "MEI": "最激励输入",
    "ZPD": "最近发展区", "FSQ": "有限标量量化", "EIG": "期望信息增益",
    "EJOP": "期望雅可比外积", "CMI": "条件互信息",
    "DNF": "析取范式", "LUT": "查找表",
    "RACG": "检索增强代码生成", "FSI": "流固耦合", "AXp": "溯因解释",
    "PSF": "点扩散函数", "EVaR": "熵风险值", "GSF": "眩光扩散函数",
    "DCT": "离散余弦变换", "DWT": "离散小波变换",
    "ETF": "等角紧框架", "MoLDS": "线性动态系统混合", "LTL": "线性时序逻辑",
    "UMR": "统一意义表示", "EBM": "能量模型", "ACA": "自治编码代理",
    "MCTS": "蒙特卡洛树搜索", "VTON": "虚拟试穿", "DPP": "行列式点过程",
    "VWFA": "视觉词形区", "VTS": "视频转语音",
    "CUA": "计算机使用智能体", "PTQ": "训练后量化", "MTF": "调制传递函数",
    "CroCo": "跨视角补全", "BRep": "边界表示", "IRLS": "迭代重加权最小二乘",
    "GLU": "门控线性单元",
    "INR": "隐式神经表示", "UOT": "非平衡最优传输", "NLOS": "非视距成像",
    "IRT": "项目反应理论",
    "BEV": "鸟瞰图",
}

# Individually reviewed English expansions for the exact-only abbreviation
# relations.  These are deliberately not enabled in rough search: expanding a
# short form such as JEPA into every word of its full name would create broad
# false positives.  The names are kept here rather than inferred from prose so
# the generated registry remains reproducible and auditable.
APPROVED_ABBREVIATION_FULL_NAMES = {
    "JEPA": ["Joint Embedding Predictive Architecture", "Joint Embedding Predictive Architectures"],
    "IRM": ["Invariant Risk Minimization"],
    "CLT": ["Central Limit Theorem"],
    "HOI": ["Human-Object Interaction"],
    "SBI": ["Simulation-Based Inference"],
    "CSI": ["Channel State Information"],
    "FNR": ["False Negative Rate"],
    "WSI": ["Whole Slide Image", "Whole-Slide Image"],
    "FHE": ["Fully Homomorphic Encryption"],
    "EQA": ["Embodied Question Answering"],
    "CCA": ["Canonical Correlation Analysis"],
    "OOD": ["Out-of-Distribution", "Out of Distribution"],
    "IGB": ["Initial Guess Bias"],
    "DOA": ["Direction of Arrival"],
    "DMFT": ["Dynamical Mean-Field Theory", "Dynamical Mean Field Theory"],
    "DTR": ["Dynamic Treatment Regime", "Dynamic Treatment Strategy"],
    "BSH": ["Bounding Sphere Hierarchy"],
    "SDP": ["Semidefinite Programming"],
    "SPML": ["Single Positive Multi-Label Learning", "Single-Positive Multi-Label Learning"],
    "PEFT": ["Parameter-Efficient Fine-Tuning", "Parameter Efficient Fine Tuning"],
    "R2V": ["Reference-to-Video", "Reference to Video"],
    "RLVR": ["Reinforcement Learning with Verifiable Rewards"],
    "GIB": ["Graph Information Bottleneck"],
    "MIL": ["Multiple Instance Learning"],
    "LTSM": ["Large Time-Series Model", "Large Time Series Model"],
    "LVLM": ["Large Vision-Language Model", "Large Vision Language Model"],
    "LSH": ["Locality-Sensitive Hashing", "Locality Sensitive Hashing"],
    "UDC": ["Under-Display Camera", "Under Display Camera"],
    "RPT": ["Reinforcement Post-Training", "Reinforcement Post Training"],
    "RFT": ["Reinforcement Fine-Tuning", "Reinforcement Fine Tuning"],
    "ToM": ["Theory of Mind"],
    "IMU": ["Inertial Measurement Unit"],
    "AST": ["Abstract Syntax Tree"],
    "CTTA": ["Continual Test-Time Adaptation", "Continual Test Time Adaptation"],
    "CBF": ["Control Barrier Function"],
    "MLM": ["Masked Language Modeling", "Masked Language Modelling"],
    "DDE": ["Delay Differential Equation"],
    "MEI": ["Most Exciting Input"],
    "ZPD": ["Zone of Proximal Development"],
    "FSQ": ["Finite Scalar Quantization", "Finite Scalar Quantisation"],
    "EIG": ["Expected Information Gain"],
    "EJOP": ["Expected Jacobian Outer Product"],
    "CMI": ["Conditional Mutual Information"],
    "DNF": ["Disjunctive Normal Form"],
    "LUT": ["Look-Up Table", "Lookup Table"],
    "RACG": ["Retrieval-Augmented Code Generation", "Retrieval Augmented Code Generation"],
    "FSI": ["Fluid-Structure Interaction", "Fluid Structure Interaction"],
    "AXp": ["Abductive Explanation"],
    "PSF": ["Point Spread Function"],
    "EVaR": ["Entropic Value-at-Risk", "Entropic Value at Risk"],
    "DPO": ["Direct Preference Optimization", "Direct Preference Optimisation"],
    "Real-ISR": ["Real-World Image Super-Resolution", "Real World Image Super Resolution"],
    "GSF": ["Glare Spread Function"],
    "DCT": ["Discrete Cosine Transform"],
    "DWT": ["Discrete Wavelet Transform"],
    "ETF": ["Equiangular Tight Frame"],
    "MoLDS": ["Mixture of Linear Dynamical Systems"],
    "LTL": ["Linear Temporal Logic"],
    "UMR": ["Unified Meaning Representation"],
    "EBM": ["Energy-Based Model", "Energy Based Model"],
    "ACA": ["Autonomous Coding Agent", "Autonomous Code Agent"],
    "MCTS": ["Monte Carlo Tree Search"],
    "VTON": ["Virtual Try-On", "Virtual Try On"],
    "DPP": ["Determinantal Point Process"],
    "VWFA": ["Visual Word Form Area"],
    "VTS": ["Video-to-Speech", "Video to Speech"],
    "CUA": ["Computer-Using Agent", "Computer Using Agent"],
    "PTQ": ["Post-Training Quantization", "Post Training Quantization", "Post-Training Quantisation"],
    "ELBO": ["Evidence Lower Bound"],
    "MTF": ["Modulation Transfer Function"],
    "CroCo": ["Cross-View Completion", "Cross View Completion"],
    "BRep": ["Boundary Representation"],
    "IRLS": ["Iteratively Reweighted Least Squares"],
    "GLU": ["Gated Linear Unit"],
    "INR": ["Implicit Neural Representation"],
    "UOT": ["Unbalanced Optimal Transport"],
    "NLOS": ["Non-Line-of-Sight Imaging", "Non Line of Sight Imaging"],
    "IRT": ["Item Response Theory"],
    "BEV": ["Bird's-Eye View", "Bird's Eye View"],
}


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = value.replace("_", " ").replace("‐", "-").replace("‑", "-").replace("–", "-")
    value = re.sub(r"[^\w\u4e00-\u9fff]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def read_index(path: Path) -> Counter[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Counter(
        str(tag).strip()
        for doc in data.get("docs", [])
        for tag in (doc.get("tags") or [])
        if str(tag).strip() and not CONF_RE.match(str(tag).strip())
    )


def read_markdown(docs_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in docs_dir.rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.startswith("---"):
            continue
        end = text.find("\n---", 3)
        if end < 0:
            continue
        try:
            meta = yaml.safe_load(text[3:end]) or {}
        except yaml.YAMLError:
            continue
        tags = meta.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        for tag in tags:
            tag = str(tag).strip()
            if tag and not CONF_RE.match(tag):
                counts[tag] += 1
    return counts


def extract_builtin_translations(template: Path) -> list[tuple[str, str]]:
    text = template.read_text(encoding="utf-8")
    match = re.search(r"var P\s*=\s*\[(.*?)\];", text, re.S)
    if not match:
        return []
    pairs = []
    for value in re.findall(r'"([^"\n]+\|[^"\n]+)"', match.group(1)):
        left, right = value.split("|", 1)
        pairs.append((left.strip(), right.strip()))
    return pairs


def is_latin_phrase(value: str) -> bool:
    return bool(re.search(r"[A-Za-z]", value)) and not bool(re.search(r"[\u4e00-\u9fff]", value))


def is_cjk_phrase(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", value)) and not bool(re.search(r"[A-Za-z]", value))


def is_abbreviation(value: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]", "", value)
    return bool(compact) and len(compact) <= 12 and " " not in value.strip() and (
        compact.upper() == compact or re.match(r"^[A-Z][A-Za-z]*[A-Z0-9][A-Za-z0-9]*$", compact)
    )


def make_rule(canonical: str, aliases: list[str], relation: str, modes: list[str], note: str = "") -> dict:
    seen = {canonical.strip()}
    clean = []
    for alias in aliases:
        alias = alias.strip()
        if alias and alias not in seen:
            seen.add(alias)
            clean.append(alias)
    rule = {"canonical": canonical.strip(), "aliases": clean, "relation": relation, "modes": modes}
    if note:
        rule["note"] = note
    return rule


def build(counts: Counter[str], docs_dir: Path, template: Path) -> dict:
    rules: list[dict] = []

    # Safe surface-form groups: case, Unicode width, underscore, hyphen and punctuation.
    format_groups: dict[str, list[str]] = defaultdict(list)
    for tag in counts:
        format_groups[norm(tag)].append(tag)
    for terms in format_groups.values():
        if len(terms) < 2:
            continue
        terms.sort(key=lambda term: (-counts[term], len(term), term.casefold(), term))
        rules.append(make_rule(terms[0], terms[1:], "format", ["rough", "exact"]))

    # Existing manually reviewed rules remain the authority for named concepts.
    curated_path = docs_dir / "tag_aliases.json"
    curated = json.loads(curated_path.read_text(encoding="utf-8"))
    for item in curated.get("rules", []):
        aliases = list(item.get("aliases", []))
        if item.get("relation") == "abbreviation":
            aliases.extend(APPROVED_ABBREVIATION_FULL_NAMES.get(item["canonical"], []))
            # Some curated rules use the Chinese expansion as canonical and
            # the abbreviation as an alias.
            aliases.extend(
                name
                for abbreviation, names in APPROVED_ABBREVIATION_FULL_NAMES.items()
                if abbreviation in aliases
                for name in names
            )
        rules.append(
            make_rule(
                item["canonical"],
                aliases,
                item.get("relation", "translation"),
                item.get("modes", ["rough", "exact"]),
                item.get("note", ""),
            )
        )

    # The site's established Chinese-English dictionary is explicit translation evidence.
    for zh, en in extract_builtin_translations(template):
        rules.append(make_rule(zh, [en], "translation", ["rough", "exact"]))

    # Explicit bilingual full-phrase tags are safe translations. Explicit short forms are
    # admitted only when the abbreviation has one unambiguous expansion in this corpus.
    abbreviation_candidates: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for tag in counts:
        match = PAREN_RE.match(tag)
        if not match:
            continue
        left, right = match.group(1).strip(), match.group(2).strip()
        if is_cjk_phrase(left) and is_latin_phrase(right):
            if is_abbreviation(right):
                abbreviation_candidates[norm(right)].append((left, tag))
            elif left in APPROVED_INLINE_TRANSLATIONS:
                rules.append(make_rule(left, [right, tag], "translation", ["rough", "exact"]))
        elif (
            is_latin_phrase(left)
            and is_cjk_phrase(right)
            and not is_abbreviation(left)
            and right in APPROVED_INLINE_TRANSLATIONS
        ):
            rules.append(make_rule(right, [left, tag], "translation", ["rough", "exact"]))

    for candidates in abbreviation_candidates.values():
        expansions = {norm(left) for left, _ in candidates}
        if len(expansions) != 1:
            continue
        left = candidates[0][0]
        abbreviation = PAREN_RE.match(candidates[0][1]).group(2).strip()
        if APPROVED_INLINE_ABBREVIATIONS.get(abbreviation) != left:
            continue
        composite_tags = [tag for _, tag in candidates]
        rules.append(
            make_rule(
                left,
                [abbreviation] + composite_tags + APPROVED_ABBREVIATION_FULL_NAMES.get(abbreviation, []),
                "abbreviation",
                ["exact"],
                "Corpus-explicit abbreviation with one unambiguous expansion.",
            )
        )

    # Deduplicate identical relation edges while retaining separate scope when needed.
    unique_rules = []
    seen_rules = set()
    for rule in rules:
        signature = (
            norm(rule["canonical"]),
            tuple(sorted(norm(a) for a in rule["aliases"])),
            rule["relation"],
            tuple(rule["modes"]),
        )
        if not rule["aliases"] or signature in seen_rules:
            continue
        seen_rules.add(signature)
        unique_rules.append(rule)
    unique_rules.sort(key=lambda r: (r["relation"], norm(r["canonical"]), r["canonical"]))

    covered_by_relation = set()
    for rule in unique_rules:
        covered_by_relation.add(norm(rule["canonical"]))
        covered_by_relation.update(norm(alias) for alias in rule["aliases"])

    tags = [
        {
            "term": term,
            "normalized": norm(term),
            "count": count,
            "status": "related" if norm(term) in covered_by_relation else "identity",
        }
        for term, count in sorted(counts.items(), key=lambda item: (norm(item[0]), item[0]))
    ]

    relation_counts = Counter(rule["relation"] for rule in unique_rules)
    return {
        "version": 2,
        "generated": date.today().isoformat(),
        "policy": {
            "translation": ["rough", "exact"],
            "format": ["rough", "exact"],
            "abbreviation": ["exact"],
            "identity": ["rough", "exact"],
        },
        "stats": {
            "observed_non_conference_tags": len(tags),
            "tag_occurrences": sum(counts.values()),
            "relation_rules": len(unique_rules),
            "relation_counts": dict(sorted(relation_counts.items())),
            "identity_tags": sum(tag["status"] == "identity" for tag in tags),
        },
        "rules": unique_rules,
        "tags": tags,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, help="Optional built search index; source Markdown is the default")
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--template", type=Path, default=Path("overrides/main.html"))
    parser.add_argument("--output", type=Path, default=Path("docs/tag_registry.json"))
    parser.add_argument("--runtime-output", type=Path, default=Path("docs/tag_search_rules.json"))
    args = parser.parse_args()

    counts = read_index(args.index) if args.index and args.index.exists() else read_markdown(args.docs)
    registry = build(counts, args.docs, args.template)
    args.output.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    runtime = {
        "version": registry["version"],
        "generated": registry["generated"],
        "policy": registry["policy"],
        "stats": registry["stats"],
        "rules": registry["rules"],
    }
    args.runtime_output.write_text(
        json.dumps(runtime, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps(registry["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
