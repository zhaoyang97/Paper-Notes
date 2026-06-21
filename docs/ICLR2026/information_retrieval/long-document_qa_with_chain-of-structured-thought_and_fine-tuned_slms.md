---
title: >-
  [论文解读] Long-Document QA with Chain-of-Structured-Thought and Fine-Tuned SLMs
description: >-
  [ICLR 2026][信息检索/RAG][长文档 QA] LiteCoST 用强 LLM 把"长文档 QA"重写成"先抽结构再答题"的可审计轨迹，再用 SFT→GRPO 双信号把这种结构优先行为蒸馏进 3B/7B 小模型，让小模型在金融/法律/科研长文档 QA 上逼平 GPT-4o，同时延迟降低 2–4 倍。
tags:
  - "ICLR 2026"
  - "信息检索/RAG"
  - "长文档 QA"
  - "结构化思维链 (CoST)"
  - "小语言模型"
  - "SFT"
  - "GRPO"
  - "结构化输出"
---

# Long-Document QA with Chain-of-Structured-Thought and Fine-Tuned SLMs

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=faECRsdRav](https://openreview.net/forum?id=faECRsdRav)  
**代码**: [https://github.com/HKUSTDial/LiteCoST](https://github.com/HKUSTDial/LiteCoST)  
**领域**: 信息检索 / 长文档问答 / 结构化信息抽取  
**关键词**: 长文档 QA, 结构化思维链 (CoST), 小语言模型, SFT, GRPO, 结构化输出  

## 一句话总结
LiteCoST 用强 LLM 把"长文档 QA"重写成"先抽结构再答题"的可审计轨迹，再用 SFT→GRPO 双信号把这种结构优先行为蒸馏进 3B/7B 小模型，让小模型在金融/法律/科研长文档 QA 上逼平 GPT-4o，同时延迟降低 2–4 倍。

## 研究背景与动机
**领域现状**：LLM 越来越多被用于文档数据分析，但直接在长、噪声大、多源的文档上推理既脆弱又不透明——在金融、法律这类高风险场景里容易漏掉证据、产生幻觉、格式漂移。一条被验证有效的路线是"先把分散证据固化成结构化数据（表/图/分块），再从结构里推导答案"，因为结构让证据可见、可核验、可复用。

**现有痛点**：(1) 证据分散在跨文档的长上下文里，直接 prompt 容易遗漏或幻觉；(2) 数值的单位/格式异构，需要归一化；(3) 长上下文推理要在整个结构上保持一致。直接让 GPT-4 / DeepSeek-R1 输出结构化产物虽然准，但**反复调用大模型 = 高 token/算力成本 + 高延迟 + 低吞吐**，还把敏感数据送上托管 API 带来隐私风险。

**核心矛盾**：换成可本地部署的小模型（SLM）能省成本、保隐私，但**现成 SLM 恰恰缺乏 CoST 所需的技能**——长上下文的 schema 感知抽取、单位/实体归一化、记录对齐、分步一致的序列化，所以朴素的 LLM→SLM 直接替换行不通。

**本文目标**：在"准确可验证"（G1）和"小模型低延迟"（G2）之间取得平衡，让 SLM 生成的结构 $S_{SLM}$ 对答题和 LLM 生成的 $S_{LLM}$ 一样有用，但延迟远低：$\text{LLM}(Q, S_{SLM}) \approx \text{LLM}(Q, S_{LLM})$ 且 $\text{Latency}(S_{SLM}) \ll \text{Latency}(S_{LLM})$。

**核心 idea**：**结构优先蒸馏**——先用强 LLM 当一次性"结构优先轨迹生成器"产出可审计的 CoST 轨迹 + 机器可校验的结构化输出（SSO），再用轻量两阶段（SFT→GRPO，带双层奖励）把这种 schema 感知的结构化推理灌进紧凑模型。

## 方法详解

### 整体框架
LiteCoST 是一个两大支柱（two-pillar）框架。**Stage A（CoST）**：强 LLM 以 CoST 模板为输入，对每个问题动态选结构、抽证据、归一化、对齐、序列化、自检自修，产出一条可审计的 CoST 轨迹（`<reasoning>`）和一个查询专属的序列化结构化输出 SSO（`<answer>`，表/图/分块），作为监督信号。**Stage B（SLM 微调）**：先 SFT 让小模型学会 schema/格式/分步纪律，再用 GRPO 配双层奖励（结果奖励 + 过程奖励）强化答案质量与推理一致性。最终 3B/7B 小模型内化"结构优先"行为，推理快且可审计。

```mermaid
flowchart LR
    Q[问题 Q + 长文档 D] --> T[CoST 模板]
    T --> A1[A1 结构分析<br/>选表/图/分块 + 动态 schema]
    A1 --> A2[A2 轨迹生成<br/>抽取/对齐/序列化]
    A2 --> A3[A3 质量验证<br/>LLM-as-Judge]
    A3 --> A4[A4 迭代精修<br/>Iterative Structuralizer]
    A4 --> DATA["(CoST 轨迹 c*, SSO S*)"]
    DATA --> SFT[Stage B-1: SFT<br/>结构/格式/分步对齐]
    SFT --> GRPO[Stage B-2: GRPO<br/>结果奖励 + 过程奖励]
    GRPO --> M[LiteCoST 小模型 3B/7B]
```

### 关键设计

**1. CoST 模板：把 QA 重写成"四步结构化"的可审计轨迹**——这是 Stage A 的核心。给定问题、文档、真值答案和 CoST 模板，强 LLM 走四步：(A1) *结构分析*先做面向问题的结构选择（统计比较选表、关系推理选图），再让 LLM 解析问题枚举任务相关的属性/实体（如 Company、Asset、Year）动态构建 schema，避免穷举整篇语料；(A2) *轨迹生成*在 schema 指导下分步抽取、对齐、序列化成确定性结构，同时吐出推理轨迹和最终结构；(A3) *质量验证*因为没有结构化真值，改用 LLM-as-Judge——用 GPT-4o 评估"结构能否答对原问题"，命中参考答案的样本才保留；(A4) *迭代精修*的核心是 Iterative Structuralizer，对低质量样本不丢弃，而是把它们连同问题/上下文递归重用、重构成"补充抽取"任务再生成，比普通微调提供更丰富的监督。最终产出 $(c^*, S^*)$ 这对高质量监督。

**2. SFT→GRPO 两阶段适配：先学结构纪律，再强化推理一致性**——Stage B 把 Stage A 的能力迁进 SLM。每个训练样本 $z=(i,d,c^*,y^*)$ 含问题、文档、CoST 轨迹和结构化输出。先做 SFT（LoRA，rank 16），让通用底座获得 CoT 驱动的信息抽取基本功，缓解直接部署时的抽取错误；再用 GRPO 做 RL。GRPO 对每个问题从旧策略采一组输出 $\{o_1,\dots,o_G\}$，按组内相对优势 $A_i=\frac{r_i-\text{mean}(r)}{\text{std}(r)}$ 优化目标 $J_{GRPO}$（带 clip 信任域和 $\beta D_{KL}$ 约束），无需价值网络即可稳定更新。

**3. 双层（结果 + 过程）奖励：把稀疏的"答对"细化到分步监督**——这是把结构化行为灌进小模型的关键奖励设计。*结果奖励*含两部分：**格式合规**用分层奖励——只有 `<reasoning>`+`<answer>` 且无冗余给软奖励 0.5，进一步有显式 Step 标签给硬奖励 1.0，否则 0；**答案正确**用混合度量 $f_{score}=\alpha\cdot S_{struct}+(1-\alpha)\cdot S_{sem}$（$\alpha=0.3$），$S_{struct}$ 用规则检查行列对齐，$S_{sem}$ 用 GPT-4o-mini 比对语义相似度。*过程奖励*针对"结果奖励太稀疏"的问题，从实体级和元组级判断每步 $s_i$ 是否与真值 $s_i^*$ 一致：$R_{process}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\text{Cons}(s_i,s_i^*\mid I_{consistency})]$，提供稠密的分步信号。总奖励是三者之和，并对过程奖励乘一个轨迹级系数 $\tilde{R}_{process}(s_i)=R_{process}(s_i)\cdot\gamma(T_i)$——正确轨迹取正值强化推理、错误/过度思考取负值抑制、格式错误取 1 把惩罚隔离到具体步骤。

## 实验关键数据

### 主实验表格（Loong 金融子集，AS=平均分 0–100，PR=完美率）

| 模型 | 规模 | Overall AS | Overall PR |
|------|------|-----------|-----------|
| LLaMA-3.2-3B (Base) | 3B | 49.37 | 0.11 |
| **LLaMA-LiteCoST (Ours)** | 3B | **76.95** (↑27.58) | **0.40** (↑0.29) |
| Qwen2-7B (Base) | 7B | 62.10 | 0.26 |
| **Qwen-LiteCoST (Ours)** | 7B | **79.93** (↑17.83) | **0.48** (↑0.22) |
| GPT-4o-mini | 8B | 78.08 | 0.51 |
| Qwen2.5-14B-Instruct | 14B | 75.60 | 0.38 |
| GPT-4o | 200B | 79.32 | 0.54 |
| DeepSeek-R1 | 671B | 78.18 | 0.46 |

Qwen-LiteCoST(7B) 的 Overall AS 超过 GPT-4o-mini(+1.85)、DeepSeek-R1(+1.75)、GPT-4o(+0.61)，用 7B 参数逼平/反超百倍量级大模型。

### 对比 SOTA 方法（Loong 金融子集 Overall AS / PR）

| 方法 | LLaMA-3B 底座 | Qwen-7B 底座 |
|------|--------------|--------------|
| StructRAG | 36.04 / 0.01 | 49.68 / 0.03 |
| Struc-bench | 49.90 / 0.11 | 73.72 / 0.44 |
| IEpile | 61.90 / 0.22 | 69.19 / 0.35 |
| **LiteCoST** | **76.95 / 0.40** | **79.93 / 0.48** |

相对最强基线 StructRAG 提升 (+30.91/+0.39) 和 (+30.47/+0.46)；相对最优微调方法分别 +15.05(over IEpile) 和 +6.41(over Strucbench)。

### 消融实验表格（奖励设计，Overall AS / PR）

| 配置 | LLaMA-Ours | Qwen-Ours |
|------|-----------|-----------|
| 完整 | 76.95 / 0.40 | 79.93 / 0.48 |
| w/o Process Reward | 75.52 / 0.37 | 77.39 / 0.46 |
| w/o Outcome Reward | 72.55 / 0.32 | 75.43 / 0.39 |

去掉任一奖励都掉点，去掉结果奖励掉得更狠（LLaMA 掉 4.4 分），说明结果与过程奖励互补。

### 关键发现
- **结构化数据普遍提升 LLM 推理**：用 LiteCoST 生成的 SSO 替代原始长文档后，Qwen2-72B/GPT-4o-mini/GPT-4o/Claude-3.5 的 Overall 分别 +12.41/+8.77/+9.04/+8.47，完美率同步上升。
- **效率**：Qwen-LiteCoST 单样本延迟 12.09s，低于 LLaMA-3.1-8B(13.19s) 和 Qwen2.5-14B(14.71s)，比 GPT-4o(21.15s) 快约 2×、比 DeepSeek-R1(44.44s) 快约 4×；要更快可用 LLaMA-LiteCoST(8.04s)。
- **低成本**：两阶段训练（LoRA 3 epoch + GRPO）总成本约 \$20，最大生成长度 2048 token。

## 亮点与洞察
- **把"答题"换成"先造结构再答题"**：用结构化中间产物当接口，天然带来证据可见、可核验、可复用，比直接长上下文推理更鲁棒，也把幻觉/格式漂移压下去。
- **强 LLM 只调用一次**：把昂贵的大模型当"一次性教师"产出可审计监督，而非在线推理依赖，从根上解决成本/延迟/隐私三难。
- **过程奖励填补 RL 稀疏信号**：实体级 + 元组级的分步一致性奖励，加上轨迹级正负缩放系数，把"只看最终答对"的稀疏监督细化成稠密的分步引导，这是小模型能逼平大模型的关键。

## 局限与展望
- **依赖 GPT-4o 当教师 + 评委**：CoST 数据生成、质量验证、语义评分都用 GPT-4o(-mini)，监督质量和评测都受其能力与偏见影响（LLM-as-Judge 的固有问题）。
- **只适用可结构化的问题**：方法显式排除开放式叙事类问题（不适合表/图表示），适用范围受限于"答案可从结构直接导出"的查询。
- **评测范围**：主分析集中在 Loong 的金融子集（法律/科研在附录），跨更多领域、更长上下文的稳健性仍需更广验证；2-hop 评测用下游 QA 间接衡量结构质量，存在间接性。
- **结构选择粒度**：结构类型限于表/图/分块，更复杂的混合结构或层级结构尚未覆盖。

## 相关工作与启发
- **QA-by-Structuring / 结构化 RAG**：与 StructRAG、GraphRAG（Edge et al.）一脉，都主张把分散证据固化成结构再推理；本文用蒸馏 + RL 把这条路线压进小模型。
- **CoT 与结构化推理**：CoST 把 Chain-of-Thought 升级成"schema 感知 + 可序列化"的结构化思维链，兼顾可读轨迹与机器可校验产物。
- **GRPO 蒸馏**：延续 DeepSeek 系 GRPO（无价值网络的组相对优势），创新点是设计了面向结构化抽取的双层（格式/答案/过程）奖励。
- **启发**：对任何"先抽取再推理"的任务（表格 QA、报表分析、知识图谱构建），"强模型造可审计轨迹 + 小模型双信号 RL 蒸馏"是一个可复用、低成本、可本地部署的范式。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — CoST 模板把 QA 重写成可审计的结构化轨迹，配双层（结果+过程）奖励的 SFT→GRPO 蒸馏，组合有新意；单个组件（结构化 RAG、GRPO、过程奖励）均有先例，故非满分。
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖效果/效率/消融/泛化多维度，含与 LLM 和 SOTA IE 方法的对比及延迟分析；主分析偏金融子集、评测依赖 LLM-as-Judge 略减分。
- **写作质量**: ⭐⭐⭐⭐ — 两支柱 + 四步 + 双层奖励的结构清晰，图表充分，动机递进流畅。
- **价值**: ⭐⭐⭐⭐ — 用 \$20 训练成本让 7B 小模型在长文档 QA 上逼平 GPT-4o 且快 2–4×、可本地部署保隐私，对金融/法律等高风险落地场景实用价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Rethinking Reasoning in Document Ranking: Why Chain-of-Thought Falls Short](rethinking_reasoning_in_document_ranking_why_chain-of-thought_falls_short.md)
- [\[ICLR 2026\] Fathom-DeepResearch: Unlocking Long Horizon Information Retrieval and Synthesis for SLMs](fathom-deepresearch_unlocking_long_horizon_information_retrieval_and_synthesis_f.md)
- [\[ACL 2026\] GIFT: Guided Fine-Tuning and Transfer for Enhancing Instruction-Tuned Language Models](../../ACL2026/information_retrieval/gift_guided_fine-tuning_and_transfer_for_enhancing_instruction-tuned_language_mo.md)
- [\[ACL 2026\] Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA](../../ACL2026/information_retrieval/navigating_large-scale_document_collections_mudabench_for_multi-document_analyti.md)
- [\[ICLR 2026\] G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)

</div>

<!-- RELATED:END -->
