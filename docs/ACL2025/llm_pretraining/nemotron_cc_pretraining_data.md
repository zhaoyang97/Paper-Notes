---
title: >-
  [论文解读] Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset
description: >-
  [ACL 2025][预训练][Common Crawl] Nemotron-CC 通过分类器集成提升高质量 token 召回、合成数据改写扩展唯一 token 数量、对高质量数据取消启发式过滤三大策略，从 Common Crawl 构建了 6.3T token 的长周期预训练数据集（含 4.4T 唯一真实 token + 1.9T 合成 token），在 15T token 训练场景下使 8B 模型 MMLU 达 70.3，超越同规模训练的 Llama 3.1 8B 的 65.3。
tags:
  - "ACL 2025"
  - "预训练"
  - "Common Crawl"
  - "数据质量"
  - "分类器集成"
  - "合成数据"
  - "长周期训练"
---

# Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset

**会议**: ACL 2025  
**arXiv**: [2412.02595](https://arxiv.org/abs/2412.02595)  
**代码**: [NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator)  
**机构**: NVIDIA
**领域**: LLM Pretraining / 预训练数据  
**关键词**: Common Crawl, 数据质量, 分类器集成, 合成数据, 长周期训练

## 一句话总结

Nemotron-CC 通过分类器集成提升高质量 token 召回、合成数据改写扩展唯一 token 数量、对高质量数据取消启发式过滤三大策略，从 Common Crawl 构建了 6.3T token 的长周期预训练数据集（含 4.4T 唯一真实 token + 1.9T 合成 token），在 15T token 训练场景下使 8B 模型 MMLU 达 70.3，超越同规模训练的 Llama 3.1 8B 的 65.3。

## 研究背景与动机

**领域现状**: 英语 Common Crawl 预训练数据集（FineWeb-Edu、DCLM）通过激进的模型过滤（model-based filtering）显著提升了短训练周期下的基准分数，DCLM 的 7B 模型在 2.6T token 训练中取得了接近闭源模型的性能。

**现有痛点**: 激进过滤丢弃了约 90% 的数据——DCLM 仅含约 1T 唯一 token，FineWeb-Edu 仅 0.2T。在 Llama 3.1 的 15T token 长训练周期中，这意味着同一样本需要被看 15 次以上，而 Muennighoff et al. 指出超过 4 个 epoch 后重复数据的收益急剧递减。

**核心矛盾**: 数据质量 vs. 数据数量的 trade-off——高质量过滤提升了短期基准分但牺牲了长训练所需的数据多样性和唯一 token 总量。

**本文切入角度**: 不走"更激进过滤"的路线，而是通过三个互补策略同时提升质量和数量：(1) 多分类器集成扩大高质量 token 的召回率，(2) 合成改写创造新的唯一 token，(3) 对高质量数据跳过启发式过滤以避免误杀。

**核心 idea**: 用分类器集成 + 分层合成数据 + 选择性过滤，打破数据质量-数量的 trade-off 瓶颈。

## 方法详解

### 整体框架

Common Crawl 99 个 snapshot 的 HTML → **JusText 提取**（比 Trafilatura 多产出 28.6% HQ token）→ 英语语言过滤（pycld2 + FastText）→ 全局模糊去重 + 精确子串去重 → **三分类器集成打分**（每个分类器输出 0-19 分，取 max）→ **退火实验分级**（20 桶 → High/MH/M/ML/Low 五级）→ 高质量数据跳过启发式过滤，低质量数据保留过滤 → **分层合成数据生成**（低质量用 Wikipedia 风格改写，高质量用 4 种 prompt 多样化）→ 最终产出 6.3T token 数据集（4.4T 唯一真实 + 1.9T 合成）。

### 关键设计

**1. 分类器集成质量打分 (Classifier Ensembling + Quality Bucketing)**

- 构建三个异构质量分类器：① Nemotron-340B 标注的教育质量分类器，② Mixtral-8x22B 标注的教育质量分类器，③ DCLM 的 fastText 信息量分类器。三者关注维度不同（教育性 vs. 信息性），交集仅 10.1%，互补性强。
- 每个分类器对所有文档排序后映射到 0-19 整数桶（每桶约 5% 文档），取三者最大值作为最终质量分数。
- 设计动机：单一分类器的 HQ 召回率仅 8-14%，集成后达到 25%——这是打破长训练数据瓶颈的关键；同时通过退火实验将 20 桶重组为 5 级（在 70% 训完的 8B 模型上用 50B token 评估每个桶的下游表现），使质量标签直接对齐下游性能而非分类器分数。

**2. 分层合成数据生成 (Stratified Synthetic Data Generation)**

- **低质量数据**（Low 级，402B token）：用 Wikipedia 风格 prompt 改写，目标是减少噪声/错误并保留有用信息，产出 336B 合成 token。
- **高质量数据**（High 级，451B token）：用四种 prompt 生成多样化变体——① Diverse QA Pairs（多形式问答，500B token），② Extract Knowledge（知识提取，304B token），③ Distill（蒸馏浓缩，158B token），④ Knowledge List（结构化知识列表，203B token），合计 1.5T token。
- 生成模型：Mistral NeMo 12B-instruct (FP8)，使用 TensorRT-LLM 加速；长文档先按 token 限制分段再逐段生成，后处理去除不完整输出和特定格式。
- 设计动机：为 HQ 数据创造全新唯一 token 以避免多 epoch 边际递减，而非用 LLM 凭空生成知识（降低幻觉风险）。

**3. 选择性启发式过滤 (Selective Heuristic Filtering)**

- 传统做法对全部数据统一应用 C4/Gopher/KenLM PPL 过滤；本文发现这些过滤会移除 18.1% 的 HQ token（FineWeb-Edu 标准）。
- 提出仅对低质量桶应用启发式过滤，对高质量桶（模型分类器高分）完全跳过，实现 HQ token 产量 +57.4%（80B → 127B per 13 snapshots）。
- 消融实验证实这不仅不降质量，还使 MMLU 提升 +2（55.5 → 57.5）。

## 实验关键数据

### 主实验：8B 模型 1T token 训练对比

| 数据集 | 唯一真实 token | MMLU | ARC-C | Hellaswag | CSQA | 10 任务均分 |
|--------|-------------|------|-------|-----------|------|-----------|
| FineWeb-Edu | 0.2T | 42.9 | 48.0 | 70.7 | 30.0 | 53.2 |
| FineWeb-Edu-2 | 1.1T | 42.4 | 44.7 | 75.4 | 25.5 | 53.2 |
| DCLM | 1.0T | 53.4 | 47.0 | 76.3 | 44.1 | 57.0 |
| Nemotron-CC (6.3T) | 4.4T | 53.0 | 50.7 | 75.9 | 47.7 | 57.8 |
| **Nemotron-CC-HQ (1.1T)** | **0.6T** | **59.0** | **52.9** | **76.6** | **55.8** | **60.1** |

Nemotron-CC-HQ 在 1T 短训练中就超 DCLM **+5.6 MMLU**、**+3.1 均分**；完整 Nemotron-CC 与 DCLM 质量持平但唯一 token 是其 **4 倍**。

### 长周期训练：8B 模型 15T token

| 模型 | MMLU | ARC-C | Hellaswag | Winogrande | CSQA | 10 任务均分 |
|------|------|-------|-----------|------------|------|-----------|
| Llama 3.1 8B | 65.3 | 55.0 | 79.3 | 74.7 | 70.6 | 64.2 |
| **Nemotron-CC 8B** | **70.3** | **58.1** | **80.8** | 73.8 | 69.9 | **64.7** |

15T 训练中 Nemotron-CC 在 MMLU 上领先 Llama 3.1 达 **+5.0**，验证了更多唯一 token 在长训练中的决定性优势。

### 消融实验：各模块贡献

| 消融对比 | 关键结论 |
|---------|---------|
| 提取器：JusText vs. Trafilatura | JusText 多产 38.8% 总 token、28.6% HQ token，下游性能无损 |
| 过滤：HQ 不过滤 vs. 全过滤 | HQ 不过滤使 MMLU +2.0（55.5→57.5），HQ token 产量 +57.4% |
| 分类器：单一 vs. 集成 | 集成将 HQ 比例从 8-14% 提升到 25%，均分最高（59.4） |
| 合成数据：LQ 改写 | 低质量改写使均分 +1.5（52.5→54.0） |
| 合成数据：HQ 多样化 | 用合成替代 4/8 重复 epoch，均分 +0.9（55.8→56.7） |

### 分类器互补性分析

| 分类器组合 | 文档数 | 占高质量联合集比例 |
|-----------|--------|------------------|
| 两分类器交集 | 1.15M | 10.1% |
| 仅 FineWeb-Edu 独有 | 4.02M | 35.4% |
| 仅 DCLM 独有 | 6.18M | 54.4% |
| 联合集（总计） | 11.36M | 100% |

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|-----------|------|
| 实用性 | 9 | 数据集完全开源，质量分级发布，NeMo-Curator 开源复现；直接可用于大规模预训练 |
| 创新性 | 7 | 单项技术（集成/改写/过滤）非全新，但三者的系统组合和"选择性过滤"思路有启发 |
| 实验充分度 | 9 | 1T/15T 两个训练周期对比，提取器/过滤器/分类器/合成数据四维消融，10 个基准覆盖全面 |
| 可复现性 | 9 | 数据集、分类器、代码库均已公开，训练细节完整（超参见附录 D），唯一门槛是计算资源 |

## 亮点与不足

**亮点**:

- "从静态启发式管线转向学习型飞轮"的理念有前瞻性——模型越强 → 数据质量越好 → 模型更强
- 退火实验直接用下游性能定义质量分级，而非依赖分类器分数的绝对值
- 合成数据不生成新知识而是改写/蒸馏/多样化现有内容，降低幻觉风险
- HQ 数据跳过启发式过滤这一"反直觉"策略，简洁但有效
- 数据集按质量级别 + 合成类型拆分发布，方便社区做课程学习实验

**不足**:

- 仅覆盖英语 Common Crawl，多语言扩展未涉及
- 合成数据用 12B 模型生成且未做事实准确性验证，更强模型 + 事实检查可能进一步提升
- 15T 训练仅在 8B 模型上验证，对 70B+ 模型的可迁移性存疑
- 未做数据集去污染（decontamination），虽然对比基线也未做但仍是潜在变量
- Medium 级数据未生成合成数据（资源约束），存在进一步提升空间
- 去重策略可能仍然不够激进，存在语义级近似重复

## 相关工作与启发
- **vs DCLM**: DCLM 用单一 fastText 分类器，仅保留约 10% 数据；Nemotron-CC 用三分类器集成保留更多高质量数据
- **vs FineWeb-Edu**: FineWeb-Edu 激进过滤到 0.2T 唯一 token，Nemotron-CC 保留 4.4T
- **vs DSIR/QuRating**: 专注数据选择但不做合成扩展，Nemotron-CC 同时做选择和合成

## 评分
- 新颖性: ⭐⭐⭐⭐ 分类器集成+合成数据+减少过滤的组合策略在预训练数据领域有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 1T和15T训练+详细消融+超越Llama 3.1
- 写作质量: ⭐⭐⭐⭐ 清晰有条理，表格丰富
- 价值: ⭐⭐⭐⭐⭐ 6.3T开源数据集+超越Llama 3.1的实验结果，极高实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](../../ICML2026/llm_pretraining/on_training_large_language_models_for_long-horizon_tasks_an_empirical_study_of_h.md)
- [\[ACL 2025\] Data Caricatures: On the Representation of African American Language in Pretraining Corpora](data_caricatures_on_the_representation_of_african_american_language_in_pretraini.md)
- [\[ACL 2025\] Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases](between_circuits_chomsky.md)
- [\[ACL 2025\] Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation](emergent_abilities_continued_pt.md)
- [\[ICLR 2026\] Beyond Length: Quantifying Long-Range Information for Long-Context LLM Pretraining Data](../../ICLR2026/llm_pretraining/beyond_length_quantifying_long-range_information_for_long-context_llm_pretrainin.md)

</div>

<!-- RELATED:END -->
