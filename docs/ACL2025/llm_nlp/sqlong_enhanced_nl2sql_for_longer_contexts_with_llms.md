---
title: >-
  [论文解读] SQLong: Enhanced NL2SQL for Longer Contexts with LLMs
description: >-
  [ACL2025][LLM 其他][NL2SQL] 提出 SQLong，一种面向长上下文场景的 NL2SQL 数据增强框架，通过向训练数据中注入采样自其他数据库的合成 CREATE TABLE 语句来扩展上下文长度，使微调后的 LLM 在大规模 Schema 场景下显著提升 SQL 生成准确率。 1. NL2SQL 在大 S…
tags:
  - "ACL2025"
  - "LLM 其他"
  - "NL2SQL"
  - "长上下文"
  - "数据增强"
  - "大规模数据库Schema"
  - "Text-to-SQL"
---

# SQLong: Enhanced NL2SQL for Longer Contexts with LLMs

**会议**: ACL2025  
**arXiv**: [2502.16747](https://arxiv.org/abs/2502.16747)  
**代码**: 待开源（论文提及计划发布）  
**领域**: LLM/NLP  
**关键词**: NL2SQL, 长上下文, 数据增强, 大规模数据库Schema, Text-to-SQL

## 一句话总结
提出 SQLong，一种面向长上下文场景的 NL2SQL 数据增强框架，通过向训练数据中注入采样自其他数据库的合成 CREATE TABLE 语句来扩展上下文长度，使微调后的 LLM 在大规模 Schema 场景下显著提升 SQL 生成准确率。

## 背景与动机

1. **NL2SQL 在大 Schema 上性能骤降**: 现有 LLM 在 Spider/BIRD 等小 Schema 基准上表现优异，但面对真实世界的大规模数据库（数百张表）时，长上下文导致性能显著下降。
2. **训练数据 Schema 规模偏小**: Spider 平均仅 5±3 张表，BIRD 平均 7±3 张表，训练数据无法覆盖真实场景中复杂的大 Schema，导致模型在长上下文泛化能力不足。
3. **缺乏长上下文 NL2SQL 评估基准**: 现有 benchmark 的输入 prompt 长度通常不超过 2000–2500 token，没有公开的大 Schema 测试集来评估模型在长上下文下的鲁棒性。
4. **采集真实大 Schema 数据困难**: 企业级数据库 Schema 涉及隐私和权限问题，难以大规模收集和公开，制约了长上下文 NL2SQL 研究的发展。
5. **RAG 方案尚不成熟**: 基于检索增强的 Schema Linking 方法虽有潜力，但尚未有成熟方案能完全解决长上下文问题，本文提出的增强方法可与 RAG 互补。
6. **LLM 上下文窗口增长带来新机遇**: Llama-3.1 支持 128k、CodeQwen 支持 64k 上下文，为直接处理大 Schema 提供了技术基础，但需要匹配的训练策略。

## 方法详解

### SQLong 三步流水线

**Step 1: Schema 收集（Schema Collection）**
从训练集中所有数据库收集全部 `CREATE TABLE` 语句及每张表的 3 行样本数据，编译成一个综合 Schema 池。

**Step 2: Schema 增强（Schema Augmentation）**
对每个训练样本（输入 prompt + 目标 SQL），从 Schema 池中随机采样若干表（表名不与原始 Schema 重复），将采样表与原始 Schema 合并后随机打乱表的顺序。核心设计：
- 注入的表来自**不同数据库**，确保多样性
- 随机打乱使原始表位置不固定，迫使模型学习在长上下文中定位相关 Schema
- **目标 SQL 保持不变**，仅输入的 Schema 上下文变长

**Step 3: 长上下文 Prompt 生成（Long-Context Prompt Generation）**
按 (task instructions, 长上下文 db schema, question) 格式组装新 prompt，确保 prompt + SQL 总长度不超过预设上下文阈值（如 32k token）。上下文长度从 4096 到 32768 按 512 步长随机采样。

### 训练策略
- 增强数据集与原始训练集合并使用
- 基座模型：CodeQwen1.5-7B-Chat（64k）、Llama-3.1-8B-Instruct（128k）
- SFT 最小化标准 log-likelihood loss
- 8×H100 80GB GPU，batch size=1，gradient accumulation=8，最多 5 epoch

### 长上下文测试集构建
- 用 BIRD 训练集 Schema 构建 Spider 系列的长上下文测试集，用 Spider 训练集 Schema 构建 BIRD 长上下文测试集（交叉构建避免数据泄露）
- 9 个上下文长度级别：8k, 16k, 24k, 32k, 40k, 48k, 56k, 64k, 128k
- 共 45 个长上下文测试集

## 实验关键数据

### 实验一：原始短上下文数据集性能（执行匹配准确率 %）

| 模型 | Spider-dev | Spider-realistic | Spider-syn | Spider-test | BIRD-dev | 平均 |
|------|-----------|-----------------|-----------|------------|---------|------|
| Qwen2-72B-Instruct | 82.7 | 80.7 | 73.0 | 82.9 | 53.7 | 74.6 |
| CodeQwen-7B (无SQLong) | 81.9 | 76.2 | 68.7 | 79.6 | 51.4 | 71.6 |
| CodeQwen-7B (SQLong) | **83.4** | **79.7** | **71.2** | **81.3** | **53.3** | **73.8** |
| Llama-70B-Instruct | 80.7 | 78.0 | 73.0 | 83.7 | 61.5 | 75.4 |
| Llama-8B (无SQLong) | 79.2 | 76.4 | 69.6 | 80.4 | 51.9 | 71.5 |
| Llama-8B (SQLong) | **83.2** | **78.0** | **73.1** | **81.8** | **53.3** | **73.9** |

**关键发现**：SQLong 微调平均提升 2.2%+；7B/8B 模型经 SQLong 微调后在短上下文上已接近甚至超越未增强的 72B/70B 模型。增强长上下文训练并未损害短上下文性能，反而有所提升。

### 实验二：长上下文测试集性能（Spider-test 执行匹配准确率 %，部分示例）

| 上下文长度 | Llama-8B (无SQLong) | Llama-8B (SQLong) | Llama-70B | 绝对提升 |
|-----------|--------------------|--------------------|-----------|---------|
| 8k | 69.9% | 77.1% | — | +7.2% |
| 24k | 59.0% | 72.3% | — | +13.3% |
| 64k | — | — | — | — |

**关键发现**：
- SQLong 微调的 Llama-8B 在 45 个长上下文测试集中有 **41 个**超越了未增强的 Llama-70B
- 平均而言，SQLong 微调带来 **11% 绝对提升**（对比无 SQLong），超过 70B 模型 **6%**
- 位置鲁棒性实验表明 SQLong 微调模型对 Schema 在 prompt 中的位置显著更鲁棒

## 亮点

- **框架简洁高效**：仅通过采样+拼接即实现数据增强，无需额外标注或模型生成，实现成本极低
- **小模型超大模型**：8B 模型经 SQLong 微调后在长上下文上全面超越 70B 未增强模型，性价比极高
- **短长兼顾**：长上下文增强训练不仅不损害短上下文性能，还带来 2.2% 的平均提升
- **系统化评估体系**：首次构建 45 个长上下文 NL2SQL 测试集（最长 128k token），填补了评估空白
- **位置鲁棒性验证**：通过控制 Schema 位置的实验，证明 SQLong 增强了模型在长上下文中的信息定位能力

## 局限与展望

1. **微调上下文长度受限于 32k**：由于计算资源限制只微调到 32k，未充分利用 Llama-3.1 的 128k 能力窗口
2. **注入 Schema 与原始查询无语义关联**：随机采样的干扰表可能过于简单，真实场景中存在语义相似的干扰表，难度更高
3. **未与 RAG Schema Linking 对比**：论文承认未与检索增强方案直接对比，无法判断 SQLong 相对于 RAG 的竞争力
4. **仅评估两个 7B/8B 基座模型**：未验证在更大模型（如 70B 微调）或更新架构上的效果
5. **增强数据质量未深入分析**：缺乏消融实验分析采样策略（如采样数量、来源多样性）的影响

## 与相关工作的对比

### vs RAG-based Schema Linking
RAG 方案通过检索相关 Schema 子集来缩短输入长度，是"做减法"策略；SQLong 通过增强训练使模型直接处理长 Schema，是"做加法"策略。两者互补：SQLong 增强模型的长上下文基础能力，RAG 在推理时进一步缩减输入。论文指出结合两者可能获得更大收益。

### vs DIN-SQL / DAIL-SQL 等 Prompt Engineering 方案
DIN-SQL、DAIL-SQL 通过精心设计 prompt 格式（如分解子问题、示例选择）提升 NL2SQL 性能，但均假设 Schema 在模型上下文窗口内。SQLong 正交于这些方法，关注的是 Schema 超出常规长度时的问题，两者可组合使用。

### vs Spider-Syn / Spider-Realistic
Spider-Syn 和 Spider-Realistic 通过同义词替换和移除显式列名来测试鲁棒性，聚焦于**语言变异**方面。SQLong 聚焦于**上下文长度**方面的鲁棒性，是不同维度的增强。实验中 SQLong 在这些变体测试集上也取得了显著提升。

## 评分
- 新颖性: ⭐⭐⭐ (数据增强思路直观简洁，但技术创新性有限，核心操作是"采样拼接")
- 实验充分度: ⭐⭐⭐⭐ (5个基准×9个长度级别=45个测试集，短长上下文全面评估，含位置鲁棒性分析)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，流水线描述完整，图表丰富)
- 价值: ⭐⭐⭐⭐ (首次系统化定义长上下文NL2SQL任务，提供实用增强方案和评估基准)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CER: Confidence Enhanced Reasoning in LLMs](cer_confidence_enhanced_reasoning.md)
- [\[ACL 2025\] Towards Enhanced Immersion and Agency for LLM-based Interactive Drama](towards_enhanced_immersion_and_agency_for_llm-based_interactive_drama.md)
- [\[ACL 2025\] ReCall: Library-Like Behavior In Language Models is Enhanced by Self-Referencing Causal Cycles](library-like_behavior_in_language_models_is_enhanced_by_self-referencing_causal_.md)
- [\[ACL 2025\] Cross-Modal Alignment for LLM-Enhanced Spoken Language Understanding](cross-modal_alignment_for_llm-enhanced_spoken_language_understanding.md)
- [\[ACL 2025\] Beyond Output Matching: Bidirectional Alignment for Enhanced In-Context Learning](beyond_output_matching_bidirectional_alignment_for_enhanced_in-context_learning.md)

</div>

<!-- RELATED:END -->
