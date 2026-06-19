---
title: >-
  [论文解读] LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs
description: >-
  [ACL 2026][图学习][关系抽取] 本文在六个关系抽取数据集上对比四个 LLM（7B-70B）和一个轻量级图解析器（124M参数），发现当文档的关系图平均边数超过约 18 条时，图解析器持续且显著优于 LLM，在最复杂的 ERFGC 数据集上 F1 差距达 13.2 个点，揭示了 LLM 在复杂语言图结构抽取上的根本局限。
tags:
  - "ACL 2026"
  - "图学习"
  - "关系抽取"
  - "图解析器"
  - "LLM局限性"
  - "语言图复杂度"
  - "有监督学习"
---

# LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs

**会议**: ACL 2026  
**arXiv**: [2604.08752](https://arxiv.org/abs/2604.08752)  
**代码**: 无  
**领域**: 信息抽取 / 关系抽取  
**关键词**: 关系抽取, 图解析器, LLM局限性, 语言图复杂度, 有监督学习

## 一句话总结

本文在六个关系抽取数据集上对比四个 LLM（7B-70B）和一个轻量级图解析器（124M参数），发现当文档的关系图平均边数超过约 18 条时，图解析器持续且显著优于 LLM，在最复杂的 ERFGC 数据集上 F1 差距达 13.2 个点，揭示了 LLM 在复杂语言图结构抽取上的根本局限。

## 研究背景与动机

**领域现状**：关系抽取（RE）是知识图谱构建的核心步骤，主流范式包括基于图的解析器（通过 GNN/注意力机制直接建模 token 间关系）和基于 LLM 的方法（通过上下文学习或有监督微调抽取 RDF 三元组）。近年来 LLM 在 RE 任务上被广泛探索。

**现有痛点**：(1) LLM 在 RE 上的研究主要集中在上下文学习（ICL）设定，有监督设定下与传统图解析器的直接对比尚属空白；(2) 现有评测多使用关系图较简单的数据集（每个文档仅 1-5 条关系），未考虑图复杂度对性能的影响；(3) LLM 的参数量比图解析器大两个数量级，但是否在所有场景下都物有所值尚不清楚。

**核心矛盾**：LLM 需要将关系抽取结果格式化为文本输出，这种序列化过程在因果注意力中引入噪声并增加相关 token 之间的距离。当图结构复杂（边数多）时，这种格式化开销会随图规模线性增长，而图解析器通过直接在 token 嵌入上建模关系绕过了这一问题。

**本文目标**：系统研究 LLM 在不同关系图复杂度下的 RE 性能，与轻量级图解析器进行公平对比。

**切入角度**：选取六个关系图复杂度差异巨大的数据集（平均边数从 1.42 到 49.19），在统一的有监督设定下进行对比实验。

**核心 idea**：LLM 的序列化输出机制是其在复杂关系图上劣于图解析器的根本原因——格式化文本稀释了注意力并增加了预测相关 token 间的距离。

## 方法详解

### 整体框架

这是一篇对照实验论文，目标是回答"有监督设定下，LLM 真的比轻量图解析器更适合关系抽取吗"。框架把两类模型放进同一套有监督流程对打：一类是图解析器，沿用 Dozat & Manning 的 biaffine attention 架构，以冻结的 BERT（110M）为编码器、上面接 14M 可训练参数的解析头，直接在 token 嵌入上建模 token 对之间的关系；另一类是四个用 LoRA 微调的 LLM（Mistral-7B、Qwen3-14B-Base、Qwen3-32B、Llama-3.3-70B），把抽取结果序列化成 RDF 三元组文本。两类模型在六个关系图复杂度差异巨大（平均边数 1.42→49.19）的数据集上训练并以精确匹配的 micro-F1 评估（三元组完全正确才算对），从而把性能随图复杂度的变化曲线刻画出来。

### 关键设计

**1. 图复杂度梯度的数据集选择：把"边数"作为可控自变量**

多数 RE 评测只用每文档 1-5 条关系的简单图，很容易高估 LLM 的能力、也看不出性能在何处崩。本文刻意按平均关系数 $\bar{k}$ 排出一条从简单到复杂的梯度：CoNLL04（$\bar{k}$=1.42，98.5% 样本 $k\le5$）、ADE（$\bar{k}$=1.59）、SciERC（$\bar{k}$=2.38）代表简单图；enEWT（$\bar{k}$=17.83，仅 25% 样本 $k\le5$）、SciDTB（$\bar{k}$=23.41）代表中等复杂度；ERFGC（$\bar{k}$=49.19，仅 3.3% 样本 $k\le5$）代表高度复杂的有向无环流程图。有了这条连续梯度，就能精确定位 LLM 性能退化的拐点——实测约在 $\bar{k}>18$ 处图解析器开始全面反超。

**2. Prompt 消融与无关 Prompt 实验：验证有监督下 prompt 内容是否还重要**

为厘清 LLM 的表现到底来自指令理解还是梯度学习，作者设计四种 prompt 配置对照：NoDesc（仅类别名）、Desc（类别名+描述）、UUID（用无意义 UUID 顶替指令）、对抗性 prompt（明确要求模型别做任务）。结果相当反直觉：UUID prompt 在 Qwen3-14B-Base 上拿到最佳平均 F1=0.692，而对抗性 prompt 在微调仅 10 步后模型就开始服从任务、无视指令。这说明有监督微调里模型是通过 LoRA 权重学任务、而非通过读指令理解任务，prompt 内容几乎无关紧要——这对"靠精心设计 prompt 提升微调效果"的常见做法提出了质疑。

**3. 公平计算资源对比：确保胜负不是训练预算或随机性造成的**

为避免被"谁训得久"或随机波动带偏，作者把算力对齐：图解析器训练 3K 步，LLM 训练单个 epoch（部分模型额外补训 3K 步作对照）；并用多个随机种子估方差——Mistral 和 Qwen3-14B 各跑 5 个种子、Qwen3-32B 跑 3 个种子。所有 LLM 的 F1 方差都很低（平均 $\sigma\le0.012$），说明观察到的差距稳定可靠，而非训练步数差异或运气导致。

### 损失函数 / 训练策略

图解析器用交叉熵损失训练 biaffine attention 头；LLM 用标准语言建模损失配合 LoRA（$r=a=16$），仅微调注意力层的 Q/K/V 权重。两者优化器均为 AdamW。

## 实验关键数据

### 主实验

**各数据集上最佳 Micro-F1 对比**

| 数据集 | 平均关系数 | 图解析器 (124M) | 最佳 LLM | LLM 型号 | 差值 |
|--------|-----------|----------------|---------|---------|------|
| CoNLL04 | 1.42 | 0.668 | 0.674 | Llama-70B | +0.6 |
| ADE | 1.59 | 0.697 | **0.836** | Qwen3-14B | +13.9 |
| SciERC | 2.38 | 0.351 | **0.444** | Qwen3-14B | +9.3 |
| enEWT | 17.83 | **0.865** | 0.851 | Llama-70B | -1.4 |
| SciDTB | 23.41 | **0.918** | 0.886 | Qwen3-14B | -3.2 |
| ERFGC | 49.19 | **0.713** | 0.606 | Llama-70B | **-13.2** (3K步) |

### 消融实验

| 配置 | 发现 | 说明 |
|------|------|------|
| Pearson r (Qwen3-14B, ERFGC) | -0.639 | 边数与 F1 强负相关 |
| Pearson r (图解析器, ERFGC) | -0.206 | 图解析器相关性弱得多 |
| UUID prompt vs NoDesc | UUID 平均更好 | 有监督下 prompt 内容无关紧要 |
| 1 epoch vs 3K steps | ERFGC: 0.514→0.581 | 多训练步数对复杂图有帮助但仍不够 |
| Qwen3-14B vs Qwen3-32B | 14B 通常更好 | 指令微调可能引入有害的 chatbot 偏置 |

### 关键发现

- 关系数阈值约 18：当平均关系数 $\bar{k} > 18$ 时，图解析器开始全面超越 LLM
- 性能差距随复杂度快速扩大：从 enEWT 的 1.4 点到 ERFGC 的 13.2 点
- 未经指令微调的 Qwen3-14B-Base 反而优于指令版本，说明 chatbot 归纳偏置对 RE 有害
- 推理速度差距同样巨大：LLM 需要数千次前向传播来抽取大图，图解析器只需一次

## 亮点与洞察

- 这是一个干净的负面结果论文——用实验精确定位了 LLM 的能力边界，而非简单宣称 LLM 不行
- UUID prompt 实验特别有启发性：在有监督微调中，模型通过梯度学习任务语义，prompt 内容被完全忽视。这对"精心设计 prompt 以提升微调效果"的做法提出了质疑
- 图解析器仅 14M 可训练参数就在复杂图上碾压 32B/70B 的 LLM，说明归纳偏置（biaffine attention 直接建模 token 对关系）在特定任务上远比模型规模重要

## 局限与展望

- 仅使用了一种图解析器架构，更多图解析器的对比可提供更全面的图景
- 缺乏对注意力层面的定性分析来直接验证"格式化噪声稀释注意力"的假设
- 未探索可能缓解 LLM 劣势的方案，如 prompt 压缩、减少格式化文本
- 计算限制导致部分大模型只在子集上评估

## 相关工作与启发

- **vs 图解析器 (Dozat & Manning 2017)**: 本文直接采用此经典架构，证明其在复杂图上的优势不减反增
- **vs LLM-based RE (Gajo & Barrón-Cedeño 2025)**: 前人研究了 LLM 的自然语言 vs 编程语言输出对 RE 的影响，本文则关注图复杂度维度
- **vs ICL-based RE (Wan et al. 2023)**: ICL 方法不微调模型，性能上限更低；本文在有监督设定下仍发现 LLM 的局限

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次在有监督设定下系统对比 LLM 和图解析器，精确定位图复杂度阈值
- 实验充分度: ⭐⭐⭐⭐ 四个 LLM、六个数据集、多种 prompt、多种训练步数，覆盖全面
- 写作质量: ⭐⭐⭐⭐ 论述逻辑清晰，数据呈现充分
- 价值: ⭐⭐⭐⭐ 为 RE 任务的模型选择提供了实证指导：简单图用 LLM，复杂图用图解析器

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Can LLMs Evaluate Complex Attribution in QA? Automatic Benchmarking using Knowledge Graphs](../../ACL2025/graph_learning/paper_2401_14640.md)
- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)
- [\[ACL 2026\] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)
- [\[ACL 2025\] Extending Complex Logical Queries on Uncertain Knowledge Graphs](../../ACL2025/graph_learning/extending_complex_logical_queries_uncertain_knowledge_graphs.md)

</div>

<!-- RELATED:END -->
