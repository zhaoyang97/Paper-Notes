---
title: >-
  [论文解读] MisMatched: A Benchmark for Scientific Natural Language Inference
description: >-
  [ACL 2025][LLM评测][scientific NLI] 引入 MisMatched——首个覆盖非 CS 领域（心理学、工程、公共卫生）的科学 NLI 评估基准，包含 2,700 对人工标注句子对，最佳 SLM 基线（SciBERT）Macro F1 仅 78.17%，最佳 LLM 基线（Phi-3）仅 57.16%，并证明训练时加入隐式关系句子对可提升模型性能。
tags:
  - "ACL 2025"
  - "LLM评测"
  - "scientific NLI"
  - "out-of-domain evaluation"
  - "cross-domain generalization"
  - "implicit relations"
  - "benchmark"
---

# MisMatched: A Benchmark for Scientific Natural Language Inference

**会议**: ACL 2025  
**arXiv**: [2506.04603](https://arxiv.org/abs/2506.04603)  
**代码**: [https://github.com/fshaik8/MisMatched](https://github.com/fshaik8/MisMatched)  
**领域**: NLI / 科学文本理解  
**关键词**: scientific NLI, out-of-domain evaluation, cross-domain generalization, implicit relations, benchmark

## 一句话总结
引入 MisMatched——首个覆盖非 CS 领域（心理学、工程、公共卫生）的科学 NLI 评估基准，包含 2,700 对人工标注句子对，最佳 SLM 基线（SciBERT）Macro F1 仅 78.17%，最佳 LLM 基线（Phi-3）仅 57.16%，并证明训练时加入隐式关系句子对可提升模型性能。

## 研究背景与动机

**领域现状**：科学 NLI 任务将研究论文中的句子对分类为四类关系——Entailment（蕴含）、Reasoning（推理）、Contrasting（对比）和 Neutral（中性）。现有数据集包括 SciNLI（ACL Anthology，NLP 领域）和 MSciNLI（5 个 CS 子领域），均通过远程监督（利用连接短语如"However"、"Therefore"自动标注）构建训练集，且仅覆盖 CS 领域。

**现有痛点**：(1) 所有现有科学 NLI 数据集仅覆盖计算机科学，非 CS 领域完全空白；(2) 训练集仅通过远程监督捕获显式关系（第二句以连接短语开头），忽略了大量隐式关系；(3) 缺乏评估模型跨领域泛化能力的跨域（OOD）测试基准。

**核心矛盾**：在 CS 领域训练的科学 NLI 模型是否能泛化到其他科学领域？远程监督中遗漏的隐式关系是否构成模型的盲区？

**本文目标** (1) 构建非 CS 领域的科学 NLI 评估基准并测试现有模型的 OOD 鲁棒性；(2) 研究隐式关系训练数据对模型性能的影响。

**切入角度**：类似 MNLI 中 mismatched 测试集的设计思路——用训练域之外的数据测试模型泛化能力。

**核心 idea**：通过构建心理学/工程/公共卫生三个非 CS 领域的科学 NLI 测试基准，揭示现有模型的跨域泛化瓶颈，并发现隐式关系训练数据可以提升性能。

## 方法详解

### 整体框架（Benchmark 设计）
MisMatched 是一个纯评估基准（仅有 dev/test，无训练集），设计为 OOD 测试——模型在现有 SciNLI/MSciNLI 的 CS 训练集上训练，在 MisMatched 的非 CS 测试集上评估。数据构建分两阶段：自动提取+远程标注 → 人工标注验证。

### 关键设计

1. **三域数据源选择与构建**:

    - 功能：从心理学和工程领域选用 Web of Science（WoS）论文，公共卫生领域从 WoS、NLM 和 PubMed 收集
    - 核心思路：Phase 1 用远程监督（连接短语映射）自动提取和标注句子对；Neutral 类通过三种策略（BothRand/FirstRand/SecondRand）随机配对非相邻句子。Phase 2 通过 COGITO 平台雇佣领域专家进行迭代式人工标注
    - 设计动机：选择与 CS 差异较大的领域来最大化 OOD 测试的挑战性

2. **迭代式人工标注流程**:

    - 功能：分多轮迭代，每轮随机采样平衡子集进行三人标注，仅保留自动标签与人工金标签一致的样本
    - 核心思路：标注者间 Fleiss-κ 达 0.72（中等偏强一致性）。总计标注 3,253 对，其中 2,791 对自动标签与人工标签一致。最终通过降采样平衡到每域每类 225 个样本，共 2,700 对
    - 设计动机：严格的质量控制确保评估基准的可靠性——仅使用自动和人工标注全部一致的样本

3. **隐式关系训练实验**:

    - 功能：将 MSciNLI 训练集中删除连接短语后仍与原标签一致的句子对定义为"隐式关系"样本，加入训练集构成 MSciNLI+
    - 核心思路：SciNLI/MSciNLI 的训练集仅包含以连接短语开头的句子对（显式关系）。而现实中许多句子对之间存在语义关系但第二句不以连接短语开头——这些就是隐式关系
    - 设计动机：如果模型只依赖连接短语作为分类线索，那么在测试时（连接短语已被删除）性能会受限

### SLM 和 LLM 基线设置
SLM：BERT、SciBERT、RoBERTa、XLNet 分别在 SciNLI/MSciNLI/MSciNLI+ 上微调。LLM：Llama-2/3、Mistral、Phi-3、GPT-4o、Gemini-1.5-Pro 使用零样本和四样本设置评估。

## 实验关键数据

### 主实验（SLM 基线，Macro F1%）

| 模型 | 训练数据 | Psychology | Engineering | Public Health | Overall |
|------|---------|------------|-------------|---------------|---------|
| BERT | MSciNLI | 68.00 | 69.23 | 66.34 | 67.89 |
| BERT | MSciNLI+ | 71.16 | 73.52 | 69.47 | 71.41 |
| SciBERT | MSciNLI | 76.98 | 76.56 | 77.97 | 77.66 |
| SciBERT | MSciNLI+ | **79.18** | 76.50 | **78.79** | **78.17** |
| RoBERTa | MSciNLI+ | 77.91 | **77.63** | **78.79** | 78.11 |

### 消融实验（LLM 基线，Macro F1%）

| 模型 | 设置 | Psychology | Engineering | Public Health | Overall |
|------|------|------------|-------------|---------------|---------|
| Phi-3 | zero-shot | 55.38 | 53.15 | 49.31 | 52.95 |
| Phi-3 | fs-MSciNLI | 58.64 | **56.76** | **55.68** | **57.16** |
| GPT-4o | zero-shot | 52.42 | 50.12 | 47.26 | 50.26 |
| GPT-4o | fs-SciNLI | **63.33** | 61.34 | 61.62 | 62.29 |
| Gemini-1.5-Pro | fs-MSciNLI+ | **63.68** | **62.57** | 62.51 | **62.95** |

### 关键发现
- SLM（SciBERT 78.17%）大幅超越开源 LLM（Phi-3 57.16%），但落后于闭源 LLM（Gemini 62.95%）——科学 NLI 中微调小模型仍有优势
- 使用 MSciNLI+ （含隐式关系）训练一致优于 MSciNLI——BERT 从 67.89% → 71.41%（+3.52%），证明隐式关系训练数据确实有帮助
- 最佳 SLM 基线仍只有 78.17%，说明非 CS 领域的科学 NLI 仍有很大提升空间
- 所有领域中 Public Health 的性能通常最低，可能因为该领域有更多领域特定术语
- LLM 在 zero-shot 下接近随机水平（~50%），few-shot 显著提升（+10%+）

## 亮点与洞察
- 基准设计思路明确——作为 OOD 测试集，类似 MNLI 的 mismatched 部分，专门为评估泛化能力而设计
- 隐式关系的发现有普遍价值——任何使用远程监督的 NLI 数据集都可能遗漏隐式关系，补充后能提升性能
- 数据构建质量控制严格——Fleiss-κ 0.72 + 迭代标注 + 仅保留自动/人工一致样本

## 局限与展望
- 仅覆盖三个非 CS 领域，未来可扩展到更多领域（如生物、化学、经济学等）
- 仅有 dev/test 无训练集——限制了领域自适应方法的探索
- Neutral 类的标注一致性较低，可能引入噪声

## 相关工作与启发
- **vs SciNLI (Sadat & Caragea, 2022)**：本文扩展到非 CS 领域，且 per-domain test 规模相同（800），但 dev 集较小
- **vs MSciNLI (Sadat & Caragea, 2024)**：后者虽扩展了 CS 子领域但仍限于 CS，本文跨出 CS 边界
- **vs MNLI (Williams et al., 2018)**：设计理念相似——matched 测试在分布内域，mismatched 测试跨域泛化

## 评分
- 新颖性: ⭐⭐⭐ 思路为现有工作的自然扩展，但填补了真实空白
- 实验充分度: ⭐⭐⭐⭐⭐ SLM+LLM 全面覆盖，含零样本/少样本/微调多种设置，含隐式关系消融
- 写作质量: ⭐⭐⭐⭐ 数据构建过程描述详细，表格信息丰富
- 价值: ⭐⭐⭐⭐ 为科学 NLI 的跨域评估提供了急需的基准，隐式关系的发现有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] BelarusianGLUE: Towards a Natural Language Understanding Benchmark for Belarusian](belarusian_glue.md)
- [\[ACL 2025\] AbGen: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research](abgen_evaluating_large_language_models_in.md)
- [\[ACL 2025\] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)
- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [\[ACL 2025\] Com2: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models](com2_causal_commonsense.md)

</div>

<!-- RELATED:END -->
