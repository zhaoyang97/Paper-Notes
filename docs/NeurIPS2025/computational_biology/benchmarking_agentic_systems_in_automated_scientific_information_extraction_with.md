---
title: >-
  [论文解读] Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX
description: >-
  [NeurIPS 2025][计算生物][化学信息提取] 构建 ChemX——10 个由领域专家手工标注和验证的多模态化学数据提取基准数据集，涵盖纳米材料和小分子两大领域，系统评估了 ChatGPT Agent、SLM-Matrix、FutureHouse、nanoMINER 等 SOTA Agent 系统以及 GPT-5/GPT-5 Thinking 等前沿 LLM；提出的单 Agent 方法通过结构化文档预处理（marker-pdf → Markdown → LLM 提取）在纳米酶数据集上达到 F1=0.61，超越所有通用多 Agent 系统，同时揭示了化学信息提取仍存在 SMILES 解析失败、术语歧义等系统性挑战。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "化学信息提取"
  - "多模态Benchmark"
  - "Agent评估"
  - "纳米材料"
  - "小分子"
---

# Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX

**会议**: NeurIPS 2025  
**arXiv**: [2510.00795](https://arxiv.org/abs/2510.00795)  
**代码**: [ChemX](https://ai-chem.github.io/ChemX)  
**领域**: 计算生物  
**关键词**: 化学信息提取, 多模态Benchmark, Agent评估, 纳米材料, 小分子

## 一句话总结
构建 ChemX——10 个由领域专家手工标注和验证的多模态化学数据提取基准数据集，涵盖纳米材料和小分子两大领域，系统评估了 ChatGPT Agent、SLM-Matrix、FutureHouse、nanoMINER 等 SOTA Agent 系统以及 GPT-5/GPT-5 Thinking 等前沿 LLM；提出的单 Agent 方法通过结构化文档预处理（marker-pdf → Markdown → LLM 提取）在纳米酶数据集上达到 F1=0.61，超越所有通用多 Agent 系统，同时揭示了化学信息提取仍存在 SMILES 解析失败、术语歧义等系统性挑战。

## 研究背景与动机

**领域现状**：机器学习在化学发现中取得了显著进展，但高度依赖结构化数据。现有的化学数据集（如 PubChem、CSD）主要面向属性预测和结构分析，不适用于评估自动化信息提取系统。近年来，基于 Agent 的自动化数据提取方法（如 nanoMINER、SLM-Matrix）已经出现，但各自局限于特定子领域。

**现有痛点**：(a) 缺乏覆盖多模态（文本+表格+图片）的化学信息提取统一基准——现有系统各自用不同数据集评估，无法横向比较；(b) 专用 Agent 系统（如 nanoMINER）虽然在单一数据集上效果好，但完全无法泛化到其他化学领域；(c) 通用 Agent 框架（ChatGPT Agent、FutureHouse）在处理化学专用术语和 SMILES 表示时错误率极高。

**核心矛盾**：化学数据的异质性（纳米材料需要合成条件/物化性质/结构参数，小分子需要 SMILES/生物活性/分子描述符）使得通用方法难以精确提取，而专用方法又无法跨域泛化——这个矛盾在没有统一基准的情况下甚至无法被量化。

**本文目标** 构建首个系统化、多领域、多模态的化学信息提取基准，并在此基础上公平评估现有 Agent 系统和 LLM 的实际提取能力，找到当前技术的瓶颈。

**切入角度**：从"化学数据的本质异质性"出发，而不是从某种特定方法出发。作者认为只有先建立一个覆盖多样化化学领域、不同难度级别的基准，才能真正推动自动化提取技术的进步。

**核心 idea**：用 10 个专家验证的多模态数据集统一评估通用和专用 Agent，揭示化学信息提取中 SMILES 解析、术语标准化、上下文依赖解析的系统性瓶颈。

## 方法详解

### 整体框架
ChemX 包含两个层面：(1) **基准数据集层**——10 个手工标注数据集，覆盖纳米材料（纳米酶、碳点、金属有机框架等）和小分子（螯合配合物、MIC/IC50 生物活性等），每个数据集含标准化 schema 和元数据；(2) **评估实验层**——选取代表性数据集（最低复杂度的纳米酶和螯合配合物），统一评估通用 LLM、ChatGPT Agent、专用多 Agent 系统，以及作者提出的单 Agent 方法。

### 关键设计

1. **ChemX 数据集体系**：

    - 功能：提供 10 个覆盖纳米材料和小分子两大领域的标注数据集
    - 核心思路：小分子数据集聚焦分子描述符（SMILES、生物活性指标 MIC/IC50、化合物元数据）；纳米材料数据集覆盖更广泛的参数（物化性质、合成条件、结构特征、应用效果）。所有数据集由领域专家手工标注并交叉验证
    - 设计动机：化学信息提取的核心难点在于数据异质性——不同领域的提取目标完全不同，需要覆盖这种多样性才能真正评估系统能力。每个数据集标注了复杂度等级，方便选择不同难度进行评估

2. **单 Agent 方法（本文提出）**：

    - 功能：在 LLM 提取之前先做结构化文档预处理，解决 OpenAI 黑盒 PDF 处理不可控的问题
    - 核心思路：使用 marker-pdf SDK 将论文 PDF 拆解为文本块、表格和图片三类元素，保持文档结构语义完整性。文本和表格转为 Markdown，图片替换为本地路径后由 GPT-4o 生成描述性文本，插入 `<DESCRIPTION_FROM_IMAGE>` 标签。最终的结构化 Markdown 文件交由 GPT-4.1/GPT-5/GPT-OSS-20b 进行提取，结果汇总为 CSV
    - 设计动机：ChatGPT Agent 等系统直接处理 PDF/截图，但预处理过程不透明、不可重复——论文图表被截图后 OCR 质量不稳定，导致提取结果波动。通过显式控制预处理流程，可以确保可复现性和语义完整性

3. **系统化评估体系**：

    - 功能：从五个维度（PDF 输入支持、输出格式可控、泛化性、端到端提取能力、多模态支持）定性比较所有系统
    - 核心思路：不仅用 Precision/Recall/F1 数值对比，还从系统能力角度分析。对于无法完成端到端提取任务的系统（如 OpenChemIE 只能提取分子 ID 和 SMILES、Eunomia 无法生成正确输出格式）直接排除
    - 设计动机：不同 Agent 系统的设计目标差异很大（有些只做分子识别，有些只做材料数据），需要先界定"完成完整提取任务"的标准，才能公平比较

### 评估指标
- 按数据集各列的 Precision / Recall / F1 分别计算，取列平均作为整体指标
- 同一 prompt 模板统一用于所有方法，确保可比性

## 实验关键数据

### 主实验（按列平均提取指标）

| 方法 | 纳米酶 Precision | 纳米酶 Recall | 纳米酶 F1 | 螯合配合物 Precision | 螯合配合物 Recall | 螯合配合物 F1 |
|------|-----------------|--------------|-----------|--------------------|--------------------|---------------|
| GPT-5 | 0.33 | 0.53 | 0.37 | 0.45 | 0.18 | 0.23 |
| GPT-5 Thinking | 0.01 | 0.04 | 0.02 | 0.22 | 0.18 | 0.19 |
| Single-agent (GPT-4.1) | 0.41 | 0.73 | 0.52 | 0.35 | 0.21 | 0.27 |
| Single-agent (GPT-5) | 0.47 | 0.75 | 0.58 | 0.32 | 0.39 | 0.35 |
| **Single-agent (GPT-OSS)** | **0.56** | **0.67** | **0.61** | 0.36 | 0.31 | 0.33 |
| ChatGPT Agent | - | - | -(违规) | **0.50** | **0.42** | **0.46** |
| SLM-Matrix | 0.14 | 0.55 | 0.22 | 0.40 | 0.38 | 0.39 |
| FutureHouse | 0.05 | 0.31 | 0.09 | 0.12 | 0.06 | 0.06 |
| nanoMINER** | **0.90** | **0.74** | **0.80** | - | - | - |

> *ChatGPT Agent 在纳米酶数据集上因"违规"而无法完成提取任务  
> **nanoMINER 仅支持纳米酶数据集，无法泛化

### 系统能力对比

| 方法 | PDF输入 | 输出格式可控 | 泛化性 | 端到端 | 多模态 |
|------|---------|-------------|--------|--------|--------|
| Single-agent (本文) | ✓ | ✓ | ✓ | ✓ | ✓ |
| ChatGPT Agent | ✓ | ✗ | ✗ | ✓ | ✓ |
| SLM-Matrix | ✓ | ✓ | ✓ | ✓ | ✓ |
| nanoMINER | ✓ | ✓ | ✗ | ✓ | ✓ |
| FutureHouse | ✗ | ✓ | ✓ | ✓ | ✓ |

### 关键发现
- **文档预处理是提升关键**：GPT-5 直接处理 PDF 的 F1 为 0.37，加了 marker-pdf 预处理后提升到 0.58（+21pp），Recall 从 0.53 跳到 0.75——说明结构化输入对提取质量有决定性影响
- **GPT-5 Thinking 反而更差**：在纳米酶上 F1 仅 0.02，"深度推理"模式在结构化提取任务上适得其反，模型倾向于过度推理而偏离提取指令
- **专用系统泛化性为零**：nanoMINER 在纳米酶上 F1=0.80 遥遥领先，但完全无法处理小分子数据集——这证实了构建统一基准的必要性
- **SMILES 解析是系统性瓶颈**：所有通用方法在小分子数据集上表现系统性偏低，核心原因是缺乏将分子结构图像转换为 SMILES 字符串的工具能力
- **ChatGPT Agent 的安全限制问题**：在纳米酶数据集上因"政策违规"拒绝提取——化学术语（如催化反应条件）触发了安全过滤

## 亮点与洞察
- **首个化学信息提取 Agent 基准**——ChemX 填补了化学领域自动化提取评估的空白，10 个数据集覆盖不同难度和领域，为社区提供了标准化评估框架。数据集托管在 HuggingFace 上可直接使用，降低了复现门槛。
- **"预处理 > 模型能力"的发现**——单 Agent 方法通过简单的 marker-pdf 结构化预处理就超越了复杂的多 Agent 系统，说明在信息提取任务中，输入质量比推理深度更重要。这个洞察可推广到其他科学文献处理任务。
- **揭示了 LLM 安全机制的意外副作用**——ChatGPT Agent 因化学术语触发安全过滤而无法完成合法的科学数据提取，暴露了当前 LLM 安全对齐在科学应用场景中的误报问题。

## 局限与展望
- **仅评估了 2 个数据集（纳米酶 + 螯合配合物）**，虽然 ChemX 有 10 个数据集，但实际 benchmark 实验的覆盖范围有限
- **闭源文章占绝大多数**，实际实验只用了开放获取论文（每个数据集 2 篇），数据量偏小
- **缺乏对 Agent 编排策略的深入分析**——仅比较了系统级结果，没有分析不同 Agent 架构（单 Agent vs 多 Agent、工具调用策略、上下文管理）对提取质量的具体影响
- **SMILES 转换问题未给出解决方案**——识别了瓶颈但没有提出缓解方法，可以集成 OSRA/MolScribe 等分子图像识别工具
- 可改进方向：扩大评估数据集覆盖、集成分子图像识别工具、分析不同 Agent 编排策略的影响

## 相关工作与启发
- **vs nanoMINER**：nanoMINER 是专为纳米酶设计的多 Agent 系统，在纳米酶上 F1=0.80 远超通用方法，但泛化能力为零——专用 vs 通用是化学 AI 的核心矛盾
- **vs FutureHouse**：FutureHouse 作为通用科学 Agent 平台，在化学提取上表现最差（F1=0.09），说明通用科学推理能力不能直接迁移到结构化信息提取
- **vs SLM-Matrix**：使用小语言模型的多 Agent 系统在两个数据集上表现中等，但在纳米酶上 Recall 较高（0.55），说明小模型也有潜力——关键是 Agent 编排策略
- 本文对 AI for Science 领域的 Agent 系统建设有直接参考价值——化学提取可作为评估 Agent 工具使用能力和领域适应能力的试金石

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统化化学信息提取 Agent 基准，填补重要空白；方法层面创新有限（预处理+LLM提取）
- 实验充分度: ⭐⭐⭐⭐ 10 个数据集构建完善，6 个系统横向对比，但实际评估仅覆盖 2 个数据集
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，实验对比全面，系统能力定性分析有价值
- 价值: ⭐⭐⭐⭐⭐ 对 AI for Science 中的自动化信息提取有直接推动，ChemX 可成为社区标准基准

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Energy Loss Functions for Physical Systems](energy_loss_functions_for_physical_systems.md)
- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)
- [\[ICML 2025\] Protein Structure Tokenization: Benchmarking and New Recipe](../../ICML2025/computational_biology/protein_structure_tokenization_benchmarking_and_new_recipe.md)
- [\[NeurIPS 2025\] Is Sequence Information All You Need for Bayesian Optimization of Antibodies?](is_sequence_information_all_you_need_for_bayesian_optimization_of_antibodies.md)
- [\[ICML 2025\] scSSL-Bench: Benchmarking Self-Supervised Learning for Single-Cell Data](../../ICML2025/computational_biology/scssl-bench_benchmarking_self-supervised_learning_for_single-cell_data.md)

</div>

<!-- RELATED:END -->
