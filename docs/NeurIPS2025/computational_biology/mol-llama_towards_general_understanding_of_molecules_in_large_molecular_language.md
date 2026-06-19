---
title: >-
  [论文解读] Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Models
description: >-
  [NeurIPS 2025][计算生物][分子语言模型] 提出 Mol-LLaMA，一个面向分子通用理解的大型分子语言模型，通过设计三类关键指令数据类型和 2D-3D 分子表示融合模块，在分子特征理解上超越 GPT-4o，具备可解释性和推理能力。 分子理解是化学和生物学的基础，对药物发现至关重要。现有分子 LLM 存在两个核…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "分子语言模型"
  - "多模态指令微调"
  - "2D-3D分子表示融合"
  - "药物发现"
  - "分子推理"
---

# Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2502.13449](https://arxiv.org/abs/2502.13449)  
**代码**: [项目主页](https://mol-llama.github.io/)  
**领域**: 计算生物  
**关键词**: 分子语言模型, 多模态指令微调, 2D-3D分子表示融合, 药物发现, 分子推理

## 一句话总结

提出 Mol-LLaMA，一个面向分子通用理解的大型分子语言模型，通过设计三类关键指令数据类型和 2D-3D 分子表示融合模块，在分子特征理解上超越 GPT-4o，具备可解释性和推理能力。

## 研究背景与动机

分子理解是化学和生物学的基础，对药物发现至关重要。现有分子 LLM 存在两个核心挑战：

**知识和推理能力不足**：现有模型依赖特定任务的公开数据库（如 PubChem 的简短描述），知识范围窄、缺乏对分子特征因果关系的深入解释，常错误预测分子分类和性质

**分子结构理解有限**：仅使用单一类型的分子编码器（2D 或 3D 图），无法同时捕获键信息/连接性（2D 优势）和空间排列/表面积/体积（3D 优势）

作者在 case study 中展示：GPT-4o 和 3D-MoLM 将苯二氮䓬类药物（Bromazepam）错误预测为喹唑啉类，而 Mol-LLaMA 准确识别其苯二氮䓬骨架并解释了与 GABA-A 受体的相互作用机制。

## 方法详解

### 整体框架

Mol-LLaMA 由四个组件组成：
1. 分子编码器（MoleculeSTM-2D + UniMol-3D）→ 捕获互补的分子表示
2. 2D-3D 融合模块（Blending Module）→ 交叉注意力整合互补信息
3. Q-Former 投影器 → 将统一分子表示投影到 LLM 空间
4. LLM 骨干（Llama-2-7B-Chat 或 Llama-3.1-8B-Instruct）+ LoRA

训练分两阶段：分子表示学习（对齐分子嵌入-文本）→ 端到端指令微调。

### 关键设计

1. **三类指令数据类型的设计（Mol-LLaMA-Instruct）**

   核心洞察：分子特征具有层级关系——结构决定化学性质，化学+结构共同决定生物学特征。基于此设计三类数据：

    - **详细结构描述（S）**：使用 IUPAC 命名提示 GPT-4o 生成详细的官能团和连接性描述，构建结构理解基础
    - **结构-特征关系解释（S2F）**：直接关联结构信息与化学/生物学特征，学习因果关系（如"为什么这个结构导致特定活性"），自然赋予模型推理和可解释能力
    - **综合对话（Conv.）**：按从结构→化学→生物的层级逐步深入，培养处理多样化用户请求的能力和逐步推理能力

   数据生成管线：利用 GPT-4o + IUPAC 名称 + PubChem 描述生成，再用 GPT-4o-as-judge 过滤事实错误，最终建立约 284K 指令样本。

2. **2D-3D 融合模块（Blending Module）**

   核心思路：2D 编码器（MoleculeSTM）擅长建模键信息和连接性，3D 编码器（UniMol）擅长捕获原子空间排列。两者提供互补信息但独立编码，需要有效融合。

   具体设计：
    - 对各编码器输出的图嵌入和节点嵌入进行拼接
    - 先做自注意力（self-attention），再做交叉注意力（cross-attention）融合互补信息
    - 拼接 2D 和 3D 融合后的嵌入，送入 Q-Former 投影

   消融实验证实：简单拼接（Concat）虽能感知原子和官能团存在，但无法正确预测连接性；融合模块能准确预测分子结构。

3. **Q-Former 投影器**

    - 使用 SciBERT 初始化的 Q-Former，通过交叉注意力在可学习 query tokens 和统一分子表示间建模
    - Q-Former 的交叉注意力天然保证置换不变性，适合图数据的建模

### 损失函数 / 训练策略

- **阶段一（分子表示学习）**：冻结 2D/3D 编码器，训练 Blending Module + Q-Former，使用分子-文本对比学习、分子-文本匹配和分子锚定文本生成三个目标，以 IUPAC 名称为文本锚点
- **阶段二（端到端指令微调）**：冻结编码器，联合训练 Blending Module + Q-Former + LLM（LoRA），在 Mol-LLaMA-Instruct 数据集上微调

## 实验关键数据

### 主实验

**通用分子理解（GPT-4o 评估，相对分数，>1 表示优于 GPT-4o）**：

| 模型 | 结构 Overall | 化学 Overall | 生物 Overall | 说明 |
|------|-------------|-------------|-------------|------|
| Mol-LLaMA (Llama3.1) | 1.125 | 1.251 | 1.744 | 全面超越 GPT-4o |
| Mol-LLaMA (Llama2) | 1.098 | 1.232 | 1.631 | 同样超越 |
| 3D-MoLM† (Llama3.1) | 0.749 | 0.875 | 1.191 | 仅生物接近 GPT-4o |
| LLaMo† (Llama3.1) | 0.442 | 0.425 | 0.705 | 远低于 GPT-4o |
| GPT-4o | 1.000 | 1.000 | 1.000 | 基准 |

**MoleculeQA 分子理解 benchmark**：

| 模型 | Structure | Source | Property | Application | Total |
|------|-----------|--------|----------|-------------|-------|
| Mol-LLaMA (Llama3.1) | **77.81** | **75.50** | **49.63** | **49.30** | **70.76** |
| 3D-MoLM† | 76.31 | 73.64 | 47.93 | 47.33 | 69.10 |
| MolCA-1.3B | 71.12 | 70.98 | 47.81 | 43.17 | 64.79 |

**PAMPA 零样本分子性质预测**：

| 模型 | Default Acc | CoT Acc | w/ Task Info Acc | Fidelity | Helpfulness |
|------|-----------|---------|-----------------|----------|-------------|
| Mol-LLaMA (Llama3.1) | 63.55% | 64.37% | **72.48%** | 0.927 | 0.966 |
| GPT-4o | 48.65% | 58.23% | 47.17% | - | - |

### 消融实验

| 配置 | 结构 Overall | 化学 Overall | 生物 Overall | 说明 |
|------|-------------|-------------|-------------|------|
| S (仅结构描述) | 1.119 | 1.166 | 1.328 | 结构理解最佳 |
| S+S2F | 1.172 | **1.285** | **1.754** | 化学/生物最佳 |
| Conv. (仅对话) | 1.166 | 0.689 | 0.887 | 多样性强，深度不足 |
| Full (完整数据) | 1.125 | 1.251 | 1.744 | 平衡最优 |
| 2D only | 0.907 | 1.137 | 1.526 | 结构理解弱 |
| 3D only | 1.071 | 1.195 | 1.632 | 不错但非最优 |
| 2D+3D Concat | 1.037 | 1.210 | 1.741 | 结构理解退化 |
| 2D+3D Blended | **1.125** | **1.251** | **1.744** | 融合模块有效 |

### 关键发现

- Mol-LLaMA 在所有五个评估维度上全面超越 GPT-4o（Overall 均>1.0）
- 结构-特征关系解释（S2F）数据对化学和生物特征的理解贡献最大
- 融合模块解决了简单拼接导致的"结构理解退化"问题
- 模型对不同分子构象具有鲁棒性（Diverse vs Fixed 性能一致）
- CoT 和任务信息提示能持续提升性能，说明模型具备真正的推理能力

## 亮点与洞察

- **数据设计的层级化思想**：受分子特征层级关系启发（结构→化学→生物），设计三类递进式指令数据，比简单堆积任务特定数据更有效
- **超越 GPT-4o**：在分子理解这个专业领域中，通过精心设计的领域数据和模态融合，7B/8B 模型全面超越 GPT-4o
- **实用的"LLM-as-judge"质量控制**：用 GPT-4o 过滤自己生成的数据中的事实错误，保证数据质量
- **2D-3D 互补的精彩案例**：消融中展示了 2D 编码器漏掉原子、3D 编码器搞错键型、简单拼接搞错连接性的具体例子

## 局限与展望

- 指令数据由 GPT-4o 生成，可能引入 GPT-4o 的偏见和知识界限
- 评估也主要依赖 GPT-4o-as-judge，存在评估者与数据生成者同源的问题
- 仅在 PubChem 分子上训练和评估，未覆盖大分子（蛋白质、核酸）
- 未与最新的分子基础模型（如 MoLFormer）做对比
- 3D 构象通过 RDKit/OpenBabel 生成而非实验结构，可能引入噪声

## 相关工作与启发

- **3D-MoLM**：3D 分子编码器 + LLM，是本文的主要比较对象
- **LLaMo**：2D 分子编码器 + LLM，结构理解能力不足
- **MoleculeSTM / UniMol**：分别作为 2D 和 3D 分子编码器被本文采用
- 启发：领域数据的精心设计比模型架构创新更重要；多表示融合需要显式的交叉注意力机制，简单拼接反而有害

## 评分

- 新颖性: ⭐⭐⭐⭐ 三类层级化指令数据设计思路新颖，融合模块虽简单但有效
- 实验充分度: ⭐⭐⭐⭐⭐ 定性定量分析全面，消融详尽（数据类型、编码器、构象鲁棒性）
- 写作质量: ⭐⭐⭐⭐ Case study 精彩，表格丰富
- 价值: ⭐⭐⭐⭐ 面向实际分子分析的通用助手，超越 GPT-4o 有应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)
- [\[NeurIPS 2025\] EDBench: Large-Scale Electron Density Data for Molecular Modeling](edbench_large-scale_electron_density_data_for_molecular_modeling.md)
- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/computational_biology/tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[NeurIPS 2025\] Omni-Mol: Multitask Molecular Model for Any-to-Any Modalities](omni-mol_multitask_molecular_model_for_any-to-any_modalities.md)
- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)

</div>

<!-- RELATED:END -->
