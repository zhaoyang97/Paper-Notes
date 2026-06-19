---
title: >-
  [论文解读] RoCoFT: Efficient Finetuning of Large Language Models with Row-Column Updates
description: >-
  [ACL 2025][LLM 其他][PEFT] 提出 RoCoFT，一种极简的参数高效微调方法：仅更新 Transformer 权重矩阵中少量行或列的参数，在 GLUE、QA、摘要生成和常识/数学推理等任务上达到与 LoRA 等 SOTA PEFT 方法相当的精度，同时更省内存和计算，并通过 Neural Tangent Kernel 理论解释了其有效性。
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "PEFT"
  - "LoRA"
  - "row-column updates"
  - "NTK"
  - "parameter-efficient finetuning"
---

# RoCoFT: Efficient Finetuning of Large Language Models with Row-Column Updates

## 基本信息

**会议**: ACL 2025  
**代码**: [Kowsher/RoCoFT](https://github.com/Kowsher/RoCoFT)  
**机构**: Nokia Bell Labs / UCF / UIC  
**领域**: 参数高效微调 / LLM  
**关键词**: PEFT, LoRA, row-column updates, NTK, parameter-efficient finetuning  

## 一句话总结

提出 RoCoFT，一种极简的参数高效微调方法：仅更新 Transformer 权重矩阵中少量行或列的参数，在 GLUE、QA、摘要生成和常识/数学推理等任务上达到与 LoRA 等 SOTA PEFT 方法相当的精度，同时更省内存和计算，并通过 Neural Tangent Kernel 理论解释了其有效性。

## 研究背景与动机

- **全量微调的困境**：随着 LLM 参数规模增长（数十亿到数千亿），为每个下游任务存储完整模型副本不切实际，且全量微调容易过拟合和灾难性遗忘
- **PEFT 方法的发展**：LoRA 通过低秩矩阵分解实现高效微调，Adapter 通过插入额外模块，Prefix/Prompt Tuning 通过添加可学习向量。这些方法虽有效但仍引入额外参数或结构
- **核心问题**：能否设计**更简单**的 PEFT 方法？更简单的方法不仅能提升效率，还能帮助我们理解 PEFT 为何有效
- **关键观察**：预训练阶段已经学到了大部分关键特征，微调只需调整极少量参数

## 方法详解

### 核心思想

RoCoFT 的方法极其直接：**只更新权重矩阵中少量行或少量列的参数**，其余参数完全冻结。

### 数学形式化

对 Transformer 中的权重矩阵 $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_{ff}$，RoCoFT 的更新可表示为：

$$\mathbf{W} = \mathbf{W}_0 + \mathbf{R} \quad \text{（行更新）}$$
$$\mathbf{W} = \mathbf{W}_0 + \mathbf{C} \quad \text{（列更新）}$$

其中 $\mathbf{R}$ 和 $\mathbf{C}$ 是受限权重矩阵，至多 $r$ 行或 $r$ 列为非零。

### 与 LoRA 的对比

| 特性 | LoRA | RoCoFT |
|------|------|--------|
| 更新形式 | $\mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}$ | $\mathbf{W} = \mathbf{W}_0 + \mathbf{R}$（或 $\mathbf{C}$） |
| 额外参数 | 需要额外矩阵 $\mathbf{A}$, $\mathbf{B}$ | **无需额外参数**，原地更新 |
| 参数量（rank=r, d×k矩阵）| $r(d+k)$ | $r \cdot k$（行）或 $r \cdot d$（列） |
| 前向计算 | 需矩阵乘法 $\mathbf{B}\mathbf{A}$ | 无额外计算 |
| 初始化问题 | 需考虑 $\mathbf{A}$, $\mathbf{B}$ 的初始化 | 无初始化问题 |

### 行/列选择策略

- **默认策略**：从头开始按序选择行或列
- **关键发现**：不同选择策略对性能影响很小，即**任意行或列均可产生相似结果**，体现了方法的鲁棒性

### NTK 理论分析

- 使用 **Neural Tangent Kernel (NTK)** 理论解释 RoCoFT 的有效性
- 核心发现：由少量行/列参数构造的 NTK 在数值上接近全参数 NTK
- 使用 NTK 核逻辑回归在多个任务上验证，受限参数集的核与全参数核的分类性能相当
- 这说明：**预训练阶段已获取了大部分微调所需的关键特征**

## 实验

### 实验设置

- **中等模型**：RoBERTa-Base/Large (GLUE)、DeBERTa-v3 (SQuAD)、BART-Large (摘要)
- **大模型**：Bloom-7B、GPT-J-6B、LLaMA2-7B、LLaMA2-13B（常识推理 + 数学推理，共 13 个数据集）
- **基线**：LoRA、AdaLoRA、IA3、Prefix-Tuning、Prompt-Tuning、BitFit、Adapter、MAM Adapter、LoRA-XS、VeRA、Diff Pruning 等

### GLUE 基准结果（RoBERTa-Base）

| 方法 | 可训练参数 | 平均分 |
|------|-----------|--------|
| Full FT | 124.6M | 83.56 |
| LoRA (r=8) | 0.89M | 84.32 |
| AdaLoRA | 1.03M | 84.06 |
| BitFit | 0.083M | 84.22 |
| SFT | 0.90M | 85.03 |
| **RoCoFT3-Row** | **0.249M** | **85.65** |
| **RoCoFT3-Column** | **0.249M** | **85.55** |

RoCoFT3 以仅 0.249M 参数（约 LoRA 的 28%）达到所有方法的最高平均分。

### LLM 推理任务结果（LLaMA2-7B）

| 方法 | 可训练参数 | 常识推理平均 | 数学推理平均 |
|------|-----------|-------------|-------------|
| LoRA | 24.30M | 75.53 | 78.52 |
| AdaLoRA | 24.90M | 74.81 | 77.48 |
| **RoCoFT3-Row** | **13.47M** | **76.46** | **79.54** |
| **RoCoFT3-Column** | **13.47M** | **76.45** | **79.35** |

在 LLaMA2-7B 上，RoCoFT 以约 55% 的参数量超越 LoRA。

### LLaMA2-13B 结果（选录）

- RoCoFT3-Row 在 13B 模型上同样表现出色，在多个任务上超越 LoRA 和 AdaLoRA
- 可训练参数约 24M（LoRA 为 44M），参数效率提升约 45%

### 消融实验

1. **行 vs 列**：行更新和列更新性能相近，无显著差异
2. **选择策略的影响**：随机选择、从头选择、从尾选择、均匀间隔选择等策略效果相似，验证了方法的鲁棒性
3. **Rank 的影响**：Rank 从 1 到 3 性能逐步提升，Rank=3 已足够达到优秀性能
4. **应用层次**：在 Q、K、V 和 FFN 所有权重矩阵上应用效果最佳

### NTK 实验验证

- 在 RoBERTa-Base 上比较全参数 NTK 与受限参数 NTK 的核分类性能
- 受限 NTK 在 GLUE 任务上与全参数 NTK 的性能差距仅 1-2%
- 从核方法视角证明了行/列参数足以捕获微调所需的核心信息

## 亮点与洞察

1. **极简设计**：可能是已知最简单的 PEFT 方法——不添加任何额外参数或模块，直接更新原始权重矩阵的子集
2. **理论支撑**：NTK 分析提供了优雅的理论解释，而不仅是纯经验性的
3. **鲁棒性**：对行/列选择策略不敏感，降低了超参搜索的负担
4. **效率优势**：无额外矩阵乘法（vs LoRA），无初始化问题，原地更新减少了内存开销
5. **深层洞察**：实验表明预训练已学到绝大部分关键特征，微调的作用只是微调极少量参数的方向

## 局限性

1. **理论局限**：NTK 理论严格来说只适用于无限宽网络，对有限宽度的实际网络只是近似
2. **Rank 上限**：当需要更大容量适应时（如领域差距大的任务），少量行/列可能不足
3. **未探索更大模型**：实验最大到 13B，未在 70B+ 的模型上验证
4. **任务覆盖**：主要评估 NLU 和推理任务，未涉及复杂生成任务（如对话、创意写作）
5. **与更新方法的组合**：未探索与量化（QLoRA）等技术的结合

## 相关工作

- **低秩方法**：LoRA (Hu et al., 2021)、AdaLoRA (Zhang et al., 2023)、VeRA (Kopiczko et al., 2023)、LoRA-XS (Bałazy et al., 2024)
- **稀疏微调**：Diff Pruning (Guo et al., 2021)、SFT (Ansell et al., 2024)、Fish Mask (Sung et al., 2021)
- **其他 PEFT**：BitFit (Zaken et al., 2021)、LayerNorm Tuning (Zhao et al., 2023)、IA3 (Liu et al., 2022)
- **NTK 理论**：Jacot et al. (2018)、Malladi et al. (2023) 将 NTK 用于分析 LLM 微调

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**：方法极其简单但有效，提出了 PEFT "能简单到什么程度"的有价值问题（+1）
- **理论深度**：NTK 分析为方法提供了理论根基，超越了纯经验工作（+0.5）
- **实验全面性**：从中等模型到大模型，覆盖 NLU/推理/摘要等多种任务（+0.5）
- **实用性**：实现简单、无额外开销、对选择策略鲁棒（+0.5）
- **扣分**：未在超大模型上验证、与 QLoRA 等组合未探索、某些复杂任务的适用性未知（-1）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)
- [\[ACL 2025\] Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition](sample-efficient_human_evaluation_of_large_language_models_via_maximum_discrepan.md)
- [\[ACL 2025\] TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models](table_lora_structure_understanding.md)
- [\[ACL 2025\] A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](a_survey_on_efficient_large_language.md)
- [\[ACL 2025\] LLM Braces: Straightening Out LLM Predictions with Relevant Sub-Updates](llm_braces_straightening.md)

</div>

<!-- RELATED:END -->
