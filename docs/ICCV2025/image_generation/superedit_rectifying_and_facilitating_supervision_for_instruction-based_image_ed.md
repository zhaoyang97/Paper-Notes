---
title: >-
  [论文解读] SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing
description: >-
  [ICCV 2025][图像生成][指令编辑] SuperEdit 通过利用扩散生成先验引导 VLM 修正编辑指令、并构建对比监督信号（正/负指令 + triplet loss）来解决指令式图像编辑中的噪声监督问题，以更少数据和更小模型超越 SmartEdit 9.19%。 指令式图像编辑的训练数据通常由自动流水线生成（LL…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "指令编辑"
  - "监督信号修正"
  - "对比学习"
  - "扩散先验"
  - "VLM"
  - "triplet loss"
---

# SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing

**会议**: ICCV 2025  
**arXiv**: [2505.02370](https://arxiv.org/abs/2505.02370)  
**领域**: 扩散模型·图像编辑  
**关键词**: 指令编辑, 监督信号修正, 对比学习, 扩散先验, VLM, triplet loss  

## 一句话总结

SuperEdit 通过利用扩散生成先验引导 VLM 修正编辑指令、并构建对比监督信号（正/负指令 + triplet loss）来解决指令式图像编辑中的噪声监督问题，以更少数据和更小模型超越 SmartEdit 9.19%。

## 研究背景与动机

指令式图像编辑的训练数据通常由自动流水线生成（LLM 修改描述 → 扩散模型生成编辑图像），但**扩散模型无法精确遵循文本指令**，导致：

- 编辑图像与编辑指令不匹配
- 原图中不需要修改的部分也被改变
- 产生**噪声监督信号**

现有方法的应对策略及其局限：

**扩大数据规模**（InstructPix2Pix）：噪声监督问题未解决

**引入大型 VLM**（SmartEdit、MGIE）：计算开销巨大（14.1B 参数）

**预训练识别任务**（InstructDiffusion）：间接缓解，不触及根本问题

SuperEdit 的关键洞察：**问题在于监督信号本身，而非模型架构**。修正指令比放大模型更直接有效。

## 方法详解

### 1. 扩散生成先验

核心发现：编辑模型在不同推理阶段生成**独立于文本的固定属性**：
- **早期阶段**：生成全局布局
- **中期阶段**：生成局部物体属性
- **晚期阶段**：生成图像细节
- **风格变化**：贯穿所有阶段

这一先验为统一的指令修正提供了基础。

### 2. 指令修正（Rectifying Supervision）

利用 GPT-4o 根据 original→edited 图像对的差异，按照四种生成属性（全局/局部/细节/风格）重新生成准确的编辑指令。

具体流程：
1. 输入原始图像和编辑图像给 GPT-4o
2. 根据扩散先验定义的四类变化引导 VLM 描述差异
3. 汇总为与图像对精确匹配的修正指令
4. 确保指令长度不超过 CLIP text encoder 的 77 token 限制

### 3. 对比监督（Facilitating Supervision）

修正指令后，编辑模型仍难以区分相近指令（如"在左边加一只猫" vs "在右边加两只猫"）。

**构建对比指令**：利用 GPT-4o 基于修正指令替换单个属性（数量/位置/物体）生成错误指令 $c^T_{neg}$。

**Triplet Loss**：

$$\mathcal{L}_{\text{triplet}} = \max\{d(\epsilon_t, \epsilon_{pos}) - d(\epsilon_t, \epsilon_{neg}) + m, 0\}$$

其中：
$$\epsilon_{pos} = \epsilon_\theta(\text{concat}(x_t, c^I), t, c^T_{pos})$$
$$\epsilon_{neg} = \epsilon_\theta(\text{concat}(x_t, c^I), t, c^T_{neg})$$

**总损失**：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{train}} + \lambda \cdot \mathcal{L}_{\text{triplet}}$$

其中 $\mathcal{L}_{\text{train}} = d(\epsilon_t, \epsilon_{pos})$ 为标准扩散损失。

## 实验

### Real-Edit 基准对比

| 方法 | 额外模块 | 预训练 | 数据量 | 模型大小 | Overall↑ |
|------|---------|--------|--------|---------|----------|
| SmartEdit | ✓ | ✓ | 1.2M | 14.1B | 3.59 |
| MGIE | ✓ | ✓ | 1.0M | 8.1B | 2.86 |
| InstructPix2Pix | ✗ | ✗ | 300K | 1.1B | 3.31 |
| **SuperEdit** | **✗** | **✗** | **40K** | **1.1B** | **3.92** |

SuperEdit 以 **30× 更少数据**和 **13× 更小模型**超越 SmartEdit 9.19%。

### GPT-4o 自动评估

| 方法 | Following Acc↑ | Preserving Acc↑ | Quality Acc↑ | Overall Acc↑ |
|------|---------------|-----------------|-------------|-------------|
| SmartEdit | 64% | 66% | 45% | 58.3% |
| **SuperEdit** | **75%** | **72%** | **55%** | **67.3%** |

在指令遵循、内容保持、图像质量三个维度上全面领先。

### 消融实验

| 配置 | Following↑ | Preserving↑ | Quality↑ | Overall↑ |
|------|-----------|------------|---------|----------|
| 原始指令 (baseline) | 52% | 53% | 50% | 51.7% |
| + 修正指令 | 70% | 68% | 52% | 63.3% |
| + 修正指令 + 对比损失 | **75%** | **72%** | **55%** | **67.3%** |

指令修正贡献了约 11.6% 的提升，对比损失额外贡献 4%。

## 亮点与洞察

1. **数据导向而非模型导向**：通过改善监督信号而非放大模型，以极小代价获得更大收益
2. **扩散先验的通用性**：发现编辑模型和 T2I 模型共享相同的阶段性生成属性
3. **对比学习的巧妙应用**：仅修改指令中的单个属性，确保正负样本的嵌入差异小而语义差异大
4. 完全开源（数据+模型），可复现性强

## 局限性

- 依赖 GPT-4o 进行指令修正和对比指令生成，存在 API 成本
- InstructPix2Pix 架构本身的局限（如 SD 1.5 的分辨率限制）
- 对比损失需要额外的 UNet 前向传播（每步多计算一次）
- 未探索与更强基础模型（SDXL、Flux）的结合

## 相关工作

- 指令编辑：InstructPix2Pix、MagicBrush、SmartEdit
- 编辑数据构建：Prompt-to-Prompt、EditBench
- 扩散模型对齐：DPO for diffusion、ReFL

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 5 |
| 写作质量 | 4 |
| **综合** | **4.2** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation](adiee_automatic_dataset_creation_and_scorer_for_instruction_guided_image_editing_evaluation.md)
- [\[ICCV 2025\] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing](early_timestep_zero-shot_candidate_selection_for_instruction-guided_image_editin.md)
- [\[ICCV 2025\] CaO2: Rectifying Inconsistencies in Diffusion-Based Dataset Distillation](cao2_rectifying_inconsistencies_in_diffusion-based_dataset_distillation.md)
- [\[CVPR 2025\] InsightEdit: Towards Better Instruction Following for Image Editing](../../CVPR2025/image_generation/insightedit_towards_better_instruction_following_for_image_editing.md)
- [\[CVPR 2026\] Inter-Edit: First Benchmark for Interactive Instruction-Based Image Editing](../../CVPR2026/image_generation/inter-edit_first_benchmark_for_interactive_instruction-based_image_editing.md)

</div>

<!-- RELATED:END -->
