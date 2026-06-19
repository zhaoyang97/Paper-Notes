---
title: >-
  [论文解读] EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models
description: >-
  [ACL 2025][LLM 其他][扩散模型] 提出 EdiText，一种基于嵌入扩散模型的可控文本编辑方法，结合 SDEdit 粗粒度编辑和 self-conditioning 细粒度编辑，实现从轻微修改到大幅改写的多尺度文本编辑控制。 研究领域现状：文本编辑是将给定参考文本修改为目标属性的任务，已有方法包括基于自回归模…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "扩散模型"
  - "text editing"
  - "SDEdit"
  - "self-conditioning"
  - "controllable generation"
---

# EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models

**会议**: ACL 2025  
**arXiv**: [2502.19765](https://arxiv.org/abs/2502.19765)  
**代码**: -  
**领域**: NLP / 文本编辑 / 扩散模型  
**关键词**: diffusion language model, text editing, SDEdit, self-conditioning, controllable generation  

## 一句话总结

提出 EdiText，一种基于嵌入扩散模型的可控文本编辑方法，结合 SDEdit 粗粒度编辑和 self-conditioning 细粒度编辑，实现从轻微修改到大幅改写的多尺度文本编辑控制。

## 研究背景与动机

**研究领域现状：** 文本编辑是将给定参考文本修改为目标属性的任务，已有方法包括基于自回归模型和非自回归模型的方案。扩散模型在图像编辑领域已展现强大的多尺度控制能力，但在文本域的编辑应用仍未被充分探索。

**现有方法的局限性：**（1）基于能量模型的方法（如 Mireshghallah et al. 2022）只能进行微调级别的控制，范围有限；（2）ParaGuide（Horvitz et al. 2024）使用分类器引导调节编辑强度，但控制范围仍然狭窄；（3）自回归模型（如 Qwen2.5）对编辑程度的指令控制响应有限，修改提示词也难以显著改变编辑幅度。

**核心问题：** 如何在文本编辑中同时实现粗粒度（大范围调整）和细粒度（精确微调）的编辑控制？

## 方法详解

### 整体框架

EdiText 使用 LD4LG（Latent Diffusion for Language Generation）作为骨干模型。该模型通过 Perceiver Resampler 编码器将离散文本压缩为固定长度的连续潜在表示，再用自回归解码器重建文本。训练一个条件扩散模型来建模潜在表示的分布，并在此基础上叠加两种互补的编辑技术。

### 关键设计

1. **SDEdit 粗粒度编辑（EdiText-CE）：** 将参考文本编码为潜在表示 $x_0$，在前向扩散过程中加噪至时间步 $t_{CE}$，然后用训练好的条件扩散模型（目标属性为条件）进行反向去噪。$t_{CE}$ 控制编辑幅度：接近 $T$ 时加噪多、原文保留少、编辑大；接近 0 时加噪少、保留多、编辑小。

2. **Self-conditioning 细粒度编辑（EdiText-FE）：** 重新诠释 self-conditioning 机制——在采样过程中，不使用模型自身上一步的预测，而是将参考文本的潜在表示作为条件注入。从 $t=T$ 到 $t_{FE}$ 使用参考文本表示作为条件，$t_{FE}$ 以下恢复正常 self-conditioning。$t_{FE}$ 越小，参考文本的影响越持久，编辑越小。

3. **粗细结合的集成编辑：** SDEdit 提供大范围但粗略的控制，self-conditioning 提供小范围但精细的控制。两者叠加使用时，先用 SDEdit 设定总体编辑范围，再用 self-conditioning 在该范围内做精细调整，实现完整的多尺度覆盖。

### 损失函数

- LD4LG 训练损失：$L(\theta) = \mathbb{E}_{t,x_0,\epsilon_t}[\lambda_t^{-1} \|x_\theta(x_t, t) - x_0\|_2^2]$，其中 $\lambda_t = 1 - \alpha_t$
- Self-conditioning 模式下额外训练：以概率 $p=0.5$ 交替训练无条件和有条件（上一步预测）两种模式

## 实验

### 主实验（毒性去除任务 - Detoxifying）

| 方法 | Hamming ↓ | SacreBLEU ↑ | BERTScore ↑ | Moderation ↓ | PerspectiveAI ↓ |
|------|-----------|-------------|-------------|-------------|-----------------|
| ParaGuide (λ=200) | 25.3 | 14.9 | 0.903 | 0.446 | 0.321 |
| ParaGuide (λ=10K) | 27.2 | 11.0 | 0.889 | 0.335 | 0.229 |
| Qwen2.5-0.5B | 27.2 | 31.1 | 0.903 | 0.347 | 0.312 |
| EdiText-CE (t=175) | 17.4 | 34.7 | 0.923 | 0.576 | 0.450 |
| EdiText-CE (t=200) | 28.9 | 7.6 | 0.865 | 0.105 | 0.136 |
| EdiText-FE (t=25) | 24.7 | 14.9 | 0.881 | 0.117 | 0.121 |

### 情感控制任务（Neg → Pos）

| 方法 | Hamming ↓ | BERTScore ↑ | Accuracy ↑ |
|------|-----------|-------------|------------|
| ParaGuide (λ=10K) | 18.0 | 0.857 | 0.89 |
| Qwen2.5-0.5B | 23.9 | 0.881 | 0.60 |
| EdiText-CE (t=200) | 15.1 | 0.879 | 0.77 |
| EdiText-CE (t=225) | 19.5 | 0.846 | **0.90** |
| EdiText-FE (t=25) | 10.7 | **0.916** | 0.60 |

### 关键发现

- **控制范围：** EdiText-CE 通过调整 $t_{CE}$ 可以覆盖从近乎无编辑到完全重写的全范围，而 ParaGuide 和 Qwen2.5 的控制范围极其有限
- **编辑质量：** 在相同保留率下，EdiText 的目标属性反映率优于或持平 baseline
- **精细控制：** EdiText-FE 提供更细腻的编辑梯度，弥补 EdiText-CE 的跳跃式变化
- **集成优势：** 粗细结合后可实现连续无间断的多尺度编辑覆盖

## 亮点

- 创新性地将 SDEdit（图像领域技术）成功迁移到文本编辑，实现粗粒度控制
- 对 self-conditioning 的重新诠释十分巧妙——将"增强生成质量"重新定位为"参考文本锚定"
- 粗+细双层控制机制互补性强，覆盖完整的编辑范围
- 方法简洁优雅，不需要额外分类器（不同于 ParaGuide）

## 局限性

- 基于嵌入扩散模型，生成文本质量仍不如当代大规模自回归模型
- LD4LG 将文本压缩为固定长度潜在表示，可能丢失长文本的细节信息
- 仅在毒性控制和情感控制两类任务上验证，泛化性待确认
- 编辑参数（$t_{CE}$、$t_{FE}$）的最优值需要针对具体任务经验性调整
- 相比自回归 LLM 指令编辑，扩散模型的推理速度更慢

## 相关工作

- **扩散语言模型：** LD4LG (Lovelace et al. 2023) 嵌入扩散；MDLM (Sahoo et al. 2024) 离散扩散
- **文本编辑：** ParaGuide (Horvitz et al. 2024) 分类器引导；Mireshghallah et al. 2022 基于 EBM
- **图像编辑迁移：** SDEdit (Meng et al. 2022) 噪声-去噪编辑框架
- **可控文本生成：** Li et al. 2022 基于扩散的约束生成；self-conditioning (Chen et al. 2023) 提升采样质量

## 评分

| 维度 | 分数 (1-10) |
|------|------------|
| 创新性 | 8 |
| 技术深度 | 7 |
| 实验充分性 | 7 |
| 写作质量 | 7 |
| 实用价值 | 6 |
| 总分 | 7.0 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models](difflm_controllable_synthetic_data_generation_via_diffusion_language_models.md)
- [\[ACL 2025\] Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](segment_level_diffusion.md)
- [\[ACL 2025\] Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions](beyond_demographics_fine-tuning_large_language_models_to_predict_individuals_sub.md)
- [\[ACL 2025\] Robust Utility-Preserving Text Anonymization Based on Large Language Models](robust_utility-preserving_text_anonymization_based_on_large_language_models.md)
- [\[ACL 2025\] HFT: Half Fine-Tuning for Large Language Models](hft_half_fine-tuning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
