---
title: >-
  [论文解读] The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models
description: >-
  [ICML 2025][多模态VLM][多模态] 发现多模态奖励模型 (MM-RM) 在训练时会过度依赖文本单模态捷径 (shortcuts)，导致分布外泛化能力差，提出 Shortcut-aware MM-RM 学习算法通过动态样本重加权来减少对单模态伪相关性的依赖，OOD 准确率从 68.1% 提升至 78.5%。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "多模态"
  - "Spurious Correlations"
  - "Shortcut Learning"
  - "Generalization"
  - "偏好对齐"
---

# The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models

**会议**: ICML 2025  
**arXiv**: [2503.03122](https://arxiv.org/abs/2503.03122)  
**代码**: [github.com/alignrm/Generalizable-MM-RM](https://github.com/alignrm/Generalizable-MM-RM)  
**领域**: 多模态奖励模型, LLM对齐, 鲁棒学习  
**关键词**: Multimodal Reward Model, Spurious Correlations, Shortcut Learning, Generalization, 偏好对齐

## 一句话总结

发现多模态奖励模型 (MM-RM) 在训练时会过度依赖文本单模态捷径 (shortcuts)，导致分布外泛化能力差，提出 Shortcut-aware MM-RM 学习算法通过动态样本重加权来减少对单模态伪相关性的依赖，OOD 准确率从 68.1% 提升至 78.5%。

## 研究背景与动机

奖励模型 (RM) 是 LLM 对齐的关键代理，多模态 RM 对于解决视觉幻觉和安全问题至关重要。然而，如果 RM 在 OOD 数据上泛化失败，会导致**奖励 hacking**——策略模型为了最大化回报而牺牲人类意图。

本文发现了一个被忽视的关键问题：**MM-RM 即使在多模态数据上训练，也会学习到文本单模态捷径**。

核心证据：
- IID 测试准确率高达 91.4%，但 OOD 仅 68.1%，差距 23.2%
- 纯文本训练的 RM 在 IID 条件下与多模态 RM 准确率相当（仅差约 1.2%）
- POVID 数据上 IID 100.0%，但 OOD 低至 47.8%（低于随机）

## 方法详解

### 问题分析框架

构建**跨分布泛化框架**：使用三个偏好数据集（VLFeedback、POVID、RLHF-V）代表不同环境 $\{D^e\}_{e \in \mathcal{E}}$，在一个环境训练、其他环境测试。

提出 **Shortcut-Failure Degradation (SFD)** 指标：衡量文本捷径失效时 MM-RM 准确率的下降程度，其值范围 14.2~57.5，均值 39.5。

### Shortcut-aware MM-RM 算法

架构为**双分支训练**：
- **主分支 $\mathcal{M}$**：标准多模态 RM（Shortcut-aware MM-RM）
- **辅助分支 $\mathcal{M}_t$**：移除图像的纯文本 RM（作为捷径代理）

**Shortcut-Failure Coefficient (SFC)** 定义：

$$\text{SFC}(\boldsymbol{x}_i, y_i) = \frac{\mathcal{L}_t(\boldsymbol{x}_i, y_i)}{\mathcal{L}(\boldsymbol{x}_i, y_i) + \mathcal{L}_t(\boldsymbol{x}_i, y_i)}$$

SFC 高→文本捷径失效→需要多模态理解→增大权重
SFC 低→文本捷径有效→可能是伪相关→降低权重

**Shortcut-aware 损失函数**：

$$\mathcal{L}_{sa} = \mathbb{E}_{(\boldsymbol{x}_i, y_i) \in \mathcal{S}_{train}}[\text{SFC}(\boldsymbol{x}_i, y_i) \cdot \mathcal{L}(\boldsymbol{x}_i, y_i)]$$

SFC 的梯度被 detach，仅作为加权系数，不参与反向传播。

### 推理阶段

训练完成后移除辅助分支，推理过程与标准 MM-RM 完全相同，**无额外开销**。

## 实验关键数据

### 跨分布泛化

| 设置 | 标准 MM-RM | Shortcut-aware MM-RM |
|------|-----------|---------------------|
| IID 平均准确率 | 91.4% | 90.2% |
| OOD 平均准确率 | 68.1% | **78.5%** |
| 提升 | — | **+10.4%** |

### Best-of-64 下游评估（VLFeedback 训练）

| 方法 | MM-Vet | LLaVA-bench | MMHal-V |
|------|--------|-------------|---------|
| 标准 | 49.0 | 80.5 | 3.70 |
| **Shortcut-aware** | **50.2** | **84.7** | **3.74** |

### 关键发现

- 平均 SFD 从标准 MM-RM 的 39.5 降至 Shortcut-aware 的显著更低值
- 在 2B、4B、8B 三个模型规模上一致有效
- Shortcut-aware MM-RM 对 reward overoptimization 更具鲁棒性

## 亮点与洞察

1. **问题定义精准**：首次系统揭示 MM-RM 中的单模态伪相关问题及其对泛化的危害
2. **SFD 指标巧妙**：用文本 RM 作为捷径代理，量化文本捷径对 MM-RM 的影响
3. **化诅咒为机遇**：将文本捷径的存在转化为训练信号——捷径失效处恰是最需多模态理解处
4. **即插即用 + 零推理开销**：辅助分支仅训练时需要，推理时移除

## 局限性

- 仅考虑文本单模态捷径，视觉单模态捷径未深入探讨
- 双分支训练增加了训练时的计算开销（约 2× 前向）
- 跨分布评估的三个数据集可能不够多样

## 相关工作

- 奖励建模（RLHF、Bradley-Terry Model）
- 伪相关/捷径学习（Geirhos et al., Arjovsky et al.）
- 多模态偏差（VQA 单模态偏差、长度偏差）
- Invariant Risk Minimization (IRM)

## 评分

⭐⭐⭐⭐ — 问题识别深刻，SFD 和 SFC 的设计优雅。实验设计（跨分布泛化矩阵）系统全面。但技术方案本质是加权重采样，方法层面的创新幅度中等。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Density-Aware Translation of Spurious Correlations in Zero-Shot VLMs](../../ICML2026/multimodal_vlm/density-aware_translation_of_spurious_correlations_in_zero-shot_vlms.md)
- [\[ICML 2025\] ELEMENTAL: Interactive Learning from Demonstrations and Vision-Language Models for Reward Design in Robotics](elemental_interactive_learning_from_demonstrations_and_vision-language_models_fo.md)
- [\[ICCV 2025\] Controlling Multimodal LLMs via Reward-guided Decoding](../../ICCV2025/multimodal_vlm/controlling_multimodal_llms_via_rewardguided_decoding.md)
- [\[ACL 2025\] InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model](../../ACL2025/multimodal_vlm/internlm-xcomposer25-reward_a_simple_yet_effective_multi-modal_reward_model.md)
- [\[ICCV 2025\] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation](../../ICCV2025/multimodal_vlm/multimodal_llms_as_customized_reward_models_for_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
