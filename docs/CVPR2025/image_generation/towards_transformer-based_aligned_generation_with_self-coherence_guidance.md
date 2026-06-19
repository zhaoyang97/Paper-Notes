---
title: >-
  [论文解读] Towards Transformer-Based Aligned Generation with Self-Coherence Guidance
description: >-
  [CVPR 2025][图像生成][文本到图像] 提出 Self-Coherence Guidance (SCG)，一种针对 Transformer 架构文本引导扩散模型的训练无关对齐方法，通过直接优化跨注意力图（而非潜变量）来改善属性绑定、细粒度属性绑定和风格绑定。 文本引导扩散模型(TGDMs)在生成与复杂文本提示语义对…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "文本到图像"
  - "对齐生成"
  - "交叉注意力优化"
  - "Transformer"
  - "训练无关"
---

# Towards Transformer-Based Aligned Generation with Self-Coherence Guidance

**会议**: CVPR 2025  
**arXiv**: [2503.17675](https://arxiv.org/abs/2503.17675)  
**代码**: [项目页面](https://scg-diffusion.github.io/scg-diffusion)  
**领域**: 图像生成  
**关键词**: 文本到图像, 对齐生成, 交叉注意力优化, Transformer扩散模型, 训练无关

## 一句话总结

提出 Self-Coherence Guidance (SCG)，一种针对 Transformer 架构文本引导扩散模型的训练无关对齐方法，通过直接优化跨注意力图（而非潜变量）来改善属性绑定、细粒度属性绑定和风格绑定。

## 研究背景与动机

文本引导扩散模型(TGDMs)在生成与复杂文本提示语义对齐的图像时经常出错，特别是属性绑定问题（如"红色长椅和黄色时钟"中颜色与对象的错误对应）。现有方法主要基于 U-Net 架构，直接迁移到 Transformer 架构效果不佳：

- **U-Net 方法迁移失效**：D&B 和 CONFORM 等 U-Net SOTA 方法直接应用于 PIXART-α 时效果有限，甚至降低生成质量
- **注意力图语义分布不同**：U-Net 的 16×16 交叉注意力图有更强的核心语义信息（低熵），而 Transformer 所有层的注意力图分辨率一致且语义信息更均匀分布
- **潜空间优化的间接性**：U-Net 方法通过损失函数优化潜变量来间接影响注意力图，在 Transformer 中这种间接控制效果更差
- **细粒度和风格绑定缺乏研究**：现有 benchmark 仅关注粗粒度属性绑定，缺乏对部件级属性和风格绑定的评估

## 方法详解

### 整体框架

SCG 是一个训练无关的方法，在 Transformer 扩散模型的推理过程中直接修改交叉注意力图。核心思想是利用上一个去噪步骤的注意力图提取概念掩码，然后将掩码应用到当前步骤的注意力图上，增强属性/风格 token 在对应概念区域的注意力权重。

### 关键设计1: 自一致性引导 — 直接优化交叉注意力图

**功能**: 动态增强属性 token 在对应概念区域的注意力权重，确保正确的属性-概念绑定。

**核心思路**: 对于概念 token $o_i$ 和属性 token $r_i$，利用上一步 $t+1$ 的交叉注意力图提取概念掩码 $M_{t+1}^{o_i}$，然后在当前步 $t$ 直接修改注意力图：

$$(\widehat{A_t})_{p,q} = \begin{cases} c \cdot (A_t)_{p,q} & \text{if } q = r_i \text{ and } M_{t+1}^{o_i}[p] = 1 \\ (A_t)_{p,q} & \text{otherwise} \end{cases}$$

其中 $c$ 是增强系数（设为 4），$p$ 是空间位置索引，$q$ 是 token 索引。

**设计动机**: Transformer 架构中所有交叉注意力图形状一致（无下采样/上采样），使直接操作注意力图成为可能。相比 U-Net 方法通过损失函数间接优化潜变量，直接修改注意力图更高效且控制更精确。

### 关键设计2: 掩码提取策略 — 粗粒度用聚类，细粒度用 LLM 规划

**功能**: 根据任务类型采用不同策略从注意力图中提取概念掩码。

**核心思路**: 
- **粗粒度属性绑定 & 风格绑定**: 对所有层平均后的注意力图应用 K-means 聚类，分为两类生成掩码
- **细粒度属性绑定**: 聚类无法有效捕获细节概念（如苹果的果肉和茎），因此利用 LLM 推断部件比例（如"果肉占 80%，茎占 20%"），根据比例在注意力图中选取对应比例的高注意力区域作为掩码

**设计动机**: 粗粒度场景中概念间边界清晰，聚类足够有效；细粒度场景中需要常识推理来确定部件比例，LLM 的通用知识正好满足这一需求。

### 关键设计3: 跨步骤的自一致性 — 利用前一步信息引导当前步

**功能**: 利用扩散模型已有的"知道画什么"的能力来指导"在哪里画"。

**核心思路**: 扩散模型通常能正确生成各个概念的部件，只是属性分配错误。通过 $t+1$ 步的概念注意力图（模型已知概念位置）来指导 $t$ 步的属性注意力图（纠正属性位置），实现自一致的引导而无需外部参考图像。

**设计动机**: 不同于 Prompt-to-Prompt 等需要参考图像引导的方法，SCG 完全自我引导——利用生成过程中模型自身对概念位置的理解来纠正属性绑定。

### 损失函数

无损失函数——SCG 是一种推理时的注意力图编辑方法，不涉及梯度计算或反向传播。

## 实验关键数据

### U-Net 方法直接迁移到 Transformer 的效果

| 方法 | image-text↑ | text-text↑ |
|------|------------|-----------|
| PIXART-α | 0.36 | 0.807 |
| + D&B (迁移) | 0.35 | 0.807 |
| + CONFORM (迁移) | 0.36 | 0.814 |

### 三类绑定任务的 text-text 相似度

| 方法 | 粗粒度↑ | 细粒度↑ | 风格↑ |
|------|--------|--------|------|
| D&B (SD) | 0.798 | 0.742 | 0.660 |
| CONFORM (SD) | 0.824 | 0.748 | 0.672 |
| PIXART-α | 0.807 | 0.720 | 0.664 |
| **SCG (Ours)** | **0.854** | **0.765** | **0.698** |

### 用户研究（正确率）

| 方法 | 粗粒度 | 细粒度 | 风格 |
|------|--------|--------|------|
| PIXART-α | 47.5% | 32.8% | 38.1% |
| D&B (迁移) | 48.2% | 35.1% | 39.4% |
| **SCG** | **71.3%** | **52.6%** | **58.7%** |

### 关键发现

- U-Net SOTA 方法(D&B, CONFORM)直接迁移到 Transformer 架构效果**几乎无改善**
- SCG 在三类绑定任务上全面超越所有基线，包括基于 SD 的 U-Net 专用方法
- 注意力熵分析证实 U-Net 16×16 层有更低熵（更强语义），而 Transformer 各层熵值更均匀
- 直接操作注意力图比优化潜变量更有效
- LLM 辅助的比例推断对细粒度属性绑定至关重要

## 亮点与洞察

1. **首次系统分析 U-Net 与 Transformer 注意力图行为差异**，解释了方法迁移失效的根本原因
2. **直接优化注意力图而非潜变量**——在 Transformer 中更高效，因为所有层分辨率一致
3. **自一致性引导不需要参考图像**，利用模型自身的概念理解能力自我引导
4. **LLM 集成用于细粒度比例推断**是跨模态能力协同的巧妙应用

## 局限与展望

- 增强系数 $c=4$ 是固定的，不同场景可能需要不同值
- K-means 聚类假设两个概念，三个及以上概念的场景需调整
- 每步增加掩码提取和注意力修改的计算开销，影响生成速度
- 仅在 PIXART-α 上验证，对 SD3、FLUX 等更新架构的适用性未知
- 组合性场景（如多个属性绑定+关系描述）的处理能力有待评估

## 相关工作与启发

- **Prompt-to-Prompt**: 通过编辑交叉注意力图实现图像编辑的经典工作
- **D&B / CONFORM**: U-Net 架构下的 SOTA 对齐生成方法
- **PIXART-α**: 基于 DiT 的高质量文本到图像模型
- **DiT**: 首个全 Transformer 架构的扩散模型

## 评分

⭐⭐⭐⭐ — 清晰地识别了 U-Net 方法迁移到 Transformer 的瓶颈，提出了简洁有效的解决方案。直接优化注意力图的思路在 Transformer 架构中自然且高效。三类绑定任务的全面评估和 benchmark 构建有贡献。但仅在 PIXART-α 上验证是不足。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)
- [\[CVPR 2025\] Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects](self-cross_diffusion_guidance_for_text-to-image_synthesis_of_similar_subjects.md)
- [\[CVPR 2025\] Rectified Diffusion Guidance for Conditional Generation](rectified_diffusion_guidance_for_conditional_generation.md)
- [\[CVPR 2025\] LaVin-DiT: Large Vision Diffusion Transformer](lavin-dit_large_vision_diffusion_transformer.md)
- [\[CVPR 2025\] AniMer: Animal Pose and Shape Estimation Using Family Aware Transformer](animer_animal_pose_and_shape_estimation_using_family_aware_transformer.md)

</div>

<!-- RELATED:END -->
