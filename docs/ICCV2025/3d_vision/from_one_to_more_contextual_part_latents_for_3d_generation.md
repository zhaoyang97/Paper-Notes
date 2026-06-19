---
title: >-
  [论文解读] From One to More: Contextual Part Latents for 3D Generation
description: >-
  [ICCV 2025][3D视觉][部件级3D生成] 提出CoPart框架，通过上下文部件潜码表示3D物体并利用互引导策略微调预训练扩散模型，实现高质量的部件级3D生成，同时支持部件编辑、铰接体生成和小场景生成。 3D原生潜在扩散范式面临三个挑战： 单一潜码表示：多数方法用一个潜码表示整个3D物体，忽略了复杂物体的多部件特性…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "部件级3D生成"
  - "扩散模型"
  - "互引导"
  - "3D VAE"
  - "部件编辑"
---

# From One to More: Contextual Part Latents for 3D Generation

**会议**: ICCV 2025  
**arXiv**: [2507.08772](https://arxiv.org/abs/2507.08772)  
**代码**: [项目页](https://copart3d.github.io)  
**领域**: 3D视觉  
**关键词**: 部件级3D生成, 扩散模型, 互引导, 3D VAE, 部件编辑

## 一句话总结

提出CoPart框架，通过上下文部件潜码表示3D物体并利用互引导策略微调预训练扩散模型，实现高质量的部件级3D生成，同时支持部件编辑、铰接体生成和小场景生成。

## 研究背景与动机

3D原生潜在扩散范式面临三个挑战：

**单一潜码表示**：多数方法用一个潜码表示整个3D物体，忽略了复杂物体的多部件特性，导致细节丢失

**忽视部件独立性**：3D设计师通常逐部件创建物体，但现有整体式表示忽略了部件间关系

**全局控制不足**：依赖全局条件（文本/图像），缺少精细的局部可控性

CoPart的"自下而上"策略：直接学习部件分布并联合生成一致的部件，而非先生成整体再分割。

## 方法详解

### 部件表示编码

每个3D部件通过**混合部件潜码**表示：
- **几何token** $\mathbf{L}_{3D} = \mathcal{E}_{3D}(P, Q) \in \mathbb{R}^{T \times D}$：由3D部件VAE从表面采样点和法线编码
- **图像token** $\mathbf{L}_{2D} = \mathcal{E}_{2D}(O_k) \in \mathbb{R}^{T \times D}$：由图像VAE从部件多视图渲染编码

### 互引导 (Mutual Guidance)

同步不同部件和不同模态的扩散过程：

**部件间同步**（Cross-Part Attention）：

$$\mathcal{G}^{p'} = \text{Attention}(\mathcal{G}^p, \{\mathcal{G}^i\}_{i=1}^N)$$

**跨模态同步**（Cross-Modality Attention）：

$$\mathcal{G}^{p'} = \mathcal{G}^p + \text{LN}(\text{Attention}(\mathcal{G}^p, \{\mathcal{F}_k^p\}_{k=1}^v))$$

$$\mathcal{F}_k^{p'} = \mathcal{F}_k^p + \text{LN}(\text{Attention}(\mathcal{F}_k^p, \mathcal{G}^p))$$

其中LN为零初始化线性层，保证训练稳定性。

### 3D包围盒条件编码

创新性地将包围盒视为六面体网格，通过预训练3D VAE编码到几何潜空间：

$$\mathbf{L}_{box}^p = \mathcal{E}_{3D}(\mathbf{P}_{box}^p, \mathbf{Q}_{box}^p)$$

通过cross-attention注入3D去噪器，同时渲染为2D线框图通过ControlNet注入2D去噪器。

### 损失函数

标准去噪损失：

$$Loss_{3D} = \frac{1}{N} \sum_{p=1}^N \mathbb{E} \|\epsilon_{3d}^p - \mathcal{N}_{3d}(\mathbf{L}_{3D}^{p,t}, \mathbf{L}_{2D}^{p,t}, t)\|_2^2$$

## 实验

### 与SOTA 3D生成器对比

| 方法 | CLIP(N-T) | CLIP(I-T) | ULIP-T | Part-CLIP(N-T) | 时间 |
|------|-----------|-----------|--------|----------------|------|
| Shap-E | 0.155 | 0.161 | 0.105 | 0.088 | 3s |
| Trellis | 0.207 | 0.236 | 0.175 | 0.127 | 10s |
| Rodin | 0.204 | 0.242 | 0.179 | 0.143 | - |
| **CoPart** | 0.201 | **0.239** | 0.174 | **0.161** | 65s |

CoPart在部件感知指标上大幅领先，整体质量与SOTA可比。

### 用户研究

| 方法 | 整体偏好 | 部件偏好 |
|------|---------|---------|
| Rodin | 33.3% | 25.5% |
| PartGen | 11.8% | 13.7% |
| **CoPart** | **54.9%** | **60.8%** |

用户更偏好CoPart的生成结果，尤其在部件质量方面优势明显。

## 亮点与洞察

1. **互引导机制**：通过attention实现部件间和跨模态信息交换，比显式的2D-3D投影更高效
2. **包围盒编码创新**：将包围盒作为网格用3D VAE编码，自然映射到几何潜空间
3. **PartVerse数据集**：91k部件/12k物体/175类别，填补大规模3D部件数据空白
4. **多应用支持**：部件编辑、铰接体生成、小场景生成均无需额外训练

## 局限性

- GPU显存限制最大部件数N=8
- 部件顺序歧义虽通过包围盒条件缓解但未完全解决
- 65s的生成时间比整体式方法慢
- PartVerse数据集的自动分割+人工后处理流程成本较高

## 相关工作

- CLAY, CraftMan: 3D原生扩散
- SALAD, DiffFacto: 部件级生成（受限于PartNet类别）
- PartGen, Part123: "自上而下"部件生成

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (部件级3D扩散+互引导)
- 技术深度: ⭐⭐⭐⭐⭐ (框架设计精巧完整)
- 实验充分度: ⭐⭐⭐⭐ (定量+用户研究+消融)
- 实用价值: ⭐⭐⭐⭐⭐ (部件编辑是刚需)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] UniPart: Part-Level 3D Generation with Unified 3D Geom-Seg Latents](../../CVPR2026/3d_vision/unipart_part-level_3d_generation_with_unified_3d_geom-seg_latents.md)
- [\[ICCV 2025\] Learning 3D Scene Analogies with Neural Contextual Scene Maps](learning_3d_scene_analogies_with_neural_contextual_scene_maps.md)
- [\[ICCV 2025\] BokehDiff: Neural Lens Blur with One-Step Diffusion](bokehdiff_neural_lens_blur_with_one-step_diffusion.md)
- [\[ICCV 2025\] Find Any Part in 3D](find_any_part_in_3d.md)
- [\[CVPR 2025\] Structured 3D Latents for Scalable and Versatile 3D Generation](../../CVPR2025/3d_vision/structured_3d_latents_for_scalable_and_versatile_3d_generation.md)

</div>

<!-- RELATED:END -->
