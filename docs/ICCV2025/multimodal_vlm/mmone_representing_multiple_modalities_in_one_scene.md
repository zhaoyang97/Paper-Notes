---
title: >-
  [论文解读] MMOne: Representing Multiple Modalities in One Scene
description: >-
  [ICCV 2025][多模态VLM][多模态场景表示] 提出 MMOne 通用框架，通过模态建模模块（含模态指示器）和多模态分解机制解决多模态场景表示中的属性差异和粒度差异问题，在单一 3DGS 表示中同时建模 RGB、热成像和语言等多种模态并均获提升。 3D 场景表示从 NeRF 到 3D 高斯泼溅（3DGS）已在 RG…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "多模态场景表示"
  - "3D高斯泼溅"
  - "模态冲突"
  - "模态分解"
  - "热成像"
---

# MMOne: Representing Multiple Modalities in One Scene

**会议**: ICCV 2025  
**arXiv**: [2507.11129](https://arxiv.org/abs/2507.11129)  
**代码**: [MMOne](https://github.com/Neal2020GitHub/MMOne)  
**领域**: 多模态VLM  
**关键词**: 多模态场景表示, 3D高斯泼溅, 模态冲突, 模态分解, 热成像  

## 一句话总结

提出 MMOne 通用框架，通过模态建模模块（含模态指示器）和多模态分解机制解决多模态场景表示中的属性差异和粒度差异问题，在单一 3DGS 表示中同时建模 RGB、热成像和语言等多种模态并均获提升。

## 研究背景与动机

3D 场景表示从 NeRF 到 3D 高斯泼溅（3DGS）已在 RGB 渲染上取得巨大成功，但将多种模态（RGB、热成像、语言）整合到统一场景表示中面临根本性挑战——**模态冲突**（Modality Conflicts）：

**属性差异（Property Disparity）**：不同模态的数据特征本质不同。例如 RGB 是 3 维颜色向量，语言需要高维特征空间；热成像中纸张不会遮挡热源，但在 RGB/语言中会遮挡

**粒度差异（Granularity Disparity）**：不同模态的信息粒度不同。热成像相对粗糙，RGB 更精细，语言在物体内部保持一致。因此对物体边界，热模态偏好少量大高斯，RGB 需要多个小高斯

现有方法的关键问题：
- 使用**共享不透明度**（shared opacity）表示所有模态，忽略了模态间的属性差异
- 使用**相同的高斯集合**表示所有模态，与不同模态的粒度差异相矛盾
- 模态特定设计仅针对特定模态，无法扩展到更多模态

作者的核心思考：**如何在同时表示多个模态时，解决模态间的本质差异？**

## 方法详解

### 整体框架

MMOne 基于 3DGS 框架，给定多视角多模态输入，逐步构建多模态场景表示。每个模态由模态建模模块表示，密集化过程集成多模态分解机制。

训练时各模态独立渲染，损失分别计算后求和：$\mathcal{L} = \sum_{i=1}^{m} \mathcal{L}_{M_i}$

### 模态建模模块（解决属性差异）

为每个模态引入两个组件：

**模态特定特征** $m_i \in \mathbb{R}^{d_m}$：不同模态使用不同维度的特征向量，适应各自的物理属性。

**模态指示器** $\alpha^m \in [0,1]$：替代共享不透明度，为每个模态独立控制不透明度。渲染公式变为：

$$M(x) = \sum_{i=1}^{N} T_i^m \cdot \alpha_i^m \cdot g_i^{2D}(x) \cdot m_i$$

$$T_i^m = \prod_{j=1}^{i-1} (1 - \alpha_j^m \cdot g_j^{2D}(x))$$

模态指示器的关键作用：
- **加权机制**：为不同模态提供不同的渲染权重
- **开关功能**：可选择性地在渲染过程中停用某些模态。当某些模态被"关闭"时，高斯的几何属性仅受剩余"活跃"模态影响

在 CUDA 光栅化过程中通过跳过特定模态的渲染实现"开关"功能，从而冻结对应模态的更新。

### 多模态分解机制（解决粒度差异）

**多模态剪枝**：
- 传统 3DGS 中低不透明度高斯直接剪枝（**Hard Prune**），但多模态场景中一个模态指示器低而另一个高时直接剪枝会损害其他模态
- 提出 **Soft Prune**：仅剪枝特定模态（将对应模态指示器设为"关"），而非删除整个高斯
- 提高单模态高斯的剪枝阈值，减少不重要的单模态高斯，鼓励学习跨模态共享属性

**多模态分解**：
3DGS 密集化时，不同模态的梯度回传到同一高斯可能互相抵消，导致次优结果。解决方案：

累积各模态梯度 $g_{m_i}$ 和 $g_{m_j}$，计算梯度差异：

$$gd_{ij} = norm(g_{m_i} - g_{m_j})$$

当梯度差异超过阈值（0.0002）时，将多模态高斯分解为多个单模态高斯，分别由各自的模态损失优化。

这样实现了将多模态信息解纠缠为**共享组件**（多模态高斯）和**模态特定组件**（单模态高斯），形成更紧凑高效的表示。

### 损失函数

总损失为各模态损失之和，具体形式因模态而异：
- **RGB**：采用 3DGS 标准 L1 + SSIM 损失
- **热成像**：L1 + SSIM + 平滑正则化
- **语言**：跟随 LangSplat 的语义特征损失

## 实验

### RGB-热成像评估

| 模态 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|------|------|------|
| RGB | 3DGS | 23.27 | 0.821 | 0.220 |
| RGB | ThermalGaussian | 24.38 | 0.846 | 0.204 |
| RGB | **MMOne** | **24.89** | **0.854** | 0.209 |
| Thermal | 3DGS | 24.11 | 0.859 | 0.214 |
| Thermal | ThermalGaussian | 25.51 | 0.879 | 0.172 |
| Thermal | **MMOne** | **25.89** | **0.890** | **0.176** |

RGB 提升 0.5dB，热成像提升 0.4dB，且仅使用 ThermalGaussian **三分之一**的高斯数量。

### RGB-语言评估

| 模态 | 方法 | PSNR↑(R) / mIoU↑(L) | 说明 |
|------|------|------|------|
| R | LangSplat | 24.02 | 先训RGB再注册语言 |
| R | LS-Joint | 23.23 | 联合训练，RGB下降 |
| R | **MMOne** | **24.35** | RGB反而超过单模态LangSplat |
| L | LangSplat | 47.6 | 基线 |
| L | LS-Joint | 55.3 | mIoU+7.7但牺牲RGB |
| L | **MMOne** | **56.6** | mIoU最佳且RGB不降 |

**关键发现**：MMOne 在 RGB 渲染质量甚至超过了只训 RGB 的 LangSplat，证明了模态间的**互利增强**。

### RGB-热成像-语言（三模态）评估

| 方法 | RGB PSNR | Thermal PSNR | Language mIoU |
|------|------|------|------|
| MM-Joint | 22.32 | 23.38 | 45.1 |
| **MMOne** | **23.19** | **24.24** | **48.1** |

### 模态冲突分析（关键消融）

| 方法 | RGB PSNR (2模态→3模态) | Thermal PSNR (2模态→3模态) |
|------|------|------|
| ThermalGaussian + Language | 22.88 → **22.32** (-0.56) | 23.90 → **23.38** (-0.52) |
| MMOne + Language | 23.12 → **23.19** (+0.07) | 24.17 → **24.24** (+0.07) |

联合训练基线加入语言后 RGB/Thermal 显著下降；MMOne 加入语言后反而**略有提升**，彻底解决了模态冲突。

### 消融实验

| 方法 | RGB PSNR | Thermal PSNR | Lang mIoU | 高斯数(×10⁴) |
|------|------|------|------|------|
| MM-Joint | 22.32 | 23.38 | 45.1 | 32.9 |
| + 模态建模 | 22.38 | 23.73 | 45.3 | 29.0 |
| + Hard Prune | 22.67 | 23.86 | 46.9 | 13.4 |
| + Soft Prune | 22.98 | 23.99 | 47.0 | 10.6 |
| + 分解机制 | **23.19** | **24.24** | **48.1** | **9.9** |

每个组件均带来一致提升。最终模型仅用基线 **30%** 的高斯数即实现全面超越。

## 亮点与洞察

1. **本质性问题的识别**：首次系统分析了多模态场景表示中的属性差异和粒度差异问题，并提供了统一的解决方案
2. **模态指示器的巧妙设计**：既是加权系数又是开关功能，一个简洁的概念解决了属性差异和粒度差异两个问题
3. **模态间互利而非冲突**：证明了合理的解纠缠可以使多模态学习互相促进而非拖累
4. **紧凑高效**：仅用三分之一的高斯数实现更好性能，说明解纠缠带来了真正的信息效率提升
5. **通用可扩展**：框架设计模态无关，可轻松添加新模态

## 局限性

1. 仅验证了 RGB、热成像、语言三种模态，对深度、触觉等其他模态的效果待验证
2. 依赖 COLMAP 提供的 RGB 相机位姿，热成像相机位姿需要精确标定
3. 多模态分解的梯度差异阈值（0.0002）为手动设置，不同场景可能需要调整
4. 未处理动态场景

## 相关工作

- **单模态场景表示**：NeRF/3DGS 在 RGB 上的进展，Thermal3D-GS 在热成像上的工作
- **双模态表示**：LERF/LangSplat 的 RGB+语言，ThermalGaussian 的 RGB+热成像
- **多模态表示**：GLS/LangSurf 利用深度辅助 RGB+语言但本质仍是双模态

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体推荐 | 8.5/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Multimodal Prompt Optimization: Why Not Leverage Multiple Modalities for MLLMs](../../ICLR2026/multimodal_vlm/multimodal_prompt_optimization_why_not_leverage_multiple_modalities_for_mllms.md)
- [\[ICCV 2025\] Synergistic Prompting for Robust Visual Recognition with Missing Modalities](synergistic_prompting_for_robust_visual_recognition_with_missing_modalities.md)
- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)
- [\[ICCV 2025\] One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models](one_perturbation_is_enough_on_generating_universal_adversarial_perturbations_aga.md)
- [\[CVPR 2025\] Recognition-Synergistic Scene Text Editing](../../CVPR2025/multimodal_vlm/recognition-synergistic_scene_text_editing.md)

</div>

<!-- RELATED:END -->
