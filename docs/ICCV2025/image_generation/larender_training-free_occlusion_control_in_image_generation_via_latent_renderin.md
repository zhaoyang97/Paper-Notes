---
title: >-
  [论文解读] LaRender: Training-Free Occlusion Control in Image Generation via Latent Rendering
description: >-
  [ICCV 2025][图像生成][遮挡控制] 提出 LaRender，一种基于体渲染原理的免训练图像生成方法，通过在潜空间中对物体特征进行"渲染"来精确控制图像中物体之间的遮挡关系。该方法仅替换预训练扩散模型的交叉注意力层，不引入任何可学习参数，在遮挡精度上显著超越现有 SOTA 方法，且能实现语义透明度控制等丰富效果。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "遮挡控制"
  - "体渲染"
  - "潜空间渲染"
  - "免训练"
  - "扩散模型"
---

# LaRender: Training-Free Occlusion Control in Image Generation via Latent Rendering

**会议**: ICCV 2025  
**arXiv**: [2508.07647](https://arxiv.org/abs/2508.07647)  
**代码**: [https://xiaohangzhan.github.io/projects/larender/](https://xiaohangzhan.github.io/projects/larender/)  
**领域**: 图像生成  
**关键词**: 遮挡控制, 体渲染, 潜空间渲染, 免训练, 扩散模型, 图像生成

## 一句话总结

提出 LaRender，一种基于体渲染原理的免训练图像生成方法，通过在潜空间中对物体特征进行"渲染"来精确控制图像中物体之间的遮挡关系。该方法仅替换预训练扩散模型的交叉注意力层，不引入任何可学习参数，在遮挡精度上显著超越现有 SOTA 方法，且能实现语义透明度控制等丰富效果。

## 研究背景与动机

**为什么遮挡控制如此重要？** 在广告创意、概念设计、复杂场景生成等应用中，物体之间的空间排列和遮挡关系必须准确表达。例如"猫在狗前面"和"狗在猫前面"应该产生明显不同的图像。

**现有方法为何无法解决遮挡控制？** 

**文本到图像方法**（SDXL、FLUX）：只能通过 prompt 描述遮挡关系（如 "a dog behind a cat"），但即使最先进的 FLUX 也无法在复杂场景中准确控制遮挡

**布局到图像方法**（MIGC、3DIS）：可以控制物体位置，但无法显式处理遮挡关系。同一个 bounding box 布局在不同遮挡关系下是相同的，因此布局信息不足以确定遮挡

**训练条件模型**：需要图像-遮挡标注配对数据，采集成本高昂

**核心洞察**：遮挡现象本质上与 3D 渲染共享相同的物理机制——无论人眼还是相机，看到的都是沿视线方向的体积积分结果。

## 方法详解

### 整体框架

LaRender 用 Latent Rendering 层替换预训练扩散模型（SDXL）中的所有交叉注意力层。给定遮挡图（occlusion graph）和 bounding box，系统执行以下步骤：

1. **拓扑排序**：根据遮挡图将物体从底到顶排序
2. **潜特征提取**：在每个交叉注意力层中，让输入特征分别与每个物体的 prompt 做注意力，获得物体级潜特征 $\mathbf{R}_i^{(l)}$
3. **透射率估计**：结合 bounding box 掩码和注意力图估计每个物体的透射率图
4. **潜空间渲染**：放置正交虚拟相机，按体渲染公式融合各物体特征

### Latent Rendering 公式

借鉴体渲染（Volume Rendering）原理，但在潜空间而非像素空间操作：

$$\mathbf{R}^{(l+1)} = \frac{1}{\mathbf{S}} \sum_{i=1}^{N} \mathbf{T}_i (1 - \exp(-\sigma_i)) \mathbf{M}_i \mathbf{R}_i^{(l)}$$

其中：
- $\mathbf{S} = \sum_{i=1}^{N} \mathbf{T}_i (1 - \exp(-\sigma_i)) \mathbf{M}_i$ 为归一化项，防止输出偏离原始分布
- $\mathbf{T}_i = \exp(-\sum_{j=1}^{i-1} \mathbf{M}_j \sigma_j)$ 为累积透射率
- $\mathbf{M}_i$ 为物体 $i$ 的透射率图（由 bounding box 掩码 × 归一化注意力图得到）
- $\sigma_i > 0$ 为语义密度标量

### 透射率图估计

结合两种信息源：
- **Bounding box 掩码**：用户提供或 LLM 解析的物体位置
- **交叉注意力图**：通过依存解析（Dependency Parsing）提取主体 token 的注意力图，归一化后与 bounding box 掩码逐元素相乘，获得更精确的物体轮廓

### 密度调度（Density Scheduling）

语义密度 $\sigma_i$ 使用反比例函数随去噪步 $t$ 变化：

$$\sigma_i(t) = \frac{D_i T}{T + 1 - t}$$

其中 $D_i \geq 0$ 为用户指定的目标密度，$T$ 为总步数。设计动机：
- **初始阶段**（$t = T$）：$\sigma_i(T) = D_i T$ 很大，退化为"不透明模式"，防止概念混淆
- **后期阶段**（$t \to 1$）：$\sigma_i \to D_i$，收敛到目标密度

这遵循了扩散模型中"概念在初期迅速形成、后期细化质量"的特性。

### 语义透明度控制

定义语义不透明度 $\alpha_i = 1 - \exp(-D_i)$，$\alpha_i \in [0, 1)$：
- $\alpha_i = 0$：物体完全透明（消失）
- $\alpha_i \to 1$：物体不透明
- $\alpha_i \in (0, 1)$：高层次"半透明"效果——不是简单的像素混合，而是概念强度控制（如森林密度、雾浓度、光照强度、镜头效果强度等）

## 实验关键数据

### 主实验：遮挡控制精度对比

| 方法 | 控制方式 | UniDet↑ | 用户AUR↑ | 用户HPSR↑ | CLIP↑ | 推理时间(s) |
|------|---------|---------|----------|-----------|-------|------------|
| SDXL | text | 0.357 | 2.36±0.49 | 0.322±0.051 | 31.12 | 7.44 |
| FLUX | text | 0.401 | 1.94±0.42 | 0.293±0.023 | 30.86 | 122 |
| MIGC | layout | 0.373 | 3.56±0.52 | 0.448±0.068 | 30.73 | 11.1 |
| 3DIS | layout | 0.337 | 2.19±0.33 | 0.311±0.051 | 30.11 | 104 |
| **LaRender** | **occlusion** | **0.416** | **4.94±0.11** | **0.767±0.070** | 30.98 | 7.46 |

LaRender 在 UniDet 指标上比最强基线 FLUX 提升 3.7%，用户评分 AUR/HPSR 大幅领先（4.94 vs 3.56），且几乎不增加推理时间（7.46s vs SDXL的7.44s）。CLIP score 仅略低于 SDXL，说明生成质量保持良好。

### 消融实验：密度调度策略

| 调度策略 | 公式 | UniDet↑ | CLIP score↑ |
|---------|------|---------|-------------|
| 固定不透明模式 | $\sigma_i(t) = D_i T$ | 0.240 | 28.32 |
| 固定密度 | $\sigma_i(t) = D_i$ | 0.393 | 30.44 |
| **反比例函数** | $\sigma_i(t) = \frac{D_i T}{T+1-t}$ | **0.416** | **30.98** |

固定不透明模式严重退化（UniDet 0.240）；固定密度打破初期概念形成；反比例调度在两个指标上均最优。

### 注意力图对透射率的影响

| 设置 | UniDet↑ | CLIP score↑ |
|------|---------|-------------|
| LaRender w/o attn. map | 0.395 | 31.00 |
| LaRender w/ attn. map | **0.416** | 30.98 |

使用注意力图精炼透射率轮廓提升遮挡精度 (+0.021)，对生成质量几乎无影响。

## 亮点与洞察

1. **物理启发的优雅设计**：将遮挡控制类比为体渲染，物理直觉清晰，不需要训练数据和额外参数
2. **免训练的实用性**：仅替换交叉注意力层，不修改模型权重，可直接用于 SDXL/FLUX 等预训练模型
3. **语义透明度的意外收获**：通过调节语义密度，不仅可以控制遮挡，还能控制雾的浓度、森林密度、光照强度等，这些"免费"效果增加了方法的实用价值
4. **LLM 辅助的输入模式**：用户只需写自然语言 prompt，LLM（DeepSeek R1 等）自动解析遮挡图和 bounding box，降低了使用门槛

## 局限性

- 当布局不合理时遮挡结果可能错误
- 有时概念会丢失或混淆，因为潜特征在融合时可能被混合或擦除以满足生成先验
- bounding box 提供的是粗糙位置控制，不追求精确的 bounding box 对齐
- 仅在 SDXL 上做主要评估，FLUX 版本的结果在补充材料中

## 相关工作与启发

- **布局到图像**（MIGC、3DIS、GLIGEN）：关注位置控制，不解决遮挡
- **3DIS**：先生成深度图再渲染纹理，但深度 ≠ 遮挡
- **MULAN/LayerFusion**：多层图像生成，但不处理多物体复杂遮挡
- **启发**：在潜空间中引入物理渲染原理是一个有潜力的方向，可推广到深度排序、光照模拟等更多物理现象的控制

## 评分 ⭐⭐⭐⭐

- 创新性：⭐⭐⭐⭐⭐ — 首次将体渲染原理引入扩散模型潜空间解决遮挡控制，思路新颖优雅
- 实验充分度：⭐⭐⭐⭐ — 两个数据集、用户研究、消融完整，但 RealOcc 仅 70 个样本
- 实用价值：⭐⭐⭐⭐ — 免训练 + 效率不降 + LLM 自动解析，可立即使用
- 写作质量：⭐⭐⭐⭐ — 方法阐述清晰，物理类比直觉性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MatchDiffusion: Training-free Generation of Match-Cuts](matchdiffusion_training-free_generation_of_match-cuts.md)
- [\[CVPR 2026\] Layer-wise Instance Binding for Regional and Occlusion Control in Text-to-Image Diffusion Transformers](../../CVPR2026/image_generation/layer-wise_instance_binding_for_regional_and_occlusion_control_in_text-to-image_.md)
- [\[ICCV 2025\] IntroStyle: Training-Free Introspective Style Attribution using Diffusion Features](introstyle_training-free_introspective_style_attribution_using_diffusion_feature.md)
- [\[ICCV 2025\] Fair Generation without Unfair Distortions: Debiasing Text-to-Image Generation with Entanglement-Free Attention](fair_generation_without_unfair_distortions_debiasing_text-to-image_generation_wi.md)
- [\[CVPR 2026\] SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation](../../CVPR2026/image_generation/seethrough3d_occlusion_aware_3d_control_in_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
