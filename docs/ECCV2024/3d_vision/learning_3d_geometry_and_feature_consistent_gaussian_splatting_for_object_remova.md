---
title: >-
  [论文解读] Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal
description: >-
  [ECCV 2024][3D视觉][3D Gaussian Splatting] 提出 GScream 框架，通过单目深度引导训练和交叉注意力特征正则化，在 3D Gaussian Splatting 表示下实现高质量的物体移除，同时保持几何一致性和纹理连贯性。 3D 物体移除是 3D 视觉中一个复杂但重要的任务…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "3D Gaussian Splatting"
  - "物体移除"
  - "深度引导"
  - "交叉注意力"
  - "辐射场编辑"
---

# Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal

**会议**: ECCV 2024  
**arXiv**: [2404.13679](https://arxiv.org/abs/2404.13679)  
**代码**: [有](https://w-ted.github.io/publications/gscream)  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, 物体移除, 深度引导, 交叉注意力, 辐射场编辑

## 一句话总结

提出 GScream 框架，通过单目深度引导训练和交叉注意力特征正则化，在 3D Gaussian Splatting 表示下实现高质量的物体移除，同时保持几何一致性和纹理连贯性。

## 研究背景与动机

3D 物体移除是 3D 视觉中一个复杂但重要的任务，在虚拟现实和内容生成领域有广泛应用。与 2D 图像修复主要关注纹理填充不同，3D 物体移除还需要处理几何补全问题，难度更大。

**现有方法的问题**：

**NeRF 基方法效率低下**：SPIn-NeRF、OR-NeRF 等虽然取得了不错效果，但隐式表示固有的训练和渲染速度慢的缺陷严重限制了实际应用。例如 OR-NeRF 需要约 6 小时训练。

**多视角修复不一致**：现有方法通常依赖 2D 修复模型生成多视角伪 GT，但不同视角的修复结果往往不一致，导致移除区域出现"鬼影"效果。

**3DGS 直接应用的困难**：虽然 3DGS 具有高效渲染优势，但将其用于物体移除面临两大挑战——（a）大量离散 Gaussian 基元导致底层几何不够精确，几何补全困难；（b）在 3DGS 框架下填充一致纹理尚未被充分探索。

**核心洞察**：增强可见区域与不可见区域（被移除物体遮挡的区域）之间的信息交互，从几何和纹理两个维度实现内容恢复。3DGS 的显式表示特性为这种交互提供了天然便利——可以直接操作 3D 空间中的 Gaussian 基元，而非仅依赖 2D 图像域的监督信号。

## 方法详解

### 整体框架

GScream 基于 Scaffold-GS（轻量化 3DGS 架构）构建，输入多视角带 mask 的图像，选择一个参考视角进行 2D 修复，然后通过两个核心模块学习移除物体后的场景表示：

1. **单目深度引导训练**：利用单目深度估计作为额外几何约束，优化 Gaussian 基元位置
2. **交叉注意力特征正则化**：通过 3D Gaussian 采样和双向交叉注意力传播纹理信息

### 关键设计

1. **在线深度对齐与监督（Online Depth Alignment）**

   **功能**：利用单目深度估计为 3DGS 提供几何约束，解决移除区域的几何不一致问题。

   **核心思路**：由于单目深度估计的尺度与 3DGS 渲染深度不同，采用在线最小二乘对齐。设计加权深度损失区分参考视角（含修复区域）和其他视角（仅背景区域）：

    $\mathcal{L}_{\text{depth}} = \frac{1}{HW} \sum M'_i \| (w\hat{D}_i + q) - D_i \|$

   其中 $w, q$ 是通过最小二乘求解的对齐参数，$M'_i$ 是针对不同视角设计的权重掩码：对参考视角，移除区域权重 $\lambda_1$、可见区域权重 $\lambda_2$；对其他视角，仅在背景区域施加权重 $\lambda_3$。

   **设计动机**：参考视角的修复图像包含完整深度信息（含移除区域），而其他视角的 mask 区域内仍有原物体，因此只能在背景区域施加深度监督。通过这种分区域加权策略，既利用了修复图的完整深度，又避免了其他视角物体区域的干扰。

2. **交叉注意力特征正则化（Cross-Attention Feature Regularization）**

   **功能**：将可见区域的准确纹理信息传播到修复区域，提升纹理连贯性。

   **核心思路**：利用 3DGS 的显式特性，先在 3D 空间中采样修复区域与周围可见区域的 Gaussian anchor，再通过双向交叉注意力促进特征交互：

    $\hat{f}_{in} = \text{Attention}(\mathbf{Q}=f_{in}, \mathbf{K}=f_{sur}, \mathbf{V}=f_{sur})$
    $\hat{f}_{sur} = \text{Attention}(\mathbf{Q}=f_{sur}, \mathbf{K}=f_{in}, \mathbf{V}=f_{in})$

   其中 $f_{in}$ 和 $f_{sur}$ 分别是修复区域和周围区域的 Gaussian anchor 特征。

   **设计动机**：不同于 NeRF 方法依赖多视角伪 GT 或视角相关效果模拟，3DGS 的显式表示允许直接在 3D 空间操作 Gaussian 特征。双向注意力确保信息双向流动：可见区域的可靠纹理传播到修复区域，同时修复区域的更新也能反馈影响周围区域，实现更自然的过渡。

3. **3D Gaussian 采样策略**

   **功能**：为交叉注意力确定哪些 Gaussian anchor 属于修复区域、哪些属于可见区域。

   **核心思路**：对每个视角 $i$，采样同时覆盖 mask 内外的 patch，将 3D anchor 的中心坐标投影到当前视角的 2D 平面，根据投影位置是否在 mask 内来分为两组。

   **设计动机**：基于 2D mask 反投影的策略简单有效，且通过不同视角的采样可以覆盖不同的 3D 区域，增强特征交互的全面性。

### 损失函数 / 训练策略

总损失为深度损失、全变分平滑损失和颜色重建损失的加权和：

$$\mathcal{L}_{\text{total}} = \lambda_{depth} \mathcal{L}_{\text{depth}} + \lambda_{tv} \mathcal{L}_{\text{tv}} + \mathcal{L}_{\text{color}}$$

- $\mathcal{L}_{\text{color}}$: L1 + SSIM 重建损失
- $\mathcal{L}_{\text{tv}}$: 深度差异的全变分损失，强制深度平滑
- 基础模型采用 Scaffold-GS，通过 anchor 特征解码 Gaussian 属性，降低存储需求

## 实验关键数据

### 主实验

在 SPIn-NeRF 数据集上的定量比较：

| 方法 | PSNR↑ | masked PSNR↑ | SSIM↑ | masked LPIPS↓ | FID↓ | 训练时间 |
|------|-------|-------------|-------|--------------|------|---------|
| SPIn-NeRF | 20.18 | 15.80 | 0.46 | 0.58 | 58.78 | ~3.0h |
| OR-NeRF | 20.32 | 15.74 | 0.54 | 0.56 | 38.69 | ~6.0h |
| **GScream** | **20.49** | **15.84** | **0.58** | **0.54** | **36.72** | **~1.2h** |

GScream 在所有指标上匹配或超越基线方法，且训练速度是 SPIn-NeRF 的 2.5 倍、OR-NeRF 的 5 倍。

### 消融实验

| 配置 | PSNR↑ | masked PSNR↑ | SSIM↑ | masked SSIM↑ | masked LPIPS↓ |
|------|-------|-------------|-------|-------------|--------------|
| w/o Cross-Attn & Mono-Depth | 20.12 | 14.87 | 0.58 | 0.19 | 0.56 |
| w/o Cross-Attn | 20.47 | 15.63 | 0.58 | 0.20 | 0.50 |
| **Full Model** | **20.49** | **15.84** | **0.58** | **0.21** | **0.54** |

- 移除深度监督：masked PSNR 从 15.84 降至 14.87（-0.97），几何质量显著下降
- 移除交叉注意力：masked PSNR 降至 15.63，纹理连贯性下降

### 关键发现

- **深度监督决定几何质量**：无深度监督时 Gaussian 基元"漂浮在空中"，渲染出现明显纹理漂浮
- **交叉注意力弥补 2D 先验不足**：仅依赖 2D 先验无法修补未见区域的纹理空洞，3D 特征交互可以将合适的纹理传播到遮挡区域
- **深度估计模型的选择有影响**：Marigold 比 MiDaS 能估计更连续的深度，进而引导更连续的 GScream 结果
- **2D 修复模型选择不关键**：LaMa 和 Stable Diffusion 都能提供合理的参考，关键在于获得合理的参考即可

## 亮点与洞察

1. **显式表示的天然优势**：首次充分利用 3DGS 显式表示特性，实现 3D 空间中的直接特征交互，这是 NeRF 隐式表示难以做到的
2. **几何-纹理协同优化**：先通过深度监督改善几何，再利用改善后的几何结构传播纹理，形成良性循环
3. **效率优势突出**：训练时间仅约 1.2 小时，相比 NeRF 方法提升数倍，具有实际应用价值

## 局限与展望

- 仅使用单张参考视角的 2D 修复结果，对于大面积遮挡可能不够充分
- 交叉注意力对 LPIPS 指标略有负面影响，说明特征传播可能引入一些高频细节偏差
- 未探索视频场景或动态物体移除
- 可以尝试结合扩散模型生成更多样化的修复先验

## 相关工作与启发

- **Scaffold-GS** 提供了高效的 3DGS 基础架构，anchor-based 设计降低了存储和计算负担
- **GaussianEditor** 是同期 3DGS 编辑工作，但缺乏 3D 域的特定约束
- 深度引导策略可推广到其他 3DGS 编辑任务（如场景补全、风格迁移）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 在 3DGS 上做物体移除是较新的方向，双向交叉注意力在 3D Gaussian 特征间传播信息的设计有创意
- **实验充分度**: ⭐⭐⭐⭐ — 两个数据集 + 多个基线对比 + 详细消融 + 额外实验（不同深度/修复模型）
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，动机阐述充分，图示直观
- **价值**: ⭐⭐⭐⭐ — 高效的 3D 物体移除方案，显著优于 NeRF 基方法且更快

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing](gaussctrl_multi-view_consistent_text-driven_3d_gaussian_splatting_editing.md)
- [\[CVPR 2026\] GOR-IS: 3D Gaussian Object Removal In the Intrinsic Space](../../CVPR2026/3d_vision/gor-is_3d_gaussian_object_removal_in_the_intrinsic_space.md)
- [\[ECCV 2024\] SlotLifter: Slot-guided Feature Lifting for Learning Object-centric Radiance Fields](slotlifter_slot-guided_feature_lifting_for_learning_object-centric_radiance_fiel.md)
- [\[ECCV 2024\] Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing](texture-gs_disentangling_the_geometry_and_texture_for_3d_gaussian_splatting_edit.md)
- [\[ECCV 2024\] Learning 3D-Aware GANs from Unposed Images with Template Feature Field](learning_3d-aware_gans_from_unposed_images_with_template_feature_field.md)

</div>

<!-- RELATED:END -->
