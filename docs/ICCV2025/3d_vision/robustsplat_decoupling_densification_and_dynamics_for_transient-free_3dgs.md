---
title: >-
  [论文解读] RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS
description: >-
  [ICCV 2025][3D视觉][3D高斯溅射] 本文发现 3DGS 的高斯致密化过程是导致瞬态物体伪影的关键因素，提出延迟高斯生长策略和尺度级联掩码自举方法来解耦致密化与动态区域建模，在多个基准数据集上实现了最优的无瞬态新视角合成效果。 3D 高斯溅射（3DGS）凭借实时渲染和照片级真实感质量，在新视角合成和三维建模领域…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D高斯溅射"
  - "新视角合成"
  - "瞬态物体去除"
  - "高斯致密化"
  - "鲁棒重建"
---

# RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS

**会议**: ICCV 2025  
**arXiv**: [2506.02751](https://arxiv.org/abs/2506.02751)  
**代码**: [https://fcyycf.github.io/RobustSplat/](https://fcyycf.github.io/RobustSplat/) (有项目页)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 新视角合成, 瞬态物体去除, 高斯致密化, 鲁棒重建

## 一句话总结

本文发现 3DGS 的高斯致密化过程是导致瞬态物体伪影的关键因素，提出延迟高斯生长策略和尺度级联掩码自举方法来解耦致密化与动态区域建模，在多个基准数据集上实现了最优的无瞬态新视角合成效果。

## 研究背景与动机

3D 高斯溅射（3DGS）凭借实时渲染和照片级真实感质量，在新视角合成和三维建模领域获得了广泛关注。然而，现实场景中不可避免地存在瞬态物体（如行人、车辆等动态干扰物），这些破坏了多视角一致性假设，导致重建结果中出现严重伪影。

**现有方法的三个范式及其不足**：
- *类别特定语义掩码*：仅能处理预定义类别（如人、车），难以泛化到各类瞬态物体
- *基于不确定性的掩码*：通过光度重建损失中的不确定性预测运动掩码，但往往不够可靠
- *基于学习的运动掩码*：用 MLP 预测掩码（输入 DINO 特征），监督信号来自光度残差或特征相似度。但训练初期 3DGS 表示欠优化，渲染过于平滑，导致掩码估计不准确

**核心矛盾**：掩码学习与高斯优化存在鸡生蛋问题——训练初期渲染质量低，导致静态区域被误判为动态（掩码过大）或动态区域未被过滤（掩码太小）。一旦早期致密化引入了建模瞬态物体的高斯，后续阶段很难移除。

**关键发现**：作者通过实验惊人地发现，**完全禁用致密化的原始 3DGS 就能达到接近 SpotLessSplats 的去瞬态效果**。这是因为没有致密化时，图像重建损失只能优化高斯的形状和颜色，初始高斯位置保持稳定，不会过拟合瞬态区域。但禁用致密化的代价是丢失细节——初始点稀疏区域的渲染过于平滑。

**切入角度**：既然"不致密化"能防止瞬态拟合但缺细节，"早期致密化"能捕获细节但会拟合瞬态，那么关键在于**延迟致密化**——先让静态场景结构稳定，再逐步添加高斯细化细节，同时配合更鲁棒的掩码监督信号。

## 方法详解

### 整体框架

RobustSplat 在标准 3DGS + 掩码 MLP 的框架上，引入两个核心设计：(1) 延迟高斯生长策略（Delayed Gaussian Growth）推迟致密化起始时间；(2) 尺度级联掩码自举（Scale-Cascaded Mask Bootstrapping）从低分辨率到高分辨率渐进式地改善掩码监督信号。

### 关键设计

1. **延迟高斯生长策略（Delayed Gaussian Growth）**

    - 功能：将 3DGS 的高斯致密化起始时间从默认的 500 次迭代推迟到 10K 次迭代
    - 核心思路：在前 10K 次迭代中，仅优化已有高斯的形状、颜色和不透明度，不允许分裂/克隆。这使得优化聚焦于静态场景的全局结构重建。实验表明，致密化开始越晚，最终效果越好——因为早期致密化会使新高斯去拟合瞬态物体
    - 设计动机：作者的分析实验（Fig. 5a）清楚显示，随着致密化进行，vanilla 3DGS 的 PSNR 逐渐下降（新高斯拟合了瞬态物体），而推迟致密化可有效缓解这一问题

2. **掩码正则化（Mask Regularization at Early Stage）**

    - 功能：在训练早期鼓励掩码 MLP 将所有区域分类为静态，随着训练推进逐渐允许检测动态区域
    - 核心思路：引入衰减正则项 $\mathcal{L}_{reg} = e^{-i/\beta_{reg}} \|1 - M_t\|$，其中 $i$ 为当前迭代步数。训练初始时该项强约束掩码趋近 1（即全部视为静态），随迭代衰减，逐步允许 MLP 学习检测瞬态区域
    - 设计动机：由于延迟致密化确保了早期优化只涉及静态场景，此正则项配合延迟策略，避免了过早的掩码学习引入偏差

3. **尺度级联掩码自举（Scale-Cascaded Mask Bootstrapping）**

    - 功能：在掩码 MLP 的训练中，从低分辨率特征/残差监督逐渐过渡到高分辨率
    - 核心思路：
        - 致密化开始前：使用 224×224 低分辨率图像提取 DINOv2 特征并计算余弦相似度作为掩码监督信号。低分辨率特征具有更大感受野，能有效抑制局部噪声，对欠重建区域更宽容
        - 致密化开始后：切换到 504×504 高分辨率图像，利用更精细的特征相似度和图像残差实现更精确的瞬态区域检测
    - 设计动机：训练初期静态区域因点稀疏而欠重建，高分辨率特征/残差会误将这些区域标记为动态。低分辨率特征天然平滑化了局部差异，对全局一致性的捕获更好（Fig. 6 清楚展示了这一特性）

### 损失函数 / 训练策略

3DGS 渲染损失保持标准设置：$\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{\text{D-SSIM}}$

掩码 MLP 优化损失：$\mathcal{L}_{MLP} = \lambda_{residual}\mathcal{L}_{residual} + \lambda_{cos}\mathcal{L}_{cos} + \lambda_{reg}\mathcal{L}_{reg}$
- $\mathcal{L}_{residual}$：图像残差鲁棒损失（来自 SpotLessSplats）
- $\mathcal{L}_{cos}$：$\|M_t - M_{cos}\|$，基于 DINOv2 特征余弦相似度的掩码监督
- $\mathcal{L}_{reg}$：指数衰减正则化，引导早期掩码全为静态

参数设置：$\lambda_{residual}=0.5$，$\lambda_{cos}=0.5$，$\lambda_{reg}=2.0$，$\beta_{reg}=2000$，延迟迭代起始 10K，总训练 30K 迭代。MLP 使用 DINOv2 ViT-S/14 提取特征。

## 实验关键数据

### 主实验

NeRF On-the-go 数据集（6个场景，低/中/高遮挡）：

| 方法 | PSNR (Mean) | SSIM (Mean) | LPIPS (Mean) |
|------|-------------|-------------|--------------|
| 3DGS | 19.09 | 0.717 | 0.248 |
| SpotLessSplats | 22.17 | 0.757 | 0.220 |
| WildGaussians | 22.45 | 0.784 | 0.190 |
| T-3DGS | 22.87 | 0.803 | 0.167 |
| **RobustSplat (Ours)** | **23.22** | **0.818** | **0.149** |

RobustNeRF 数据集（4个室内场景）：

| 方法 | PSNR (Mean) | SSIM (Mean) | LPIPS (Mean) |
|------|-------------|-------------|--------------|
| 3DGS | 26.21 | 0.864 | 0.168 |
| SpotLessSplats | 28.58 | 0.875 | 0.162 |
| T-3DGS | 28.25 | 0.888 | 0.149 |
| **RobustSplat (Ours)** | **29.36** | **0.895** | **0.135** |

### 消融实验

NeRF On-the-go 数据集消融（以 PSNR 为指标）：

| 配置 | Mountain | Corner | Patio | Spot | Patio-High | 说明 |
|------|----------|--------|-------|------|------------|------|
| 3DGS | 19.21 | 22.65 | 17.04 | 18.54 | 17.04 | 基线 |
| + Mask | 19.81 | 25.05 | 21.23 | 24.75 | 22.19 | 加掩码学习 |
| + Mask + DG | 20.85 | 26.01 | 21.49 | 25.61 | 22.74 | 加延迟生长 |
| + Mask + MB | 20.78 | 25.52 | 20.88 | 25.25 | 22.11 | 加掩码自举 |
| Full Model | **21.15** | **26.42** | **21.63** | **26.21** | **22.87** | DG+MB协同 |

### 关键发现

- 延迟高斯生长（DG）对所有场景均有稳定提升，平均 PSNR 比"仅掩码"高约 0.8dB
- 尺度级联掩码自举（MB）单独使用时提升略小，但与 DG 结合后效果最优，说明两者互补
- 在高遮挡场景（Spot, Patio-High）中提升最为显著，PSNR 比 vanilla 3DGS 提升 7-8dB
- RobustSplat 在所有 6 个 NeRF On-the-go 场景的 3 个指标上均取得最优

## 亮点与洞察

- 发现"禁用致密化即可去瞬态"这一反直觉现象，是本文最重要的分析贡献，揭示了 3DGS 过拟合瞬态的机制
- 延迟致密化思路极其简洁——仅需修改致密化起始时间这一超参数，无需额外网络结构
- 从低分辨率到高分辨率的级联监督策略体现了"粗到细"的经典思想，在处理欠重建-过检测权衡中非常有效
- 整体方案无需引入额外的大模型（如 SAM、Stable Diffusion），仅用 DINOv2 ViT-S/14，计算效率高

## 局限与展望

- 延迟时间（10K 迭代）和低/高分辨率切换点是手动设置的固定参数，能否自适应调整有待探索
- 对于非常密集的瞬态遮挡（>50%），延迟策略可能不足以让静态结构先稳定
- 掩码 MLP 仅使用 DINOv2 特征，引入多尺度或多模态特征（如深度先验）可能进一步提升精度
- 目前仅在静态场景+瞬态干扰设置下验证，对于真正的动态场景（如 4D 重建）需要更复杂的建模

## 相关工作与启发

- SpotLessSplats：使用 Stable Diffusion 特征的两种聚类策略预测掩码，效果好但计算开销大
- WildGaussians：基于 DINO 特征预测不确定性转换为掩码，但早期掩码不准确
- T-3DGS：引入无监督瞬态检测器和视频目标分割模块，适用于视频但复杂度高
- 启发：问题的根源分析（致密化→过拟合瞬态）比直接设计复杂模块更有价值，简洁的方案往往更鲁棒

## 评分

- 新颖性: ⭐⭐⭐⭐ （核心观察新颖，方法简洁有效，但整体框架设计较直觉）
- 实验充分度: ⭐⭐⭐⭐⭐ （两个标准基准，详细消融，多基线对比）
- 写作质量: ⭐⭐⭐⭐⭐ （动机分析清晰，Figure 2 的实验分析非常有说服力）
- 价值: ⭐⭐⭐⭐ （为 wild 场景下的 3DGS 提供了简洁高效的解决方案）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt](3dgs_lm_faster_gaussian_splatting_optimization_with_levenberg_marquardt.md)
- [\[CVPR 2026\] EDGS: Eliminating Densification for Efficient Convergence of 3DGS](../../CVPR2026/3d_vision/edgs_eliminating_densification_for_efficient_convergence_of_3dgs.md)
- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)
- [\[ICCV 2025\] LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling](localdygs_multi-view_global_dynamic_scene_modeling_via_adaptive_local_implicit_f.md)
- [\[CVPR 2025\] HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting](../../CVPR2025/3d_vision/hybridgs_decoupling_transients_and_statics_with_2d_and_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
