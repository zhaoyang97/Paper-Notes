---
title: >-
  [论文解读] Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging
description: >-
  [ICCV 2025][3D视觉][3D生成] 提出 Hi3DGen 框架，以法线图作为中间表示桥接 2D 图像到 3D 几何的映射，通过噪声注入回归式法线估计器（NiRNE）和法线正则化潜在扩散（NoRLD）两大核心组件，显著提升生成 3D 模型的几何细节保真度。 从 2D 图像生成高保真 3D 模型是计算机视觉的核心任务…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D生成"
  - "法线图桥接"
  - "噪声注入回归"
  - "双流架构"
  - "潜在扩散正则化"
  - "高保真几何"
---

# Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging

**会议**: ICCV 2025  
**arXiv**: [2503.22236](https://arxiv.org/abs/2503.22236)  
**领域**: 其他  
**关键词**: 3D生成, 法线图桥接, 噪声注入回归, 双流架构, 潜在扩散正则化, 高保真几何

## 一句话总结

提出 Hi3DGen 框架，以法线图作为中间表示桥接 2D 图像到 3D 几何的映射，通过噪声注入回归式法线估计器（NiRNE）和法线正则化潜在扩散（NoRLD）两大核心组件，显著提升生成 3D 模型的几何细节保真度。

## 研究背景与动机

从 2D 图像生成高保真 3D 模型是计算机视觉的核心任务，但现有方法在细粒度几何细节还原方面仍面临三大难题：

**高质量 3D 训练数据匮乏**：Objaverse 等数据集中高质量、富含复杂几何细节的 3D 资产严重不足，导致网络生成的模型过于简化

**域差距**：训练数据通常来自合成渲染，与真实世界图像存在显著风格差异，推理时性能明显下降

**RGB 图像的固有歧义**：光照、着色和复杂纹理使得从 RGB 图像中提取精确几何信息非常困难

现有的直接从 RGB 到 3D 的方法（如 CRM、InstantMesh、CraftsMan 等）无法充分保留输入图像中的细粒度几何特征。法线图作为 2.5D 表示，编码了表面朝向信息，天然具有更清晰的几何线索，且可以借助强大的 2D 先验来缓解域差距。

**核心洞察**：将 image-to-3D 分解为 image-to-normal 和 normal-to-geometry 两步，利用法线图作为桥接（normal bridging），可以同时缓解域差距和几何歧义问题。

## 方法详解

### 整体框架

Hi3DGen 包含三个核心组件：
- **NiRNE**（Noise-injected Regressive Normal Estimation）：从图像估计高质量法线图
- **NoRLD**（Normal-Regularized Latent Diffusion）：利用法线正则化来增强 3D 潜在扩散学习
- **DetailVerse 数据集**：合成的高质量 3D 数据集，提供丰富的几何细节训练数据

### 关键设计 1：噪声注入回归式法线估计（NiRNE）

**问题分析**：扩散式方法估计的法线更锐利但不稳定且有伪细节；回归式方法稳定但缺乏锐度。作者从频域角度分析了扩散方法产生锐利结果的根本原因。

**频域分析**：在扩散过程 $x_t = x_0 + \int_0^t g(s) dw_t$ 中，由于自然图像具有低通特性 $|\hat{x}_0(\omega)|^2 \propto |\omega|^{-\alpha}$，高频分量的 SNR 衰减更快。这意味着扩散模型在高频区域获得更强的监督信号，促使其更关注锐利细节。

**噪声注入**：受此启发，将噪声注入技术集成到回归框架中，使其对高频信息更敏感，兼顾锐利性和稳定性。

**双流架构**：
- **Clean Stream**：处理原始无噪声图像，稳健地捕获低频细节（整体结构信息）
- **Noisy Stream**：处理噪声注入后的图像，专注学习高频细节（边缘和细腻纹理）
- 两个流的特征以 ControlNet 风格拼接后送入解码器进行回归预测

**域特定训练策略**：
- **第一阶段**：使用真实域数据训练完整网络，学习低频信息以获得强泛化能力
- **第二阶段**：冻结 Clean Stream，仅用合成域数据微调 Noisy Stream，学习高频细节作为残差

这一设计巧妙地利用了真实数据（泛化好但高频标签噪声大）和合成数据（高频标签精确但有域差距）各自的优势。

### 关键设计 2：法线正则化潜在扩散（NoRLD）

现有 3D 潜在扩散方法（如 Trellis、CRM）仅在高度压缩的潜在空间进行监督，几何细节容易丢失。

**核心思想**：在扩散训练过程中，在线地对预测的潜在码进行法线图正则化，提供显式的 3D 几何监督：

$$\mathcal{L}_{\text{NoRLD}} = \mathcal{L}_{\text{LDM}} + \lambda \cdot \mathcal{R}_{\text{Normal}}(\hat{x}_0)$$

其中法线正则化项为：

$$\mathcal{R}_{\text{Normal}}(\hat{x}_0) = \mathbb{E}_v \left[ \| R_v(D(\hat{x}_0)) - N_v \|^2 \right]$$

即将预测的潜在码 $\hat{x}_0$ 解码为 3D 几何，从视点 $v$ 渲染法线图，与真值法线图对比。这一正则化在扩散训练中**在线进行**（非后处理），主动引导扩散网络学习包含丰富细节的分布。

### 损失函数

- **NiRNE**：标准回归损失（L2），域特定两阶段训练
- **NoRLD**：Flow matching 损失 + 法线渲染正则化损失
- $\mathcal{L}_{\text{LDM}}$：标准速度场匹配损失
- $\lambda \cdot \mathcal{R}_{\text{Normal}}$：在线法线正则化

### DetailVerse 数据集

通过 3D 数据合成 pipeline 构建的高质量数据集，弥补了 Objaverse 中高质量资产的不足：
- 具有高语义多样性、几何结构多样性和表面细节丰富性
- 为 NiRNE 提供干净的法线标签，为 NoRLD 提供高保真 3D 训练数据

## 实验关键数据

### 主实验

论文展示了与多种 SOTA 方法的对比结果（缓存中实验部分截断，基于论文描述总结）：

- Hi3DGen 在生成几何细节保真度方面超越了所有 SOTA 方法
- 对比方法包括：CRM、InstantMesh、CraftsMan、Trellis 等
- 在真实世界输入图像上表现尤为突出，得益于域差距的缓解

### 法线估计对比

NiRNE 同时实现了扩散式方法的锐利性和回归式方法的稳定性：
- 相比 StableNormal：稳定性更好，不会产生伪细节
- 相比 Marigold：锐利程度相当，但推理速度更快（单步回归 vs 多步扩散）
- 相比传统回归方法（如 DSINE）：细节更丰富锐利

### 关键发现

- 法线桥接策略有效缓解了域差距，使模型对各种风格的输入图像都能产生高保真几何
- 在线法线正则化显著提升了潜在扩散学习的几何细节保持能力
- DetailVerse 数据集的引入对高频细节的学习至关重要

## 亮点与洞察

1. **频域分析很有启发性**：从 SNR 衰减角度解释了扩散模型为何更擅长高频的原因，并据此将噪声注入引入回归框架，是一个非常优雅的理论-实践结合
2. **双流解耦设计精巧**：将低频泛化能力和高频细节能力解耦到两个独立流中，配合域特定训练策略，充分利用了不同数据源的优势
3. **在线法线正则化是关键创新**：不同于 CraftsMan 的后处理正则化或 Trellis 仅在 VAE 训练中使用法线损失，NoRLD 在扩散训练中在线引入法线监督，更直接有效
4. **问题分解的思路值得借鉴**：将困难的 image-to-3D 分解为两个相对容易的子问题，利用中间表示（法线图）降低整体学习难度
5. **数据补充策略实用**：合成 DetailVerse 数据集来弥补现有数据集的不足，是解决高质量 3D 数据匮乏的可行方案

## 局限性

1. 缓存截断，无法看到完整的定量实验和消融研究结果
2. 双流架构 + VAE 解码器使得法线正则化的计算开销较大
3. 依赖 DetailVerse 数据集的质量和多样性，可能在数据覆盖不到的类别上效果受限
4. 法线图作为中间表示虽然缓解了歧义，但仍然无法完全消除深度歧义（法线编码方向而非距离）
5. 两阶段 pipeline（先估法线再生成 3D）的误差可能累积

## 相关工作

- **3D 生成**：CRM、InstantMesh、CraftsMan、Trellis、Unique3D 等直接 image-to-3D 方法
- **法线估计**：扩散式（Marigold、GeoWizard、StableNormal）vs 回归式（DSINE、Metric3D）
- **法线在 3D 中的应用**：SDS 优化中的法线渲染损失、多视图法线融合
- **3D 数据集**：Objaverse、Objaverse-XL、MVImgNet

## 评分

- **新颖性**: ★★★★☆ — 法线桥接思路虽非首创，但噪声注入回归和在线法线正则化是显著创新
- **技术深度**: ★★★★★ — 频域分析严谨，双流设计和域特定训练策略理论依据充分
- **实验质量**: ★★★☆☆ — 缓存截断无法完整评估，但从方法描述看实验设计合理
- **实用性**: ★★★★☆ — 生成的高保真 3D 模型在游戏、影视、3D 打印等领域有直接应用价值
- **表达清晰度**: ★★★★★ — 论文结构清晰，频域分析和方法动机解释透彻

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation](bridging_3d_anomaly_localization_and_repair_via_high-qualit.md)
- [\[ICCV 2025\] GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting](gazegaussian_high-fidelity_gaze_redirection_with_3d_gaussian_splatting.md)
- [\[CVPR 2026\] LATTICE: Democratize High-Fidelity 3D Generation at Scale](../../CVPR2026/3d_vision/lattice_democratize_high-fidelity_3d_generation_at_scale.md)
- [\[ICCV 2025\] SegmentDreamer: Towards High-Fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation](segmentdreamer_towards_high-fidelity_text-to-3d_synthesis_with_segmented_consist.md)
- [\[CVPR 2025\] Geometry in Style: 3D Stylization via Surface Normal Deformation](../../CVPR2025/3d_vision/geometry_in_style_3d_stylization_via_surface_normal_deformation.md)

</div>

<!-- RELATED:END -->
