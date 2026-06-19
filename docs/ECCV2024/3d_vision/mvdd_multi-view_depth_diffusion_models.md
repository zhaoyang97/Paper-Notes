---
title: >-
  [论文解读] MVDD: Multi-View Depth Diffusion Models
description: >-
  [ECCV 2024][3D视觉][扩散模型] 提出MVDD，一个基于多视角深度图表示的扩散模型，通过极线"线段"注意力和去噪深度融合实现3D一致的高质量形状生成，可生成20K+点的稠密点云。 领域现状： 3D形状生成是AIGC的重要方向。现有生成模型包括基于隐式函数（AutoSDF、3D-LDM）、基于体素（Vox-Dif…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "扩散模型"
  - "多视角深度"
  - "3D形状生成"
  - "极线注意力"
  - "深度补全"
---

# MVDD: Multi-View Depth Diffusion Models

**会议**: ECCV 2024  
**arXiv**: [2312.04875](https://arxiv.org/abs/2312.04875)  
**代码**: [项目页面](https://mvdepth.github.io/)  
**领域**: 图像生成  
**关键词**: 扩散模型, 多视角深度, 3D形状生成, 极线注意力, 深度补全

## 一句话总结

提出MVDD，一个基于多视角深度图表示的扩散模型，通过极线"线段"注意力和去噪深度融合实现3D一致的高质量形状生成，可生成20K+点的稠密点云。

## 研究背景与动机

**领域现状**: 3D形状生成是AIGC的重要方向。现有生成模型包括基于隐式函数（AutoSDF、3D-LDM）、基于体素（Vox-Diff）和基于点云（DPM、PVD、LION）的方法。扩散模型在2D图像生成取得巨大成功，但在3D生成中复制这一成功仍具挑战。

**现有痛点**: (a) 隐式方法计算量随分辨率立方增长，或产生过度平滑的形状；(b) 点云扩散模型在非结构化数据上训练极慢（>10000 epochs），且只能生成约2048个点，无法捕捉精细细节；(c) 多视角RGB扩散的Janus问题和3D不一致性。

**核心矛盾**: 高质量3D生成需要高分辨率和精细细节，但现有点云/体素/隐式表示要么受限于分辨率，要么训练代价过高。需要一种既适合扩散框架又能高效表示复杂3D形状的表示。

**本文目标**: 设计一种能够生成3D一致的多视角深度图的扩散模型，用于高质量点云和mesh生成。

**切入角度**: 多视角深度图是一种将3D表面"注册"到2D网格的表示，天然适配2D扩散架构，同时能生成比体素/点云方法更高分辨率的输出。

**核心 idea**: 用多视角深度图代替点云/隐式函数作为3D扩散模型的生成目标，配合极线线段注意力解决跨视角一致性。

## 方法详解

### 整体框架

MVDD将3D形状$\mathcal{X}$表示为$N$张多视角深度图$\mathbf{x} \in \mathbb{R}^{N \times H \times W}$。前向过程对每张深度图独立加噪$T$步，反向过程通过U-Net去噪。关键是在去噪过程中引入跨视角条件化，使每个视角的去噪步以邻近视角为条件：

$$p_\theta(\mathbf{x}_{t-1}^v | \mathbf{x}_t^v, \mathbf{x}_t^{r_1:r_R}) := \mathcal{N}(\mathbf{x}_{t-1}^v; \mu_\theta(\mathbf{x}_t^v, \mathbf{x}_t^{r_1:r_R}, t), \beta_t \mathbf{I})$$

最终将多视角深度图反投影融合得到稠密点云（20K+点），可选用SAP重建高质量mesh。

### 关键设计

1. **极线"线段"注意力（Epipolar Line Segment Attention）**: 不同于MVDream的全注意力或SyncDreamer的极线注意力，MVDD利用当前步的深度估计值来缩小注意力范围。对于源视角$v$上的像素$v_{ij}$，先将其深度值$\mathbf{x}_t^{v_{ij}}$反投影到3D空间得到点$\rho^{v_{ij}}$：

$$\rho^{v_{ij}} = \mathbf{x}_t^{v_{ij}} A^{-1} v_{ij}$$

然后在该3D点沿射线方向取$k-1$个等间距点，投影到邻近视角$r$上形成"线段"，仅在这些位置采样特征作为K和V。设计动机：因为有当前步的深度估计，无需搜索整条极线，只需在估计位置附近搜索，兼顾效率和效果。

2. **可见性阈值过滤**: 通过检查反投影深度与邻近视角预测深度的差异来判断可见性：

$$M(r_{mn}) = \|z(\pi_{v \to r} \rho^{v_{ij}}) - \mathbf{x}_t^{r_{mn}}\| < \tau$$

不满足条件的位置attention权重被设为极小值。

3. **深度值拼接**: 将采样点的深度值$\{z(\rho_1^{v_{ij}}), \ldots, z(\rho_k^{v_{ij}})\}$拼接到V的特征维度，使模型能感知这些点的空间位置。直觉：如果$v_{ij}$的几何特征与某个采样点$\rho_1^{v_{ij}}$的特征匹配度高，则去噪后的深度应向该点的深度值靠拢。

4. **去噪深度融合（Denoising Depth Fusion）**: 即使有极线注意力保证语义一致，反投影的3D点仍可能不完美对齐产生"双层"问题。借鉴MVS方法，在去噪步的U-Net输出后进行深度平均：将每个像素从其他可见视角反投影的深度进行平均，再加噪进入下一步。可见性判断采用双阈值：

$$\|v_{ij} - v_{\tilde{i}\tilde{j}}\| < \psi_{\max}, \quad \frac{|\mathbf{x}^{v_{ij}} - z(\rho^{v_{\tilde{i}\tilde{j}}})|}{|\mathbf{x}^{v_{ij}}|} < \epsilon_\theta$$

仅在最后20步应用融合，最后一步额外进行深度过滤去除不可见点。

### 损失函数 / 训练策略

标准DDPM目标函数，预测噪声：

$$L_t = \mathbb{E}_{t, \mathbf{x}_0, \epsilon_t} \left[\|\epsilon_t - \epsilon_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \epsilon_t, t)\|^2\right]$$

训练设置：$T=1000$步，cosine调度，深度图分辨率$128 \times 128$，8个视角，8×A100训练约3000 epochs。

## 实验关键数据

### 主实验

**ShapeNet 无条件生成 (1-NNA EMD↓)**:

| 类别 | DPM | PVD | LION | 3D-LDM | IM-GAN | **MVDD** |
|------|-----|-----|------|--------|--------|----------|
| Airplane | 73.47 | 64.89 | 63.49 | 80.10 | 64.04 | **62.50** |
| Car | 80.33 | 71.29 | 65.70 | - | 57.04 | **56.80** |
| Chair | 65.73 | 56.14 | 57.31 | 65.30 | 55.54 | **54.51** |

**深度补全 (EMD×10²↓)**:

| 类别 | PointFlow | PVD | DPF-Net | **MVDD** |
|------|-----------|-----|---------|----------|
| Airplane | 1.180 | 1.030 | 1.105 | **0.900** |
| Chair | 3.649 | 2.939 | 3.320 | **2.400** |
| Car | 2.851 | 2.146 | 2.318 | **1.460** |

### 消融实验

**极线注意力设计消融（Chair, 1-NNA CD↓）**:

| 组件 | 完整模型 | 无线段注意力 | 无深度拼接 | 无阈值过滤 |
|------|----------|-------------|-----------|-----------|
| 1-NNA↓ | 最优 | 退化明显 | 质量下降 | 小幅退化 |

去噪深度融合的有效性在Fig.4中直观展示：不使用融合时出现明显的"双层"伪影。

### 关键发现

- MVDD生成的点云密度是现有点云扩散模型的**10倍**（20K+ vs 2048），能捕捉椅子横条、飞机薄翼等精细结构
- 随着点云密度增加，LION等稀疏方法性能急剧下降，MVDD保持稳定
- 深度补全任务上全面超越所有baseline，证明模型学到了真实的3D形状先验
- 可作为3D先验用于GAN反演等下游任务，防止几何塌缩

## 亮点与洞察

- **表示选择的洞察**: 多视角深度图将3D生成"降维"为2D生成，完美适配成熟的2D扩散架构，比在非结构化点云上去噪更高效
- **利用中间结果**: 极线"线段"注意力巧妙利用了扩散中间步的深度估计来缩小搜索范围，这在之前的多视角扩散中没有被利用
- **多功能性**: 同一个无条件生成模型可以直接用于深度补全和作为3D先验，展现了出色的灵活性

## 局限与展望

- $128 \times 128$的深度图分辨率仍有提升空间，更高分辨率可提供更多几何细节
- 8个视角的固定相机配置可能不适用于所有场景（如薄结构物体需要更多视角）
- 训练数据仅限ShapeNet单类别，未展示跨类别泛化能力
- 与当前text-to-3D主流方法（SDS-based）的结合尚未探索
- 推理需1000步去噪，速度仍有优化空间

## 相关工作与启发

- **DDPM/DDIM**: 基础扩散框架
- **MVDream**: 多视角RGB扩散，使用3D self-attention，MVDD借鉴了跨视角交互的思路但针对深度图设计了更高效的注意力
- **PVD**: 点云扩散baseline，直接在点位置上去噪，训练慢且点数受限
- **SAP**: 用于从点云重建mesh的后处理方法

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将多视角深度表示引入3D扩散生成，极线线段注意力设计巧妙
- **实验充分度**: ⭐⭐⭐⭐ — 生成/补全/GAN先验三个任务，定量+定性全面
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，模块设计有明确的物理直觉
- **价值**: ⭐⭐⭐⭐ — 多视角深度作为3D表示的潜力值得更多研究跟进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Diffusion Models for Monocular Depth Estimation: Overcoming Challenging Conditions](diffusion_models_for_monocular_depth_estimation_overcoming_challenging_condition.md)
- [\[ECCV 2024\] SEDiff: Structure Extraction for Domain Adaptive Depth Estimation via Denoising Diffusion Models](sediff_structure_extraction_for_domain_adaptive_depth_estimation_via_denoising_d.md)
- [\[ECCV 2024\] Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [\[ECCV 2024\] DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)
- [\[ECCV 2024\] Open-Vocabulary 3D Semantic Segmentation with Text-to-Image Diffusion Models](open-vocabulary_3d_semantic_segmentation_with_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
