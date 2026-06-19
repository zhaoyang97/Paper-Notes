---
title: >-
  [论文解读] Material Anything: Generating Materials for Any 3D Object via Diffusion
description: >-
  [CVPR 2025][3D视觉][PBR材质生成] 提出 Material Anything，一个全自动的统一扩散框架，通过三头 U-Net 架构、置信度掩码和渲染损失适配预训练图像扩散模型生成 PBR 材质（albedo/roughness/metallic/bump），配合置信度引导的渐进式多视角生成策略和 UV 空间精化模型，为不同光照条件（无纹理/纯 albedo/扫描/生成）下的 3D 物体统一生成高质量材质贴图。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "PBR材质生成"
  - "扩散模型"
  - "置信度掩码"
  - "三头U-Net"
  - "UV空间精化"
---

# Material Anything: Generating Materials for Any 3D Object via Diffusion

**会议**: CVPR 2025  
**arXiv**: [2411.15138](https://arxiv.org/abs/2411.15138)  
**代码**: 未开源  
**领域**: 三维视觉 / 材质生成  
**关键词**: PBR材质生成, 扩散模型, 置信度掩码, 三头U-Net, UV空间精化

## 一句话总结

提出 Material Anything，一个全自动的统一扩散框架，通过三头 U-Net 架构、置信度掩码和渲染损失适配预训练图像扩散模型生成 PBR 材质（albedo/roughness/metallic/bump），配合置信度引导的渐进式多视角生成策略和 UV 空间精化模型，为不同光照条件（无纹理/纯 albedo/扫描/生成）下的 3D 物体统一生成高质量材质贴图。

## 研究背景与动机

**领域现状**：PBR（物理渲染）材质是 3D 物体在不同光照下保持一致外观的关键。手工创建材质耗时费力，现有 3D 纹理生成方法（TEXTure, Paint3D, SyncMVD）生成的纹理包含烘焙的光照信息（高光/阴影），缺乏真正的材质建模。

**现有痛点**：(1) 基于优化的方法（NvDiffRec, DreamMat）每个物体需要独立优化，耗时且难以自动化；(2) 基于检索的方法（Make-it-Real）依赖 SAM+GPT 分割和材质匹配，系统脆弱且无法处理精细区域；(3) 所有方法都对光照条件敏感——真实扫描有复杂光照、生成纹理有不真实光照、albedo 无光照——缺乏统一处理方案。

**核心矛盾**：如何用一个统一模型自动处理各种光照条件下的材质生成，同时保证多视角一致性？

**本文切入角度**：将 3D 材质生成重新定义为基于图像的材质估计任务，利用预训练图像扩散模型的强大先验，通过置信度掩码动态指示光照可靠程度——高置信度区域利用光照线索估计材质、低置信度区域基于语义生成材质。

## 方法详解

### 整体框架

两阶段管线：(1) **图像空间材质扩散**——对 3D 物体的每个渲染视角，以渲染图像、法线图和置信度掩码为条件，用三头 U-Net 生成 albedo/roughness-metallic/bump 三组材质图；(2) **UV 空间材质精化**——将多视角材质投影到 UV 空间后用精化扩散模型填充遮挡区域和消除接缝。对无纹理物体先用图像扩散模型生成粗纹理再处理。

### 关键设计

1. **三头扩散 U-Net (Triple-Head Architecture)**:
    - 将 U-Net 的初始卷积层 + 第一个 DownBlock 和最后一个 UpBlock + 输出卷积分为三个独立分支
    - 三个分支分别输出 albedo (RGB)、roughness-metallic (R=1, G=roughness, B=metallic)、bump (RGB)
    - 中间层共享以保持材质间的一致性，分支头避免材质间的耦合干扰
    - 设计动机：直接输出 12 通道 latent 的 vanilla U-Net 会产生耦合效应（如 bump 被 albedo 着色），而训练单独的材质 VAE 在有限 PBR 数据上不可行

2. **置信度掩码 (Confidence Mask)**:
    - 对不同光照条件设置不同置信度值引导模型行为：
        - 真实扫描（可靠光照）→ confidence=1，模型利用光照线索**估计**材质
        - 无光照/纯 albedo → confidence=0，模型基于语义**生成**材质
        - 生成纹理（不可靠光照）→ 已知区域 confidence=1、其他区域 confidence=0，自适应切换
    - 训练时通过数据增强模拟各种光照条件（随机光源类型 + 图像拼接 + 降质），置信度掩码标记降质区域
    - 设计动机：一个模型统一"材质估计"和"材质生成"两个任务，置信度掩码是切换器

3. **置信度引导的渐进式多视角生成**:
    - 逐视角生成材质，每个新视角用之前视角的已知材质初始化噪声 latent：$\hat{z}_t = \hat{z}_t \cdot (1-\hat{m}) + z_t \cdot \hat{m}$
    - 对生成光照的视角，额外用置信度掩码标记已知区域 (confidence=1) 和需生成区域 (confidence=0)
    - 最后将所有视角材质 bake 到 UV 空间，用 UV 精化扩散模型（输入粗材质 + CCM 坐标图）填孔和消缝
    - 设计动机：渐进式生成避免了多视角扩散的高分辨率/多通道瓶颈，置信度掩码同时服务于光照适配和多视角一致性

### 损失函数

- **v-prediction 损失**：$\mathcal{L}_v = \|\hat{V}_\theta(z_t; c, y) - v_t\|_2^2$（三头分别预测）
- **渲染损失**：解码材质图 → 随机光照条件下可微渲染 → 感知损失 $\mathcal{L}_p = \sum_l \|\phi_l(\hat{r}) - \phi_l(r)\|_2^2$（VGG 多层特征匹配）
- **L2 材质重建损失**：各材质通道的 L2 损失
- 渲染损失是稳定训练的关键——弥合自然图像与材质图之间的域差距

## 实验关键数据

### 主实验表

**定量比较 (FID/CLIP)：**

| 方法 | 类型 | FID↓ | CLIP Score↑ |
|------|------|------|-------------|
| Text2Tex | 学习/无纹理 | 116.41 | 30.33 |
| SyncMVD | 学习/无纹理 | 118.46 | 30.66 |
| NvDiffRec | 优化/无纹理 | 103.81 | 30.14 |
| DreamMat | 优化/无纹理 | 113.34 | 30.64 |
| **Ours** | **学习/无纹理** | **100.63** | **31.06** |
| Make-it-Real | 检索/有纹理 | 104.38 | 88.62 |
| **Ours** | **学习/有纹理** | **101.19** | **89.70** |

- 无纹理和有纹理两种设定下 FID/CLIP 均最优
- 与 Rodin Gen-1 和 Tripo3D（使用大规模数据）可比的结果

### 消融表

**三头 U-Net + 渲染损失消融 (材质 RMSE ↓)：**

| 配置 | Albedo | Roughness | Metallic | Bump |
|------|--------|-----------|----------|------|
| W/O Triple-head | 0.0800 | 0.1196 | 0.1584 | 0.0824 |
| W/O Rendering Loss | 0.1442 | 0.1943 | 0.2594 | 0.0716 |
| **Full** | **0.0604** | **0.0877** | **0.1193** | **0.0313** |

**置信度掩码消融 (Mean RMSE ↓)：**

| 配置 | 无光照 | 真实光照 | 不真实光照 | 平均 |
|------|--------|---------|-----------|------|
| W/O Confidence | 0.1521 | 0.1074 | 0.1111 | 0.1235 |
| **Full** | **0.1102** | **0.0747** | **0.0847** | **0.0899** |

### 关键发现

- 渲染损失是最关键组件——去除后所有材质 RMSE 大幅退化，尤其 metallic 恶化 2.2×
- 三头架构有效解耦材质——vanilla U-Net 的 bump 图被 albedo 着色
- 置信度掩码在三种光照条件下都改善了材质质量（平均 RMSE 降低 27%）
- 渐进式生成 + 置信度掩码在生成光照场景下消除了多视角材质的明显不一致

## 亮点与洞察

- **统一框架设计出色**：一个模型同时处理四种不同光照条件（无纹理/纯 albedo/扫描/生成），省去了过去各自为政的复杂管线
- **置信度掩码是优雅的设计**：作为模型"开关"在估计和生成模式间平滑切换，同时复用为多视角一致性工具
- **渲染损失的关键作用**：将材质通过可微渲染闭环到图像空间比较，是跨域训练（自然图像→材质图）的稳定器
- **Material3D 数据集**：80K 高质量 PBR 物体，多种光照条件渲染，为社区提供了材质生成的训练基础

## 局限性

- 材质分辨率受限于 latent diffusion 的分辨率（细节不如高分辨率手工材质）
- 对极端反射/透明材质的建模能力有限
- 渐进式多视角生成仍存在纹理接缝问题（虽然 UV 精化可缓解）
- 生成材质的物理准确性未验证（与真实测量的 BRDF 比较）
- 依赖预训练扩散模型的类别覆盖范围

## 相关工作与启发

- **从纹理生成到材质生成的升级**：纹理生成方法（Paint3D, TEXTure）的输出包含烘焙光照，Material Anything 证明了直接生成解耦的 PBR 材质是可行且更实用的
- **置信度机制的通用性**：置信度掩码的设计思路可推广到其他需要统一有监督/无监督信号的生成任务
- **与 3D 生成管线的协作**：可作为 3D 生成管线（如 LRM/InstantMesh）的下游模块，为生成的 3D 模型添加真实材质

## 评分

⭐⭐⭐⭐ — 统一框架设计优雅（置信度掩码同时解决光照适配和多视角一致性），在四种场景下均优于专用方法，工程价值高。三头架构和渲染损失的设计虽不算新颖但组合有效。材质分辨率和物理准确性有进步空间。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MVPaint: Synchronized Multi-View Diffusion for Painting Anything 3D](mvpaint_synchronized_multi-view_diffusion_for_painting_anything_3d.md)
- [\[CVPR 2025\] HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D](hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)
- [\[CVPR 2026\] MatMart: Material Reconstruction of 3D Objects via Diffusion](../../CVPR2026/3d_vision/matmart_material_reconstruction_of_3d_objects_via_diffusion.md)
- [\[CVPR 2025\] MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model](mvgenmaster_scaling_multi-view_generation_from_any_image_via_3d_priors_enhanced_.md)
- [\[CVPR 2025\] Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)

</div>

<!-- RELATED:END -->
