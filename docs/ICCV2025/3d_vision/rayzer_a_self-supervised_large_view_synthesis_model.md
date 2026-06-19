---
title: >-
  [论文解读] RayZer: A Self-supervised Large View Synthesis Model
description: >-
  [ICCV 2025][3D视觉][自监督学习] 提出 RayZer，一个无需任何3D监督（无相机位姿/无场景几何标注）的自监督多视角3D视觉模型，通过将图像解耦为相机参数和场景表示实现3D感知自编码，在新视角合成任务上达到甚至超越依赖位姿标注的"oracle"方法。 自监督学习已驱动了 LLM、VLM 和视觉生成领域的基础…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "自监督学习"
  - "新视角合成"
  - "相机位姿估计"
  - "Plücker光线"
  - "Transformer"
---

# RayZer: A Self-supervised Large View Synthesis Model

**会议**: ICCV 2025  
**arXiv**: [2505.00702](https://arxiv.org/abs/2505.00702)  
**代码**: [Project Page](https://hwjiang1510.github.io/RayZer/)  
**领域**: 3D视觉  
**关键词**: 自监督学习, 新视角合成, 相机位姿估计, Plücker光线, Transformer

## 一句话总结

提出 RayZer，一个无需任何3D监督（无相机位姿/无场景几何标注）的自监督多视角3D视觉模型，通过将图像解耦为相机参数和场景表示实现3D感知自编码，在新视角合成任务上达到甚至超越依赖位姿标注的"oracle"方法。

## 研究背景与动机

自监督学习已驱动了 LLM、VLM 和视觉生成领域的基础模型崛起，但 **3D 视觉模型仍严重依赖真值3D几何和相机位姿标注**。这些标注通常来自耗时的优化方法（如COLMAP），且并不总是准确。这种依赖限制了3D模型的学习可扩展性和有效性。

关键问题是：**如果不提供任何3D监督，3D视觉模型能走多远？**

现有方法的具体限制包括：
- GS-LRM、LVSM 等"oracle"方法需要真值位姿进行训练和推理
- MegaSynth、Stereo4D 等使用合成数据扩展训练规模，但数据策划仍然费力
- COLMAP提供的位姿标注本身可能存在噪声，反而限制了依赖它的模型的上限
- RUST 等自监督方法使用隐式位姿表示，难以实现位姿-场景解耦，且不显式3D感知

RayZer 的核心洞察：**用模型自身预测的相机位姿来渲染目标视图并提供光度监督**，而不是使用真值位姿。这将自监督训练转化为一种3D感知的图像自编码问题。

## 方法详解

### 整体框架

RayZer 接收无位姿、无标定的多视角图像 $\mathcal{I} = \{I_i \in \mathbb{R}^{H \times W \times 3} | i = 1, ..., K\}$ 作为输入，依次预测：（1）相机参数（内参+位姿）→（2）Plücker光线图 →（3）潜在场景表示 →（4）渲染新视角。全程纯Transformer架构，24层（相机估计8层，场景编码8层，渲染解码8层）。

自监督训练的关键信息流控制：将输入图像分为两个不重叠子集 $\mathcal{I}_\mathcal{A}$（用于场景重建）和 $\mathcal{I}_\mathcal{B}$（提供监督），避免平凡解。

### 关键设计

1. **相机估计器 (Camera Estimator)**：

    - 使用可学习的相机token $\mathbf{p} \in \mathbb{R}^{K \times d}$（每视图一个），与图像token $\mathbf{f} \in \mathbb{R}^{Khw \times d}$ 拼接后输入全自注意力Transformer层
    - 选择一个参考视图（恒等旋转+零平移），其余视图预测相对位姿
    - 旋转使用6D连续表示，通过MLP预测：$p_i = \text{MLP}_{pose}([\mathbf{p}_i^*, \mathbf{p}_c^*])$
    - 内参用单一焦距值参数化：$\text{focal} = \text{MLP}_{focal}(\mathbf{p}_c^*)$
    - **设计动机**：低维、几何定义良好的SE(3)参数化有助于信息解耦；先预测位姿再重建场景（pose-first范式）提供了更好的互相正则化

2. **Plücker光线桥接**：将预测的SE(3)位姿和内参转换为像素对齐的Plücker光线图 $\mathcal{R} \in \mathbb{R}^{K \times H \times W \times 6}$。这是RayZer中**唯一的3D先验**，它同时编码了：

    - 2D像素与光线的对齐关系
    - 3D光线几何（方向和原点）
    - 相机、像素和场景之间的物理关系
   
   光线图经线性层token化后与图像token融合：$\mathbf{x}_\mathcal{A} = \text{MLP}_{fuse}([\mathbf{f}_\mathcal{A}, \mathbf{r}_\mathcal{A}])$
    - **关键细节**：使用原始图像token $\mathbf{f}$ 而非相机估计器输出的 $\mathbf{f}^*$，防止 $\mathcal{I}_\mathcal{B}$ 的信息泄露

3. **潜在集合场景表示与全Transformer渲染**：

    - 场景表示为可学习的token集合 $\mathbf{z} \in \mathbb{R}^{L \times d}$，不显式3D感知，3D属性完全通过学习获得
    - 场景重建：$\{\mathbf{z}^*, \mathbf{x}_\mathcal{A}^*\} = \mathcal{E}_{scene}(\{\mathbf{z}, \mathbf{x}_\mathcal{A}\})$
    - 渲染：给定目标相机的Plücker光线token $\mathbf{r}$，通过渲染解码器生成图像：$\hat{I} = \text{MLP}_{rgb}(\mathbf{r}^*)$
    - 与传统渲染公式 $v = R(\text{scene}, \text{ray})$ 类比，只是这里"渲染方程"是参数化的可学习模型

### 损失函数 / 训练策略

纯光度自监督损失：

$$\mathcal{L} = \frac{1}{K_\mathcal{B}} \sum_{\hat{I} \in \hat{\mathcal{I}}_\mathcal{B}} (\text{MSE}(I, \hat{I}) + \lambda \cdot \text{Percep}(I, \hat{I}))$$

其中 $\lambda = 0.2$ 为感知损失权重。训练学习率 $4 \times 10^{-4}$，余弦调度器，50,000次迭代，batch size 256，分辨率 $256 \times 256$，patch size 16。两个子集 $\mathcal{I}_\mathcal{A}$ 和 $\mathcal{I}_\mathcal{B}$ 在训练中随机采样。

## 实验关键数据

### 主实验

| 数据集 | 方法 | 训练监督 | 需GT位姿 | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|---------|---------|-------|-------|--------|
| DL3DV | GS-LRM | 2D+Camera | 是 | 23.49 | 0.712 | 0.252 |
| DL3DV | LVSM | 2D+Camera | 是 | 23.69 | 0.723 | 0.242 |
| DL3DV | **RayZer** | **2D only** | **否** | **24.36** | **0.757** | **0.209** |
| RealEstate | GS-LRM | 2D+Camera | 是 | 24.25 | 0.770 | 0.227 |
| RealEstate | LVSM | 2D+Camera | 是 | 27.00 | 0.851 | 0.157 |
| RealEstate | **RayZer** | **2D only** | **否** | **27.48** | **0.861** | **0.146** |
| Objaverse | LVSM | 2D+GT | 是 | **32.34** | **0.950** | **0.050** |
| Objaverse | RayZer | 2D only | 否 | 31.52 | 0.945 | 0.052 |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|------|-------|-------|--------|------|
| RayZer (完整) | **24.36** | **0.757** | **0.209** | pose-first + Plücker + latent |
| 3DGS表示 | - | - | failed | 显式3D表示训练不收敛 |
| 无Plücker光线,用SE(3) | 22.73 | 0.687 | 0.249 | 缺少像素级条件 |
| 无显式位姿,用隐式特征 | 23.13 | 0.700 | 0.251 | 信息泄露+不可内插 |
| Scene-first范式 | 13.31 | 0.338 | 0.732 | 先建场景再估位姿，崩溃 |

### 关键发现

- **RayZer在DL3DV和RealEstate上超越oracle方法**：这些数据集的位姿来自COLMAP，本身有噪声。自监督学习可以找到比COLMAP更有利于视图合成的位姿空间
- **在Objaverse（完美GT位姿）上略低于LVSM**：差距很小（32.34 vs 31.52），证实当GT位姿完美时oracle方法有优势
- **Pose-first范式至关重要**：scene-first范式的PSNR仅13.31，完全崩塌
- **Plücker光线是关键3D先验**：比直接编码SE(3)位姿效果好1.6dB PSNR
- 自监督学习的位姿可以进行几何内插，生成连续相机轨迹的新视角

## 亮点与洞察

- **自监督3D模型可媲美甚至超越有监督方法**：这一反直觉的结果说明COLMAP位姿的噪声是有监督方法的瓶颈
- **极简的3D先验**：唯一的3D先验就是Plücker光线结构，其余全由数据驱动学习
- **信息流控制精妙**：使用原始图像token而非相机估计器输出，防止信息泄露；两子集分离确保非平凡解
- **潜在表示优于显式3D**：3DGS表示完全无法收敛于自监督训练，验证了潜在表示在this设置下的必要性

## 局限与展望

- 学到的位姿空间与真实位姿空间不完全对应，限制了位姿估计的直接应用
- 连续视频帧训练效果好于无序图像集，说明对输入顺序仍有一定依赖
- 仅在256分辨率上训练和评估，高分辨率场景的表现有待验证
- 每个数据集单独训练，跨数据集的零样本泛化能力尚未验证
- 静态场景假设限制了对动态场景的处理能力

## 相关工作与启发

- 与RUST的三个关键区别：(1) pose-first vs scene-first, (2) 显式SE(3) vs 隐式位姿, (3) 纯自注意力 vs 交叉注意力
- LVSM的潜在集合场景表示是关键灵感来源，RayZer证明了该表示无需位姿监督也能有效学习
- 能够使用视频资源（在线视频丰富）进行训练，比依赖无序图像集更有扩展潜力

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 零3D监督下达到oracle级别性能，自监督3D学习的重要里程碑
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集(DL3DV/RealEstate/Objaverse)全面对比，消融+位姿分析透彻
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑缜密，动机清晰，设计选择有充分理由
- 价值: ⭐⭐⭐⭐⭐ 证明了3D视觉模型摆脱监督学习的可行性，开辟新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SpatialDreamer: Self-supervised Stereo Video Synthesis from Monocular Input](../../CVPR2025/3d_vision/spatialdreamer_self-supervised_stereo_video_synthesis_from_monocular_input.md)
- [\[CVPR 2026\] From None to All: Self-Supervised 3D Reconstruction via Novel View Synthesis](../../CVPR2026/3d_vision/from_none_to_all_self-supervised_3d_reconstruction_via_novel_view_synthesis.md)
- [\[ICCV 2025\] HumanOLAT: A Large-Scale Dataset for Full-Body Human Relighting and Novel-View Synthesis](humanolat_a_large-scale_dataset_for_full-body_human_relighting_and_novel-view_sy.md)
- [\[ICCV 2025\] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)
- [\[ICCV 2025\] SHeaP: Self-Supervised Head Geometry Predictor Learned via 2D Gaussians](sheap_self-supervised_head_geometry_predictor_learned_via_2d_gaussians.md)

</div>

<!-- RELATED:END -->
