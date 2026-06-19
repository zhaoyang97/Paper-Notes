---
title: >-
  [论文解读] TriTex: Learning Texture from a Single Mesh via Triplane Semantic Features
description: >-
  [CVPR 2025][3D视觉][纹理迁移] 提出 TriTex，一种从单个纹理网格学习体积纹理场（volumetric texture field）的方法，利用Diff3F语义特征投影到三平面（triplane）表示中，通过卷积网络和MLP实现语义感知的前馈式纹理迁移，在推理速度和纹理保真度上超越现有方法。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "纹理迁移"
  - "三平面表示"
  - "语义特征"
  - "单样本学习"
  - "3D网格"
---

# TriTex: Learning Texture from a Single Mesh via Triplane Semantic Features

**会议**: CVPR 2025  
**arXiv**: [2503.16630](https://arxiv.org/abs/2503.16630)  
**代码**: [项目主页](https://danacohen95.github.io/TriTex/)  
**领域**: 3D视觉  
**关键词**: 纹理迁移, 三平面表示, 语义特征, 单样本学习, 3D网格

## 一句话总结

提出 TriTex，一种从单个纹理网格学习体积纹理场（volumetric texture field）的方法，利用Diff3F语义特征投影到三平面（triplane）表示中，通过卷积网络和MLP实现语义感知的前馈式纹理迁移，在推理速度和纹理保真度上超越现有方法。

## 研究背景与动机

- 3D纹理迁移（将源网格的语义纹理应用到目标网格）是游戏开发、仿真和视频制作中的基本需求
- 现有的扩散模型方法（TEXTure, EASI-TEX等）擅长纹理生成，但难以忠实保留源纹理的外观
- SDS优化类方法（Latent-NeRF, Paint-it）处理单个物体需要较长时间，不适合大规模场景
- 迭代depth-conditioned inpainting方法容易产生视角不一致的伪影
- 基于IP-Adapter的方法（MVEdit, EASI-TEX）只能从参考图像获得模糊灵感，偏离源纹理
- 需要大规模3D数据集训练的方法（Texturify, AUV-net）限制了类别和数据可用性
- 缺乏仅从单个纹理网格学习并泛化到同类新目标网格的高效方法
- 纹理迁移需要隐式或显式的语义对应，而非简单的像素级复制

## 方法详解

### 整体框架

TriTex 的架构接收一个带预提取Diff3F语义特征的3D网格和一个3D查询点，输出该点的颜色。首先从6个正交视图将语义特征投影到三平面 $\mathcal{T} \in \mathbb{R}^{3 \times W \times H \times 2D}$，经三平面感知卷积块处理生成 $\mathcal{T}'$。推理时，对目标网格的查询点从三个平面采样特征，拼接后通过着色MLP $c: \mathbb{R}^{3D'} \to [0,1]^3$ 输出RGB颜色。训练仅在单个纹理网格上通过渲染重建损失完成。

### 关键设计

**设计一：Diff3F语义特征 + 三平面投影**
- **功能**：建立源和目标网格之间的语义对应关系
- **核心思路**：利用Diff3F（冻结的扩散模型+DINO特征）为网格提取零样本3D语义描述符，然后从6个正交方向（正反各3个）投影到三平面表示中。三平面感知卷积块在三个平面间交叉聚合特征（将每个平面沿轴平均后复制到其他平面），实现跨平面信息交互
- **设计动机**：Diff3F特征具有跨形状的语义一致性，使得在单个网格上学到的语义→颜色映射可以泛化到几何差异显著的同类物体；三平面表示允许使用高效的2D卷积处理3D信息

**设计二：单样本训练策略与数据增强**
- **功能**：仅从单个纹理网格学习，同时避免对特定三平面投影的过拟合
- **核心思路**：训练使用渲染重建损失 $\mathcal{L} = \mathbb{E}_\theta[\mathcal{L}_{MSE}(\theta) + \delta_{app}\mathcal{L}_{app}(\theta)]$，比较随机视角下预测纹理和真值纹理。两级数据增强：(1) 预处理阶段对网格施加3D变换并重新提取特征；(2) 训练时施加平移、缩放和小旋转扰动
- **设计动机**：单样本学习的核心挑战是泛化能力，数据增强扩大了语义特征的分布范围，防止网络过拟合到源网格特定的三平面投影

**设计三：着色神经场（Coloring Neural Field）**
- **功能**：将三平面语义特征映射到RGB颜色
- **核心思路**：对任意3D查询点，从三个处理后的平面通过双线性插值采样特征，拼接为单一向量，通过轻量MLP映射到 $[0,1]^3$ 颜色空间。位置编码提升细节生成能力（三平面分辨率仅 $32 \times 32$）
- **设计动机**：MLP的隐式表示天然具有空间连续性和泛化性，比直接在UV空间操作更鲁棒

### 损失函数

总损失 $\mathcal{L} = \mathbb{E}_\theta[\mathcal{L}_{MSE}(\theta) + \delta_{app}\mathcal{L}_{app}(\theta)]$，其中MSE损失保证像素级准确性，感知损失 $\mathcal{L}_{app}$ 强调高级语义特征对齐，改善纹理真实感。

## 实验关键数据

### 主实验：纹理迁移质量比较

| 方法 | SIFID↓ | CLIP sim.↑ | 推理时间 |
|------|--------|-----------|---------|
| TEXTure | 0.34 | 0.84 | 5 min |
| MVEdit | 0.38 | 0.84 | 1 min |
| EASI-TEX | 0.29 | 0.85 | 15 min |
| **TriTex** | **0.22** | **0.87** | **1 min** |

### 消融实验：各组件贡献

| 设置 | SIFID↓ | CLIP sim.↑ |
|------|--------|-----------|
| w/o network (最近邻) | 0.23 | 0.86 |
| w/o $\mathcal{L}_{MSE}$ | 0.23 | 0.86 |
| w/o $\mathcal{L}_{app}$ | 0.28 | 0.85 |
| w/o augmentations | 0.21 | 0.85 |
| **TriTex (full)** | **0.22** | **0.87** |

### 关键发现
- TriTex在SIFID（0.22 vs 0.29）和CLIP similarity（0.87 vs 0.85）上均优于最佳基线EASI-TEX
- 在Amazon Mechanical Turk用户研究中，TriTex在所有三个对比中被强烈偏好（>65%投票）
- 去除感知损失导致模糊输出（SIFID升至0.28），说明高级特征对齐至关重要
- 去除增强导致CLIP score下降（0.85），验证了泛化能力依赖增强
- 推理仅需1分钟，与MVEdit相当但质量更优

## 亮点与洞察

1. **单样本+语义对应的优雅组合**：借助预训练语义特征（Diff3F）将单样本学习转化为语义映射学习
2. **三平面表示的实用性**：在3D和2D操作之间取得良好平衡，支持高效的2D卷积处理
3. **跨形状泛化能力强**：尽管仅在一个网格上训练，对同类物体的显著形状变化仍保持良好迁移
4. **前馈推理速度快**：无需逐物体优化，适合大规模场景应用

## 局限与展望

- 缺乏生成能力：无法生成源纹理中不存在的新细节（如源犬无舌头则目标犬舌头无法着色）
- 不利用text-to-image先验：在语义特征匹配模糊时无法补偿
- 要求目标形状与源形状朝向一致，启用任意旋转会降低细节保持能力
- 跨类别迁移效果取决于语义重叠程度
- 未来可结合cross-attention处理参考图像，实现基于图像的前馈纹理化

## 相关工作与启发

- 与Texturify/AUV-net需要大规模数据集不同，TriTex仅需单个纹理网格
- 与Splice将DINO特征用于2D颜色迁移类似，TriTex将其扩展到3D语义纹理迁移
- Diff3F特征的跨形状语义一致性是方法成功的关键基础

## 评分

⭐⭐⭐⭐ — 方法简洁高效，单样本学习+三平面语义映射的设计巧妙；实验充分（定量+用户研究），在纹理保真度和速度上均有优势。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] BLADE: Single-view Body Mesh Learning through Accurate Depth Estimation](blade_single-view_body_mesh_estimation_through_accurate_depth_estimation.md)
- [\[CVPR 2025\] Coherent 3D Portrait Video Reconstruction via Triplane Fusion](coherent_3d_portrait_video_reconstruction_via_triplane_fusion.md)
- [\[CVPR 2025\] StdGEN: Semantic-Decomposed 3D Character Generation from Single Images](stdgen_semantic-decomposed_3d_character_generation_from_single_images.md)
- [\[CVPR 2025\] P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning](p-slcr_unsupervised_point_cloud_semantic_segmentation_via_prototypes_structure_l.md)
- [\[CVPR 2025\] Doppelgangers++: Improved Visual Disambiguation with Geometric 3D Features](doppelgangers_improved_visual_disambiguation_with_geometric_3d_features.md)

</div>

<!-- RELATED:END -->
