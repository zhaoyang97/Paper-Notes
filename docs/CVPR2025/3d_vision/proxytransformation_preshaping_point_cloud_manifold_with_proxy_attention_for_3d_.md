---
title: >-
  [论文解读] ProxyTransformation: Preshaping Point Cloud Manifold with Proxy Attention for 3D Visual Grounding
description: >-
  [CVPR 2025][3D视觉][3D视觉定位] 提出Proxy Transformation，通过可变形点云聚类和代理注意力机制，利用文本信息引导子流形平移、图像信息引导子流形内部变换，在训练前高效增强点云流形结构，在自我中心3D视觉定位任务上实现7.49%的显著提升。 自我中心3D视觉定位（ego-centric 3D…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D视觉定位"
  - "点云增强"
  - "可变形聚类"
  - "代理注意力"
  - "流形变换"
---

# ProxyTransformation: Preshaping Point Cloud Manifold with Proxy Attention for 3D Visual Grounding

**会议**: CVPR 2025  
**arXiv**: [2502.19247](https://arxiv.org/abs/2502.19247)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 3D视觉定位, 点云增强, 可变形聚类, 代理注意力, 流形变换

## 一句话总结

提出Proxy Transformation，通过可变形点云聚类和代理注意力机制，利用文本信息引导子流形平移、图像信息引导子流形内部变换，在训练前高效增强点云流形结构，在自我中心3D视觉定位任务上实现7.49%的显著提升。

## 研究背景与动机

自我中心3D视觉定位（ego-centric 3DVG）是具身智能的核心感知能力，需要从多视角RGB-D观测中根据语言描述定位3D目标。然而面临几个关键挑战：

1. **点云质量差**：从深度传感器重建的点云包含大量噪声（如非朗伯表面的深度误差），且由于计算限制只能稀疏采样（约2%），导致目标区域的流形结构被破坏
2. **背景冗余**：采样点中大量为背景区域，前景目标点密度不足
3. **现有增强方法不适用**：传统点云去噪/补全方法需要耗时的预处理，不适合实时场景；且只在单一点云模态上工作，无法利用任务中可用的多模态信息
4. **全局变换不可行**：场景级点云不同局部区域的流形结构差异大，无法用单一全局变换处理

核心问题是：如何在不增加离线预处理开销的前提下，利用多模态信息实时增强点云子流形？

## 方法详解

### 整体框架

在EmbodiedScan基线上添加Proxy Transformation模块：首先通过可变形点云聚类定位关键子流形区域，然后利用Proxy Attention联合文本和图像信息为每个子流形学习变换矩阵和平移向量，最终将变换后的点云送入3D backbone。

### 关键设计

**设计一：可变形点云聚类（Deformable Point Clustering）**

- **功能**：自适应地选择最关键的点云子区域进行变换
- **核心思路**：先初始化3D均匀网格作为参考点，以参考点为中心进行球查询得到初始聚类。然后用3D偏移网络（轻量CNN）预测每个参考点的偏移量，使聚类中心向关键区域移动。偏移后重新聚类得到更优的子流形区域
- **设计动机**：均匀网格提供稳定的空间先验，弥补稀疏采样导致的几何信息损失；可变形偏移使聚类中心自适应移动到前景/目标区域，增加多样性

$$\hat{q}_t = q_t + \Gamma_{offset}(\mathcal{N}_t)$$

**设计二：代理注意力（Proxy Attention）**

- **功能**：以线性复杂度实现跨模态特征交互
- **核心思路**：引入代理token $P$（可以是文本特征或图像特征），将标准 $O(N^2)$ 自注意力分解为两步：先由代理token压缩key-value（$\text{Attn}(P,K,V)$），再由原始query从压缩表示中广播信息（$\text{Attn}(Q,P,V^P)$），复杂度降为 $O(Nnd)$
- **设计动机**：场景级点云token数量 $N$ 巨大，标准注意力不可行。代理注意力通过少量proxy token（$n \ll N$）作为信息瓶颈，实现线性复杂度的全局交互

$$O^P = \sigma(QP^T) \sigma(PK^T) V$$

**设计三：文本引导平移 + 图像引导变换的双路分解**

- **功能**：利用不同模态的互补信息分别优化子流形间和子流形内的几何结构
- **核心思路**：文本特征包含全局空间关系信息（"椅子在桌子旁边"），用作代理token引导学习子流形间的平移向量 $T \in \mathbb{R}^{n \times 3}$；图像特征包含局部细粒度语义（纹理、姿态），用作代理token引导学习子流形内部的线性变换矩阵 $M \in \mathbb{R}^{n \times 3 \times 3}$
- **设计动机**：任意3D空间变换可分解为线性变换+平移。文本擅长表达空间关系，适合引导平移；图像擅长捕获局部细节，适合引导内部结构调整

$$\hat{\mathcal{P}} = \mathcal{M} \odot \mathcal{P} \oplus \mathcal{T}$$

### 损失函数

使用与基线相同的9-DoF边界框回归损失，Proxy Transformation模块通过端到端训练间接学习最优变换。

## 实验关键数据

### EmbodiedScan验证集主实验

| 方法 | 训练集 | Easy AP25 | Hard AP25 | Overall AP25 | Overall AP50 |
|------|--------|-----------|-----------|-------------|-------------|
| EmbodiedScan | Full | 39.82 | 31.02 | 39.10 | 18.48 |
| EmbodiedScan | Mini | 33.87 | 30.49 | 33.59 | 14.40 |
| DenseG | Mini | 40.17 | 34.38 | 39.70 | 18.31 |
| **ProxyTransformation** | **Mini** | **41.66** | **34.38** | **41.08** | **19.00** |

### 消融实验

| 组件 | Easy AP25 | Hard AP25 | Overall AP25 |
|------|-----------|-----------|-------------|
| Baseline | 37.05 | 30.60 | 36.53 |
| + Grid Prior | 40.39 | 32.60 | 39.76 |
| + Offsets | 40.59 | 32.18 | 39.91 |
| + Proxy Transformation | **41.66** | **34.38** | **41.08** |

### 注意力效率对比

| 注意力类型 | FLOPs | Params | Overall AP25 |
|-----------|-------|--------|-------------|
| Self-Attention | 8.36G | 2.52M | 40.14 |
| **Proxy Attention** | **5.29G** | **1.82M** | **41.08** |

### 关键发现

1. 仅用Mini训练集（~20%数据），ProxyTransformation超越了Full训练集基线，证明流形增强的有效性
2. Grid Prior单独贡献+3.23 Overall AP25，说明均匀网格先验对稀疏点云的补偿非常重要
3. Proxy Attention以约63%的FLOPs超越Self-Attention，证明代理机制的效率-精度优势
4. 注意力块计算开销减少40.6%，对实时应用意义重大

## 亮点与洞察

1. **多模态信息的精准分工**：文本→全局平移、图像→局部变换的设计非常对应两种模态的信息特性
2. **点云增强的新范式**：在feature learning之前通过坐标变换增强点云，而非在特征空间中处理，更直接且计算高效
3. **代理注意力的通用性**：作为一种通用的线性复杂度注意力机制，可以根据任务需求选择不同的proxy token

## 局限与展望

1. 变换参数（聚类数、每聚类点数等）需要手动设置，对不同场景可能需要调整
2. 子流形变换可能导致重叠或自交叉，缺乏几何合理性约束
3. 仅在EmbodiedScan基准上验证，未测试更广泛的3D理解任务
4. 可以探索将Proxy Transformation扩展到动态场景（如视频级3DVG）

## 相关工作与启发

- **EmbodiedScan**：ego-centric 3DVG的基准框架，本文在其上构建
- **Deformable DETR/DCNv3**：可变形偏移的思想从2D扩展到3D点云聚类
- **Linear Attention**：代理注意力与线性注意力有相似的复杂度优势，但通过代理token的选择获得更好的表达能力
- 启发：在特征学习之前对原始数据进行几何变换增强，是一种被低估的预处理策略

## 评分

⭐⭐⭐⭐ — 多模态引导点云增强的想法新颖且直觉性强，代理注意力设计优雅高效。用20%数据超越全量训练基线的结果很有说服力。不足在于仅在单一基准上验证且缺少代码开源。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Text-Guided Sparse Voxel Pruning for Efficient 3D Visual Grounding](text-guided_sparse_voxel_pruning_for_efficient_3d_visual_grounding.md)
- [\[CVPR 2025\] SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](seeground_see_and_ground_for_zero-shot_open-vocabulary_3d_visual_grounding.md)
- [\[CVPR 2025\] Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions](grounding_3d_object_affordance_with_language_instructions_visual_observations_an.md)
- [\[CVPR 2026\] PV-Ground: Text-Guided Point-Voxel Interaction for 3D Visual Grounding](../../CVPR2026/3d_vision/pv-ground_text-guided_point-voxel_interaction_for_3d_visual_grounding.md)
- [\[CVPR 2026\] EG-3DVG: Expression and Geometry Aware Grounding Decoder for 3D Visual Grounding](../../CVPR2026/3d_vision/eg-3dvg_expression_and_geometry_aware_grounding_decoder_for_3d_visual_grounding.md)

</div>

<!-- RELATED:END -->
