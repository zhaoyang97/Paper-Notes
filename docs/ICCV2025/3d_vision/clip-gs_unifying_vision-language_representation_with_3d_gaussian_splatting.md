---
title: >-
  [论文解读] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting
description: >-
  [ICCV 2025][3D视觉][3D Gaussian Splatting] 提出 CLIP-GS，首个基于 3D Gaussian Splatting (3DGS) 的多模态表示学习框架。通过 GS Tokenizer 将 3DGS 序列化为 token，结合图像投票损失 (Image Voting Loss) 进行多模态对齐，在跨模态检索、零样本和少样本 3D 分类任务上全面超越基于点云的方法。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D Gaussian Splatting"
  - "多模态表示学习"
  - "CLIP"
  - "对比学习"
  - "零样本分类"
---

# CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting

**会议**: ICCV 2025  
**arXiv**: [2412.19142](https://arxiv.org/abs/2412.19142)  
**代码**: 无  
**领域**: 3D 视觉  
**关键词**: 3D Gaussian Splatting, 多模态表示学习, CLIP, 对比学习, 零样本分类

## 一句话总结

提出 CLIP-GS，首个基于 3D Gaussian Splatting (3DGS) 的多模态表示学习框架。通过 GS Tokenizer 将 3DGS 序列化为 token，结合图像投票损失 (Image Voting Loss) 进行多模态对齐，在跨模态检索、零样本和少样本 3D 分类任务上全面超越基于点云的方法。

## 研究背景与动机

3D 表示学习是 3D 视觉的基础课题。现有 3D 多模态模型（如 ULIP、OpenShape、Uni3D）主要处理**点云**输入。然而，点云作为稀疏空间表示，无法描述 3D 物体的纹理信息，重建能力有限。相比之下，3D Gaussian Splatting (3DGS) 作为新兴的 3D 表示技术，通过显式的高斯点（包含位置、旋转、缩放、颜色和不透明度属性）表示物体，具有更优的空间精度和几何形状捕获能力。

核心问题在于：**如何设计一个处理 3DGS 输入的 3D 编码器，并将其与 CLIP 的视觉-文本表示空间对齐？**

主要挑战包括：
- 3DGS 的属性维度（14维：位置3+颜色3+不透明度1+缩放3+旋转4）远多于点云（3维位置+可选颜色），直接输入现有网络效果差
- 不同视角渲染的图像特征差异大，单图对比学习可能导致次优优化
- 大规模 3DGS 数据的生成和存储成本高

## 方法详解

### 整体框架

CLIP-GS 包含三个核心流程：
1. **可扩展三元组生成**：从 Objaverse 筛选 ~240K 3D 模型，为每个生成 3DGS + 36 张渲染图 + 1 个文本描述
2. **GS 特征提取**：FPS & kNN 形成高斯 patch → GS Tokenizer 生成序列化 token → Transformer 层输出 3DGS 嵌入
3. **多模态对齐**：冻结 EVA-CLIP 的图像和文本编码器，仅训练 3DGS 编码器，通过对比损失对齐三种模态

### 关键设计

1. **GS Tokenizer**: 3DGS 的每个高斯点是 14 维向量（SH degree=0 时）。首先用 FPS & kNN 将 3DGS 组织成 $g$ 个局部 patch，每个 patch 包含 $n$ 个邻近高斯点。Tokenizer 中用 Sigmoid 归一化不透明度和缩放属性，将旋转四元数线性化为 3×3 旋转矩阵，然后采用**多路排序策略**（xyz-order、Hilbert曲线、Z-order）重组 patch。GS Refinement Block 分两路处理：位置+颜色属性经点云编码器提取特征，全部高斯属性经 1×3 卷积+BN+ReLU 提取特征，最后融合得到 GS token。

2. **Image Voting Loss**: 解决不同视角渲染图像特征差异大的问题。对每个 3DGS 采样 K=5 张不同视角的图像，利用预训练 EVA-CLIP 计算每张图像与文本描述的语义一致性得分 $S_i$，作为该图像在对比损失中的权重。得分计算为文本嵌入与图像嵌入的余弦相似度。这样，与文本语义更一致的视角获得更高权重，引导梯度优化方向：$\mathcal{L}_{\text{img}} = -\frac{1}{2N}\sum_{i=1}^N S_i \cdot (\text{Contra}(E_i^G, E^I) + \text{Contra}(E_i^I, E^G))$。

3. **高效 3DGS 生成**: 将球谐函数（SH）degree 设为 0（每个高斯点仅保留单一 RGB 颜色），大幅降低存储需求。用点云初始化 3DGS 的位置和颜色属性，仅需 5,000 次迭代即可完成优化。大多数 3DGS 包含 10K-20K 个高斯点。

### 损失函数 / 训练策略

总损失包含两部分：
- **3D-Text 对比损失** $\mathcal{L}_{\text{text}}$：3DGS 与文本一对一对应，标准对比学习
- **3D-Image 投票损失** $\mathcal{L}_{\text{img}}$：一对多关系，用投票机制加权

$$\mathcal{L} = \mathcal{L}_{\text{text}} + \mathcal{L}_{\text{img}}$$

EVA-CLIP 的图像和文本编码器完全冻结，仅训练 3DGS 编码器 $F^G$。Transformer 层使用点云模型预训练权重初始化。

## 实验关键数据

### 主实验（多模态检索，Objaverse-GS）

| 方法 | 3D表示 | Text→3D R@1 | 3D→Text R@1 | Image→3D R@1 | 3D→Image R@1 |
|---|---|---|---|---|---|
| ULIP 2 | 点云 | 4.5 | 5.3 | 5.6 | 25.0 |
| OpenShape-PointBERT | 点云 | 24.4 | 22.6 | 61.6 | 53.8 |
| Uni3D | 点云 | 27.8 | 23.1 | 65.1 | 49.3 |
| **CLIP-GS** | **3DGS** | **36.8** | **30.0** | **75.6** | **56.9** |

CLIP-GS 在 Text→3D 上提升 +9.0，Image→3D 上提升 +10.5。

### 消融实验（设计选择消融，Objaverse-GS 零样本）

| 配置 | 3D表示 | Top1 | Top3 | Top5 |
|---|---|---|---|---|
| Uni3D (基线) | P&C | 33.6 | 52.3 | 60.1 |
| + Fine-tune | P&C | 46.9 | 68.5 | 75.9 |
| + Fine-tune | 3DGS (全属性) | 44.8 | 66.3 | 74.1 |
| + GS Tokenizer | 3DGS | 47.9 | 69.9 | 76.8 |
| + Image Voting Loss | 3DGS | **48.5** | **70.3** | **77.5** |

直接使用全部 3DGS 属性反而降低性能（44.8 vs 46.9），GS Tokenizer 有效融合额外属性（+3.1 Top1），Image Voting Loss 进一步提升（+0.6 Top1）。

### 关键发现

1. **3DGS 全面优于点云**：在检索、零样本分类、少样本分类所有任务上，CLIP-GS 均超越最佳点云方法
2. **少样本分类**：在 ModelNet-GS 10-shot/10-way 上达 95.4%（±0.2），超越 PointRWKV 的 94.8%（±2.8），且标准差显著更小
3. **零样本分类**：仅用 ~240K 样本训练，远少于点云方法的百万级样本，即展现出优秀的零样本能力
4. **点云预训练权重有效**：使用点云预训练的 Transformer 权重初始化优于 2D 图像预训练权重

## 亮点与洞察

- **方向正确**：将多模态表示学习从点云扩展到 3DGS 是自然且有价值的方向，3DGS 的纹理和几何表达能力确实转化为了更好的下游性能
- **GS Tokenizer 设计巧妙**：分两路处理位置/颜色和高斯特有属性，既复用了点云编码器的预训练知识，又有效利用了 3DGS 的额外信息
- **Image Voting Loss 解决了一个实际问题**：不同视角渲染图像语义不一致是 3D-2D 对齐的固有挑战
- **数据效率高**：仅 ~240K 样本即可训练出超越百万级训练的点云模型

## 局限与展望

- 仅在对象级 3DGS（Objaverse、ModelNet）上验证，未扩展到场景级 3DGS
- SH degree 固定为 0，丢失了视角相关的颜色变化信息
- 240K 的训练数据规模仍偏小，随着 3DGS 数据积累可进一步提升
- 未探索 3DGS 编码器的更大参数规模（仅用了 ~88M 的 Base 模型）
- 缺少实际应用场景演示（如机器人导航、自动驾驶中的场景理解）

## 相关工作与启发

- 继承了 Uni3D 的训练范式（冻结 CLIP 编码器，训练 3D 编码器），但将输入从点云换为 3DGS
- GS Tokenizer 中的多路排序策略（Hilbert 曲线、Z-order）借鉴了空间填充曲线在序列化3D数据中的应用
- 点云预训练权重的有效迁移表明 3DGS 和点云在空间结构上有足够的共性

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个将 3DGS 引入多模态对齐预训练的工作，GS Tokenizer 和 Image Voting Loss 设计合理
- **实验充分度**: ⭐⭐⭐⭐ 检索+零样本+少样本+消融覆盖全面，与多个基线对比公平
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，方法介绍详细，图示直观
- **价值**: ⭐⭐⭐⭐ 开拓了 3DGS 多模态学习的新方向，实验结果有说服力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Online Language Splatting](online_language_splatting.md)
- [\[ICCV 2025\] 3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation](3d_gaussian_map_with_openset_semantic_grouping_for_visionlan.md)
- [\[ICCV 2025\] PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations](pcr-gs_colmap-free_3d_gaussian_splatting_via_pose_co-regularizations.md)
- [\[ICML 2025\] LaGa: Tackling View-Dependent Semantics in 3D Language Gaussian Splatting](../../ICML2025/3d_vision/tackling_view-dependent_semantics_in_3d_language_gaussian_splatting.md)
- [\[ICCV 2025\] AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting](autoocc_automatic_openended_semantic_occupancy_annotation_vi.md)

</div>

<!-- RELATED:END -->
