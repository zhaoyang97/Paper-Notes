---
title: >-
  [论文解读] DACoN: DINO for Anime Paint Bucket Colorization with Any Number of Reference Images
description: >-
  [ICCV 2025][视频生成][动画上色] 提出DACoN，利用DINOv2基础模型的语义特征与U-Net的高分辨率空间特征融合，实现支持任意数量参考图像的动画线稿自动上色，在关键帧和连续帧上色任务中均超越现有方法。 动画制作中，线稿着色是耗时且重复的人工作业。现有方法面临三大瓶颈： 参考图像数量受限：之前的方法（如An…
tags:
  - "ICCV 2025"
  - "视频生成"
  - "动画上色"
  - "DINOv2"
  - "线稿着色"
  - "多参考图像"
  - "语义对应"
---

# DACoN: DINO for Anime Paint Bucket Colorization with Any Number of Reference Images

**会议**: ICCV 2025  
**arXiv**: [2509.14685](https://arxiv.org/abs/2509.14685)  
**代码**: [https://github.com/kzmngt/DACoN](https://github.com/kzmngt/DACoN)  
**领域**: 视频生成  
**关键词**: 动画上色, DINOv2, 线稿着色, 多参考图像, 语义对应

## 一句话总结

提出DACoN，利用DINOv2基础模型的语义特征与U-Net的高分辨率空间特征融合，实现支持任意数量参考图像的动画线稿自动上色，在关键帧和连续帧上色任务中均超越现有方法。

## 研究背景与动机

动画制作中，线稿着色是耗时且重复的人工作业。现有方法面临三大瓶颈：

**参考图像数量受限**：之前的方法（如AnT）依赖Multiplex Transformer，固定了输入结构，最多支持1-2张参考图像，无法利用多角度色彩设计图的完整信息

**关键帧上色精度低**：关键帧之间构图差异大（姿态、视角、遮挡变化），线稿仅包含轮廓信息，难以建立语义对应，导致精度远低于连续帧上色

**生成式方法的局限**：基于扩散模型的方法可能生成参考图中不存在的颜色，且输出为合成图像而非分段层，难以后期修改

本文的关键发现是：**DINOv2能在线稿上捕捉部件级语义信息**（如手臂、头发等），即使线稿仅包含轮廓，DINO特征的PCA可视化仍能展现清晰的语义分区。这为基于对应关系的上色提供了强大的先验。

## 方法详解

### 整体框架

DACoN的工作流程：给定 $K$ 张参考图像和目标线稿 → 分别提取每张图像的特征（模型不查看其他图像，因此无数量限制）→ 通过segment pooling获得逐区段特征 → 计算参考-目标区段间余弦相似度 → 将最相似参考区段的颜色传播到目标。

设参考图像为 $L_k \in \mathbb{R}^{H \times W \times 3}$（$k=1,...,K$），目标图像为 $L_t$，各自的区段mask为 $m_k \in \mathbb{R}^{M_k \times H \times W}$ 和 $m_t \in \mathbb{R}^{N \times H \times W}$，参考图颜色信息为 $c_k \in \mathbb{R}^{M_k \times 3}$。

### 关键设计

1. **DINOv2 + U-Net双流特征提取**：

    - **DINOv2编码器**（冻结）：提取 $C_d=1024$ 维低分辨率语义特征 $D \in \mathbb{R}^{H_d \times W_d \times C_d}$，输入尺寸518×518
    - **U-Net编码器**（可训练）：提取 $C_u=128$ 维高分辨率空间特征 $U \in \mathbb{R}^{H \times W \times C_u}$，输入尺寸512×512
    - 设计动机：DINOv2提供全局语义理解能力（"这是手臂"），U-Net提供局部精细特征（"这个区段的精确形状和位置"）

2. **Segment Pooling与特征融合**：

    - 将特征图resize到原始分辨率，通过区段mask逐元素相乘后空间平均：
    $d = \text{avg}(D \odot m), \quad u = \text{avg}(U \odot m)$
    - 对DINO特征通过MLP降维 $C_d \to C_u$，然后与CNN特征拼接
    - 再经MLP融合得到最终区段特征 $f_k \in \mathbb{R}^{M_k \times C_u}$
    - 关键优势：**每张图像独立处理**，特征提取不依赖其他图像，因此参考图像数量无限制

3. **区段对应与颜色传播**：

    - 将所有 $K$ 张参考图的区段特征沿区段维度拼接：$f_r \in \mathbb{R}^{M \times C_u}$，$M = \sum_k M_k$
    - 计算参考与目标特征的余弦相似度矩阵 $\hat{S} \in \mathbb{R}^{M \times N}$
    - 用argmax找到每个目标区段最相似的参考区段，传播其颜色
    - 应用温度为 $T=0.1$ 的softmax增强分布锐化

### 损失函数 / 训练策略

总损失为两项加权和：
$$\mathcal{L}_{\text{final}} = \lambda_{ce} \mathcal{L}_{ce} + \lambda_{dc} \mathcal{L}_{dc}$$

- **交叉熵损失** $\mathcal{L}_{ce}$：将上色视为分类问题，对softmax后的颜色概率计算交叉熵，ground truth缺失的颜色不参与计算
- **DINO引导的特征一致性损失** $\mathcal{L}_{dc}$：鼓励模型最终的相似度矩阵 $\hat{S}$ 与纯DINO特征的相似度矩阵 $\hat{S}'$ 保持一致：
$$\mathcal{L}_{dc} = \frac{1}{MN} \sum_{m,n} |\hat{S}_{m,n} - \hat{S}'_{m,n}|$$

超参设置：$\lambda_{ce}=0.5$, $\lambda_{dc}=0.2$。训练5个epoch，batch size=2，Adam优化器，学习率 $1 \times 10^{-4}$，单卡RTX 4090训练14小时。

## 实验关键数据

### 主实验（关键帧上色，3D渲染测试集，8个角色子集）

| 方法 | 参考数 | Acc | Acc-Thresh | Pix-Acc | Pix-F-Acc |
|------|--------|-----|------------|---------|-----------|
| ColorFlow | 1 | 12.10 | 13.13 | 53.75 | 7.51 |
| MangaNinja | 1 | 17.39 | 19.52 | 8.68 | 34.51 |
| AniDoc | 1 | 23.98 | 27.06 | 78.71 | 50.25 |
| BasicPBC-Ref | 1 | - | 63.46 | 95.31 | 82.69 |
| **DACoN (本文)** | 1 | **70.61** | **75.15** | **97.82** | **92.16** |
| **DACoN (本文)** | 5 | **75.49** | **79.55** | **98.43** | **94.72** |
| **DACoN (本文)** | max | **76.66** | **80.68** | **98.60** | **95.04** |

### 消融实验

| 配置 | Acc (1-ref) | Acc (5-ref) | 说明 |
|------|-------------|-------------|------|
| DACoN完整版 | 67.87 | 73.25 | 完整测试集 |
| 无颜色信息 (Mono) | 64.32 | 70.26 | 去除线条颜色 |
| 去除 $\mathcal{L}_{dc}$ | 63.95 | 72.57 | 去除DINO一致性损失 |

连续帧上色（3D渲染）：

| 方法 | Acc | Acc-Thresh | Pix-Acc |
|------|-----|------------|---------|
| AniDoc | 31.20 | 36.36 | 92.01 |
| LVCD | 37.63 | 43.77 | 66.20 |
| AnT (Cadmium) | 66.34 | 77.13 | 97.65 |
| BasicPBC | 82.66 | 87.26 | 99.05 |
| **DACoN (本文)** | **84.76** | **88.23** | **99.27** |

### 关键发现

- **多参考图像的收益**：从1张增加到5张参考图，关键帧精度提升约5个百分点（67.87→73.25），说明多视角颜色信息确实重要
- **选择性参考不必要**：即使部分参考图与目标相似度低，加入后也不会降低精度，简化了整体流程
- **前后帧同时参考效果最佳**：在连续帧任务中，同时使用前一帧和后一帧作参考（±1 frame）将精度从84.71提升到88.93
- **DINO一致性损失对关键帧至关重要**：去掉该损失后关键帧精度下降约4个百分点，但对连续帧影响较小
- **线条颜色信息有价值**：去除线条颜色（Mono模式）导致精度下降3-5个百分点，因为阴影和高光区域需要颜色线条来区分

## 亮点与洞察

- 发现DINOv2在线稿上的语义能力是一个重要的经验贡献，为动画领域引入基础模型提供了新方向
- 通过独立处理每张图像来消除参考数量限制，设计极其简洁有效
- 统一模型同时处理关键帧和连续帧两种任务，避免了维护多个模型
- 在实际动画制作场景中，分段式上色保留了可编辑性，比生成式方法更实用

## 局限与展望

- 角色离镜头太近导致身体部位超出画面时，U-Net无法获取完整线稿信息
- 极端姿态下性能下降，因为色彩设计图主要展示直立姿势
- 小的背景区域被前景部分包围时可能产生前景-背景误判
- DINOv2编码器被冻结，可能限制了对动画风格的适应能力

## 相关工作与启发

- 与BasicPBC相比，DACoN在连续帧上的提升虽然不大（+2%），但在关键帧上色上的跨越式提升（精度从63.46到75.15）才是核心贡献
- 基础模型在创意领域（动画、插画）的特征提取能力被严重低估，本文提供了有力证据
- 启发点：其他需要语义对应的创意任务（如漫画翻译、风格迁移）也可探索DINO特征

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将DINOv2引入动画上色是新颖的跨领域应用，但整体技术框架相对直接
- **实验充分度**: ⭐⭐⭐⭐ 覆盖关键帧和连续帧两种任务、多种比较方法、消融实验和失败分析
- **写作质量**: ⭐⭐⭐⭐ 方法描述清晰，DINO特征可视化直观有说服力
- **价值**: ⭐⭐⭐⭐⭐ 对实际动画制作工作流有直接帮助，开源代码增强了实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DIVE: Taming DINO for Subject-Driven Video Editing](dive_taming_dino_for_subject-driven_video_editing.md)
- [\[ICCV 2025\] SteerX: Creating Any Camera-Free 3D and 4D Scenes with Geometric Steering](steerx_creating_any_camera-free_3d_and_4d_scenes_with_geometric_steering.md)
- [\[CVPR 2026\] Composing Concepts from Images and Videos via Concept-prompt Binding](../../CVPR2026/video_generation/composing_concepts_from_images_and_videos_via_concept-prompt_binding.md)
- [\[CVPR 2026\] Scaling Zero-Shot Reference-to-Video Generation](../../CVPR2026/video_generation/scaling_zero-shot_reference-to-video_generation.md)
- [\[CVPR 2026\] ReDirector: Creating Any-Length Video Retakes with Rotary Camera Encoding](../../CVPR2026/video_generation/redirector_creating_any-length_video_retakes_with_rotary_camera_encoding.md)

</div>

<!-- RELATED:END -->
