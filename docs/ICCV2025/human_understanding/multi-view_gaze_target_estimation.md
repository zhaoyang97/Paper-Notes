---
title: >-
  [论文解读] Multi-view Gaze Target Estimation
description: >-
  [ICCV 2025][人体理解][注视目标估计] 本文首次将注视目标估计（GTE）从单视角扩展到多视角，通过头部信息聚合（HIA）、基于不确定性的注视选择（UGS）和基于极线的场景注意力（ESA）三个模块融合多相机信息，在自建 MVGT 数据集上显著超越单视角 SOTA，并实现了单视角方法无法处理的跨视角估计。
tags:
  - "ICCV 2025"
  - "人体理解"
  - "注视目标估计"
  - "多视角"
  - "跨视角"
  - "极线注意力"
  - "不确定性"
---

# Multi-view Gaze Target Estimation

**会议**: ICCV 2025  
**arXiv**: [2508.05857](https://arxiv.org/abs/2508.05857)  
**代码**: [https://www3.cs.stonybrook.edu/~cvl/multiview_gte.html](https://www3.cs.stonybrook.edu/~cvl/multiview_gte.html)  
**领域**: 人体理解  
**关键词**: 注视目标估计, 多视角, 跨视角, 极线注意力, 不确定性

## 一句话总结
本文首次将注视目标估计（GTE）从单视角扩展到多视角，通过头部信息聚合（HIA）、基于不确定性的注视选择（UGS）和基于极线的场景注意力（ESA）三个模块融合多相机信息，在自建 MVGT 数据集上显著超越单视角 SOTA，并实现了单视角方法无法处理的跨视角估计。

## 研究背景与动机

**领域现状**：注视目标估计（GTE）旨在通过普通场景相机判断一个人正在看哪里。近年来基于深度学习的方法取得了长足进步，包括利用深度图、3D 视场角热力图、Transformer 架构等。

**现有痛点**：所有现有方法都局限于单视角。当主体面部被遮挡、视野内存在多个候选目标、或注视目标在画面外时，单视角方法都无能为力。

**核心矛盾**：单相机视野有限，既无法获取被遮挡头部的外观信息来精确估计注视方向，也无法预测画面外的注视目标。

**本文目标** (a) 如何利用多相机视角中对面部更清晰的拍摄来提升注视方向估计精度；(b) 如何在不同视角间传播场景背景信息；(c) 如何处理主体和目标不在同一相机中的跨视角场景。

**切入角度**：多相机系统在超市、教室等场景广泛存在，天然提供了更广的覆盖范围和多角度观察。作者利用已标定的多相机几何关系，在特征层面高效融合多视角信息，而非做完整的 3D 重建。

**核心 idea**：通过头部跨视角注意力聚合面部信息、不确定性驱动选择可靠注视方向、极线约束注意力共享场景上下文，实现多视角 GTE。

## 方法详解

### 整体框架
输入是一对标定相机的图像 $\mathbf{I}_1, \mathbf{I}_2$、对应的头部边界框和相机内外参。模型首先通过 HIA 模块聚合两视角的头部特征，然后 UGS 模块选择更可靠的注视向量生成 FoV 热力图，最后多视角场景编码器（含 ESA 模块）融合跨视角场景特征，经注视解码器输出目标热力图和 in/out 概率。

### 关键设计

1. **Head Information Aggregation（HIA）模块**:

    - 功能：从另一视角的头部图像中聚合面部外观信息，增强本视角的头部嵌入
    - 核心思路：用 ResNet-18 提取两视角的头部特征 $\mathbf{F}^h$，将其展平为 token，通过跨注意力（cross-attention）聚合。关键创新在将相对相机旋转矩阵 $\mathbf{R}_{21} = \mathbf{R}_1 \mathbf{R}_2^{-1}$ 拼接到 Key 和 Value 中，让模型感知两视角的几何关系：$\tilde{\mathbf{F}}^h_1 = \mathbf{F}^h_1 + \text{CrossAtt}(\mathbf{Q}^h_1, \mathbf{K}^h_1, \mathbf{V}^h_1)$
    - 设计动机：当一个视角中主体背对相机时，另一视角可能拍到正面；几何关系编码让模型理解两视角头部姿态的对应关系

2. **Uncertainty-based Gaze Selection（UGS）模块**:

    - 功能：从两个视角预测的注视向量中选择更可靠的那一个
    - 核心思路：扩展注视估计器同时预测不确定性分数 $\sigma$，使用不确定性感知损失 $\mathcal{L}_{gaze} = \frac{1}{2\sigma^2}(1 - \cos(\mathbf{g}, \hat{\mathbf{g}})) + \frac{1}{2}\log(\sigma^2)$。选择 $\sigma$ 更小的视角的注视向量，通过相机变换 $\mathbf{g}'_j = \mathbf{R}_j \mathbf{R}_i^{-1} \mathbf{g}_i$ 转换到另一视角
    - 设计动机：即使 HIA 增强了信息传播，两视角预测质量仍可能不同（如一个视角面部被遮挡）。Aleatoric Uncertainty 自然地反映了输入图像的注视估计难度

3. **Epipolar-based Scene Attention（ESA）模块**:

    - 功能：在多视角场景编码器中传播跨视角的场景背景信息
    - 核心思路：对场景特征中的每个 token，计算其在另一视角中的极线（通过基础矩阵 $\mathbb{F}$），沿极线均匀采样 48 个特征向量做跨注意力。比密集跨注意力更高效，且利用了极线几何约束
    - 设计动机：极线约束确保只关注 3D 空间中对应位置附近的特征，特别有利于处理遮挡区域（被遮挡物体的信息可从另一视角获取）

### 损失函数 / 训练策略
总损失 $\mathcal{L} = \alpha \mathcal{L}_{hm} + \beta \mathcal{L}_{io} + \lambda \mathcal{L}_{gaze}$，其中 $\mathcal{L}_{hm}$ 是热力图 MSE 损失，$\mathcal{L}_{io}$ 是 in/out 二分类 BCE 损失，$\mathcal{L}_{gaze}$ 是不确定性感知注视损失。模型先在 GazeFollow 数据集上预训练单视角版本，再在 MVGT 上微调完整多视角模型。

此外，论文还提出了跨视角 GTE 的扩展：当主体只在参考视角、目标只在主视角时，通过 Dust3R 预重建场景获取绝对深度，将参考视角的注视向量转换到主视角生成 FoV 热力图，并增加 feature transform 模块和可学习的 "outside embedding"。

## 实验关键数据

### 主实验
在 MVGT 数据集上的 leave-one-scene-out 评估（参考视角中头部可见 + 目标可见时的结果）：

| 方法 | Dist. ↓ | AP ↑ | 特点 |
|------|---------|------|------|
| Chong [CVPR'20] | 0.159 | 0.855 | 仅RGB输入 |
| Miao [CVPR'22] | 0.141 | 0.886 | 使用深度 |
| Tafasca [ECCV'24] | 0.149 | 0.893 | FoV热力图 |
| Ours-Single | 0.151 | 0.877 | 本文单视角基线 |
| **Ours** | **0.129** | **0.909** | 多视角完整模型 |

当参考视角头部可见时，本文方法相对单视角基线 Dist. 降低 14.6%；当面部正面/侧面可见时，误差降低率分别为 23.7% 和 23.2%。

### 消融实验

| 配置 | Dist. ↓ (Head+Target可见) | AP ↑ |
|------|---------------------------|------|
| 单视角基线 | 0.151 | 0.877 |
| + σ 不确定性 | 0.145 | 0.874 |
| + HIA | 0.135 | 0.896 |
| + HIA + UGS | 0.130 | 0.902 |
| + HIA + UGS + ESA (完整) | 0.129 | 0.909 |

### 关键发现
- **HIA 贡献最大**：加入 HIA 后 Dist. 从 0.145 降至 0.135（降 6.9%），说明融合另一视角的头部外观是最有效的改进来源
- **UGS 在头部不可见时作用更小**：当参考视角无头部时 UGS 无法发挥选择优势
- **ESA 在目标可见时更有效**：证实了其作用是从另一视角补充目标区域的场景信息
- **跨视角估计**：本文方法在跨视角任务上 Dist.=0.188, AP=0.820，远优于适配的 Recasens 方法（Dist.=0.271, AP=0.542）

## 亮点与洞察
- **极线约束的高效跨视角注意力**：不做密集全图注意力，而是沿极线采样，既利用了多视角几何先验又大幅降低了计算量。这个设计可迁移到其他多视角理解任务
- **不确定性驱动的多视角融合**：用 aleatoric uncertainty 自动评估每个视角的预测质量，优雅地解决了"哪个视角更可靠"的问题。这种思路在多传感器融合中普遍适用
- **非侵入式数据采集协议**：用激光笔标注注视目标后关闭激光再拍照，避免了图像中的标注伪影，比传统的主观标注更精确

## 局限与展望
- **依赖已标定的相机参数**：实际部署中相机标定可能不准确或动态变化，需要研究几何感知特征学习来降低对精确标定的依赖
- **跨视角任务依赖预重建**：需要 Dust3R 预重建场景获取绝对深度，增加了部署成本
- **数据集规模有限**：仅 4 个场景、28 名受试者，泛化性待验证
- **仅处理成对视角**：虽然多于两个视角可通过聚合多对结果实现，但缺乏端到端的多视角（>2）融合方案

## 相关工作与启发
- **vs Tafasca [ECCV'24]**：Tafasca 用 FoV 热力图作为注视先验，本文在此基础上通过 UGS 选择更可靠的注视向量来生成更高质量的 FoV 热力图
- **vs 多视角 3D 姿态估计**：姿态估计中的多视角方法通常要求大量视角重叠，而 GTE 场景中视角重叠可能很少
- **vs DVGaze [ICCV'23]**：DVGaze 探索了双视角注视方向估计，但要求面部可见且经过矫正，无法直接用于 GTE

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次探索多视角 GTE 任务，三个模块设计合理但各自并非全新
- 实验充分度: ⭐⭐⭐⭐ 消融详尽，跨视角实验有说服力，但数据集较小
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，方法描述流畅，图示精美
- 价值: ⭐⭐⭐⭐ 开辟了多视角 GTE 新方向，数据集和代码公开

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Gaze Target Estimation Anywhere with Concepts](../../CVPR2026/human_understanding/gaze_target_estimation_anywhere_with_concepts.md)
- [\[AAAI 2026\] Toward Gaze Target Detection in Young Autistic Children](../../AAAI2026/human_understanding/toward_gaze_target_detection_of_young_autistic_children.md)
- [\[ECCV 2024\] Gaze Target Detection Based on Head-Local-Global Coordination](../../ECCV2024/human_understanding/gaze_target_detection_based_on_head-local-global_coordination.md)
- [\[CVPR 2025\] Enhancing 3D Gaze Estimation in the Wild Using Weak Supervision with Gaze Following Labels](../../CVPR2025/human_understanding/enhancing_3d_gaze_estimation_in_the_wild_using_weak_supervision_with_gaze_follow.md)
- [\[CVPR 2025\] GA3CE: Unconstrained 3D Gaze Estimation with Gaze-Aware 3D Context Encoding](../../CVPR2025/human_understanding/ga3ce_unconstrained_3d_gaze_estimation_with_gaze-aware_3d_context_encoding.md)

</div>

<!-- RELATED:END -->
