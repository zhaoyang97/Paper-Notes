---
title: >-
  [论文解读] unMORE: Unsupervised Multi-Object Segmentation via Center-Boundary Reasoning
description: >-
  [ICML2025][语义分割][图像分割] 提出 unMORE，通过学习三层物体中心表征（存在性/中心场/边界距离场）并设计无网络的多目标推理模块，实现无监督多目标分割，在 COCO 等 6 个数据集上大幅超越所有无监督方法。 无监督多目标分割旨在不依赖人工标注、从单张图像中发现并分割多个物体。现有方法主要分两类： Slo…
tags:
  - "ICML2025"
  - "语义分割"
  - "图像分割"
  - "object-centric representation"
  - "center field"
  - "boundary distance field"
  - "multi-object reasoning"
---

# unMORE: Unsupervised Multi-Object Segmentation via Center-Boundary Reasoning

**会议**: ICML2025  
**arXiv**: [2506.01778](https://arxiv.org/abs/2506.01778)  
**代码**: [GitHub](https://github.com/vLAR-group/unMORE)  
**领域**: 无监督分割 / 多目标发现  
**关键词**: unsupervised segmentation, object-centric representation, center field, boundary distance field, multi-object reasoning

## 一句话总结

提出 unMORE，通过学习三层物体中心表征（存在性/中心场/边界距离场）并设计无网络的多目标推理模块，实现无监督多目标分割，在 COCO 等 6 个数据集上大幅超越所有无监督方法。

## 研究背景与动机

无监督多目标分割旨在不依赖人工标注、从单张图像中发现并分割多个物体。现有方法主要分两类：

**Slot-based 方法**（如 SlotAttention）：依赖图像重建目标学习物体表征，在合成数据上有效但难以扩展到复杂真实场景

**自监督特征蒸馏方法**（如 TokenCut、CutLER、CuVLER）：利用 DINO/v2 预训练特征的物体定位线索发现多目标，但仍存在 **欠分割** 问题——倾向于将相邻物体聚合为一个

核心挑战在于：(1) 物体性（objectness）定义不明确；(2) 缺乏有效的多目标搜索机制。本文类比人类婴幼儿从单物体图像学习物体概念、再在复杂场景中识别多物体的能力，提出两阶段流水线。

## 方法详解

### 整体框架

unMORE 是一个两阶段流水线：

- **阶段一**：在 ImageNet 单物体图像上训练 Objectness Network，学习三层物体中心表征
- **阶段二**：利用训练好的（冻结的）Objectness Network，通过无网络的多目标推理模块在场景图像中发现多个物体

### 数据准备

利用 CuVLER 的 VoteCut 方法，对 ImageNet 每张图像提取 DINO/v2 patch 特征 → 构建亲和矩阵 → Normalized Cut → 选择最显著的前景 mask 作为粗略物体掩码。

### 三层物体中心表征

**1. 物体存在性分数 $f^e$**：二值分类，图像含有效物体则为 1，否则为 0。对 ImageNet 图像裁剪最大背景区域作为负样本。

**2. 物体中心场 $\boldsymbol{f}^c \in \mathbb{R}^{H \times W \times 2}$**：mask 内每个像素分配一个指向物体包围盒中心 $[C_h, C_w]$ 的单位向量，mask 外为零向量：

$$\boldsymbol{f}^c_{(h,w)} = \begin{cases} \frac{[h,w] - [C_h, C_w]}{\|[h,w] - [C_h, C_w]\|}, & \text{if } M_{(h,w)}=1 \\ [0,0], & \text{otherwise} \end{cases}$$

**3. 物体边界距离场 $\boldsymbol{f}^b \in \mathbb{R}^{H \times W \times 1}$**：先算带符号距离场（mask 内正、外负、边界零），再分别对前景/背景归一化：

$$\boldsymbol{f}^b_{(h,w)} = \begin{cases} S_{(h,w)} / \max(\boldsymbol{S} * \boldsymbol{M}), & \text{if } M_{(h,w)}=1 \\ S_{(h,w)} / |\min(\boldsymbol{S} * (\boldsymbol{1}-\boldsymbol{M}))|, & \text{otherwise} \end{cases}$$

关键性质：通过 $\boldsymbol{f}^b$ 的梯度可反推最大符号距离值 $S_{(\hat h,\hat w)} = 1 / \|[\partial \boldsymbol{f}^b / \partial h, \partial \boldsymbol{f}^b / \partial w]\|$，用于边界推理。

### Objectness Network 结构与训练

- **存在性分支**：ResNet50 二分类器 → 预测 $\tilde{f^e}$
- **中心场 + 边界距离场分支**：DPT-Large + 两个 CNN head → 分别预测 $\tilde{\boldsymbol{f}^c}$、$\tilde{\boldsymbol{f}^b}$
- **总损失**：

$$\ell = \text{CE}(\tilde{f^e}, f^e) + \ell_2(\tilde{\boldsymbol{f}^c}, \boldsymbol{f}^c) + \ell_1(\tilde{\boldsymbol{f}^b}, \boldsymbol{f}^b)$$

### 多目标推理模块（无网络）

**Step 0 — 初始提案生成**：在场景图像上均匀随机初始化 T 个 bounding box 提案，统一缩放到 128×128。

**Step 1 — 存在性检查**：查询每个提案的 $f^e_p$，低于阈值 $\tau^e$ 则丢弃。

**Step 2 — 中心推理**：查询中心场 $\boldsymbol{f}^c_p$，用预定义的 5×5 反中心核卷积生成 **anti-center map** $\boldsymbol{f}^{ac}_p$。若最大反中心值 > $\tau^c$，说明含 ≥2 个拥挤物体，在该位置将提案分裂为上/下/左/右 4 个子提案，回到 Step 1；否则用连通分量分割。

**Step 3 — 边界推理**：对单物体提案，取边界距离场四条边框的最大值，利用梯度反推实际像素距离，迭代更新提案的四个角点坐标——正值扩张、负值收缩，直到收敛到紧致包围盒。

最终对所有收敛提案做 NMS 去重，取边界距离场正值区域与中心场非零区域的并集作为物体 mask。

### 可选：训练检测器

将发现的物体作为伪标签训练 class-agnostic detector（Mask R-CNN），即 unMORE 的完整版本。

## 实验关键数据

### COCO* val（含额外 197 类标注）主要结果

| 方法 | 类型 | AP50^box | AP^box | AP50^mask | AP^mask |
|------|------|----------|--------|-----------|---------|
| VoteCut | 无学习 | 10.8 | 5.5 | 9.5 | 4.6 |
| DINOSAUR | SlotAtt | 2.0 | 0.6 | 1.1 | 0.3 |
| **unMORE_disc** | **Obj.Net** | **19.1** | **10.1** | **17.8** | **9.5** |
| CutLER | Det.×3 | 26.0 | 14.7 | 22.7 | 11.8 |
| CuVLER | Det.×2 | 28.0 | 15.5 | 24.4 | 12.6 |
| **unMORE** | **Obj.Net+Det.×1** | **32.6** | **18.0** | **29.6** | **15.5** |

- unMORE_disc（不训练检测器）已超越所有无学习基线约 **2× AP**
- unMORE（训练单轮检测器）相比 CuVLER AP^box 提升 +2.5，AP^mask 提升 +2.9

### 零样本跨数据集泛化（Table 2 摘要）

| 数据集 | CutLER AP50^box | CuVLER AP50^box | **unMORE AP50^box** |
|--------|-----------------|-----------------|---------------------|
| COCO20K | 22.4 | 24.1 | **25.9** |
| LVIS | 8.5 | 8.9 | **10.4** |

在 KITTI、VOC、Object365、OpenImages 上同样取得最佳无监督结果。

### 拥挤场景优势

所有基线在拥挤图像上性能崩溃，unMORE 由于中心推理的分裂机制能有效分离相邻物体，表现显著优于基线。

## 亮点与洞察

1. **三层表征设计精巧**：存在性→中心→边界三个层级分别回答"有没有/在哪/什么形状"，类比人类物体认知过程
2. **无网络推理模块**：Step 2/3 完全不含可学习参数，仅依赖中心场和边界距离场的几何性质进行迭代推理
3. **反中心核设计**：5×5 核向外辐射的单位向量与中心场做卷积，巧妙检测拥挤物体间的分界点
4. **边界距离场梯度性质**：归一化后梯度的倒数即可恢复物体尺寸，直接用于提案扩张/收缩
5. **COCO* 补充标注**：手动为 COCO val 补充 197 类标注，更公平评估无监督方法

## 局限与展望

1. **依赖预训练特征质量**：粗略 mask 来自 DINO/v2 + VoteCut，若预训练特征在特定领域（如医学影像）定位能力弱则效果受限
2. **推理效率**：Step 2/3 的迭代推理需要对每个提案多次前向传播，大量提案时计算开销较大
3. **仅类无关检测**：不提供类别信息，需额外分类器完成语义分割
4. **阈值敏感性**：$\tau^e$、$\tau^c$ 等超参数需要调优，论文未充分讨论跨数据集鲁棒性
5. **边界距离场对遮挡敏感**：严重遮挡时 mask 不完整，可能导致边界距离场估计偏差

## 相关工作与启发

- **CutLER / CuVLER**：强基线，但依赖多轮自训练检测器；unMORE 仅一轮即超越
- **SlotAttention / DINOSAUR**：重建目标驱动的物体发现，在真实场景严重失败
- **DeepSDF / Park et al.**：边界距离场（SDF）在 3D 重建中成功，本文首次将其用于 2D 无监督物体发现
- 启发：将物体中心表征的显式学习与无参数迭代推理结合，或许可推广到 3D 点云场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — 三层表征+无网络推理的组合非常新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 6 个数据集全面评估，含消融和 COCO* 补充标注
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图示优秀
- 价值: ⭐⭐⭐⭐ — 无监督分割新 SOTA，拥挤场景突破性进展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Pixel-Level Reasoning Segmentation via Multi-turn Conversations](../../ACL2025/segmentation/pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)
- [\[CVPR 2025\] M3-VOS: Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation](../../CVPR2025/segmentation/m3-vos_multi-phase_multi-transition_and_multi-scenery_video_object_segmentation.md)
- [\[ICLR 2026\] RegionReasoner: Region-Grounded Multi-Round Visual Reasoning](../../ICLR2026/segmentation/regionreasoner_region-grounded_multi-round_visual_reasoning.md)
- [\[ICCV 2025\] Ensemble Foreground Management for Unsupervised Object Discovery](../../ICCV2025/segmentation/ensemble_foreground_management_for_unsupervised_object_discovery.md)
- [\[ECCV 2024\] VISA: Reasoning Video Object Segmentation via Large Language Models](../../ECCV2024/segmentation/visa_reasoning_video_object_segmentation_via_large_language_models.md)

</div>

<!-- RELATED:END -->
