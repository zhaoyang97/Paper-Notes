---
title: >-
  [论文解读] O-MaMa: Learning Object Mask Matching between Egocentric and Exocentric Views
description: >-
  [ICCV 2025][语义分割][图像分割] 将跨视角（ego-exo）物体分割任务重新定义为 mask matching 问题，利用 FastSAM 生成候选 mask、DINOv2 提取语义特征、对比学习匹配跨视角物体，在 Ego-Exo4D 基准上以仅 1% 可训练参数实现 SOTA。 多智能体协作（多机器人操作、A…
tags:
  - "ICCV 2025"
  - "语义分割"
  - "图像分割"
  - "Mask Matching"
  - "Ego-Exo Correspondences"
  - "对比学习"
  - "DINOv2"
---

# O-MaMa: Learning Object Mask Matching between Egocentric and Exocentric Views

**会议**: ICCV 2025  
**arXiv**: [2506.06026](https://arxiv.org/abs/2506.06026)  
**代码**: [Maria-SanVil/O-MaMa](https://github.com/Maria-SanVil/O-MaMa)  
**领域**: 图像分割  
**关键词**: Cross-View Segmentation, Mask Matching, Ego-Exo Correspondences, Contrastive Learning, DINOv2

## 一句话总结

将跨视角（ego-exo）物体分割任务重新定义为 mask matching 问题，利用 FastSAM 生成候选 mask、DINOv2 提取语义特征、对比学习匹配跨视角物体，在 Ego-Exo4D 基准上以仅 1% 可训练参数实现 SOTA。

## 研究背景与动机

多智能体协作（多机器人操作、AR 助手、人机协作）需要建立第一人称（egocentric）和第三人称（exocentric）视角之间的物体对应关系。虽然单图分割已非常成熟，但跨视角分割面临独特挑战：

**剧烈视角变换**：ego 视角捕捉手-物交互细节但动态强、运动模糊严重；exo 视角覆盖全场景但物体尺度差异大

**遮挡与域偏移**：不同相机光学特性、成像条件导致的域差异

**传统几何匹配失效**：即使 SOTA 的 RoMa 在 ego-exo 场景下也仅有 67.6% 的成功率

核心洞察：与其让模型从零做跨视角分割（pixel-level 预测），不如利用 SAM 模型的零样本分割能力先生成高质量候选 mask，然后只需解决"哪个候选 mask 对应目标物体"的 matching 问题。

## 方法详解

### 整体框架

O-MaMa pipeline：
1. 在目标视角用 FastSAM 生成 $N$ 个候选 mask $\{\mathcal{M}_n\}_{n=1}^N$
2. Mask-Context Encoder 提取每个 mask 的描述子
3. Ego↔Exo Cross Attention 融合跨视角全局信息
4. 通过 Mask Matching Contrastive Loss 学习视角不变特征
5. 推理时选择与源 mask 嵌入最相似的候选 mask

### 关键设计

1. **Mask-Context Encoder（Mask 上下文编码器）**：

    - 使用 DINOv2 ViT-B/14 提取密集特征图 $\psi(I)$，上采样 4× 以保持细粒度
    - **物体描述子** $\mathbf{o}_n$：在 mask 区域上对 DINOv2 特征做平均池化
    - **上下文描述子** $\mathbf{c}_n$：在扩展 bounding box 区域上做平均池化，引入周围环境信息以辅助跨视角定位
    - 设计动机：DINOv2 的自监督特征具有优秀的语义理解和物体分解能力。实验证明 Avg-Pool(mask) 优于 Avg-Pool(bbox)、Max-Pool(bbox)、Centroid、CLIP 特征

2. **Hard Negative Adjacent Mining（邻近硬负例挖掘）**：

    - 问题：邻近物体共享相似上下文但物体本身不同，简单的全局负采样不足以学到区分能力
    - 使用 Delaunay 三角剖分构建 mask 段的邻接图
    - 取每个物体的 1 阶和 2 阶邻居：$\mathcal{O}_n^- = \mathcal{N}(\mathbf{o}_n) \cup \mathcal{N}^2(\mathbf{o}_n)$
    - 从邻居集合中采样硬负例进行对比学习
    - 消融显示此策略带来 +4.2 IoU (Ego2Exo) 和 +1.2 IoU (Exo2Ego) 提升

3. **Ego↔Exo Cross Attention（跨视角交叉注意力）**：

    - 将候选 mask 描述子 $\mathbf{o}_n$ 作为 Query，源图像的完整 DINOv2 特征图 $\psi(I^S)$ 作为 Key/Value
    - 计算标准 cross attention：$\hat{\mathbf{o}}_n = \text{Softmax}(\frac{\mathbf{o}_n W_Q \cdot (\psi(I^S) W_K)^\top}{\sqrt{d}}) \cdot \psi(I^S) W_V$
    - 加入可学习位置编码和 LayerNorm
    - 同样计算源 mask 在目标视角的跨视角嵌入 $\hat{\mathbf{o}}_S$
    - 设计动机：上下文嵌入只包含局部信息，缺乏全局跨视角语义关联

### 损失函数 / 训练策略

- **Mask Matching Contrastive Loss**：基于 InfoNCE，从邻近硬负例中采样 batch $\mathcal{B}$

  $$\mathcal{L}_M(\rho^+, \rho_S) = -\log \frac{\exp(\text{sim}(f_\theta(\rho^+), f_\theta(\rho_S))/\tau)}{\sum_{n=1}^{|\mathcal{B}|} \exp(\text{sim}(f_\theta(\rho_n), f_\theta(\rho_S))/\tau)}$$

- 最终描述子 $\rho_n = [\hat{\mathbf{o}}_n; \mathbf{c}_n; \mathbf{o}_n]$（跨视角嵌入 + 上下文 + 物体），通过浅层 MLP $f_\theta$ 映射到共享潜在空间
- 优化器：AdamW，lr=$8 \times 10^{-5}$，cosine annealing，batch size=24 图像对，每张目标图采样 32 个候选 mask
- 设备：2× NVIDIA RTX 4090

## 实验关键数据

### 主实验 (表格)

**Ego-Exo4D Correspondences v2 Test Split**

| 方法 | Ego2Exo IoU ↑ | Exo2Ego IoU ↑ | Total IoU ↑ | 训练参数(M) |
|------|--------------|--------------|-------------|------------|
| XMem + XSegTx | 34.9 | 25.0 | 30.0 | 67.1 |
| PSALM (zero-shot) | 7.4 | 2.1 | 4.8 | 0 |
| k-NN baseline | 31.9 | 30.9 | 31.4 | 0 |
| **O-MaMa** | **42.6** | **44.1** | **43.4** | **11.6** |

**Ego-Exo4D Correspondences v1 Val Split**

| 方法 | Ego2Exo IoU ↑ | Exo2Ego IoU ↑ | Total IoU ↑ | 训练参数(M) |
|------|--------------|--------------|-------------|------------|
| PSALM (fine-tuned) | 41.3 | 44.1 | 42.7 | 1587.1 |
| ObjectRelator | 44.3 | 50.9 | 47.6 | 1587.3 |
| **O-MaMa** | **50.1** | **54.2** | **52.1** | **11.6** |

O-MaMa 超越 ObjectRelator(SOTA) +13.1%(Ego2Exo) / +6.5%(Exo2Ego)，但仅用 **1%** 训练参数。

### 消融实验 (表格)

**各模块消融 (10% 验证集)**

| 配置 | $\mathcal{L}_M$ | Context | Adj.Neg | CrossAttn | Ego2Exo IoU | Exo2Ego IoU | Total IoU |
|------|------|---------|---------|-----------|-------------|-------------|-----------|
| Baseline | ✗ | ✗ | ✗ | ✗ | 35.2 | 34.9 | 35.1 |
| A | ✓ | ✗ | ✗ | ✗ | 42.2 | 44.7 | 43.5 |
| C | ✓ | ✓ | ✓ | ✗ | 46.9 | 45.6 | 46.3 |
| E (full) | ✓ | ✓ | ✓ | ✓ | **48.3** | **49.6** | **49.0** |

相对 baseline 的 IoU 提升：Ego2Exo +37.2%，Exo2Ego +42.1%。

**Mask 描述子比较**

| 描述子 | k-NN Ego2Exo | k-NN Exo2Ego | 学习后 Ego2Exo | 学习后 Exo2Ego |
|--------|-------------|-------------|---------------|---------------|
| Avg-Pool(Mask)-DINOv2 | **35.2** | **34.9** | **42.2** | **44.7** |
| Avg-Pool(BBox)-DINOv2 | 21.8 | 21.2 | 27.8 | 44.1 |
| Avg-Pool(BBox)-CLIP | 24.5 | 23.9 | 27.5 | 40.4 |
| Centroid-DINOv2 | 25.6 | 24.1 | - | - |

DINOv2 mask 池化特征远优于 CLIP 和其他池化策略。

### 关键发现

- **问题重定义是最大贡献**：将跨视角分割转化为 mask matching，使得零样本 k-NN baseline（40.5 IoU）即超过许多训练模型
- **几何约束帮助有限**：RoMa 成功率仅 67.6%，几何匹配相比对比学习提升微弱（35.2→35.4 vs 35.2→42.2）
- **DINOv2 > CLIP**：在此任务中 DINOv2 的细粒度语义特征优于 CLIP 的粗粒度表示
- **小物体仍具挑战**：O-MaMa 在中大型物体上表现优异，但极小物体的 mask 描述子难以提取有效信息
- **推理速度适中**：平均 250ms（其中 FastSAM 70ms）

## 亮点与洞察

1. **问题重定义的力量**：将困难的 pixel-level 跨视角分割转化为 mask 级别的检索/匹配问题，大幅降低了任务难度，使得轻量模型即可达到 SOTA
2. **DINOv2 的物体分解能力**：自监督预训练的 DINOv2 提供了极其强大的 object-level 语义表示，甚至零样本即可超越专门训练的模型
3. **Delaunay 三角剖分硬负例挖掘**：巧妙利用空间邻近关系增强对比学习的区分能力，比随机负采样更有效
4. **参数效率极高**：11.6M 训练参数 vs ObjectRelator 的 1587.3M，说明 foundation model 的特征质量已经足够好，只需极少的任务适配

## 局限与展望

- FastSAM 可能产生不完整的分割（只覆盖物体的一部分），导致匹配正确但 IoU 不够高
- 极小物体的 mask 描述子信息不足，是当前的主要瓶颈
- 未利用视频的时序信息（当前每帧独立处理），加入时序连续性可能进一步提升
- 依赖 FastSAM 的候选质量——如果目标物体未被任何候选 mask 覆盖则无法匹配
- 未探索 SAM2 等更强的分割模型作为候选生成器

## 相关工作与启发

- **Ego-Exo4D**：提供大规模同步 ego-exo 视频数据集和对应的 Correspondences 基准
- **ObjectRelator**：微调 PSALM (LLM-based) 做跨视角分割，参数量巨大
- **FastSAM / SAM**：提供高质量零样本分割能力，是本方法的基础
- **DINOv2**：自监督视觉基础模型，提供物体级语义表示
- 启发：当 foundation model 已能提供足够好的基础能力（分割、特征提取）时，轻量化的任务适配（如对比学习 + 匹配）可能是更优的范式

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将跨视角分割重定义为 mask matching 是简洁有效的创新
- **实验充分度**: ⭐⭐⭐⭐⭐ 两个数据集分割、多基线对比、完整消融（模块/描述子/几何约束）、任务级分析
- **写作质量**: ⭐⭐⭐⭐ 方法直观易懂，架构图清晰，实验分析细致
- **价值**: ⭐⭐⭐⭐⭐ 1% 参数达 SOTA、问题重定义思想对跨视角理解任务有广泛启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Learning Precise Affordances from Egocentric Videos for Robotic Manipulation](learning_precise_affordances_from_egocentric_videos_for_robotic_manipulation.md)
- [\[CVPR 2025\] LiVOS: Light Video Object Segmentation with Gated Linear Matching](../../CVPR2025/segmentation/livos_light_video_object_segmentation_with_gated_linear_matching.md)
- [\[ICCV 2025\] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)
- [\[CVPR 2026\] Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction](../../CVPR2026/segmentation/learning_cross-view_object_correspondence_via_cycle-consistent_mask_prediction.md)
- [\[NeurIPS 2025\] Robust Egocentric Referring Video Object Segmentation via Dual-Modal Causal Intervention](../../NeurIPS2025/segmentation/robust_egocentric_referring_video_object_segmentation_via_dual-modal_causal_inte.md)

</div>

<!-- RELATED:END -->
