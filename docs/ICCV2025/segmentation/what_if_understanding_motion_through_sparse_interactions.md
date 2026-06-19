---
title: >-
  [论文解读] What If: Understanding Motion Through Sparse Interactions
description: >-
  [ICCV 2025][语义分割][运动理解] 提出 Flow Poke Transformer (FPT)，直接预测场景中物体运动的多模态概率分布：（而非单一确定性结果），通过稀疏"戳动 (poke)"交互条件化，实现可解释的运动理解和运动部件分割。 理解场景动态的核心挑战在于：真实世界的运动是不确定的、多模态的：…
tags:
  - "ICCV 2025"
  - "语义分割"
  - "运动理解"
  - "光流分布预测"
  - "稀疏交互"
  - "运动部件分割"
  - "Transformer"
---

# What If: Understanding Motion Through Sparse Interactions

**会议**: ICCV 2025  
**arXiv**: [2510.12777](https://arxiv.org/abs/2510.12777)  
**代码**: [compvis.github.io/flow-poke-transformer](https://compvis.github.io/flow-poke-transformer)  
**领域**: 图像分割  
**关键词**: 运动理解, 光流分布预测, 稀疏交互, 运动部件分割, Transformer

## 一句话总结

提出 Flow Poke Transformer (FPT)，直接预测场景中物体运动的**多模态概率分布**（而非单一确定性结果），通过稀疏"戳动 (poke)"交互条件化，实现可解释的运动理解和运动部件分割。

## 研究背景与动机

理解场景动态的核心挑战在于：**真实世界的运动是不确定的、多模态的**。人类视觉智能不是预测一个确定的未来，而是推断物体可能运动的多种方式。

现有运动预测方法的问题：

**密集视频预测**（如 diffusion video generation）必须选择*一种*轨迹输出，忽略了运动的多模态性——即使产出逼真帧，也不代表理解了物理过程

**确定性光流预测**（如 RAFT）仅估计给定两帧间的运动，需要未来帧，无法预测未来

**现有条件运动生成**（如 DragAPart, PuppetMaster）在 RGB 空间直接合成结果，底层运动表征难以访问，且不提供不确定性估计

核心动机：设计一个**直接输出运动概率分布**的模型，而非只能从分布中采样。这样可以：
- 直接量化不确定性
- 识别多模态运动
- 通过稀疏 poke 探索场景中的物理交互

## 方法详解

### 整体框架

给定图像 $\mathcal{I}$，建模条件分布 $p(\mathbf{f}(\mathbf{q})|\mathcal{P}, \mathcal{I})$：查询点 $\mathbf{q}$ 的运动 $\mathbf{f}(\mathbf{q}) \in \mathbb{R}^2$ 在给定一组 poke $\mathcal{P} = \{(\mathbf{p}_i, \mathbf{f}(\mathbf{p}_i))\}_{i=1}^{N_p}$ 和图像条件下的分布。架构由**图像编码器**（ViT-Base + DINOv2-R 初始化）和 **Flow Poke Transformer**（ViT-Base）组成，总参数量 220M。

### 关键设计

1. **稀疏运动学建模**: 每个 poke $(\mathbf{p}_i, \mathbf{f}(\mathbf{p}_i))$ 和查询点 $\mathbf{q}_j$ 作为独立 token。Poke 的运动通过 Fourier embedding 编码，位置通过 RoPE 相对位置编码实现**任意精度的非网格位置**。查询 token 仅注意力关注自身和 poke（不关注其他查询），使得多查询可**并行高效预测**。

2. **高斯混合模型 (GMM) 输出分布**: Transformer 输出端的投影头直接预测一个 $N$ 分量的 GMM：
$$p_\theta = \sum_{n=1}^N \pi^{(n)} \cdot \mathcal{N}(\boldsymbol{\mu}^{(n)}, \boldsymbol{\Sigma}^{(n)})$$
关键改进：使用**完整协方差矩阵** $\boldsymbol{\Sigma}^{(n)} \in \mathbb{R}^{2 \times 2}$（通过预测下三角矩阵 $\mathbf{L}^{(n)}$ 保证正定性），而非 GIVT 的对角协方差，大幅增加建模自由度。

3. **Query-Causal Attention**: 训练时对 poke 使用因果注意力掩码，查询仅注意对应的 poke 子集。将计算复杂度从 $\mathcal{O}(N_p^2 \cdot N_q^2)$ 降至 $\mathcal{O}(N_p^2 + N_p \cdot N_q)$，实现高效训练。

4. **相机运动适应**: 使用自适应归一化层（AdaIN），根据相机是否静止来条件化模型，避免相机运动主导学习到的运动分布。

5. **运动部件分割（Moving Part Segmentation）**: 利用 KL 散度度量一个 poke 对查询点运动分布的影响：
$$D_{KL}(p_\theta(\mathbf{f}(\mathbf{q})|(\mathbf{p}, \mathbf{f}(\mathbf{p})), \mathcal{I}) \parallel p_\theta(\mathbf{f}(\mathbf{q})|\mathcal{I}))$$
KL 散度为 0 表示运动独立，非 0 表示受 poke 影响——直接量化运动部件的关联性。

### 损失函数 / 训练策略

直接最小化地面真值光流的**负对数似然**：
$$\mathcal{L} = -\log p_\theta(\mathbf{f}(\mathbf{q})|\mathcal{P}, \mathcal{I}) = -\log\left(\sum_{n=1}^N \pi^{(n)} \mathcal{N}(\mathbf{f}(\mathbf{q})|\boldsymbol{\mu}^{(n)}_\theta, \boldsymbol{\Sigma}^{(n)}_\theta)\right)$$

训练细节：
- 数据集：WebVid 3.8M 子集（通用预训练），另一版本使用 5M 开放视频
- 光流真值：CoTracker3 / TAPNext 在 $48^2$ 网格上提取的光学跟踪
- AdamW，lr=5e-5，batch size 32→128，800k steps
- 每张图像采样 0~128 个 poke 和 15 个随机查询点
- 训练耗时：2×H200 训练 7 天（或优化配置 8×H200 训练 24h）

## 实验关键数据

### 主实验：人脸运动生成（TalkingHead-1KH）

| 方法 | 训练数据 | 1 Poke EPE↓ | 10 Pokes EPE↓ | 100 Pokes EPE↓ |
|------|---------|------------|--------------|----------------|
| InstantDrag | 人脸专用 | 9.24 | 8.39 | 7.29 |
| Motion-I2V | 通用 (Zero-Shot) | 29.08 | 20.90 | n/a |
| **FPT (本文)** | **通用 (Zero-Shot)** | **7.64** | **4.20** | **2.51** |

零样本通用模型即超越人脸专用模型 InstantDrag，且随 poke 数增加提升更明显。

### 铰接物体运动估计（Drag-A-Move）

| 方法 | 训练集 | EPE↓ | PCK↑ | 分割 mIoU↑ |
|------|--------|-----|------|-----------|
| DragAPart | DAM (专用) | 9.69 | 0.514 | 0.273 |
| PuppetMaster | DAM+OAHQ (专用) | 9.62 | 0.472 | 0.112 |
| FPT (zero-shot) | 通用 | 12.74 | 0.191 | 0.287 |
| **FPT (fine-tuned)** | **通用→DAM** | **3.57** | **0.834** | **0.572** |

微调后 EPE **降低 63%**，运动部件分割 mIoU 达 0.572（vs DragAPart 0.273）。

### 消融实验

| 预测模式的不确定性校准 | 说明 |
|----------------------|------|
| Pearson $\rho$ = 0.66（采样） | 预测不确定性与实际误差强相关 |
| Pearson $\rho$ = 0.64（均值） | 均值预测同样准确 |
| Pearson $\rho$ = 0.62（最高置信模态） | 置信度越高越准确 |

| 多模态分析 | 说明 |
|-----------|------|
| 模态多样性高 | 模态变化覆盖 poke 幅度的大部分 |
| 最接近GT模态的置信度 > 平均 | 模型的置信度预测有意义 |
| 更多 poke → 更单峰 | 条件信息足够时分布自然收敛 |

### 关键发现

- FPT 预测的**不确定性与真实误差高度相关**（Pearson $\rho > 0.6$），确认了概率分布预测的可靠性
- **通用预训练即可泛化**：在人脸运动任务上零样本超越专用模型
- 运动部件分割**不需要特殊训练**，直接从分布对比中自然涌现
- 单 H200 推理延迟 <25ms，吞吐量 >160k predictions/s，适用于实时场景

## 亮点与洞察

- **概率分布直接可访问**：与 diffusion/GAN 只能采样不同，FPT 的 GMM 输出可直接读取概率密度、计算 KL 散度、识别模态
- **稀疏→高效**：稀疏 token 建模避免了密集预测的计算开销，同时保留了丰富的运动语义
- **运动部件分割从运动理解中自然涌现**：无需额外标注或模块，仅靠 KL 散度即可量化部件关联，概念优雅
- query-causal attention 的设计大幅减少训练开销，可复用于其他稀疏条件预测任务

## 局限与展望

- 在卡通/动画图像上泛化较差（训练数据主要是真实视频）
- 有时会错误地将物体阴影与物体运动耦合
- 当前仅建模 2D 运动分布，扩展到 3D 运动是未来方向
- 自回归密集采样虽然可以生成一致的全局运动，但速度较慢

## 相关工作与启发

- 基于 GIVT 的 GMM 输出头设计，但扩展为完整协方差矩阵，增强建模能力
- Poke 概念源自 iPoke (Blattmann et al. 2021)，但从 RGB 合成升级为概率分布建模
- RoPE 位置编码使模型支持任意精度的非网格位置，具有较强的通用性

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 直接预测运动概率分布 + 运动部件分割涌现
- 实验充分度：⭐⭐⭐⭐ — 多域评估 + 不确定性分析 + 分割
- 实用性：⭐⭐⭐⭐ — 实时推理 + 通用预训练 + 多下游任务
- 总体：⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Enhancing Transformers Through Conditioned Embedded Tokens](enhancing_transformers_through_conditioned_embedded_tokens.md)
- [\[ICCV 2025\] Temporal Rate Reduction Clustering for Human Motion Segmentation](temporal_rate_reduction_clustering_for_human_motion_segmentation.md)
- [\[ICCV 2025\] Personalized OVSS: Understanding Personal Concept in Open-Vocabulary Semantic Segmentation](understanding_personal_concept_in_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] MOVE: Motion-Guided Few-Shot Video Object Segmentation](move_motion-guided_few-shot_video_object_segmentation.md)
- [\[ICCV 2025\] Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation](skeleton_motion_words_for_unsupervised_skeleton-based_temporal_action_segmentati.md)

</div>

<!-- RELATED:END -->
