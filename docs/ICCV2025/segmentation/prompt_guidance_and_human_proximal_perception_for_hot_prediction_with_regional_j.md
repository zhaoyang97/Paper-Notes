---
title: >-
  [论文解读] Prompt Guidance and Human Proximal Perception for HOT Prediction with Regional Joint Loss
description: >-
  [ICCV 2025][语义分割][人体-物体接触检测] 提出 P3HOT 框架，通过文本 prompt 引导关注人体接触部位、深度感知模块过滤无关背景、以及 Regional Joint Loss 保证区域内类别一致性，在 HOT（Human-Object Contact）检测任务上取得 SOTA。
tags:
  - "ICCV 2025"
  - "语义分割"
  - "人体-物体接触检测"
  - "文本引导"
  - "深度感知"
  - "区域联合损失"
---

# Prompt Guidance and Human Proximal Perception for HOT Prediction with Regional Joint Loss

**会议**: ICCV 2025  
**arXiv**: [2507.01630](https://arxiv.org/abs/2507.01630)  
**代码**: [github.com/YuxiaoWang-AI/P3HOT](https://github.com/YuxiaoWang-AI/P3HOT)  
**领域**: 图像分割  
**关键词**: 人体-物体接触检测, 文本引导, 深度感知, 区域联合损失, 语义分割

## 一句话总结

提出 P3HOT 框架，通过文本 prompt 引导关注人体接触部位、深度感知模块过滤无关背景、以及 Regional Joint Loss 保证区域内类别一致性，在 HOT（Human-Object Contact）检测任务上取得 SOTA。

## 研究背景与动机

HOT（Human-Object conTact）检测源自 HOI（Human-Object Interaction），旨在识别人体**具体哪些部位**与物体接触，将人体分为 18 个类别（17 个部位 + 背景）。该任务对人机交互、VR、手势识别等领域具有重要价值。

现有方法（DHOT、PIHOT）的不足：

**单一模态**：仅用图像特征，忽视文本引导的语义信息

**区域类别不一致**：交叉熵损失无法约束区域内类别一致性，导致在某类别区域内出现其他类别（如手掌区域内出现小块"头部"预测）

**无交互区域干扰**：在交互较少的区域过度分割

**评估指标缺陷**：C-Acc. 指标有缺陷——若将整张图预测为某接触类，C-Acc. 可达 100%

核心动机：引入多模态信息（文本 prompt + 深度伪 3D 信息）提升 HOT 检测能力，并设计专门的损失函数确保分割的区域一致性。

## 方法详解

### 整体框架

P3HOT 由四部分组成：
- **图像编码器**（ResNet-50 + CLIP 初始化 + Attention Pooling）
- **文本编码器**（CLIP Text Encoder，冻结参数）
- **人体近端感知 (HPP) 模块**（SAM + ZoeDepth + 可学习参数 $\tau$）
- **图像解码器**（逐级上采样 + 多尺度特征融合）

### 关键设计

1. **文本 Prompt 引导机制**: 构建 17 条模板 "A [body part] of the human body is in contact with an object"，分别替换 17 个人体部位。通过 CLIP 文本编码器提取文本特征 $\mathbf{F}_{TE} \in \mathbb{R}^{17 \times 1024}$，计算与图像特征 $\mathbf{F}_{IE}$ 的余弦相似度 $\mathbf{S}$：
$$\mathbf{S} = \frac{\mathbf{F}_{IE} \cdot \mathbf{F}_{TE}^T}{\|\mathbf{F}_{IE}\| \cdot \|\mathbf{F}_{TE}\|}$$
相似度 $\mathbf{S}$ 按通道乘以解码器输出，增强文本所关注部位的响应。**核心思想**：利用图文匹配度动态调整各身体部位通道的权重。

2. **人体近端感知 (HPP) 模块**: 解决2D视角下人体与物体重叠的不确定性问题：

    - 使用 SAM（text prompt "person"）生成人体掩码 $\mathbf{M}$
    - 使用 ZoeDepth 提取深度图 $\mathbf{D}$，归一化到 $[0,1]$
    - 计算每个人体的平均深度 $\mathbf{m}_i^{da}$
    - 通过可学习参数 $\tau$ 确定深度范围 $[\mathbf{m}_i^{da} - \tau, \mathbf{m}_i^{da} + \tau]$
    - 生成过滤掩码保留人体及其周围物体，排除无关背景
   
   为使 $\tau$ 可反向传播，用 ReLU 替代硬阈值比较：
    $\Theta_i = (D_{Norm} - (\mathbf{m}_i^{da} - \tau)) \otimes ((\mathbf{m}_i^{da} + \tau) - D_{Norm})$
    $FM = FM + \sum_{i=1}^N \text{ReLU}(\Theta_i)$

3. **多尺度特征融合解码器**: 将编码器四个 block 的输出（$S_i = \{4, 8, 16, 32\}$ 下采样率）逐级上采样并拼接：
$$\mathbf{x}_{i-1} = \text{DoubleConv}(\text{Up}(\mathbf{x}_i) \text{ⓒ} \mathbf{F}_{i-1})$$
弥补下采样导致的纹理细节丢失。

### 损失函数 / 训练策略

**Regional Joint Loss (RJLoss)** 由两部分组成：

**(a) Local Joint Loss** — 在 GT 定义的每个类别区域内，惩罚出现其他类别：
$$\mathcal{L}_c^L = \frac{\sum(|\mathbf{O}_c - \mathbf{GT}_c| \otimes \mathbf{GT}_c)}{\sum \mathbf{GT}_c}$$

**(b) Global Joint Loss** — 在整个预测图上，找到被某类别包围的连通区域（closed region），惩罚其中的错误类别。通过连通区域分析，检测被边界像素围住的"孤岛"区域：
$$\mathcal{L}_c^G = \sum(\neg \mathbf{O}_c \otimes \mathbf{O}_c^M)$$

**总损失**：
$$\mathcal{L} = \text{CE}(\mathbf{O}, \mathbf{GT}) + \alpha \mathcal{L}^L + \beta \mathcal{L}^G + \gamma \text{BE}(\mathbf{S}, \mathbb{C})$$
其中 $\alpha = 0.3, \beta = 0.1, \gamma = 1.0$。BE 为图文匹配的二元交叉熵损失。

训练环境：8×NVIDIA A6000，AdamW 优化器，batch size 4/GPU。

## 实验关键数据

### 主实验

| 方法 | 数据集 | SC-Acc. | mIoU | wIoU | AD-Acc. (新指标) |
|------|--------|---------|------|------|-----------------|
| DHOT-Full | HOT-Annotated | 40.7 | 21.5 | 26.0 | - |
| PIHOT | HOT-Annotated | 45.3 | 23.6 | 28.6 | 31.3 |
| **P3HOT (本文)** | **HOT-Annotated** | **46.0** | **25.6** | **30.2** | **42.3** |
| DHOT-Full | HOT-Generated | 30.4 | 13.9 | 16.7 | - |
| PIHOT | HOT-Generated | 34.9 | 16.9 | 21.2 | 25.4 |
| **P3HOT (本文)** | **HOT-Generated** | **35.2** | **18.0** | **23.1** | **30.6** |

提升幅度：HOT-Annotated 上 SC-Acc. +0.7, mIoU +2.0, wIoU +1.6, **AD-Acc. +11.0**。

### 消融实验

**各组件贡献**（HOT-Annotated）：

| 配置 | SC-Acc. | mIoU | AD-Acc. | 说明 |
|------|---------|------|---------|------|
| Baseline | 37.2 | 19.0 | 30.3 | 仅编码器+解码器 |
| +Fine (多尺度融合) | 38.9 | 20.1 | 34.1 | +3.8 AD-Acc. |
| +TE (文本编码器) | 40.3 | 21.0 | 37.1 | +3.0 AD-Acc. |
| +TE+DM+SAM (HPP) | 43.2 | 23.1 | 38.9 | HPP 贡献显著 |
| +RJLoss | **46.0** | **25.6** | **42.3** | RJLoss 再加 3.4 |

**损失函数影响**：

| 损失配置 | SC-Acc. | mIoU | AD-Acc. |
|---------|---------|------|---------|
| CE only | 43.2 | 23.1 | 38.9 |
| +BE | 44.5 | 23.8 | 40.1 |
| **+RJLoss** | **46.0** | **25.6** | **42.3** |

**深度图归一化**：

| 深度范围 | SC-Acc. | AD-Acc. | 说明 |
|---------|---------|---------|------|
| $D \in R$ (未归一化) | 44.1 | 40.2 | $\tau$ 难以优化 |
| **$D \in [0,1]$** | **46.0** | **42.3** | 归一化后最优 |

### 关键发现

- **RJLoss 的贡献最大**：从 CE-only → +RJLoss，AD-Acc. 提升 3.4 分，证明区域一致性约束的重要性
- 提出的 **AD-Acc. 指标**修正了 C-Acc. 的严重缺陷（整图预测单类即可 100%）
- **深度归一化**对 $\tau$ 学习至关重要：未归一化时不同图像深度范围差异大，$\tau$ 无法收敛
- HPP 模块从伪 3D 角度过滤无关区域，比仅用 SAM 掩码或仅用深度效果好

## 亮点与洞察

- **AD-Acc. 新评估指标**：同时考虑正负样本，解决 C-Acc. 缺陷，为 HOT 社区提供更公平的评估
- **RJLoss 的连通区域分析思路独特**：通过检测被某类别包围的"孤岛"来定位区域内错分，在分割任务中有广泛适用性
- **可学习深度阈值 $\tau$**：通过 ReLU 化替代硬阈值实现可微分，巧妙解决深度范围自适应问题
- **文本引导首次引入 HOT**：为后续多模态 HOT 研究奠定基础

## 局限与展望

- ResNet-50 backbone 较老，可尝试更强的视觉编码器（ViT, Swin）
- SAM 和 ZoeDepth 作为外部模块增加推理开销，可探索轻量化替代
- 仅在 HOT-Annotated 和 HOT-Generated 两个数据集上验证，数据规模有限
- 文本 prompt 是固定模板，可探索动态生成或上下文感知的 prompt

## 相关工作与启发

- 文本引导受 CLIP 在 HOI 检测中应用（GEN-VLKT, FreeA）的启发
- 深度感知机制与 3D HOT 检测（PROX+SMPL-X）形成互补：2D 方法更高效，适用范围更广
- RJLoss 的区域一致性思路可迁移到医学图像分割、语义分割等任务

## 评分

- 新颖性：⭐⭐⭐ — 各组件单独看不算全新，组合有价值
- 实验充分度：⭐⭐⭐⭐ — 消融充分，多维度分析
- 实用性：⭐⭐⭐⭐ — 代码开源，AD-Acc. 指标实用
- 总体：⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance](../../NeurIPS2025/segmentation/haodiff_human-aware_one-step_diffusion_via_dual-prompt_guidance.md)
- [\[ICCV 2025\] Joint Self-Supervised Video Alignment and Action Segmentation](joint_self-supervised_video_alignment_and_action_segmentation.md)
- [\[ICCV 2025\] Temporal Rate Reduction Clustering for Human Motion Segmentation](temporal_rate_reduction_clustering_for_human_motion_segmentation.md)
- [\[ICCV 2025\] ConformalSAM: Unlocking the Potential of Foundational Segmentation Models in Semi-Supervised Semantic Segmentation with Conformal Prediction](conformalsam_unlocking_the_potential_of_foundational_segmentation_models_in_semi.md)
- [\[ICCV 2025\] Advancing Visual Large Language Model for Multi-granular Versatile Perception](advancing_visual_large_language_model_for_multi-granular_versatile_perception.md)

</div>

<!-- RELATED:END -->
