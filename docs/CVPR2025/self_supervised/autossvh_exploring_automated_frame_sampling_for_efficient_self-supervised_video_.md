---
title: "AutoSSVH: Automated Frame Sampling for Self-Supervised Video Hashing"
description: "提出AutoSSVH框架，通过Grade-Net评分+Gumbel-Softmax可微分采样+梯度反转对抗训练实现自动关键帧选择，UCF101 64-bit提升36.5~46.7%"
tags:
  - CVPR2025
  - 视频哈希
  - 自监督学习
  - 帧采样
  - 自监督
---

# AutoSSVH: Automated Frame Sampling for Self-Supervised Video Hashing

**会议**: CVPR 2025  
**机构**: 哈尔滨工业大学（深圳） / 清华大学 / 鹏城实验室  
**关键词**: 视频哈希、帧采样、Gumbel-Softmax、梯度反转、自监督  

## 研究背景与动机

视频检索是多媒体应用的核心需求，但视频数据的海量性和高维性使得高效检索成为巨大挑战。**视频哈希**通过将视频映射为紧凑的二进制码（如64-bit），实现了近似最近邻的快速检索。

自监督视频哈希（Self-Supervised Video Hashing, SSVH）无需标注数据，通过预设代理任务学习哈希函数。但一个被忽视的关键问题是**帧采样**：一段视频通常包含数百到数千帧，但哈希网络只能处理有限帧数（通常8~16帧）。

现有方法几乎都使用**均匀采样**或**随机采样**，但这存在明显问题：
- 均匀采样可能选中大量冗余帧（如静止场景）
- 随机采样不可复现，引入不必要的噪声
- 两者都忽略了帧的信息量差异——关键动作帧和无关背景帧被同等对待

理想的采样策略应该是**自适应的**：根据帧的信息内容选择最有价值的帧子集。但传统的帧选择方法（如基于手工特征的关键帧检测）无法与下游哈希任务端到端优化。

AutoSSVH的核心动机是：**让帧采样策略与哈希学习联合优化，同时保持无监督。**

## 方法详解

### 系统概览

AutoSSVH由三个主要组件构成：(1) Grade-Net帧评分网络，(2) Gumbel-Softmax可微分TopK采样，(3) Transformer哈希网络。

### 组件1：Grade-Net帧评分

Grade-Net是一个轻量级MLP，为视频中的每一帧分配一个重要性分数：

$$g_i = \text{MLP}(f_i), \quad f_i = \text{ResNet}(I_i)$$

其中 $f_i$ 是第 $i$ 帧的预提取CNN特征，$g_i \in \mathbb{R}$ 是标量重要性分数。

**关键设计**：Grade-Net没有直接的监督信号。它的训练信号完全来自下游哈希任务的反向传播。

### 组件2：Gumbel-Softmax可微分TopK采样

标准的TopK操作是不可微分的，无法进行梯度传播。AutoSSVH使用Gumbel-Softmax技巧实现可微分的离散采样：

$$y_i = \frac{\exp((g_i + G_i) / \tau)}{\sum_j \exp((g_j + G_j) / \tau)}$$

其中 $G_i \sim \text{Gumbel}(0, 1)$ 是Gumbel噪声，$\tau$ 是温度参数。

在训练时，使用Straight-Through Estimator：前向传播取硬TopK（离散选择），反向传播使用软Gumbel-Softmax权重（连续梯度）。

温度 $\tau$ 随训练进行逐步退火：$\tau_t = \max(\tau_{\min}, \tau_0 \cdot \exp(-\gamma t))$。

### 梯度反转层（GRL）

为防止Grade-Net退化为简单的帧特征差异度量（可能与哈希目标不一致），AutoSSVH引入梯度反转层进行对抗训练：

$$\text{GRL}(x) = x \quad (\text{前向}), \quad \frac{\partial \text{GRL}}{\partial x} = -\lambda I \quad (\text{反向})$$

GRL放置在Grade-Net和一个判别器之间。判别器试图从采样的帧中恢复视频语义，而Grade-Net在GRL作用下反向优化——选择能让判别器困惑的帧，迫使哈希网络不能依赖简单的帧选择策略，从而学到更鲁棒的表征。

### 组件3：Transformer哈希网络

| 层 | 功能 | 输出维度 |
|----|------|---------|
| Transformer Encoder (6层) | 帧间关系建模 | 512 |
| Transformer Decoder (2层) | 哈希码生成 | 512 |
| tanh软哈希层 | 连续→近似二进制 | L (哈希位数) |

**P2Set对比损失**：每个视频生成多个哈希码（通过不同采样），使用组件投票法确定哈希中心，然后拉近同视频的不同采样结果、推远不同视频的哈希码：

$$\mathcal{L}_{\text{P2Set}} = -\log \frac{\exp(\text{sim}(h, c^+) / \tau)}{\exp(\text{sim}(h, c^+) / \tau) + \sum_j \exp(\text{sim}(h, c_j^-) / \tau)}$$

## 实验结果

### 视频检索性能 (MAP@20)

| 方法 | UCF101 16-bit | UCF101 32-bit | UCF101 64-bit | HMDB51 64-bit |
|------|-------------|-------------|-------------|--------------|
| ITQ | 0.0412 | 0.0518 | 0.0623 | 0.0312 |
| SSVH | 0.0856 | 0.1234 | 0.1567 | 0.0678 |
| MCMSH | 0.1023 | 0.1456 | 0.1834 | 0.0812 |
| **AutoSSVH** | **0.1589** | **0.2134** | **0.2693** | **0.1221** |
| vs MCMSH 提升 | +55.3% | +46.6% | +46.8% | +50.4% |

### 跨数据集检索

| 方法 | UCF→HMDB GMAP | HMDB→UCF GMAP |
|------|-------------|-------------|
| MCMSH | 0.0646 | 0.0723 |
| **AutoSSVH** | **0.0780** | **0.0856** |
| 提升 | +20.7% | +18.4% |

### 采样策略对比

| 采样方法 | UCF101 64-bit MAP | 训练时间（相对） |
|---------|-----------------|--------------|
| 均匀采样 | 0.2134 | 1.0× |
| 随机采样 | 0.2067 | 1.0× |
| 关键帧检测(预处理) | 0.2289 | 1.5× |
| **AutoSSVH (端到端)** | **0.2693** | **1.2×** |

端到端学习的采样策略以微小的额外计算代价带来了26%的性能提升。

### 消融实验

| 配置 | MAP |
|------|-----|
| 无Grade-Net（均匀采样） | 0.2134 |
| +Grade-Net（无GRL） | 0.2412 |
| +Grade-Net + GRL | 0.2578 |
| +Grade-Net + GRL + P2Set | **0.2693** |

## 创新点

1. **可微分帧采样**：基于Gumbel-Softmax的端到端帧选择，首次将采样策略纳入哈希学习循环
2. **对抗训练正则化**：GRL防止采样策略退化，提升泛化性
3. **P2Set对比学习**：利用多次采样构建正样本对，充分利用帧选择的随机性

## 局限性

- Gumbel-Softmax的温度退火策略需要精心调整
- 预提取CNN特征增加了存储需求
- 在超长视频（>1000帧）上的效率有待验证

## 总结

AutoSSVH提出了自监督视频哈希中帧选择的自动化方案，通过可微分采样将帧选择与哈希学习统一优化。36~50%的性能提升证明了"选对帧"比"处理所有帧"更重要，这一发现对视频理解领域具有普遍借鉴意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] VideoSSR: Video Self-Supervised Reinforcement Learning](../../CVPR2026/self_supervised/videossr_video_self-supervised_reinforcement_learning.md)
- [\[CVPR 2026\] Towards Stable Self-Supervised Object Representations in Unconstrained Egocentric Video](../../CVPR2026/self_supervised/towards_stable_self-supervised_object_representations_in_unconstrained_egocentri.md)
- [\[CVPR 2026\] TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation](../../CVPR2026/self_supervised/teflow_enabling_multi-frame_supervision_for_self-supervised_feed-forward_scene_f.md)
- [\[CVPR 2026\] Progressive Mask Distillation for Self-supervised Video Representation](../../CVPR2026/self_supervised/progressive_mask_distillation_for_self-supervised_video_representation.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)

</div>

<!-- RELATED:END -->
