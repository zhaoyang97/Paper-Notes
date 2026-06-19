---
title: >-
  [论文解读] Soft Self-Labeling and Potts Relaxations for Weakly-Supervised Segmentation
description: >-
  [CVPR 2025][语义分割][弱监督语义分割] 本文提出一种基于软伪标签的自标注方法，通过系统性评估多种 Potts 松弛形式和交叉熵变体，在标准网络架构上仅使用涂鸦（3% 像素）监督就实现了接近甚至超过全像素监督的分割性能，无需任何网络结构修改。 领域现状：语义分割的全监督需要大量像素级标注，成本极高…
tags:
  - "CVPR 2025"
  - "语义分割"
  - "弱监督语义分割"
  - "软伪标签"
  - "Potts 松弛"
  - "碰撞交叉熵"
  - "涂鸦监督"
---

# Soft Self-Labeling and Potts Relaxations for Weakly-Supervised Segmentation

**会议**: CVPR 2025  
**arXiv**: [2507.01721](https://arxiv.org/abs/2507.01721)  
**代码**: [https://vision.cs.uwaterloo.ca/code](https://vision.cs.uwaterloo.ca/code)  
**领域**: 图像分割 / 弱监督学习  
**关键词**: 弱监督语义分割, 软伪标签, Potts 松弛, 碰撞交叉熵, 涂鸦监督

## 一句话总结

本文提出一种基于软伪标签的自标注方法，通过系统性评估多种 Potts 松弛形式和交叉熵变体，在标准网络架构上仅使用涂鸦（3% 像素）监督就实现了接近甚至超过全像素监督的分割性能，无需任何网络结构修改。

## 研究背景与动机

**领域现状**：语义分割的全监督需要大量像素级标注，成本极高。弱监督方法（图像级标签、涂鸦、边界框）是降低标注成本的重要方向。其中涂鸦监督比图像级标签信息量略多但标注成本相近，且此前已被证明可以在不修改分割模型的情况下接近全监督性能。涂鸦监督的方法论核心是设计无监督/自监督损失函数和更强的优化算法。

**现有痛点**：（1）直接用梯度下降优化 Potts 松弛（最常见的无监督分割损失）效果受限，即使凸松弛在与凹的熵项结合时也很有挑战性；（2）现有自标注方法使用**硬伪标签**（one-hot 分布），无法表达类别估计的不确定性和误差，导致错误信号传播；（3）涂鸦监督领域缺乏对不同 Potts 松弛形式和交叉熵变体的系统性比较。

**核心矛盾**：在自标注框架中，伪标签既需要足够确定以指导网络训练，又需要能表达不确定性以避免将错误标签强行传播给网络。硬伪标签无法实现这种平衡。

**本文目标**：（1）将软伪标签引入有原则保证收敛的自标注框架；（2）系统评估不同 Potts 松弛和交叉熵变体；（3）在标准架构上证明软自标注可以超越复杂的专用弱监督方法。

**切入角度**：基于 ADM（Alternating Direction Method）分裂方法，将原始弱监督损失分解为两个子问题——网络训练子问题和伪标签优化子问题，后者中伪标签被允许取软值（概率分布），而非被限制为 one-hot。

**核心 idea**：用软分类分布作为伪标签，配合新提出的碰撞交叉熵和归一化二次/碰撞散度等 Potts 松弛形式，在有收敛保证的自标注框架中实现更好的弱监督分割。

## 方法详解

### 整体框架

整体框架围绕一个联合损失函数进行交替优化。该损失包含三项：（1）已标注像素（涂鸦）上的 NLL 损失；（2）未标注像素上连接网络预测 $\sigma_i$ 和软伪标签 $y_i$ 的交叉熵项 $H(\sigma_i, y_i)$；（3）未标注像素上伪标签之间的 Potts 松弛正则化 $P(y_i, y_j)$。交替优化过程中：固定伪标签更新网络参数（标准训练子问题），固定网络参数更新软伪标签（Potts 优化子问题），两个子问题迭代求解保证联合损失的收敛。

### 关键设计

1. **软伪标签自标注框架**:

    - 功能：将伪标签从 one-hot 扩展为一般概率分布，能够表达类别估计的不确定性
    - 核心思路：通过 ADM 分裂将弱监督损失 $-\sum_{i \in S} \ln \sigma_i^{\bar{y}_i} + \eta \sum_{i \notin S} H(\sigma_i) + \lambda \sum_{ij \in \mathcal{N}} P(\sigma_i, \sigma_j)$ 转化为引入辅助变量 $y_i \in \Delta^K$ 的联合损失。伪标签 $y_i$ 是定义在 K 类概率单纯形上的一般分布，在种子像素上约束为与真值一致 $y_i = \bar{y}_i$，在未标注像素上自由优化。用 KL 散度和熵的组合将 $\sigma_i \approx y_i$ 的约束融入损失，最终得到形式简洁的联合损失
    - 设计动机：硬伪标签将分类错误不可逆地传播给网络；软伪标签在边界处自然地保持高不确定性，减少边界附近的错误监督信号。这与硬标签 graph cut 求解器相比，能更好地处理模糊区域

2. **Potts 松弛系统性研究（6 种形式）**:

    - 功能：为 Potts 模型在连续域上的优化提供多种选择，解决不同松弛形式的梯度消失和局部最优问题
    - 核心思路：研究了 6 种松弛形式。基本形式：双线性 $P_{BL} = 1 - p^\top q$（紧但非凸），二次 $P_Q = \frac{1}{2}\|p-q\|^2$（凸但不紧）。新提出归一化二次 $P_{NQ} = 1 - \frac{p^\top q}{\|p\|\|q\|}$，结合了两者优点。对数版本：碰撞交叉熵 $P_{CCE} = -\ln p^\top q$，碰撞散度 $P_{CD} = -\ln \frac{p^\top q}{\|p\|\|q\|}$，对数二次 $P_{LQ} = -\ln(1-\frac{\|p-q\|^2}{2})$。对数版本解决了基本形式的梯度消失问题
    - 设计动机：通过分析两个典型的"移动"场景（同区域重分类和边界区域标签切换），揭示双线性松弛在前者产生局部最优而二次松弛在后者产生局部最优。归一化消除了这两个问题，对数变换解决了平坦区域的梯度消失

3. **碰撞交叉熵（Collision Cross-Entropy）**:

    - 功能：连接网络预测与软伪标签的损失函数，对标签不确定性具有鲁棒性
    - 核心思路：定义为 $H_{CCE}(y_i, \sigma_i) = -\ln \sum_k \sigma_i^k y_i^k = -\ln \sigma^\top y$。内积 $\sigma^\top y$ 可解释为预测类别 C 和真实类别 T 相等的概率 $\Pr(C=T) = \sum_k \Pr(C=k)\Pr(T=k)$。该损失最大化"碰撞概率"而非强制分布相等，且关于两个参数对称——既不强迫预测模仿伪标签的不确定性，也不让伪标签模仿预测的不确定性
    - 设计动机：标准交叉熵 $H_{CE}$ 会强迫网络复制伪标签的不确定性（当 $y=(0.5,0.5)$ 时网络也学习输出 $(0.5,0.5)$）；反向交叉熵 $H_{RCE}$ 解决了这个方向但伪标签反过来模仿预测。碰撞交叉熵因对称性一劳永逸地解决了双方的不确定性模仿问题

### 损失函数 / 训练策略

使用 DeepLabv3+ 架构，backbone 为 ResNet-101/MobileNetV2/ViT。先用涂鸦上的交叉熵损失预热网络参数，然后切换到联合损失交替优化 60 轮。SGD 优化器，初始学习率 0.0007，多项式衰减。最佳超参数：$\eta=0.3$，$\lambda=6$，使用碰撞交叉熵 $H_{CCE}$ 和碰撞散度 $P_{CD}$。伪标签子问题使用 GPU 友好的梯度下降求解。

## 实验关键数据

### 主实验

| 方法 | 架构 | 监督 | PASCAL VOC mIoU |
|------|------|------|-----------------|
| Full supervision | V3+ (R101) | 全像素 | 76.6 |
| Full supervision | V3+ (R101, bs16) | 全像素 | 78.9 |
| Full supervision | ViT-linear | 全像素 | 81.4 |
| **Soft SL (ours)** | V3+ (R101) | **涂鸦** | **76.7** |
| **Soft SL (ours)** | ViT-linear | **涂鸦** | **81.6** |
| Hard SL [29] | V3+ (R101) | 涂鸦 | 69.6 |
| GD baseline [38] | V3+ (R101) | 涂鸦 | 69.5 |
| BPG [41] | V2 (△) | 涂鸦 | 76.0 |

### 消融实验（Potts 松弛对比，使用 MobileNetV2）

| 松弛形式 | scribble=0 | scribble=0.5 | scribble=1.0 |
|----------|-----------|-------------|-------------|
| $P_{BL}$ (双线性) | 56.42 | 63.81 | 67.24 |
| $P_Q$ (二次) | 58.92 | 67.81 | 71.05 |
| $P_{NQ}$ (归一化二次) | 59.01 | 67.80 | 71.12 |
| $P_{CCE}$ (碰撞交叉熵) | 56.40 | 63.81 | 67.41 |
| $P_{CD}$ (碰撞散度) | **59.04** | **67.84** | **71.22** |
| $P_{LQ}$ (对数二次) | 59.03 | 67.81 | 71.21 |

### 关键发现

- **软自标注用涂鸦可以超越全像素监督**：在 ViT-linear 架构上，涂鸦监督（81.6 mIoU）超过了全像素监督（81.4 mIoU），这是一个惊人的结果
- **碰撞交叉熵一致胜出**：在不同监督水平下，$H_{CCE}$ 始终优于标准交叉熵和反向交叉熵，验证了其对标签不确定性的鲁棒性
- **对数版松弛优于基本版**：对数变换一致性地改善了所有松弛形式的性能，原因是解决了梯度消失问题并鼓励边界处更平滑的过渡
- **最近邻（NN）优于稠密邻域（DN）**：NN 的 71.1% vs DN 的 67.9%，因为大邻域使 Potts 退化为容积势，不利于边缘对齐
- **归一化松弛消除了基本形式的局部最优**：$P_{NQ}$ 和 $P_{CD}$ 同时避免了双线性和二次松弛各自的问题

## 亮点与洞察

- **3% 像素标注超越 100% 标注**：这是方法论上的重大突破，说明好的无监督损失设计和优化方法可以比简单的全监督更有效，因为涂鸦加正则化避免了全监督中的标注噪声
- **碰撞交叉熵的概率解释**：将软标签间的对齐理解为"两个随机变量相等的概率"而非"分布相等"，这种视角转换非常优雅，可广泛应用于其他使用软标签/伪标签的场景
- **方法的通用性**：纯优化方法论层面的贡献，不依赖特定架构或训练技巧，可以直接迁移到其他弱监督问题（如点监督、框监督）

## 局限与展望

- 伪标签子问题使用梯度下降求解，对于非凸松弛可能陷入局部最优，开发专门的凸优化或组合优化求解器可能进一步提升
- 仅在 PASCAL VOC、Cityscapes、ADE20k 上评估，缺乏在更大规模数据集上的验证
- 邻域系统的选择（NN vs DN）对结果影响显著，目前依赖人工选择
- 碰撞交叉熵虽然鲁棒但不是最严格的信息论量，理论基础可以进一步加强

## 相关工作与启发

- **vs GCRF [28]**: 最早使用 graph cut 求解硬伪标签的涂鸦分割方法。本文从硬扩展到软伪标签，且有收敛保证的联合损失形式
- **vs ADM/Trust-Region [30,29]**: 这些方法已经使用 ADM 分裂但仅限于硬伪标签。本文是首个在原则性框架中使用软伪标签并研究其在 Potts 松弛中表现的工作
- **vs BPG [41]**: BPG 使用修改过的架构和软伪标签提议但没有收敛保证。本文在标准架构上用有证明的方法达到更好效果

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 碰撞交叉熵是全新的损失设计，Potts 松弛的系统性研究填补了理论空白
- 实验充分度: ⭐⭐⭐⭐ 系统性评估了多种组合，但数据集范围可以更广
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨清晰，理论与实验紧密结合，是一篇优秀的方法论论文
- 价值: ⭐⭐⭐⭐⭐ 涂鸦超越全监督是标志性结果，碰撞交叉熵有广泛应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation](exploring_clips_dense_knowledge_for_weakly_supervised_semantic_segmentation.md)
- [\[CVPR 2026\] Frequency-Aware Affinity for Weakly Supervised Semantic Segmentation](../../CVPR2026/segmentation/frequency-aware_affinity_for_weakly_supervised_semantic_segmentation.md)
- [\[AAAI 2026\] SSR: Semantic and Spatial Rectification for CLIP-based Weakly Supervised Segmentation](../../AAAI2026/segmentation/ssr_semantic_and_spatial_rectification_for_clip-based_weakly_supervised_segmenta.md)
- [\[CVPR 2026\] Leveraging Class Distributions in CLIP for Weakly Supervised Semantic Segmentation](../../CVPR2026/segmentation/leveraging_class_distributions_in_clip_for_weakly_supervised_semantic_segmentati.md)
- [\[ICCV 2025\] Joint Self-Supervised Video Alignment and Action Segmentation](../../ICCV2025/segmentation/joint_self-supervised_video_alignment_and_action_segmentation.md)

</div>

<!-- RELATED:END -->
