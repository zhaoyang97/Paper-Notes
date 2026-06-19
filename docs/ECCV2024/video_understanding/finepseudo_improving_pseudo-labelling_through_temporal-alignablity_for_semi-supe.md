---
title: >-
  [论文解读] FinePseudo: Improving Pseudo-Labelling through Temporal-Alignability for Semi-Supervised Fine-Grained Action Recognition
description: >-
  [ECCV 2024][视频理解][细粒度动作识别] 提出 FinePseudo 框架，利用基于时序对齐性（temporal alignability）的度量学习来改善伪标签质量，首次系统性地解决半监督细粒度动作识别问题，在四个细粒度数据集上显著超越现有方法。 细粒度动作识别（Fine-Grained Action Reco…
tags:
  - "ECCV 2024"
  - "视频理解"
  - "细粒度动作识别"
  - "伪标签"
  - "时序对齐"
  - "半监督学习"
  - "度量学习"
---

# FinePseudo: Improving Pseudo-Labelling through Temporal-Alignability for Semi-Supervised Fine-Grained Action Recognition

**会议**: ECCV 2024  
**arXiv**: [2409.01448](https://arxiv.org/abs/2409.01448)  
**代码**: [有](https://daveishan.github.io/finepsuedo-webpage/)  
**领域**: 视频理解 / 半监督学习  
**关键词**: 细粒度动作识别, 伪标签, 时序对齐, 半监督学习, 度量学习

## 一句话总结

提出 FinePseudo 框架，利用基于时序对齐性（temporal alignability）的度量学习来改善伪标签质量，首次系统性地解决半监督细粒度动作识别问题，在四个细粒度数据集上显著超越现有方法。

## 研究背景与动机

细粒度动作识别（Fine-Grained Action Recognition, FGAR）在体育分析、手术视频、AR/VR 等实际应用中至关重要。与粗粒度动作（如"弹吉他" vs "投标枪"）不同，细粒度动作（如跳水的不同类别）之间的差异仅体现在动作阶段（action phases）的微小变化上——例如起跳、空中翻转、入水三个阶段中，仅入水姿势不同就可能改变动作类别。

然而，细粒度动作的标注极其昂贵，需要专家反复观看才能准确标注。这使得半监督学习成为 FGAR 的自然选择。但现有半监督视频方法存在两个核心问题：

**场景偏置依赖**：现有方法（如 token-mix、CutMix 等增强策略）主要利用场景上下文来区分动作。细粒度动作通常发生在相同场景中（如所有跳水动作都在跳台），这些策略失效。

**时序粒度不足**：视频级自监督方法（如 TimeBal）学习的表示缺乏帧级别的动作阶段信息，而这正是 FGAR 的关键。

作者进行了一个关键的初步实验：用帧级视频编码器提取嵌入后，比较不同距离度量在区分同类/异类细粒度动作对上的能力。结果发现：
- 时序池化后的 cosine 距离丧失了时序细节
- 逐帧 cosine 距离无法处理不同动作阶段时长差异
- **DTW（动态时间规整）对齐距离**能够进行阶段对阶段的比较，显著提升了区分能力

这一发现此前从未在 FGAR 的有限标签设定中被探索过。

## 方法详解

### 整体框架

FinePseudo 是一个基于伪标签的协同训练（co-training）框架，包含两个分支：

- **动作编码器** $f_E$：学习视频级高层语义特征（动作分类）
- **对齐性编码器** $f_A$：帧级视频编码器（VTN），学习基于动作阶段的低层帧内表示

训练流程分为三个阶段：(1) 无标签数据上的自监督预训练；(2) 有标签数据上的对齐性验证度量学习；(3) 协同伪标签自训练。

### 关键设计

1. **基于对齐性验证的度量学习（Alignability-Verification Metric Learning）**

   核心假设：同类细粒度视频比异类视频更"可对齐"（alignable）。

   对于视频对 $U, V$，通过 $f_A$ 提取帧级嵌入 $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{T \times F}$，构建代价矩阵 $\mathbb{C}(i,j) = h(\mathbf{u}(i), \mathbf{v}(j))$（cosine 距离），然后用 softDTW 计算可微对齐距离：

    $D(\mathbf{u}, \mathbf{v}) = \mathbb{C}(i,j) + \gamma\text{-smooth-min}(\Pi_{\text{cost}}(i,j))$

   基于此对齐距离，使用三元组损失进行度量学习：

    $\mathcal{L}_{AT} = \sum_{i=1}^{N} [D(\mathbf{v}^{(i)}, \mathbf{v}^{(j)}) - D(\mathbf{v}^{(i)}, \mathbf{v}^{(k)}) + m]$

   其中正对来自同类，负对来自异类，并使用 hard-negative mining。

2. **可学习对齐性分数（Learnable Alignability Score）**

   将对齐距离 $D$ 通过非线性缩放函数和 sigmoid 映射到 $[0,1]$：

    $S(\mathbf{u}, \mathbf{v}) = \varsigma(f_S(D(\mathbf{u}, \mathbf{v})))$

   用二元交叉熵训练这个分数函数：

    $\mathcal{L}_{Score} = -[y_A \log(S) + (1-y_A)\log(1-S)]$

   总的对齐性训练目标为 $\mathcal{L}_{AV} = \mathcal{L}_{AT} + \omega \mathcal{L}_{Score}$。这个分数相比原始 DTW 距离提供了更好的类区分能力。

3. **协同伪标签（Collaborative Pseudo-Labeling）**

   对每个无标签视频 $U$，同时从两个编码器获取预测：
    - $\mathbf{p}_E$：$f_E$ 的分类头直接输出
    - $\mathbf{p}_A$：基于非参数分类器——计算 $U$ 的嵌入与每个类别标签样本的平均对齐性分数 $\bar{S}_c$，再通过带温度的 softmax：

    $\mathbf{p}_A(c) = \frac{\exp(\bar{S}_c / \tau)}{\sum_j \exp(\bar{S}_j / \tau)}$

   最终预测 $\mathbf{p} = \mathbf{p}_A + \mathbf{p}_E$，超过置信度阈值 $\theta$ 的样本被赋予伪标签加入训练集。两个分支提供互补信息（视频级 vs 对齐性），迭代更新伪标签。

### 损失函数 / 训练策略

- $f_A$ 先用 GITDL 自监督预训练学习帧间动态，再用 $\mathcal{L}_{AV}$ 在标签数据上训练
- $f_E$ 用标准交叉熵 $\mathcal{L}_{CE}$ 训练
- 自训练阶段迭代进行：生成协同伪标签 → 扩充标签集 → 重新训练两个编码器

## 实验关键数据

### 主实验

在四个细粒度数据集上，使用 R2plus1D-18 骨干网络：

| 数据集 | 标签比例 | FinePseudo | TimeBal (之前SOTA) | 提升 |
|--------|---------|------------|-------------------|------|
| Diving48 | 5% | **20.9** | 15.8 | +5.1 |
| Diving48 | 10% | **37.6** | 33.7 | +3.9 |
| Diving48 | 20% | **60.4** | 56.3 | +4.1 |
| FineGym99 | 5% | **49.2** | 44.4 | +4.8 |
| FineGym99 | 10% | **69.9** | 65.9 | +4.0 |
| FineGym288 | 5% | **41.7** | 37.3 | +4.4 |
| FineDiving | 5% | **28.4** | 25.1 | +3.3 |

使用 AIM-ViTB (CLIP初始化) 在 Diving48 上：

| 方法 | 5% | 10% | 20% |
|------|-----|------|------|
| SVFormer | 38.00 | 56.02 | 76.20 |
| TimeBal | 38.12 | 55.80 | 76.01 |
| **FinePseudo** | **43.02** | **60.79** | **80.02** |

### 消融实验

| 配置 | 10% Acc | 20% Acc | 说明 |
|------|---------|---------|------|
| 仅 $f_E$ + PL | 33.40 | 54.00 | 基线伪标签 |
| 仅 $f_A$ | 32.82 | 51.05 | 对齐编码器单独不够强 |
| $f_E$ + $\mathcal{L}_{AT}$ only | 33.73 | 55.67 | 三元组损失贡献 |
| $f_E$ + $\mathcal{L}_{Score}$ only | 36.11 | 59.32 | 分数损失贡献更大 |
| 无SSL预训练 | 35.23 | 58.64 | SSL预训练有帮助 |
| **完整 FinePseudo** | **37.64** | **60.40** | 所有组件协同最优 |

### 关键发现

- FinePseudo 在所有细粒度数据集上一致超越先前方法 4-5% 绝对精度
- 在粗粒度数据集（K400、SSv2）上也表现出竞争力或略有提升
- 在开放世界设定（含未知类别的无标签数据）中，非参数分类器能有效过滤未知类，展现鲁棒性
- 对齐性分数损失 $\mathcal{L}_{Score}$ 比三元组损失 $\mathcal{L}_{AT}$ 单独贡献更大，但两者结合最佳

## 亮点与洞察

- **核心洞察优雅**：将"可对齐性"从传统的类内对齐假设扩展为跨类判别度量，用于半监督设定
- **互补性强**：视频级语义预测与帧级对齐性预测提供正交信息源
- **通用性**：方法同时适用于细粒度和粗粒度动作识别
- **开放世界鲁棒**：非参数分类器天然适合处理未知类别

## 局限与展望

- softDTW 计算复杂度为 $O(T^2)$，对长视频可能带来效率问题
- 依赖 SSL 预训练权重（TCLR/Kinetics400），对预训练质量有要求
- 对齐性假设对非结构化动作（无明确阶段的日常动作）可能效果减弱
- 未探索无标签数据中类别不平衡的影响

## 相关工作与启发

- 与 LAV（Learning by Aligning Videos）等对齐方法不同，本文的"对齐性验证"不假设视频一定可对齐，而是学习判别对齐难度
- 对齐性分数可作为通用的视频相似度度量，潜在应用于检索、质量评估等任务
- 协同伪标签的思路可推广到其他需要互补信息源的半监督场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将对齐性引入半监督FGAR是原创且有效的洞察
- **实验充分度**: ⭐⭐⭐⭐ — 4个细粒度 + 2个粗粒度数据集，消融详尽，开放世界设定新颖
- **写作质量**: ⭐⭐⭐⭐ — 动机从初步实验出发，逻辑清晰
- **价值**: ⭐⭐⭐⭐ — 首次系统研究半监督FGAR，方法具有实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] FineTec: Fine-Grained Action Recognition Under Temporal Corruption via Skeleton Decomposition and Sequence Completion](../../AAAI2026/video_understanding/finetec_fine-grained_action_recognition_under_temporal_corruption_via_skeleton_d.md)
- [\[ECCV 2024\] SA-DVAE: Improving Zero-Shot Skeleton-Based Action Recognition by Disentangled Variational Autoencoders](sa-dvae_improving_zero-shot_skeleton-based_action_recognition_by_disentangled_va.md)
- [\[ECCV 2024\] Leveraging Temporal Contextualization for Video Action Recognition](leveraging_temporal_contextualization_for_video_action_recognition.md)
- [\[CVPR 2026\] MA-Bench: Towards Fine-grained Micro-Action Understanding](../../CVPR2026/video_understanding/ma-bench_towards_fine-grained_micro-action_understanding.md)
- [\[ECCV 2024\] Referring Atomic Video Action Recognition](referring_atomic_video_action_recognition.md)

</div>

<!-- RELATED:END -->
