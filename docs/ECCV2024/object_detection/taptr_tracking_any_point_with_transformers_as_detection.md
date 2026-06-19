---
title: >-
  [论文解读] TAPTR: Tracking Any Point with Transformers as Detection
description: >-
  [ECCV 2024][目标检测][点跟踪] TAPTR 将 Tracking Any Point (TAP) 任务重新建模为类 DETR 的检测问题，将每个跟踪点表示为包含位置和内容的 point query，通过多层 Transformer 解码器逐层优化，结合 cost volume 和滑动窗口特征更新策略，在 TAP-Vid 基准上达到 SOTA 且推理速度更快。
tags:
  - "ECCV 2024"
  - "目标检测"
  - "点跟踪"
  - "Transformer"
  - "DETR"
  - "光流"
  - "TAP"
---

# TAPTR: Tracking Any Point with Transformers as Detection

**会议**: ECCV 2024  
**arXiv**: [2403.13042](https://arxiv.org/abs/2403.13042)  
**代码**: [https://taptr.github.io](https://taptr.github.io)  
**领域**: 视频理解  
**关键词**: 点跟踪, Transformer, DETR, optical flow, TAP

## 一句话总结

TAPTR 将 Tracking Any Point (TAP) 任务重新建模为类 DETR 的检测问题，将每个跟踪点表示为包含位置和内容的 point query，通过多层 Transformer 解码器逐层优化，结合 cost volume 和滑动窗口特征更新策略，在 TAP-Vid 基准上达到 SOTA 且推理速度更快。

## 研究背景与动机

**领域现状**: 理解视频中每个像素的运动是计算机视觉的基础任务。光流估计 (Optical Flow) 是主流方法但仅处理连续两帧间的对应关系，无法应对遮挡。语义关键点跟踪可以处理遮挡但目标语义类别受限（如人体关节）。近期 TAP (Tracking Any Point) 任务被提出，要求在整个视频中跟踪任意用户指定的点，其代表性方法包括 PIPs、TAP-Net、TAPIR 和 CoTracker。

**现有痛点**: 现有 TAP 方法（PIPs、TAPIR、CoTracker）对跟踪点的建模方式不够清晰——将 flow vector、flow embedding、visibility、content feature、cost volume 等多种特征简单拼接成一个"黑箱"向量，送入 MLP 或 Transformer 期望模型自行解读和利用。这种方式缺乏结构化设计，不利于理解和优化。

**核心矛盾**: TAP 任务需要同时处理长距离时序建模（遮挡恢复）和精细的低层特征匹配（精确定位），需要一个既简洁又强大的统一框架。大多数先前方法独立处理每个跟踪点，忽略了属于同一物体的点之间可提供的上下文信息。

**本文目标** 设计一个概念简洁、含义清晰的点跟踪框架，使每个组件都有明确的物理意义，同时在性能和速度上超越现有方法。

**切入角度**: 观察到点跟踪与目标检测/跟踪具有高度相似性——在每一帧中，跟踪点本质上是需要检测的目标。因此可以直接借用 DETR 系列算法的成熟设计。

**核心 idea**: 将每个跟踪点建模为 DETR-like 的 point query（位置部分 + 内容部分），通过 Transformer 解码器逐层优化，自然地复用检测任务中已被充分验证的自注意力、交叉注意力和迭代优化机制。

## 方法详解

### 整体框架

TAPTR 由四个主要部分组成：(1) **Video Preparation** — 使用 CNN backbone + Transformer Encoder 提取多尺度特征图；(2) **Query Preparation** — 为每帧中的每个跟踪点准备初始位置、内容特征和 cost volume；(3) **Point Decoder** — 多层 Transformer 解码器并行处理滑动窗口内所有帧；(4) **Window Post-processing** — 在窗口间更新和传播 query 的显式状态和内容特征。采用滑动窗口策略，每次处理 $W$ 帧（默认 $W=8$，步长 4）。

### 关键设计

1. **Point Query 建模**:

    - **功能**: 在每帧中，每个跟踪点被表示为一个 point query $q_t^i = [f_t^i, l_t^i]$，包含内容特征 $f_t^i$ 和位置 $l_t^i$ 两部分，含义清晰。
    - **核心思路**: 内容特征通过在跟踪点首次出现的帧的多尺度特征图上进行双线性插值获得：$f_e^i = \text{MLP}(\text{Cat}(\text{Bili}(F_{e^i,1}, l_e^i), \ldots, \text{Bili}(F_{e^i,S}, l_e^i)))$。所有属于同一跟踪点的 query 共享初始内容特征和初始位置。
    - **设计动机**: 区别于先前方法将所有信息拼接成黑箱向量，DETR-like 的 query（位置+内容）具有明确的物理含义——位置描述"在哪"，内容描述"是什么"，与检测任务的 query 设计一致。

2. **Cost Volume 聚合**:

    - **功能**: 计算 point query 与图像特征的内积得到视觉相似度图，在解码器中局部采样 cost volume 来增强 query 的内容特征。
    - **核心思路**: Cost volume 在解码器开始前一次性计算 $C_{t,s}^i = \text{InnerProd}(F_{t,s}, f_t^i)$，在解码器层间作为静态特征图重复使用。采用 RAFT 风格的局部 grid sampling：$c_{t,s}^i = \text{GridSample}(C_{t,s}^i, \text{Grid}(l_t^i, G))$，采样后的 cost vector 与内容特征拼接经 MLP 融合。
    - **设计动机**: (a) 点跟踪比目标检测需要更多局部低层特征来精确定位；(b) 不同于先前方法在每次内容特征更新后重新计算 cost volume，一次计算保持了解码器的简洁性和多层优化目标的稳定性；(c) Cost volume 在光流和立体匹配中已被充分验证。实验证明一次性计算 + 窗口间更新优于逐迭代更新（DAVIS 上 +1.1 AJ, RGB-Stacking 上 +2.8 AJ）。

3. **Point Decoder 多模块设计**:

    - **Visual Feature Enhancer (交叉注意力)**: 使用 2D deformable attention 在跟踪点周围采样多尺度局部图像特征，补充 cost volume 缺少的详细几何信息。
    - **Point Query 自注意力**: 同一帧内所有 point queries 通过自注意力交互，添加正弦位置编码（降低温度 $\tau$ 以获得更尖锐的位置嵌入，适应点跟踪的高精度需求）。降温 100 倍带来 1.1 AJ 提升。
    - **Temporal Attention**: 属于同一跟踪点的 queries 沿时间维度进行密集注意力交互 $f^i \Leftarrow \text{Attention}(f^i, f^i)$，$f^i \in \mathbb{R}^{W \times C}$，建模短期时序信息。
    - **Residual Content Update**: 位置更新采用 DETR 式的 Sigmoid 迭代优化；内容特征采用残差更新 $f_t^i \Leftarrow f_t^i + \text{MLP}(\text{Cat}(f_t^i, f_{e^i}^i))$，始终参考初始特征避免漂移。比 DETR 原始的直接更新好 1.7 AJ。

4. **窗口后处理与特征漂移缓解**:

    - **功能**: 在滑动窗口间更新轨迹状态，同时传递内容特征以保留长程时序信息，但需解决特征漂移问题。
    - **核心思路**: 训练时以 0.6 的概率随机丢弃特征更新（Random Drop），迫使网络适应有/无特征更新两种情况。推理时保持特征更新开启但以动态频率（$T/24$ 窗口间隔）丢弃特征 padding，在时序信息保留与漂移控制之间取得最佳平衡。
    - **设计动机**: 直接无限制更新内容特征会在长视频上导致严重漂移（RGB-Stacking 上 AJ 从 60.8 暴降到 23.1），因为训练时视频只有 24 帧但推理时可能很长，存在长度不一致问题。

### 损失函数 / 训练策略

- **损失函数**: 全序列多层损失，在获得完整轨迹后计算：$\text{Loss} = (\omega_V \cdot \text{CE}(V, \tilde{V}) + \sum_{d=1}^{D} \omega_L \cdot \text{L1}(L_d, \tilde{L})) / N$
- 位置使用 L1 loss，每层解码器输出都有监督（辅助损失）；visibility 使用交叉熵损失，仅在最后一层预测
- 在 TAP-Vid-Kubric 合成数据上训练（11000 个 24 帧视频），随机采样 700-800 个点
- 训练分辨率 $512 \times 512$，评估分辨率 $256 \times 256$
- 使用 ResNet50 backbone，2 层 Transformer Encoder，6 层解码器
- AdamW 优化器 + EMA，8 块 A100 训练约 36000 步，学习率 2e-4，梯度累积 4 次（等效 batch size 32）

## 实验关键数据

### 主实验

在 TAP-Vid 基准上的 SOTA 对比 (First mode)：

| 方法 | DAVIS AJ | DAVIS $<\delta_{avg}^x$ | DAVIS OA | RGB-Stacking AJ | Kinetics AJ | 速度 (PPS) |
|---|---|---|---|---|---|---|
| TAPIR | 56.2 | 70.0 | 86.5 | 55.5 | 49.6 | - |
| CoTracker-All | 60.7 | 75.7 | 88.1 | - | - | 15.7 |
| CoTracker-Single | 62.2 | 75.7 | 89.3 | - | - | 0.8 |
| BootsTAP† (extra 15M data) | 61.4 | 74.0 | 88.4 | - | 54.7 | - |
| **TAPTR (ours)** | **63.0** | **76.1** | **91.1** | **60.8** | 49.0 | **20.4** |

关键对比: TAPTR 超越 CoTracker-All 2.3 AJ (63.0 vs 60.7)，超越 CoTracker-Single 0.8 AJ (63.0 vs 62.2) 且速度快 **25 倍** (20.4 vs 0.8 PPS)。

### 消融实验

**关键组件消融 (DAVIS, Table 2)**:

| 配置 | AJ | $<\delta_{avg}^x$ | OA | 说明 |
|---|---|---|---|---|
| Full TAPTR | 63.0 | 76.1 | 91.1 | 完整模型 |
| - Small Temperature | 61.9 | 75.4 | 90.3 | 降温对自注意力很重要 (-1.1) |
| - Transformer Encoder | 60.9 | 75.2 | 88.9 | 影响visibility估计 (-1.0) |
| - Self Attention | 58.4 | 72.1 | 88.3 | 点间交互很重要 (-2.5) |
| - Temporal Attention | 51.6 | 66.7 | 84.5 | 时序信息最关键 (-6.8) |
| - Cost Volume | 46.8 | 61.3 | 82.4 | 视觉相似度基础 (-4.8) |
| - Cross Attention | 50.0 | 65.0 | 83.4 | 几何信息补充 (-1.6) |
| 原始DETR更新 (vs残差) | 45.1 | 60.0 | 82.4 | 残差更新至关重要 (-1.7) |

**Cost Volume 更新频率消融 (Table 6)**:

| 更新策略 | DAVIS AJ | RGB-Stacking AJ | 说明 |
|---|---|---|---|
| 每次迭代更新 | 61.9 | 58.0 | 过于频繁，优化目标不稳定 |
| 从不更新 | 63.0 | 58.3 | 短视频可以但长视频信息不足 |
| **每窗口更新** | **63.0** | **60.8** | 最佳：稳定性 + 长程信息 |

### 关键发现

1. **时序信息是最关键因素**: 移除 Temporal Attention + 窗口间特征更新导致 10.1 AJ 下降（首要贡献）。
2. **Cost Volume 作为不可或缺的基础感知**: 移除后下降 4.8 AJ，与光流领域的经验一致。
3. **低温位置编码对点级精度至关重要**: 将温度降低 100 倍，使邻近 query 获得更多注意力，提升 1.1 AJ。
4. **特征漂移是长视频的核心挑战**: 不受限制的特征更新在 RGB-Stacking（长视频）上 AJ 从 60.8 暴跌至 23.1。
5. **多层监督必不可少**: 仅最后一层监督 vs 全层监督：AJ 55.4 vs 63.0，差距 7.6。
6. **残差内容更新优于直接更新**: 始终参考初始特征避免噪声累积，提升 1.7 AJ。
7. **解码器层数呈正收益**: 2层→4层→6层 对应 AJ 58.2→61.6→63.0。

## 亮点与洞察

- **检测-跟踪统一视角**: 将点跟踪重新定义为逐帧点检测问题，充分复用 DETR 成熟的 query 机制，概念简洁而性能强大。
- **清晰的点建模**: 相比先前方法的黑箱向量拼接，position + content 的 query 设计含义明确，每个操作（自注意力、交叉注意力、时序注意力）都有清晰的物理解释。
- **Cost Volume 的一次性计算策略**: 反直觉地不在解码器内频繁更新 cost volume，反而保持了优化目标的稳定性，是一个值得借鉴的工程洞察。
- **特征漂移的 Random Drop 缓解**: 简单而有效的训练-推理不一致问题解决方案，通过随机丢弃迫使网络学习鲁棒表示。
- **消融实验极其详尽**: 逐一验证每个组件的贡献，为后续工作提供了宝贵的参考基线。

## 局限与展望

- **训练数据仅为合成数据**: 仅在 Kubric 合成数据上训练，如何利用真实世界的检测/分割标注数据来辅助 TAP 训练是一个开放问题。
- **Kinetics 数据集性能不突出**: TAPTR 在 Kinetics 上 (49.0 AJ) 低于 BootsTAP (54.7)，可能因为 BootsTAP 使用了额外 15M 真实视频数据。
- **滑动窗口的固定设计**: 窗口大小 (W=8) 和步长 (4) 是固定的，自适应窗口策略可能在不同视频复杂度下更优。
- **Backbone 仅用 ResNet50**: 可探索更强的 backbone（如 Swin Transformer、ConvNeXt）或自监督预训练的特征。
- **点间交互的效率**: 全局自注意力在跟踪大量点时复杂度为 $O(N^2)$，大规模点跟踪场景可能需要更高效的交互方式。

## 相关工作与启发

- **DETR 系列**: 直接借用了 DETR / Deformable DETR / DINO 的 query 设计、多层迭代优化和辅助损失，验证了检测框架在跟踪任务上的强大迁移性。
- **RAFT**: Cost volume 的局部 grid sampling 策略直接来自 RAFT，是光流估计到点跟踪的关键桥梁。
- **CoTracker**: 首个使用 Transformer 进行多点联合跟踪的方法，但 TAPTR 通过更清晰的 query 建模和位置编码设计超越了它。
- **PIPs / TAPIR**: 迭代优化的点跟踪范式的先驱，但独立处理每个点，缺乏点间交互。
- **DAB-DETR**: 动态 anchor box 的 query 设计启发了 TAPTR 的位置更新方式。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将点跟踪统一到 DETR 框架的视角新颖且自然，但核心组件均来自已有方法的组合
- **实验充分度**: ⭐⭐⭐⭐⭐ 消融实验覆盖所有关键组件，每个设计决策都有定量验证，BADJA 等额外数据集的补充评估
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，与目标检测的类比直观易懂，公式符号统一
- **价值**: ⭐⭐⭐⭐ 为 TAP 任务提供了一个简洁强大的基线，消融研究对后续工作具有很高的参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Projecting Points to Axes: Oriented Object Detection via Point-Axis Representation](projecting_points_to_axes_oriented_object_detection_via_point-axis_representatio.md)
- [\[ECCV 2024\] Shifted Autoencoders for Point Annotation Restoration in Object Counting](shifted_autoencoders_for_point_annotation_restoration_in_object_counting.md)
- [\[ECCV 2024\] WALKER: Self-supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs](walker_self-supervised_multiple_object_tracking_by_walking_on_temporal_appearanc.md)
- [\[ECCV 2024\] Stepwise Multi-grained Boundary Detector for Point-Supervised Temporal Action Localization](stepwise_multi-grained_boundary_detector_for_point-supervised_temporal_action_lo.md)
- [\[CVPR 2025\] Mr. DETR++: Instructive Multi-Route Training for Detection Transformers with MoE](../../CVPR2025/object_detection/mr_detr_instructive_multi-route_training_for_detection_transformers.md)

</div>

<!-- RELATED:END -->
