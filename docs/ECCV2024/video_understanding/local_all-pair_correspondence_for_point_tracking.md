---
title: >-
  [论文解读] Local All-Pair Correspondence for Point Tracking
description: >-
  [ECCV 2024][视频理解][点跟踪] 本文提出LocoTrack，通过局部4D相关性体（local 4D correlation）实现视频中任意点的全对应匹配，结合轻量级相关性编码器和长度可泛化的Transformer，在所有TAP-Vid基准测试上达到最高精度，同时比SOTA方法快近6倍。
tags:
  - "ECCV 2024"
  - "视频理解"
  - "点跟踪"
  - "4D相关性"
  - "全对应匹配"
  - "TAP-Vid"
  - "Transformer"
---

# Local All-Pair Correspondence for Point Tracking

**会议**: ECCV 2024  
**arXiv**: [2407.15420](https://arxiv.org/abs/2407.15420)  
**代码**: [https://github.com/KU-CVLAB/LocoTrack](https://github.com/KU-CVLAB/LocoTrack)  
**领域**: 视频理解 / 点跟踪  
**关键词**: 点跟踪, 4D相关性, 全对应匹配, TAP-Vid, 高效Transformer

## 一句话总结
本文提出LocoTrack，通过局部4D相关性体（local 4D correlation）实现视频中任意点的全对应匹配，结合轻量级相关性编码器和长度可泛化的Transformer，在所有TAP-Vid基准测试上达到最高精度，同时比SOTA方法快近6倍。

## 研究背景与动机

**领域现状**：视频任意点跟踪（Tracking Any Point, TAP）是近年兴起的基础视觉任务，给定视频和一个查询点，需要在所有帧中找到该点的对应位置及遮挡状态。代表方法如TAPIR和CoTracker通常使用局部2D相关性图——将查询点的特征与目标帧的局部区域做点到区域的相似度匹配来定位对应点。

**现有痛点**：局部2D相关性本质上是"一对多"匹配——一个查询点与一个区域内所有像素匹配。这在同质区域（如白墙）或重复纹理（如栅栏）中会产生严重的匹配歧义：相似度图上出现多个响应峰值，无法确定哪个才是正确对应。现有方法试图通过时间上下文（MLP-Mixer、1D卷积、Transformer）来缓解歧义，但在严重遮挡或复杂场景中，空间上的匹配歧义仍然是性能瓶颈。

**核心矛盾**：稠密对应方法（如光流估计中的RAFT）使用全局4D相关性体（所有像素对的相似度），通过双向匹配一致性和匹配平滑性可以有效消歧，但计算复杂度为$O(H^2W^2)$，在高分辨率下不可行。如何将4D相关性的消歧优势引入点跟踪，同时保持计算效率？

**本文目标** （1）如何利用4D相关性的双向匹配和平滑性先验来消除点跟踪中的匹配歧义？（2）如何高效地处理高维的4D相关性体？（3）如何设计可处理任意长度视频的时间建模架构？

**切入角度**：作者观察到不需要全局4D相关性——只需在查询点和预测位置的局部邻域内计算all-pair相关性，就能获得足够的匹配消歧信息。这种"局部4D相关性"将问题从不可行的全局计算简化为局部计算，复杂度大幅降低。

**核心 idea**：将点跟踪从点到区域的2D相关性提升为区域到区域的局部4D相关性，利用all-pair匹配的双向一致性和平滑性先验消除歧义，同时保持高效计算。

## 方法详解

### 整体框架
LocoTrack采用两阶段架构：**Track初始化阶段**使用全局2D相关性图确定每帧的初始对应位置；**Track精化阶段**通过迭代方式使用局部4D相关性体和Transformer逐步精化轨迹。输入为视频$\mathcal{V}$和查询点$q=(x_q, y_q, t_q)$，输出为所有帧上的轨迹$\mathcal{T}$和遮挡概率$\mathcal{O}$。特征提取使用带实例归一化的ResNet18，产出多尺度金字塔特征图。

### 关键设计

1. **局部4D相关性体（Local 4D Correlation Volume）**:

    - 功能：在查询点和目标位置的局部邻域内建立all-pair密集对应关系，提供双向匹配和平滑性先验用于消歧
    - 核心思路：在精化阶段的第$k$次迭代，围绕当前预测位置$\mathcal{T}^k_t$取半径$r_p$的目标区域，围绕查询点取半径$r_q$的查询区域，计算两个区域之间所有像素对的余弦相似度，形成$L_t \in \mathbb{R}^{h_p \times w_p \times h_q \times w_q}$的4D张量。其中$h_p = w_p = 2r_p+1$, $h_q = w_q = 2r_q+1$（实验中$r_p = r_q = 3$，即$7 \times 7 \times 7 \times 7$的4D体）。相比2D相关性只有"查询点→目标区域"的单向信息，4D相关性提供了双向的all-pair匹配信息
    - 设计动机：2D相关性在遇到重复纹理时会产生多个响应峰值无法消歧。4D相关性提供两个关键先验：（1）**双向一致性**——如果A对应B，则B也应对应A；（2）**匹配平滑性**——相邻点的对应关系应空间连续。这两个先验在稠密对应文献中已被证明能有效消歧

2. **轻量级4D相关性编码器（Correlation Encoder）**:

    - 功能：将高维4D相关性体压缩为紧凑的特征嵌入，同时保留消歧信息
    - 核心思路：直接处理4D张量的计算和参数开销过大。编码器将4D相关性分解为两个对称分支处理：一个分支将查询维度作为空间维度、目标维度展平为通道维度；另一个分支反过来。每个分支使用堆叠的步进2D卷积+组归一化+ReLU逐步降低空间维度，最终全局平均池化得到紧凑向量。两个分支的输出拼接形成相关性嵌入$E_t^k$。这种分解策略使得只需2D卷积就能有效处理4D数据
    - 设计动机：4D张量的直接处理需要4D卷积，参数量和计算量巨大。受启发于稠密对应文献中Cost Aggregation的思路，通过维度分解将4D处理拆解为两组2D处理，大幅降低复杂度的同时保留了双向匹配信息

3. **长度可泛化的Transformer（Length-Generalizable Transformer）**:

    - 功能：整合时间上下文信息，使模型能处理任意长度的视频而无需滑窗推理
    - 核心思路：堆叠3层Transformer对相关性嵌入序列做自注意力。关键创新在位置编码：不使用正弦位置编码（会在序列长度与训练时不同步时性能退化），而是使用相对位置偏置（ALiBi风格）。为了让偏置能区分左右方向（token A在B之前还是之后），将注意力头分为两组——一组只编码左侧token的相对位置，另一组只编码右侧。偏置函数$b(t_1, t_2; h) = -s_h |t_1 - t_2|$，其中缩放因子$s_h$按注意力头不同设置
    - 设计动机：MLP-Mixer无法处理变长序列需要滑窗推理，1D卷积需要大量堆叠才能获得足够感受野。Transformer仅用单层就有全局感受野，加上相对位置偏置后可泛化到训练时未见的序列长度，完全消除了滑窗推理的需求和由此带来的计算冗余

### 损失函数 / 训练策略
使用Huber损失监督轨迹位置的精度，交叉熵损失监督遮挡分类。多个精化迭代的预测都参与损失计算，采用指数衰减权重（后续迭代权重更高）。在Panning MOVi-E合成数据集上训练，先训练初始化阶段100K步，再训练精化阶段300K步。使用AdamW优化器（lr=1e-3）和余弦退化调度。

## 实验关键数据

### 主实验（Strided Query Mode, 256×256）

| 数据集 | 指标(AJ↑) | LocoTrack-S | LocoTrack-B | TAPIR | CoTracker | 提升 |
|--------|----------|-------------|-------------|-------|-----------|------|
| TAP-Vid-DAVIS | AJ | 66.9 | 67.8 | 61.3 | 65.9* | +1.9 vs CoTracker |
| TAP-Vid-Kinetics | AJ | 59.6 | 59.5 | 57.2 | - | +2.3 vs TAPIR |
| TAP-Vid-RGB-Stacking | AJ | 77.4 | 77.1 | 62.7 | - | +14.4 vs TAPIR |

*CoTracker在384×512分辨率下的结果

| 模型 | 吞吐量(points/s)↑ | 参数量(M) | 每点FLOPs(G) |
|------|-------------------|----------|-------------|
| LocoTrack-S | 7,244 | 8.2 | 1.08 |
| LocoTrack-B | 4,359 | 11.5 | 2.10 |
| TAPIR | 2,097 | 29.3 | 5.12 |
| CoTracker | 1,147 | 45.5 | 4.65 |

### 消融实验

| 配置 | AJ↑ | δ_avg↑ | OA↑ | 说明 |
|------|-----|--------|-----|------|
| 2D相关性（无邻域） | 65.0 | 77.2 | 89.0 | 基线2D方法 |
| 随机采样邻域 | 65.7 | 77.8 | 88.9 | 随机点不如密集网格 |
| 水平线邻域 | 66.5 | 78.4 | 89.4 | 1D邻域有限提升 |
| 规则网格(r=2) | 67.2 | 79.1 | 89.5 | 较小4D区域 |
| 规则网格(r=3, Ours) | 67.8 | 79.6 | 89.9 | 最优配置 |
| 正弦位置编码 | 61.9 | 73.9 | 83.5 | 变长泛化差 |
| 相对位置偏置(Ours) | 67.8 | 79.6 | 89.9 | 支持任意长度 |

### 关键发现
- **4D vs 2D相关性**：从2D升级到4D带来+2.8 AJ提升，证明all-pair匹配的消歧优势
- **密集采样的重要性**：随机和线形邻域不如规则网格，说明密集且空间连续的all-pair匹配才能充分利用平滑性先验
- **位置编码的巨大影响**：正弦编码在变长推理时AJ降到61.9，相对位置偏置保持67.8，差距5.9，说明长度泛化是关键
- **单次迭代即超TAPIR**：LocoTrack仅用1次精化迭代就超过TAPIR（4次迭代），且速度快9倍
- **效率优势显著**：LocoTrack-S参数量仅8.2M（CoTracker的1/5.5），吞吐量7,244 points/s（CoTracker的6.3倍）

## 亮点与洞察
- **从2D到4D的范式转变**：将点跟踪从"点到区域"的2D匹配提升为"区域到区域"的4D匹配，这不仅是维度的增加，更引入了双向一致性和匹配平滑性两个强先验，这个思路可以推广到其他需要消歧的匹配任务
- **维度分解编码器**：将4D处理拆解为两组对称的2D处理，既优雅又高效，后续工作可以借鉴这种分解思路处理其他高维张量
- **长度泛化的实用价值**：通过相对位置偏置消除滑窗推理需求，使模型能直接处理任意长度视频，这在实际应用中非常重要。左右分组的设计解决了方向感知问题

## 局限与展望
- 在高分辨率（384×512）时某些数据集上性能反而下降，作者归因于局部相关性的有效感受野在高分辨率下变小——可以考虑自适应调整局部区域大小
- 局部区域大小$r_p = r_q = 3$是固定的，面对大位移场景可能不够覆盖正确对应；可以设计多尺度的4D相关性
- 仅用合成数据训练，真实世界数据上的泛化可能有限，加入真实数据微调可能进一步提升
- 遮挡预测的准确率（OA）提升有限（+0.9），4D相关性对遮挡判断的贡献不如对位置精度的贡献大

## 相关工作与启发
- **vs TAPIR**: TAPIR使用2D相关性+1D卷积做时间建模，需要多次迭代才能收敛；LocoTrack用4D相关性提供更强的空间先验，1次迭代即可超越TAPIR
- **vs CoTracker**: CoTracker通过跟踪额外的支撑点来利用空间上下文，引入显著计算开销（参数量45.5M vs 8.2M）；LocoTrack通过4D相关性在不增加额外跟踪点的情况下获得更强的空间上下文
- **vs RAFT（光流）**: RAFT使用全局4D相关性做光流估计；LocoTrack将这一思路引入点跟踪，通过局部化使其在长视频中可行
- 4D相关性还在少样本分割（HSNet）、视频目标分割（XMem）等任务中展现优势，值得在更多视觉任务中探索

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将稠密对应中的4D相关性成功引入点跟踪，局部化处理和分解编码器设计都很巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 在所有TAP-Vid基准上全面评测，消融实验覆盖了核心设计的每个组件
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，从2D/4D的对比到局部化的动机，再到编码器和Transformer的设计，层层推进
- 价值: ⭐⭐⭐⭐⭐ 在精度和效率上全面超越SOTA，实用性强，对后续点跟踪研究有深远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Matching Every Pair to Track Every Point: PairFormer for All-Pairs Tracking and Video Trajectory Fields](../../CVPR2026/video_understanding/matching_every_pair_to_track_every_point_pairformer_for_all-pairs_tracking_and_v.md)
- [\[ECCV 2024\] Self-Supervised Any-Point Tracking by Contrastive Random Walks](self-supervised_any-point_tracking_by_contrastive_random_walks.md)
- [\[CVPR 2026\] Generative Point Tracking and Forecasting](../../CVPR2026/video_understanding/generative_point_tracking_and_forecasting.md)
- [\[ECCV 2024\] DINO-Tracker: Taming DINO for Self-Supervised Point Tracking in a Single Video](dino-tracker_taming_dino_for_self-supervised_point_tracking_in_a_single_video.md)
- [\[ECCV 2024\] Exploring the Feature Extraction and Relation Modeling For Light-Weight Transformer Tracking](exploring_the_feature_extraction_and_relation_modeling_for_light-weight_transfor.md)

</div>

<!-- RELATED:END -->
