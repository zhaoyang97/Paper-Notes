---
title: >-
  [论文解读] Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding for Unsupervised Video Anomaly Detection
description: >-
  [ECCV 2024][视频理解][视频异常检测] 提出一个将加权单类分类 (wOCC) 与弱监督 (WS) 模型交替训练的无监督视频异常检测框架，通过软标签缓解训练波动、自适应阈值策略逐步优化分割阈值，无需任何人工标注即可实现接近弱监督方法的性能。 视频异常检测 (VAD) 主要有两大范式： - 单类分类 (OCC)：仅使…
tags:
  - "ECCV 2024"
  - "视频理解"
  - "视频异常检测"
  - "无监督学习"
  - "单类分类"
  - "弱监督学习"
  - "自适应阈值"
---

# Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding for Unsupervised Video Anomaly Detection

**会议**: ECCV 2024  
**arXiv**: [2401.13551](https://arxiv.org/abs/2401.13551)  
**代码**: [github.com/benedictstar/Joint-VAD](https://github.com/benedictstar/Joint-VAD)  
**领域**: 人体理解  
**关键词**: 视频异常检测, 无监督学习, 单类分类, 弱监督学习, 自适应阈值

## 一句话总结

提出一个将加权单类分类 (wOCC) 与弱监督 (WS) 模型交替训练的无监督视频异常检测框架，通过软标签缓解训练波动、自适应阈值策略逐步优化分割阈值，无需任何人工标注即可实现接近弱监督方法的性能。

## 研究背景与动机

视频异常检测 (VAD) 主要有两大范式：
- **单类分类 (OCC)**：仅使用正常数据训练，但需要人工筛除异常数据
- **弱监督 (WS)**：使用视频级标注训练，标注成本高且异常类别有界

两者都依赖人工标注，且无法覆盖所有异常类型。**无监督 VAD (UVAD)** 旨在完全不依赖标注，但现有方法（如 GCL）采用的自编码器和全连接分类器能力有限。

本文核心洞察：OCC 和 WS 方法各自快速发展，能否直接将最新的 OCC 和 WS 模型组合成一个 UVAD 框架？挑战在于：(1) 伪标签的随机性导致训练波动；(2) 需要阈值将伪标签分为正常/异常，引入人工超参数依赖。

## 方法详解

### 整体框架

框架包含可重复运行的 **wOCC-WS 交替训练模块**：
1. wOCC 模型生成异常分数 → 划分伪标签给 WS 模型
2. WS 模型生成异常分数 → 作为软标签给 wOCC 模型
3. 模块收敛后，用自适应阈值更新机制调整阈值，启动下一轮模块
4. 重复直至停止条件满足

### 关键设计

1. **加权单类分类 wOCC**：解决训练波动问题的核心。传统 OCC 用硬标签 {0,1} 区分正常/异常数据来训练，硬标签可能突变（0→1 或 1→0），导致训练不稳定。wOCC 引入软标签 $w_{X_i} \in [0,1]$ 作为加权：
    $\mathcal{L}_{wocc} = -(1 - w_{X_i}) \log(p_Z(f_{STG\text{-}NF}(X_i)))$
   权重越小表示越正常，对应样本在似然建模中权重越大。软标签变化连续（如 0.7→0.6 而不是 1→0），显著减少训练波动。以 STG-NF 为例，将原始仅在正常数据上的负对数似然扩展为对全部数据的加权版本。

2. **伪标签交互机制**：wOCC → WS 方向使用排序+阈值方式生成硬标签：将 snippet 按异常分数降序排列，排名前 $T_{ws}$ 个标记为异常。WS → wOCC 方向直接使用 WS 模型的异常分数作为软标签 $w_{X_i} = \hat{x}_i$，无需额外阈值。

3. **自适应阈值策略**：解决阈值依赖问题。核心保证阈值单调递减 $T_{ws}^1 \geq T_{ws}^2 \geq \cdots$：

    - 初始阈值 $T_{ws}^1 = R\% \times N$（如 R=30），设为足够大的值
    - 每轮模块产生多个 wOCC 模型，每个识别出 R% 高异常分数的 snippet 集合 $A_j$
    - 下一轮阈值通过求交集计数：$T_{ws}^{i+1} = \text{Num}(A_1 \cap A_2 \cap \cdots \cap A_{M_i})$
    - 交集操作确保阈值单调递减：早期模型少则交集大，随着更多模型参与共识收紧
    - **停止准则**：当阈值变化率降至初始变化率的 Q%（默认10%）以下时停止

4. **初始化策略**：首个模块用 Beta 分布采样 $w_X \sim \text{Beta}(1, 5)$ 随机初始化软标签，确保大部分权重接近0（正常），少量接近1（异常），符合数据先验。后续模块用前一模块最后的 WS 模型输出初始化。

### 损失函数 / 训练策略

- **wOCC 损失**：加权负对数似然（以 STG-NF 为例）
- **WS 损失**：Top-k MIL ranking loss + BCE 分类损失（以 RTFM 为例）
- **交替训练**：每次训练一个模型一个 epoch 后切换，整个训练约 30 个 epoch，耗时约 2.5 小时
- 每个新模块重新初始化模型参数（不继承），避免累积错误

## 实验关键数据

### 主实验

**ShanghaiTech 数据集 (AUC %)**:

| 方法 | 监督类型 | 特征 | AUC |
|------|---------|------|-----|
| GCL | 无监督 | I3D | 76.14 |
| STG-NF (全数据) | 无监督 | - | 80.29 |
| **OurwOCC** | **无监督** | - | **82.57** |
| **OurWS** | **无监督** | I3D | **88.18** |
| STG-NF | OCC | - | 85.90 |
| RTFM | WS | I3D | 96.10 |

**UBnormal 数据集 (AUC %)**:

| 方法 | AUC |
|------|-----|
| STG-NF (全数据) | 70.48 |
| **OurwOCC** | **74.76** |
| **OurWS** | 63.10 |

### 消融实验

**wOCC vs OCC + 自适应阈值消融 (ShanghaiTech)**:

| Weighted OCC | 自适应阈值 | RTFM AUC | STG-NF AUC |
|:---:|:---:|:---:|:---:|
| ✗ | ✗ | 82.06 | 80.52 |
| ✓ | ✗ | 83.48 | 81.78 |
| ✗ | ✓ | 85.86 | 81.94 |
| **✓** | **✓** | **88.18** | **82.57** |

**不同 OCC 模型组合 (WS=RTFM)**:

| OCC 模型 | OurwOCC | OurWS(RTFM) |
|---------|---------|-------------|
| AE | 70.99 | 78.90 |
| Jigsaw | 81.23 | 85.35 |
| **STG-NF** | **82.57** | **88.18** |

### 关键发现

- wOCC 比直接使用 OCC 显著减少训练波动，AUC 收敛更高
- 自适应阈值经过约6个模块后收敛，不同初始 R% 值最终收敛到相似阈值
- 异构组合 (OCC+WS) 优于同构组合 (OCC+OCC 或 WS+WS)
- 退化为有监督时，wOCC 优于原始 OCC（86.37 vs 85.90），说明加权机制本身有价值
- 整体训练时间仅约 2.5 小时，与原始单模型训练时间相当

## 亮点与洞察

- **框架的灵活性**：可即插即用最新的 OCC 和 WS 模型，随领域进步而升级
- **软标签的稳定性洞察**：发现硬标签突变是训练波动根源，用软标签自然解决，简洁有效
- **自适应阈值的数学保证**：交集操作保证单调递减，不依赖特定异常分数范围
- **收敛性分析**：给出了为什么从随机初始化开始 wOCC 仍能学到有意义表示的分析——正常数据远多于异常数据这一先验足够启动学习

## 局限与展望

- R% 参数虽然不敏感但仍需根据数据集粗略设置
- 在 UBnormal 上 WS 模型的 AUC (63.10%) 低于 wOCC (74.76%)，对异常比例高的数据适应性有限
- 未探索 Transformer 架构的 OCC/WS 模型组合
- 停止准则基于阈值变化率而非直接的性能指标，极端情况下可能不是最优停止点
- 仅在姿态类异常检测上验证，未测试外观类异常场景

## 相关工作与启发

- GCL 是之前唯一的 UVAD 方法，但其网络架构能力有限
- STG-NF 作为 OCC 模型通过正则化流建模正常姿态分布，启发了加权似然的设计
- RTFM 的 Top-k MIL 机制为 WS 端提供了强基线
- 自训练 (self-training) 和伪标签在半监督学习中已广泛使用，本文将其推广到完全无监督场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ — wOCC-WS 交替框架 + 自适应阈值组合有新意，软标签思路简单有效
- **实验充分度**: ⭐⭐⭐⭐ — 多种 OCC/WS 模型组合、消融详尽，但数据集仅两个
- **写作质量**: ⭐⭐⭐⭐ — 逻辑清晰，公式推导完整，图示直观
- **价值**: ⭐⭐⭐⭐ — 提出了可持续升级的 UVAD 框架，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Learning Anomalies with Normality Prior for Unsupervised Video Anomaly Detection](learning_anomalies_with_normality_prior_for_unsupervised_video_anomaly_detection.md)
- [\[CVPR 2026\] Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](../../CVPR2026/video_understanding/weakly_supervised_video_anomaly_detection_with_anomaly-connected_components_and_.md)
- [\[ECCV 2024\] ActionSwitch: Class-agnostic Detection of Simultaneous Actions in Streaming Videos](actionswitch_class-agnostic_detection_of_simultaneous_actions_in_streaming_video.md)
- [\[CVPR 2026\] The Road Less Seen: Segment Exploration for Weakly Supervised Video Anomaly Detection](../../CVPR2026/video_understanding/the_road_less_seen_segment_exploration_for_weakly_supervised_video_anomaly_detec.md)
- [\[AAAI 2026\] RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection](../../AAAI2026/video_understanding/refinevad_semantic-guided_feature_recalibration_for_weakly_supervised_video_anom.md)

</div>

<!-- RELATED:END -->
