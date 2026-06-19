---
title: >-
  [论文解读] Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance
description: >-
  [ECCV 2024][人群计数] 提出辅助点引导 (APG) 策略和隐式特征插值 (IFI) 模块，通过在真值点附近显式生成辅助正负样本来稳定 point-based 人群计数方法中 proposal-target 匹配过程的不稳定性，在多个数据集上取得 SOTA。 人群计数与定位方法大致分为三类：基于密度图 (map-b…
tags:
  - "ECCV 2024"
  - "人群计数"
  - "人群定位"
  - "点监督"
  - "匹配不稳定性"
  - "隐式特征插值"
---

# Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance

**会议**: ECCV 2024  
**arXiv**: [2405.10589](https://arxiv.org/abs/2405.10589)  
**代码**: 论文中提及将公开  
**领域**: 人体理解  
**关键词**: 人群计数, 人群定位, 点监督, 匹配不稳定性, 隐式特征插值

## 一句话总结

提出辅助点引导 (APG) 策略和隐式特征插值 (IFI) 模块，通过在真值点附近显式生成辅助正负样本来稳定 point-based 人群计数方法中 proposal-target 匹配过程的不稳定性，在多个数据集上取得 SOTA。

## 研究背景与动机

人群计数与定位方法大致分为三类：基于密度图 (map-based)、基于检测 (detection-based) 和基于点 (point-based)。基于点的方法（如 P2PNet、CLTR、PET）因端到端可训练、无需复杂后处理而受到关注，但存在一个关键问题：**proposal-target 匹配不稳定**。

具体而言，在每个训练 epoch 中，大量目标点会与不同的 point proposal 匹配，导致每个 proposal 的学习目标模糊不清。作者定义了 Instability Rate (IR) 来衡量这种不稳定性，发现现有方法（如 Matcher）在训练过程中 IR 居高不下。这种不稳定性的根源在于缺乏有效的学习策略来指导网络一致地选择最合适的 proposal，最终导致区域性低估或高估。

## 方法详解

### 整体框架

APGCC 采用 VGG-16 作为骨干网络，提取 conv3 和 conv4 层特征，经 ASPP 模块增强多尺度表示后，通过隐式特征插值 (IFI) 获取任意位置的特征表示，最后送入预测头输出置信度和偏移量。总损失为：

$$\mathcal{L}_{overall} = \mathcal{L}_{point} + \lambda_5 \mathcal{L}_{APG}$$

### 关键设计

1. **Auxiliary Point Guidance (APG)**：核心创新，解决匹配不稳定问题。对于每个真值点 $(x,y)$，在其附近生成辅助正样本 $A_{pos}$ 和辅助负样本 $A_{neg}$：

    - **辅助正样本**：在真值坐标附近 $[-n_{pos}, n_{pos}]$ 范围内随机偏移生成，训练目标为置信度接近1、预测偏移能回到真值位置
    - **辅助负样本**：在 $[-n_{neg}, -n_{pos}] \cup [n_{pos}, n_{neg}]$ 范围内生成，训练目标为置信度接近0且偏移量趋零，防止负样本通过偏移"越界"接近真值
    - 设计动机：通过显式告知网络"靠近真值的 proposal 应为正、远离的应为负"，引导匹配过程趋于稳定，使同一真值点在不同 epoch 被一致地匹配到相同 proposal

2. **Implicit Feature Interpolation (IFI)**：由于辅助点位于任意位置而非网格点上，传统双线性插值不够灵活。IFI 利用隐式函数进行连续特征表示：

    - 对任意位置 $(x,y)$，找到最近4个 latent feature $Z_i^*$，计算距离 $\delta_i^*$
    - 将特征和距离拼接后送入 MLP $f_\theta$，并引入位置编码 $\phi(\cdot)$ 增强高频信息捕获
    - 最终特征通过面积加权求和：$F_{proposal}(x,y) = \sum_{i=1}^{4} \frac{S_i}{S} f_\theta(Z_i^*, \delta_i^*, \phi(\delta_i^*))$
    - 相比传统上采样，IFI 用更少参数实现更精准的特征学习

### 损失函数 / 训练策略

- **点损失** $\mathcal{L}_{point}$：包含 Cross Entropy 分类损失 $\mathcal{L}_{cls}$ 和 Euclidean 回归损失 $\mathcal{L}_{loc}$
- **APG 正样本损失** $\mathcal{L}_{APG}^{pos}$：最大化辅助正样本的置信度 + 最小化其预测位置与真值的距离
- **APG 负样本损失** $\mathcal{L}_{APG}^{neg}$：最小化辅助负样本的置信度 + 约束其偏移量趋零
- **训练细节**：Adam 优化器，学习率 $10^{-4}$（骨干 $10^{-5}$），batch size 8，stride $s=8$，$(k_{pos}, k_{neg}) = (2, 2)$，随机范围 $(n_{pos}, n_{neg}) = (2, 8)$

## 实验关键数据

### 主实验

| 数据集 | 指标 | APGCC | PET (前SOTA) | 提升 |
|--------|------|-------|-------------|------|
| SHHA | MAE/MSE | **48.8/76.7** | 49.3/78.7 | -0.5/-2.0 |
| SHHB | MAE/MSE | **5.6/8.7** | 6.1/9.6 | -0.5/-0.9 |
| UCF_CC_50 | MAE/MSE | **154.8/205.5** | 159.9/223.7 | -5.1/-18.2 |
| UCF-QNRF | MSE | **136.6** | 144.3 | -7.7 |
| JHU-Crowd++ | MAE/MSE | **54.3/225.9** | 58.5/238.0 | -4.2/-12.1 |
| NWPU | MAE/MSE | **71.7/284.4** | 74.4/328.5 | -2.7/-44.1 |

**定位性能 (NWPU)**:

| 方法 | F1(σ_l) | P(σ_l) | F1(σ_s) |
|------|---------|--------|---------|
| PET | 74.2% | 75.2% | 67.5% |
| **APGCC** | **76.4%** | **79.2%** | **68.9%** |

**定位性能 (SHHA)**:

| 方法 | F1(σ=4) | F1(σ=8) |
|------|---------|---------|
| CLTR | 43.2% | 74.2% |
| **APGCC** | **48.7%** | **78.4%** |

### 消融实验

| 配置 | MAE | 说明 |
|------|-----|------|
| Matcher only | 基线 | 仅使用匈牙利匹配 |
| Nearest Point | 计数严重低估 | 多真值映射到同一 proposal |
| APG only | 次优 | 推理时无辅助点参考导致过度依赖 |
| **Matcher + APG** | **48.8** | 兼顾分配与引导 |

**IFI 消融**:

| 配置 | MAE | 说明 |
|------|-----|------|
| 最近邻 (无MLP) | 较差 | 特征上下文不足 |
| 双线性插值 (无MLP) | 中等 | 缺少距离连续变换 |
| IFI 单参考点 | 次优 | 参考点不足 |
| IFI 无位置编码 | 次优 | 丢失高频信息 |
| **IFI (完整)** | **最优** | 所有组件协同 |

### 关键发现

- APG 显著降低了匹配不稳定率 (IR)，且正负样本配比 (2,2) 已足够
- APG 仅在训练时使用，不增加推理开销
- IFI 相比传统上采样用更少参数实现了更好性能
- 在严格定位阈值 (σ=4) 下，APGCC 的 F1 比 CLTR 提升 5.5%

## 亮点与洞察

- **问题定义精准**：通过 Instability Rate 量化了 point-based 方法长期被忽视的匹配不稳定问题
- **辅助点思路巧妙**：不改变匹配机制本身，而是通过额外监督信号"教"网络如何做出正确选择
- **训练/推理解耦**：APG 仅训练时起作用，推理时无额外开销，工程友好
- IFI 提供了一种通用的任意位置特征提取方案，可迁移至其他需要非网格位置特征的任务

## 局限与展望

- 辅助点的随机范围 $(n_{pos}, n_{neg})$ 需要根据数据集调参
- stride 固定为8，无自适应多尺度 stride 机制
- 未探索将 APG 策略应用于 DETR 系列检测器等 set prediction 框架
- 隐式特征插值的 MLP 参数量虽小但仍增加了一定计算

## 相关工作与启发

- P2PNet 最早提出 point-based 人群计数框架，但未解决匹配不稳定问题
- DETR 中的 Hungarian matching 同样存在早期不稳定性，APG 的思路或可借鉴
- 隐式函数 (LIIF, NeRF 等) 在连续表示上的成功启发了 IFI 的设计

## 评分

- **新颖性**: ⭐⭐⭐⭐ — APG 策略新颖且直觉清晰，对匹配不稳定性的分析深入
- **实验充分度**: ⭐⭐⭐⭐⭐ — 6个数据集全面评估，计数+定位双重任务，消融详尽
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，问题动机阐述充分
- **价值**: ⭐⭐⭐⭐ — 对 point-based 方法有实质性改进，APG 思路可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Mahalanobis Distance-Based Multi-View Optimal Transport for Multi-View Crowd Localization](mahalanobis_distance-based_multi-view_optimal_transport_for_multi-view_crowd_loc.md)
- [\[NeurIPS 2025\] Fixed-Point RNNs: Interpolating from Diagonal to Dense](../../NeurIPS2025/others/fixed-point_rnns_interpolating_from_diagonal_to_dense.md)
- [\[ICCV 2025\] A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks](../../ICCV2025/others/a_linear_n-point_solver_for_structure_and_motion_from_asynchronous_tracks.md)
- [\[ICML 2025\] Fixed-Confidence Multiple Change Point Identification under Bandit Feedback](../../ICML2025/others/fixed-confidence_multiple_change_point_identification_under_bandit_feedback.md)
- [\[ACL 2025\] A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior](../../ACL2025/others/a_spatio-temporal_point_process_for_fine-grained_modeling_of_reading_behavior.md)

</div>

<!-- RELATED:END -->
