---
title: >-
  [论文解读] SPACE: SPike-Aware Consistency Enhancement for Test-Time Adaptation in Spiking Neural Networks
description: >-
  [NeurIPS 2025][SNN] 提出SPACE，首个专为脉冲神经网络(SNN)设计的无源单样本测试时自适应(TTA)方法，通过最大化增强样本间脉冲行为特征图的一致性，在多个数据集和架构上实现鲁棒适应。 - SNN作为生物合理的ANN替代，具有能效和时间处理优势，但对分布偏移高度敏感 - 实验证明：SNN在分布偏移下的…
tags:
  - "NeurIPS 2025"
  - "SNN"
  - "测试时自适应"
  - "脉冲一致性"
  - "分布偏移"
  - "单样本适应"
---

# SPACE: SPike-Aware Consistency Enhancement for Test-Time Adaptation in Spiking Neural Networks

**会议**: NeurIPS 2025  
**arXiv**: [2504.02298](https://arxiv.org/abs/2504.02298)  
**代码**: [GitHub](https://github.com/ethanxyluo/SPACE)  
**领域**: 脉冲神经网络, 测试时自适应, 领域鲁棒性  
**关键词**: SNN, 测试时自适应, 脉冲一致性, 分布偏移, 单样本适应

## 一句话总结
提出SPACE，首个专为脉冲神经网络(SNN)设计的无源单样本测试时自适应(TTA)方法，通过最大化增强样本间脉冲行为特征图的一致性，在多个数据集和架构上实现鲁棒适应。

## 研究背景与动机
- SNN作为生物合理的ANN替代，具有能效和时间处理优势，但对分布偏移高度敏感
- 实验证明：SNN在分布偏移下的精度损失显著大于同架构的ANN（CIFAR-10上SNN损失22.74% vs ANN损失13.28%）
- 现有ANN-TTA方法不适用于SNN：
    - MEMO：仅操作输出概率，无法控制精细的时间脉冲动态
    - SITA：依赖BN层统计更新，但许多SNN架构没有BN
    - SHOT/TAST：需要小批量目标数据或批级统计
- 需要一种直接利用脉冲动态的SNN特定TTA方法

## 方法详解

### 整体框架
四步流程：
1. 从单个测试样本生成增强批次
2. 通过模型获取脉冲计数局部特征图
3. 最大化增强样本间特征图相似度来适应模型
4. 用适应后的模型预测原始样本标签

### 关键设计

#### 脉冲感知特征图
- 选择特征提取器$E_{\theta_E}$中保留空间支持的最深层
- 对每个增强视图$\mathbf{x}_i$，在LIF神经元的二进制脉冲输出$\mathbf{O} \in \{0,1\}$上沿时间维度聚合
- 得到脉冲感知特征图$\mathbf{F}(\mathbf{x}_i) \in \mathbb{R}^{C \times D}$（C=通道数，D=空间维度H×W）

选择总脉冲计数的三个理由：
1. SNN通常运行数十到数千步，逐步匹配冗余且昂贵
2. 聚合使损失曲面平滑——即使脉冲在时间上抖动但计数匹配，损失不变
3. 分布偏移主要表现为中间层脉冲率和空间激活支持的变化

#### 特征图对齐
- 通道级局部向量通过softmax归一化为概率分布$\mathbf{P}_c \in \Delta^{D-1}$
- 两个增强视图的相似度：通道级内积平均值
$$\bar{\mathcal{S}}(i,j|\mathbf{x}) = \frac{1}{C}\sum_{c=1}^{C}\sum_{d=1}^{D}\mathbf{P}_{c,d}(\mathbf{x}_i)\mathbf{P}_{c,d}(\mathbf{x}_j)$$

### 损失函数 / 训练策略
$$\mathcal{L}(\theta_E;\mathbf{x}) = \sum_{1 \leq j < i \leq M}(1 - \bar{\mathcal{S}}(i,j|\mathbf{x}))$$

- 仅更新提取器参数$\theta_E$（分类器冻结）
- 单步SGD更新
- 增强策略：AugMix，批次大小32(CIFAR)/64(ImageNet)
- 每个测试样本仅更新一次

## 实验关键数据

### 主实验（CIFAR-10-C，最高损坏级别5，Top-1准确率%）

| 方法 | Gauss. | Shot | Impl. | Avg |
|-----|--------|------|-------|-----|
| No Adapt (VGG9) | 72.38 | 74.70 | 58.57 | 66.57 |
| SITA | 73.06 | 74.15 | 58.41 | 66.41 |
| MEMO | 77.73 | 79.50 | 65.74 | 69.20 |
| **SPACE** | **77.98** | **79.34** | **69.41** | **71.03** |

### 跨架构泛化（准确率提升，百分点）

| 数据集 | SNN-VGG9 | SNN-ResNet11 | Spike Transformer V3 | SNN-ConvLSTM |
|-------|----------|-------------|---------------------|-------------|
| CIFAR-10-C | +4.46 | +2.19 | +0.65 | +1.30 |
| CIFAR-100-C | +1.72 | +1.03 | +0.29 | 适用 |
| ImageNet系列 | - | - | +0.53~1.97 | - |

### SNN vs ANN分布偏移敏感性对比

| 架构 | 数据集 | ANN精度损失 | SNN精度损失 |
|-----|-------|-----------|-----------|
| VGG9 | CIFAR-10 | 13.28 | **22.74** |
| VGG11 | CIFAR-100 | 22.54 | **28.14** |
| VGG11 | Tiny-ImageNet | 41.48 | **43.80** |
| Transformer | ImageNet | 9.67 | **11.96** |

### 关键发现
- SPACE在4种SNN架构(CNN/Transformer/ConvLSTM)上一致性地提升性能
- 在Fog等低对比度腐蚀上提升最大（VGG9 CIFAR-10-C: 43.57→52.80）
- 在Contrast腐蚀上几乎无法帮助（22.54→23.85），因为极端对比度变化导致脉冲率崩溃
- 脉冲级别的对齐比输出级别的MEMO提供更稳定的适应信号
- 在DVS Gesture神经形态数据集上也有效

## 亮点与洞察
1. **首个SNN-TTA方法**：填补了SNN测试时适应的重要空白
2. **利用SNN固有特性**：直接在脉冲动态上操作，而非将ANN方法直接移植
3. **无BN依赖**：不要求模型包含批归一化层，适用于更广泛的SNN架构
4. **计算高效**：单步SGD+脉冲计数聚合避免逐时间步匹配
5. **理论动机**：信息瓶颈视角（抑制增强特异性变异）+ 流形去噪视角

## 局限与展望
- 对极端腐蚀(如Contrast)效果有限——脉冲率可能崩溃
- AugMix增强质量影响适应效果
- 仅在分类任务上评估，未扩展到检测/分割
- 核嵌入/MMD版本未带来显著收益但增加了开销
- 未来可探索与SNN特异性数据增强的结合

## 相关工作与启发
- MEMO/SITA作为ANN-TTA先驱，但在SNN上受限
- SNN-SFDA (Guo 2023)采用批处理/epoch设置，不适合逐样本在线适应
- 本文将TTA从ANN范式扩展到SNN范式的第一步

## 评分
- 新颖性：⭐⭐⭐⭐ （首个SNN-TTA方法，清晰的问题定义）
- 技术深度：⭐⭐⭐⭐ （脉冲动态对齐设计合理，理论动机扎实）
- 实验充分性：⭐⭐⭐⭐⭐ （4种架构×多数据集×详细消融）
- 写作质量：⭐⭐⭐⭐ （清晰全面）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Test-Time Adaptation by Causal Trimming](test-time_adaptation_by_causal_trimming.md)
- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](../../CVPR2026/others/neural_collapse_in_test-time_adaptation.md)
- [\[ICML 2025\] Ranked Entropy Minimization for Continual Test-Time Adaptation](../../ICML2025/others/ranked_entropy_minimization_for_continual_test-time_adaptation.md)
- [\[ECCV 2024\] MemBN: Robust Test-Time Adaptation via Batch Norm with Statistics Memory](../../ECCV2024/others/membn_robust_test-time_adaptation_via_batch_norm_with_statistics_memory.md)
- [\[ICML 2025\] Beyond Entropy: Region Confidence Proxy for Wild Test-Time Adaptation](../../ICML2025/others/beyond_entropy_region_confidence_proxy_for_wild_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
