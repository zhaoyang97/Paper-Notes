---
title: >-
  [论文解读] Model Poisoning Attacks to Federated Learning via Multi-Round Consistency
description: >-
  [CVPR 2025][优化/理论][联邦学习] 发现现有联邦学习模型投毒攻击因跨轮次方向不一致导致效果自相抵消，提出 PoisonedFL 通过固定随机方向向量 + 动态幅度调节 + 假设检验机制实现多轮一致性攻击，在无需任何真实客户端信息的前提下击穿 8 种 SOTA 防御。 领域现状： 联邦学习（FL）允许多客户端协作…
tags:
  - "CVPR 2025"
  - "优化/理论"
  - "联邦学习"
  - "model poisoning"
  - "multi-round consistency"
  - "untargeted attack"
  - "Byzantine robustness"
---

# Model Poisoning Attacks to Federated Learning via Multi-Round Consistency

**会议**: CVPR 2025  
**arXiv**: [2404.15611](https://arxiv.org/abs/2404.15611)  
**代码**: [xyq7/PoisonedFL](https://github.com/xyq7/PoisonedFL/)  
**领域**: 优化  
**关键词**: federated learning, model poisoning, multi-round consistency, untargeted attack, Byzantine robustness

## 一句话总结

发现现有联邦学习模型投毒攻击因跨轮次方向不一致导致效果自相抵消，提出 PoisonedFL 通过固定随机方向向量 + 动态幅度调节 + 假设检验机制实现多轮一致性攻击，在无需任何真实客户端信息的前提下击穿 8 种 SOTA 防御。

## 研究背景与动机

**领域现状**: 联邦学习（FL）允许多客户端协作训练而不共享数据，但其分布式特性使其天然容易受到模型投毒攻击（恶意客户端发送精心构造的梯度更新来破坏全局模型）。

**核心问题**: 现有模型投毒攻击存在两大局限：
1. **效果次优**: 现有攻击（如 LIE、Fang、Min-Max）仅保证单轮内恶意更新一致，但跨轮之间参数的攻击方向（+1/-1）频繁翻转（flipping rate 高），导致多轮后效果自相抵消
2. **假设过强**: 多数攻击需要获取真实客户端的模型更新或本地数据，要求大规模入侵真实设备

**关键洞察**: 论文观察到，当一个模型沿随机方向被大幅移动时，精度会严重退化。因此如果能使多轮聚合更新沿同一随机方向累积，最终模型将被推向随机猜测水平。而现有攻击因方向不一致，聚合后的累积效果被大幅削弱。

## 方法详解

### 整体框架

PoisonedFL 的核心思想: 在所有假客户端上发送方向一致的恶意更新 $\mathbf{g}_i^t = \mathbf{k}^t \odot \mathbf{s}$，其中 $\mathbf{s}$ 是一开始随机选定并固定不变的方向向量（每维为 +1 或 -1），$\mathbf{k}^t$ 是动态调整的非负幅度向量。方向一致保证多轮累积而非抵消，幅度动态调整保证不被防御过滤。

### 关键设计 1：多轮一致性优化目标

- **全轮优化**: 最大化总聚合更新 $\|\sum_{t=1}^T \mathbf{g}^t\|$，约束 $\text{sign}(\sum_t \mathbf{g}^t) = \mathbf{s}$
- **转化为逐轮**: 每轮约束 $\text{sign}(\mathbf{g}^t) = \mathbf{s}$，使各轮聚合更新方向相同，幅度自然累加
- **核心区别**: 现有攻击的聚合更新方向在各轮间反复翻转导致抵消；PoisonedFL 保持方向一致使效果逐轮叠加

### 关键设计 2：动态幅度调整

幅度向量 $\mathbf{k}^t = \lambda^t \cdot \mathbf{v}^t$ 分解为两个部分：

- **单位幅度向量 $\mathbf{v}^t$**: 通过上一轮的全局模型差 $\mathbf{g}^{t-1}$ 减去归一化后的恶意更新来估计真实客户端的模型更新分布，使恶意更新维度分布与真实客户端相似，避免被异常检测过滤
- **缩放因子 $\lambda^t = c^t \cdot \|\mathbf{w}^{t-1} - \mathbf{w}^{t-2}\|$**: 与上轮聚合更新幅度成正比，避免幅度过大/过小

### 关键设计 3：基于假设检验的自适应 $c^t$ 调节

- **零假设 $H_0$**: 攻击在过去 $e$ 轮未成功（方向 $\mathbf{s}$ 未体现在聚合更新中）
- **备择假设 $H_1$**: 攻击成功
- **检验方法**: 计算 $\mathbf{w}^{t-1} - \mathbf{w}^{t-e}$ 中与 $\mathbf{s}$ 方向一致的维度数 $X$，在 $H_0$ 下 $X \sim Bin(d, 0.5)$，若 $p > 0.01$ 则拒绝 $H_1$，将 $c^t = \beta \cdot c^{t-1}$（$\beta < 1$）降低幅度以提高隐蔽性
- **效果**: 无需知道服务器部署了哪种防御，自适应调整攻击强度

### 损失函数

本文是攻击方法，无传统训练损失。核心是逐轮求解优化问题：
$$\max_{\mathbf{g}_i^t} \|\mathbf{g}^t\|, \quad \text{s.t.} \; \text{sign}(\mathbf{g}^t) = \mathbf{s}$$

通过上述三个组件的近似求解实现。

## 实验关键数据

### 主实验表

PoisonedFL vs 7 种攻击 × 9 种防御 × 5 数据集，测试错误率（%）越高攻击越成功（Table 1 部分）:

| 防御 \ 攻击 | No Attack | Fang | LIE | Min-Max | MPAF | **PoisonedFL** |
|---|---|---|---|---|---|---|
| **MNIST-FedAvg** | 2.11 | 13.66 | 2.28 | 97.89 | 90.04 | **90.02** |
| **MNIST-Multi-Krum** | 2.13 | 5.98 | 2.34 | 6.23 | 2.80 | **75.28** |
| **MNIST-FLTrust** | 3.43 | 4.00 | 3.41 | 12.56 | 3.43 | **88.65** |
| **MNIST-FLAME** | 2.86 | 2.66 | 2.61 | 2.60 | 2.72 | **88.59** |
| **MNIST-FLCert** | 3.34 | 4.61 | 2.83 | 4.57 | 6.46 | **88.06** |
| **FashionMNIST-FLTrust** | 16.73 | 17.52 | 12.50 | 21.32 | 16.83 | **88.41** |
| **Purchase-Multi-Krum** | 11.09 | 16.78 | 11.87 | 14.56 | 12.11 | **73.59** |

- PoisonedFL 在所有数据集 × 所有防御组合上均实现最高或接近最高的测试错误率
- 尤其在最强防御（FLTrust、FLAME、FLCert）上，其它攻击几乎无效，PoisonedFL 仍能达到 70-90% 错误率

### 消融表

- 去除多轮一致性（即随机方向每轮变化）：攻击效果大幅下降
- 均匀幅度（$\mathbf{v}^t$ 各维相同）vs 动态 $\mathbf{v}^t$ vs 完整方法：动态估计显著优于均匀
- 去除假设检验（固定 $c^t$）：对部分防御效果下降

### 关键发现

1. **FL 系统远比之前认为的脆弱**: 即使部署 SOTA 防御，PoisonedFL（无任何关于真实客户端的信息）仍可使模型退化到随机猜测
2. 现有攻击失败的根本原因不是单轮攻击强度不够，而是多轮间方向不一致导致累积效果被抵消
3. 即使设计针对性防御（检测方向一致性），PoisonedFL 仍可通过微调 $\mathbf{s}$ 来规避
4. FLDetector（检测多轮不一致）对 PoisonedFL 完全无效，因为 PoisonedFL 的恶意更新恰好是跨轮一致的

## 亮点与洞察

1. **洞察深刻**: 指出现有攻击"自相抵消"的根本缺陷，从全局视角审视多轮训练的累积效应，思路简洁但影响巨大
2. **最小假设**: 仅需注入假客户端，不需要知道防御方式、不需要真实客户端数据——这是最现实的威胁模型
3. **假设检验的巧妙运用**: 用统计方法自动判断攻击是否被过滤，无需对防御做任何假设
4. **方法极其简单**: 核心就是固定一个随机方向向量并保持跨轮一致，简单到令人发指却效果惊人

## 局限性

1. 主要在untargeted attack场景验证，targeted attack（后门攻击）的适用性未探讨
2. 评估以图像分类为主，NLP/推荐系统等领域的有效性待验证
3. 攻击者需要能注入假客户端，某些封闭FL系统可能不适用
4. 未考虑差分隐私等可能削弱攻击效果的额外防护措施

## 相关工作与启发

- **LIE/Fang/Min-Max/Min-Sum**: 需要真实客户端信息的攻击，本文证明即使有这些优势仍不如 PoisonedFL
- **MPAF**: 同样不需要真实客户端信息的攻击，但缺乏多轮一致性导致效果远逊
- **FLTrust/FLAME/FLCert**: 各种 SOTA 防御，均被 PoisonedFL 突破
- **启发**: 安全研究中"累积效应"的视角非常重要——单轮看起来温和的扰动，在方向一致时多轮累积可以产生灾难性影响；这提示防御方也需关注跨时间维度的一致性检测

## 评分

⭐⭐⭐⭐ — 洞察深刻（多轮自抵消问题），方法极简但效果摧毁性，在最弱假设下击穿所有SOTA防御，对FL安全社区敲响警钟。扣1星因为仅聚焦 untargeted attack，且实验场景偏学术（小模型+标准数据集）。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients](../../NeurIPS2025/optimization/efficient_federated_learning_against_byzantine_attacks_and_data_heterogeneity_vi.md)
- [\[CVPR 2025\] Federated Learning with Domain Shift Eraser](federated_learning_with_domain_shift_eraser.md)
- [\[NeurIPS 2025\] FedQS: Optimizing Gradient and Model Aggregation for Semi-Asynchronous Federated Learning](../../NeurIPS2025/optimization/fedqs_optimizing_gradient_and_model_aggregation_for_semi-asynchronous_federated_.md)
- [\[CVPR 2025\] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning](scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)
- [\[CVPR 2026\] DC-Merge: Improving Model Merging with Directional Consistency](../../CVPR2026/optimization/dc-merge_improving_model_merging_with_directional_consistency.md)

</div>

<!-- RELATED:END -->
