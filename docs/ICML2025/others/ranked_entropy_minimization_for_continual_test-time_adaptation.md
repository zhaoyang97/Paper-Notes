---
title: >-
  [论文解读] Ranked Entropy Minimization for Continual Test-Time Adaptation
description: >-
  [ICML 2025][测试时自适应] 提出 Ranked Entropy Minimization (REM)，通过渐进式遮挡策略构建预测难度的显式排序结构，结合遮挡一致性损失和熵排序损失，解决了熵最小化方法在持续测试时自适应(CTTA)中的模型崩塌问题，同时保持了计算效率。 问题定义 测试时自适应(TTA)旨在推理阶段在…
tags:
  - "ICML 2025"
  - "测试时自适应"
  - "持续测试时自适应"
  - "熵最小化"
  - "模型崩塌"
  - "Transformer"
---

# Ranked Entropy Minimization for Continual Test-Time Adaptation

**会议**: ICML 2025  
**arXiv**: [2505.16441](https://arxiv.org/abs/2505.16441)  
**代码**: [https://github.com/pilsHan/rem](https://github.com/pilsHan/rem)  
**领域**: LLM评测  
**关键词**: 测试时自适应, 持续测试时自适应, 熵最小化, 模型崩塌, Vision Transformer

## 一句话总结

提出 Ranked Entropy Minimization (REM)，通过渐进式遮挡策略构建预测难度的显式排序结构，结合遮挡一致性损失和熵排序损失，解决了熵最小化方法在持续测试时自适应(CTTA)中的模型崩塌问题，同时保持了计算效率。

## 研究背景与动机

### 问题定义

测试时自适应(TTA)旨在推理阶段在线适应目标域的分布偏移。持续测试时自适应(CTTA)进一步要求模型在连续的域变化中进行顺序适应，且不重置模型参数，因此灾难性遗忘问题更加突出。

### 现有方法的局限

CTTA 的主流方法分为两大流派：

**熵最小化(EM)**：以 Tent 为代表，最小化预测熵实现自适应。计算效率高（仅更新 BN 层），但存在 **模型崩塌** 风险——模型收敛到对所有输入预测同一类别的平凡解。

**一致性正则化(CR)**：以 CoTTA 为代表，采用教师-学生框架保证稳定性，但需要额外模型和大量前向传播，计算代价过高。

本质上存在一个 **效率-稳定性权衡**：EM 高效但不稳定，CR 稳定但低效。

### 关键观察

作者发现熵最小化的梯度存在两种平凡解：(1) 均匀分布 $\hat{p}_{t,c} = 1/C$；(2) 完全确信预测 $\hat{p}_{t,c} \in \{0,1\}$。在实践中，模型倾向于收敛到对某单一类别的完全确信预测，导致模型崩塌（Tent 在 impulse noise 域适应后性能剧烈下降）。

### 核心直觉

如果将图像中的关键目标遮挡掉，模型的预测准确率会下降、预测熵会上升——这是一个符合直觉且可利用的显式排序关系。灵感来自芝诺的 "Achilles 与乌龟" 悖论：原始预测（乌龟）保持领先地位，遮挡预测（Achilles）努力追赶，从而防止熵的急剧下降导致崩塌。

## 方法详解

### 整体框架

REM 由三个核心组件构成：

1. **显式遮挡链 (Explicit Mask Chaining)**：利用 ViT 自注意力机制，按注意力分数从高到低渐进遮挡 patch，构建预测难度的排序结构
2. **遮挡一致性损失 (Masked Consistency Loss, MCL)**：确保高遮挡率预测与低遮挡率预测保持一致
3. **熵排序损失 (Entropy Ranking Loss, ERL)**：确保低遮挡率预测的熵低于高遮挡率预测的熵

总损失为两者的线性组合，仅使用单一模型、仅更新 BN 层。

### 关键设计

#### 显式遮挡链 (Explicit Mask Chaining)

利用 ViT 最后一层的自注意力结构计算注意力分数：

$$A = \sum_{h=1}^{H} \text{Softmax}\left(\frac{Q_{h,cls} K_{h,img}^\top}{\sqrt{d}}\right)$$

其中 $Q_{h,cls}$ 是 CLS token 的 query，$K_{h,img}$ 是图像 token 的 key。注意力分数高的 patch 大概率包含目标对象。

按注意力分数降序排列后，定义遮挡链 $\{x_{m_1}, x_{m_2}, \cdots, x_{m_N}\}$，满足 $0 \leq m_1 \leq m_2 \leq \cdots \leq m_N \leq 1$。实验验证随着遮挡比例增大，误差和熵单调递增，特别是在低遮挡比例区域呈近似线性关系。

**设计要点**：

- 遮挡前景（目标区域）而非背景或随机遮挡，因为只有前景遮挡才能建立可靠的排序关系
- 默认使用 $M_N = \{0, 5\%, 10\%\}$，即原始图像 + 两级遮挡
- 利用 ViT 自身的注意力无需额外计算

#### 遮挡一致性损失 (MCL)

$$\mathcal{L}_{MCL} = \sum_{i < j}^{M_N} \mathcal{H}(f_t(x_j), \mathbf{sg}(f_t(x_i)))$$

其中 $\mathcal{H}(p, q)$ 是交叉熵，$\mathbf{sg}$ 是 stop-gradient 操作。

**设计思路**：

- 让高遮挡率预测向低遮挡率预测靠拢，间接降低预测熵
- 相比直接的 EM，使用交叉熵而非自身熵作为损失，避免了急剧的熵变化和过度自信
- 相比 CR 方法，不需要额外的教师模型和不确定性估计，通过显式排序结构在单一模型内生成多样预测
- 让模型学习被遮挡区域的上下文信息

#### 熵排序损失 (ERL)

$$\mathcal{L}_{ERL} = \sum_{i < j}^{M_N} \max(0, \mathcal{S}(f_t(x_i)) - \mathbf{sg}(\mathcal{S}(f_t(x_j))) + \mathsf{m})$$

其中 $\mathsf{m}$ 是 margin 超参数。

**设计思路**：

- 维护 "低遮挡 → 低熵" 的排序结构，防止高遮挡率预测过度自信导致偏向背景信息
- 借鉴排序损失在神经网络校准中的成功经验（RankMixup 等），有效缓解过度自信
- 对违反排序约束的样本直接降低熵，加速适应
- 与 MCL 互补：MCL 间接降熵但可能适应较慢，ERL 直接降熵补充适应速度

### 损失函数 / 训练策略

**总损失函数**：

$$\mathcal{L}_{REM} = \mathcal{L}_{MCL} + \lambda \cdot \mathcal{L}_{ERL}$$

**训练策略**：

- 仅更新 ViT 的 normalization 层参数（约 0.03M）
- 使用 Adam 优化器，学习率 1e-3（ImageNetC 和 CIFARC）
- 单一模型，无需 EMA 教师模型或源模型保存
- 每个样本需要 3 次前向传播（原始 + 2 级遮挡），1 次反向传播
- 默认超参数：$\lambda = 1$，$\mathsf{m} = 0$，$M_N = \{0, 5\%, 10\%\}$

## 实验关键数据

### 主实验

| 数据集 | 指标 | REM | 之前SOTA (Continual-MAE) | Source | 提升 (vs Source) |
|---|---|---|---|---|---|
| ImageNetC (CTTA) | Mean Error ↓ | **39.2%** | 42.5% | 55.8% | +16.6% |
| CIFAR10C (CTTA) | Mean Error ↓ | **最优** | - | 28.2% | 显著提升 |
| CIFAR100C (CTTA) | Mean Error ↓ | **最优** | - | 35.4% | 显著提升 |
| ImageNetC (TTA, imbalanced) | Mean Acc ↑ | **63.3%** | - | 29.9% | +33.4% |
| ImageNetC (TTA, BS=1) | Mean Acc ↑ | **60.1%** | - | 29.9% | +30.2% |
| ImageNetC (TTA, mixed L5) | Mean Acc ↑ | **62.4%** | - | 29.9% | +32.5% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|---|---|---|
| REM (MCL + ERL) | 最优，无崩塌 | 两个损失有机配合，稳定性最佳 |
| 仅 MCL (移除 ERL) | 性能下降，CIFARC 早期崩塌 | 间接降熵不足以维持适应速度 |
| 仅 ERL (移除 MCL) | 性能下降，CIFARC 早期崩塌 | 缺少一致性约束不够稳定 |
| 前景遮挡 (REM) | 最优，无崩塌 | 满足显式排序的直觉假设 |
| 背景遮挡 | 性能下降 | 排序关系不成立 |
| 随机遮挡 | 性能下降 | 排序关系不可靠 |

### 效率对比

| 方法 | 训练参数量 | 总时间 | 前向传播次数 | 模型数量 | Error (%) |
|---|---|---|---|---|---|
| Tent (EM) | 0.04M | 8m35s | 1 | 1 | 51.0 |
| CoTTA (CR) | 86.4M | 33m23s | 3~35 | 2 | 54.8 |
| ViDA (CR) | 93.7M | 54m48s | 1 | 2 | 43.4 |
| Continual-MAE (CR) | 86.5M | 59m56s | 1 | 2 | 42.5 |
| **REM (本文)** | **0.03M** | **17m21s** | **3** | **1** | **39.2** |

### 关键发现

1. **效率优势显著**：相比 Continual-MAE，REM 性能提升 3.3% 的同时，仅用 30% 的计算时间和 0.03% 的训练参数
2. **校准误差低**：REM 的 ECE 为 8.7%，在所有高性能方法中最低（ViDA 14.6%），说明在降低错误率的同时避免了过度自信
3. **学习率鲁棒**：REM 在不同学习率设置下均表现稳定，方便根据应用需求灵活选择适应速度
4. **前向/后向迁移权衡**：高学习率在已见域表现好（42.1%），低学习率在未见域泛化好（41.4%），与监督学习上界差距仅 3.5%
5. **广泛适用性**：可扩展到 CNN 架构（使用 Feature Activation 或 Grad-CAM 替代注意力分数）、CLIP 视觉语言模型

## 亮点与洞察

1. **直觉简单但有效**：核心思想——遮挡目标导致熵升高——简单直观，却能衍生出完整的方法论框架。这类 "把不可控的分布变化转化为可控的排序结构" 的思路值得借鉴
2. **融合 EM 和 CR 的优势**：REM 用遮挡链替代数据增强实现一致性，用排序损失替代直接熵最小化，保留了 EM 的效率和 CR 的稳定性
3. **单模型方案**：不需要教师模型、EMA 模型或源模型保存，从工程部署角度极具吸引力
4. **利用 ViT 内在结构**：注意力分数天然指示目标位置，遮挡策略无需额外计算即可获得语义有意义的增强

## 局限与展望

1. **理论支撑不足**：显式遮挡导致熵增的假设缺乏严格数学证明，虽然实验验证了统计显著性，但个别样本可能存在反例
2. **学习率敏感的适应速度权衡**：快速适应和泛化之间的 trade-off 未完全解决
3. **简单域适应速度受限**：TVD 分析显示，对于较简单的域（如 brightness、JPEG），遮挡前后预测差异小导致损失低、适应慢。作者建议未来引入自适应损失权重
4. **遮挡比例固定**：当前使用固定的遮挡比例集合，未根据域差异动态调整
5. **ViT 依赖性**：虽然可扩展到 CNN，但核心方法设计基于 ViT 注意力机制，通用性需进一步验证

## 相关工作与启发

- **Tent (Wang et al., 2021)**：TTA 的开创性工作，提出仅更新 BN 层的熵最小化策略，是本文的直接改进对象
- **CoTTA (Wang et al., 2022)**：CTTA 基准方法，教师-学生框架 + 随机参数恢复，稳定但低效
- **SAR (Niu et al., 2023)**：首次观察到模型崩塌现象，提出平坦极小值适应和大梯度样本过滤
- **Continual-MAE (Liu et al., 2024)**：当前 SOTA，用 MC dropout 估计不确定性 + 掩码自编码器，但计算代价大
- **RankMixup (Noh et al., 2023)**：排序损失在网络校准中的应用，为 ERL 设计提供启发

## 评分

| 维度 | 分数 | 说明 |
|---|---|---|
| 创新性 | ⭐⭐⭐⭐ | 将不可控的分布变化转化为可控的排序结构，思路新颖实用 |
| 技术深度 | ⭐⭐⭐⭐ | 两个互补损失函数设计合理，消融实验充分验证每个组件 |
| 实验质量 | ⭐⭐⭐⭐⭐ | 多数据集、多场景、多架构全面验证，效率对比令人信服 |
| 写作质量 | ⭐⭐⭐⭐ | 动机清晰，Achilles 与乌龟的类比生动，图表设计好 |
| 实用价值 | ⭐⭐⭐⭐⭐ | 0.03M 参数、单模型、即插即用，工程部署价值极高 |
| **综合** | **⭐⭐⭐⭐☆** | 简洁优雅的方法，以极低代价实现 SOTA，值得重点关注 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Beyond Entropy: Region Confidence Proxy for Wild Test-Time Adaptation](beyond_entropy_region_confidence_proxy_for_wild_test-time_adaptation.md)
- [\[NeurIPS 2025\] Test-Time Adaptation by Causal Trimming](../../NeurIPS2025/others/test-time_adaptation_by_causal_trimming.md)
- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](../../CVPR2026/others/neural_collapse_in_test-time_adaptation.md)
- [\[ICML 2025\] Discrepancy Minimization in Input-Sparsity Time](discrepancy_minimization_in_input-sparsity_time.md)
- [\[NeurIPS 2025\] SPACE: SPike-Aware Consistency Enhancement for Test-Time Adaptation in Spiking Neural Networks](../../NeurIPS2025/others/space_spike-aware_consistency_enhancement_for_test-time_adaptation_in_spiking_ne.md)

</div>

<!-- RELATED:END -->
