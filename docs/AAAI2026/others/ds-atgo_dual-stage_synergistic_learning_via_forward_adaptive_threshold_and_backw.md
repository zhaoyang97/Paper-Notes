---
title: >-
  [论文解读] DS-ATGO: Dual-Stage Synergistic Learning via Forward Adaptive Threshold and Backward Gradient Optimization for Spiking Neural Networks
description: >-
  [AAAI 2026][脉冲神经网络] 针对SNN训练中因膜电位分布偏移导致的脉冲发放不均衡和梯度消失问题，提出前向自适应阈值+后向阈值驱动梯度优化的双阶段协同学习算法DS-ATGO，在CIFAR10/100和ImageNet上以低时延实现SOTA性能。 脉冲神经网络（SNN）作为一种受生物启发的计算范式…
tags:
  - "AAAI 2026"
  - "脉冲神经网络"
  - "自适应阈值"
  - "代理梯度优化"
  - "膜电位动力学"
  - "低时延推理"
---

# DS-ATGO: Dual-Stage Synergistic Learning via Forward Adaptive Threshold and Backward Gradient Optimization for Spiking Neural Networks

**会议**: AAAI 2026  
**arXiv**: [2511.13050](https://arxiv.org/abs/2511.13050)  
**代码**: [github.com/jqjiang1999/DS-ATGO](https://github.com/jqjiang1999/DS-ATGO)  
**领域**: 脉冲神经网络 / 神经形态计算  
**关键词**: 脉冲神经网络, 自适应阈值, 代理梯度优化, 膜电位动力学, 低时延推理

## 一句话总结

针对SNN训练中因膜电位分布偏移导致的脉冲发放不均衡和梯度消失问题，提出前向自适应阈值+后向阈值驱动梯度优化的双阶段协同学习算法DS-ATGO，在CIFAR10/100和ImageNet上以低时延实现SOTA性能。

## 研究背景与动机

脉冲神经网络（SNN）作为一种受生物启发的计算范式，使用离散脉冲信号进行异步计算，具有天然的时空信息处理能力和高能效特性。但SNN的直接训练面临一个根本挑战：脉冲发放函数（Heaviside函数）不可微分，导致梯度反向传播受阻。

**代理梯度（SG）学习**是目前最主流的SNN训练方法，用连续平滑函数替代Heaviside函数来近似梯度。但现有SG方法存在两个关键问题:

### 问题一：固定阈值导致脉冲发放不均衡

神经元只有当膜电位超过阈值 $V_{th}$ 时才发放脉冲。但随着脉冲在层间传播，膜电位分布会发生偏移：
- 当膜电位远小于阈值时：过度稀疏发放 → "脉冲消失问题"
- 当膜电位远大于阈值时：过度发放 → 丧失对输入模式的区分能力

神经科学研究已发现，生物神经元的阈值并非固定，而是展现"阈值可塑性"（threshold plasticity），根据神经元活动历史动态调整。

### 问题二：固定SG导致梯度信号衰减

代理梯度函数（如矩形函数）只在阈值附近一个有限区间内提供非零梯度。当膜电位分布偏移时：
- 方差过小→大量膜电位落入梯度区间→梯度近似误差累积
- 方差过大→大部分膜电位落在梯度区间外→梯度消失

**核心洞察**：膜电位、阈值和代理梯度三者之间存在内在关联，但现有方法要么只调阈值、要么只调SG，忽略了这种协同关系。

## 方法详解

### 整体框架

DS-ATGO采用双阶段协同学习框架（如Fig. 3所示）：
- **前向传播（绿色）**：自适应阈值（AT）机制根据每个时间步的膜电位分布动态调整阈值
- **后向传播（黄色）**：阈值驱动的梯度优化（TGO）方法根据自适应阈值的变化动态缩放代理梯度宽度

### 关键设计

#### 1. **自适应阈值机制（AT）**

**核心定理（Theorem 1）**：当膜电位 $U(t) \sim N(\mu, \sigma^2)$ 时，若设定阈值为 $V_{th} = \mu + \sigma$，则膜电位超过阈值的概率恒为 $P = 1 - \Phi(1) \approx 15.87\%$，**与 $\mu$ 和 $\sigma$ 的具体值无关**。

因此，只要让阈值跟随膜电位分布的均值+标准差变化，就能保证稳定的发放率。具体公式：

$$\Delta V_{th}^l(t)_n = f_c \cdot (\mathbb{E}(U^l(t)_n) + \sqrt{\mathbb{VAR}(U^l(t)_n)})$$

其中 $f_c$ 是控制能效与性能平衡的因子。

**推理阈值稳定化**：受Batch Normalization启发，使用滑动平均稳定推理阶段的阈值：
$$\Delta V_{th}^l(t) = m \cdot \Delta V_{th}^l(t)_n + (1-m) \cdot \Delta V_{th}^l(t)$$
动量系数 $m=0.1$。

**设计动机**：阈值与膜电位建立正相关关系，保留了生物神经元的自适应特性，使每层每时间步的发放率保持在适度活跃状态（约15%±1.62%）。

#### 2. **阈值驱动的梯度优化（TGO）**

自适应阈值虽然移动了SG的中心位置，但SG的**宽度**仍是固定的，导致梯度匹配不精确。TGO根据自适应阈值与初始阈值的差异来动态调节SG宽度：

$$k = \begin{cases} (1 - \tanh(V_{th} - \Delta V_{th})) \cdot k, & \Delta V_{th} < V_{th} \\ (1 + \tanh(\Delta V_{th} - V_{th})) \cdot k, & \Delta V_{th} \geq V_{th} \end{cases}$$

**两种情况的直觉**：
- $\Delta V_{th} < V_{th}$（膜电位分布集中）：缩小SG宽度，减少落入梯度区间的神经元比例，抑制梯度近似误差累积
- $\Delta V_{th} \geq V_{th}$（膜电位分布分散）：扩大SG宽度，增加获得梯度的神经元比例，缓解梯度信息丢失

**关键创新**：通过自适应阈值建立了SG与膜电位分布的桥梁，实现了时空对齐的梯度优化。

### 损失函数 / 训练策略

- 输出层使用累积膜电位（不泄露不发放），取T个时间步的平均
- 使用标准交叉熵损失训练
- 优化器：SGD with cosine annealing
- 时间常数 $\tau$ 设为逐层可学习
- 不引入额外的推理开销（AT和TGO仅涉及阈值和梯度的缩放）

## 实验关键数据

### 主实验

| 数据集 | 方法 | 架构 | 时间步 | 精度(%) |
|--------|------|------|--------|---------|
| CIFAR10 | **DS-ATGO** | ResNet-19 | **2** | **96.91±0.12** |
| CIFAR10 | MPD-AGL | ResNet-19 | 4/2 | 96.35/96.18 |
| CIFAR10 | DeepTAGE | ResNet-18 | 4 | 95.86 |
| CIFAR10 | LT-SNN | Spikformer-4-256 | 4 | 95.19 |
| CIFAR100 | **DS-ATGO** | ResNet-19 | **2** | **80.59±0.17** |
| CIFAR100 | SNN-ViT | VGG-16 | 4 | 80.01 |
| CIFAR100 | MPD-AGL | ResNet-19 | 4/2 | 79.72/78.84 |
| CIFAR10-DVS | **DS-ATGO** | VGGSNN | 10 | **83.70±0.41** |
| CIFAR10-DVS | MPD-AGL | VGGSNN | 10 | 82.50 |
| ImageNet | **DS-ATGO** | ResNet-18 | 4 | **68.86±0.25** |
| ImageNet | DeepTAGE | ResNet-18 | 4 | 68.52 |

DS-ATGO在所有数据集上均达到SOTA，且在CIFAR10/100上仅用**2个时间步**即超越使用4-6个时间步的竞争方法。

### 消融实验

| 配置 | CIFAR10 提升 | CIFAR10-DVS 提升 | 说明 |
|------|-------------|-----------------|------|
| Vanilla-SNN | 基线 | 基线 | 固定阈值+固定SG |
| w/ AT only | +0.71% | +0.90% | 自适应阈值改善信息编码 |
| w/ TGO only | +1.05% | +1.40% | 梯度优化比阈值调整更有效 |
| **w/ AT+TGO** | **+1.71%** | **+2.30%** | 双阶段协同效果显著 |
| 估计分布 vs 真实分布 | 真实更优 | 真实更优 | 定理推导存在近似误差 |
| 有/无滑动平均 | 有更优 | 有更优（尤其DVS） | 稳定推理阈值 |

### 关键发现

1. **发放率稳定性**：DS-ATGO的各层发放率维持在15%±1.62%的小幅波动范围内，而Vanilla-SNN各层波动剧烈（平均10.66%，层间差异大），DIET-SNN和LTMD虽提高了发放率但波动更大。
2. **梯度可用比例**：DS-ATGO将ResNet-19每层的梯度可用率提升至38.53%以上，深层仍保持36.54%，而Vanilla-SNN深层降至不足15.13%。
3. **损失景观**：DS-ATGO产生更平坦、更稀疏的损失景观，表明更好的泛化能力。Vanilla-SNN有两个局部极小值且轮廓线有明显突起。
4. **TGO > AT**：单独使用TGO的效果超过单独使用AT，因为TGO在时间维度上同步调整SG以准确捕获膜电位偏离，增强了参数更新的效率和正确性。

## 亮点与洞察

- **生物学启发的理论保证**：Theorem 1给出了自适应阈值设定的数学依据（$\mu+\sigma$ 原则），保证恒定发放概率
- **双阶段协同设计**：前向（阈值）和后向（梯度）不是独立的，而是通过阈值变化传递信息，实现了真正的协同优化
- **低时延优势**：2个时间步即达SOTA，对SNN部署至关重要（更少时间步 = 更短延迟 + 更低能耗）
- **零推理开销**：AT的滑动平均和TGO的梯度缩放不增加推理计算量

## 局限与展望

- 仅在分类任务上验证，未扩展到检测/分割等密集预测任务
- Theorem 1假设膜电位服从正态分布，对于高度非线性的深层网络可能不总是成立
- 因子 $f_c$ 需要手动设定，不同任务可能需要不同的值
- ImageNet精度（68.86%）与ANN仍有较大差距，SNN的可扩展性仍是开放问题
- 未讨论能效方面的具体数据（SNN的核心卖点之一）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 阈值-梯度协同优化的视角新颖，但各组件的单独想法已有前人工作
- **实验充分度**: ⭐⭐⭐⭐⭐ — 消融全面（发放率/梯度比例/损失景观），4个数据集覆盖静态+神经形态
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，图表直观，但部分符号稍显冗余
- **价值**: ⭐⭐⭐⭐ — 对SNN社区很有价值，低时延SOTA有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)
- [\[ICLR 2026\] Breaking Gradient Temporal Collinearity for Robust Spiking Neural Networks](../../ICLR2026/others/breaking_gradient_temporal_collinearity_for_robust_spiking_neural_networks.md)
- [\[AAAI 2026\] TDSNNs: Competitive Topographic Deep Spiking Neural Networks for Visual Cortex Modeling](tdsnns_competitive_topographic_deep_spiking_neural_networks_for_visual_cortex_mo.md)
- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)
- [\[ICML 2026\] Bullet Trains: Parallelizing Training of Temporally Precise Spiking Neural Networks](../../ICML2026/others/bullet_trains_parallelizing_training_of_temporally_precise_spiking_neural_networ.md)

</div>

<!-- RELATED:END -->
