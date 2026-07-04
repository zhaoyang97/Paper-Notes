---
title: >-
  [论文解读] Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting
description: >-
  [AAAI 2026 (Oral)][时间序列][多变量时间序列] 提出 Sonnet，通过可学习小波变换将输入映射到时频域，引入基于谱相干性的多变量注意力（MVCA）建模变量间依赖关系，并利用 Koopman 算子进行稳定的时间演化预测，在 47 个预测任务中的 34 个取得最优，平均 MAE 降低 2.2%。
tags:
  - "AAAI 2026 (Oral)"
  - "时间序列"
  - "多变量时间序列"
  - "频谱分析"
  - "小波变换"
  - "Koopman 算子"
  - "谱相干注意力"
  - "外生变量"
---

# Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting

**会议**: AAAI 2026 (Oral)  
**arXiv**: [2505.15312](https://arxiv.org/abs/2505.15312)  
**代码**: [https://github.com/ClaudiaShu/Sonnet](https://github.com/ClaudiaShu/Sonnet)  
**领域**: 时间序列预测  
**关键词**: 多变量时间序列, 频谱分析, 小波变换, Koopman 算子, 谱相干注意力, 外生变量

## 一句话总结
提出 Sonnet，通过可学习小波变换将输入映射到时频域，引入基于谱相干性的多变量注意力（MVCA）建模变量间依赖关系，并利用 Koopman 算子进行稳定的时间演化预测，在 47 个预测任务中的 34 个取得最优，平均 MAE 降低 2.2%。

## 研究背景与动机

**领域现状**：多变量时间序列（MTS）预测利用外生变量预测单一目标变量，广泛应用于气象、流感预测、电力消费等领域。Transformer 架构虽然能捕获长程依赖，但在建模变量间复杂关系方面存在不足。

**现有痛点**：  
   - iTransformer/Samformer 等方法将序列沿时间维度嵌入再做变量间注意力，破坏了时间信息；  
   - Crossformer/ModernTCN 虽然尝试双维度依赖建模，但 GPU 开销大；  
   - TimeXer/DeformTime 仅在卷积核感受野内捕获变量间依赖，范围有限；  
   - 频域方法（FEDformer、FiLM）聚焦于变量内部的季节性建模，忽略了变量间的频谱关联。

**核心矛盾**：现有注意力机制在 MTS 任务中并不总是有效——实验发现移除部分模型的注意力模块反而不会显著降低性能，说明 vanilla attention 无法有效捕获变量间信息。

**本文目标** 设计一种在频谱域建模变量间依赖的新架构，兼顾时间信息保留和跨变量关联捕获。

**切入角度**：谱相干性（spectral coherence）是信号处理中衡量两个信号在不同频率上相关性的经典工具，将其引入注意力机制可以量化变量间的频域依赖。

**核心 idea**：用可学习小波做时频分解 → 谱相干注意力捕获变量间频域依赖 → Koopman 算子线性化时间演化 → 小波逆变换重建。

## 方法详解

### 整体框架
Sonnet 由四个核心模块串联：**联合嵌入** → **可学习小波变换** → **多变量谱相干注意力（MVCA）** → **Koopman 时间演化** → **小波逆变换重建** → **卷积解码器**。输入为外生变量矩阵 $\mathbf{X} \in \mathbb{R}^{L \times C}$ 和内生变量 $\mathbf{y} \in \mathbb{R}^{L}$，通过超参数 $\alpha$ 控制两者在嵌入中的维度分配，最终输出预测序列 $\hat{\mathbf{y}} \in \mathbb{R}^{H}$。

### 关键设计 1：可学习小波变换（Learnable Wavelet Transform）
- 定义 $K$ 个可学习小波原子（atom），每个原子由参数化的高斯包络 × 调频余弦构成：$\mathbf{M}_k = \exp(-\mathbf{w}_\alpha \mathbf{t}^2) \times \cos(\mathbf{w}_\beta \mathbf{t} + \mathbf{w}_\gamma \mathbf{t}^2)$
- 三组可学习权重 $\mathbf{w}_\alpha, \mathbf{w}_\beta, \mathbf{w}_\gamma$ 分别控制高斯窗宽度、线性频率调制、二次频率调制
- 通过逐元素乘法将嵌入投影到小波空间：$\mathbf{P}_k = \mathbf{E} \odot \mathbf{M}_k^\top$
- **优势**：不同于固定母小波，可学习参数使原子自适应数据的局部时频结构；$K$ 个原子自然对应多头注意力的 $K$ 个头

### 关键设计 2：多变量谱相干注意力（MVCA）
- 对每个注意力头，用 Q/K/V 线性投影后，沿**变量维度**做 FFT 转换到频域
- 计算交叉谱密度 $\mathbf{P}_{qk} = \mathbf{Q}_f \odot \mathbf{K}_f^*$ 和功率谱密度 $\mathbf{P}_{qq}, \mathbf{P}_{kk}$
- 归一化得到谱相干性：$\mathbf{C}_{qk} = |\bar{\mathbf{P}}_{qk}|^2 / (\bar{\mathbf{P}}_{qq} \cdot \bar{\mathbf{P}}_{kk} + \epsilon)$
- 谱相干值经 Softmax 归一化后作为注意力权重，按元素乘以 Value，再经过 2 层 MLP + 残差连接
- **核心思想**：谱相干性衡量了 Q 和 K 在多个频率上的线性相关性，相干性越高表示变量间依赖越强；这比 dot-product attention 更能捕获频域中的变量关联

### 关键设计 3：Koopman 时间演化
- 初始化可学习复数矩阵 $\mathbf{S}$，每次前向传播做 QR 分解保留酉矩阵 $\mathbf{U}$（保证 $\mathbf{U}^\dagger \mathbf{U} = \mathbf{I}$）
- 学习相位向量 $\mathbf{p}$，构造对角矩阵 $\mathbf{D} = \text{diag}(e^{ip_k})$
- Koopman 算子 $\mathbf{K} = \mathbf{U} \mathbf{D} \mathbf{U}^\dagger$，作用于 MVCA 输出的复数形式
- **优势**：酉矩阵保证不放大/失真数据；一次全局投影替代逐步递归，减少误差累积

### 损失函数
均方误差（MSE），在所有时间步 $\{t+1, \ldots, t+H\}$ 上计算，使用 Adam 优化器，线性学习率衰减。

## 实验

### 主实验结果

| 任务 | H | Sonnet MAE | DeformTime MAE | ModernTCN MAE | PatchTST MAE | iTransformer MAE |
|------|---|-----------|----------------|---------------|-------------|-----------------|
| ELEC | 12 | **0.1040** | 0.1162 | 0.1596 | 0.1419 | 0.1468 |
| ELEC | 36 | **0.1389** | 0.1729 | 0.2065 | 0.1659 | 0.1791 |
| ILI-ENG | 7 | **1.4791** | 1.6417 | 1.9489 | 2.3115 | 2.3084 |
| ILI-ENG | 28 | **2.7481** | 2.7228 | 3.3611 | 4.9964 | 4.8125 |
| ILI-US2 | 7 | **0.3806** | 0.4122 | 0.4398 | 0.7097 | 0.6507 |
| WEA-HK | 4 | **0.6389** | 0.6804 | 0.7004 | 1.1488 | 0.8048 |
| WEA-LD | 4 | **1.7231** | 1.8753 | 1.9456 | 2.7602 | 2.1509 |

- 47 个预测任务中，Sonnet 在 34 个上最优、9 个次优
- 整体平均 MAE 降低 2.2%（vs. DeformTime），统计显著（p = 5e-4）
- 在 ILI + WEA 等高难度任务上，MAE 降低 3.5%（ILI）和 2.0%（WEA）

### MVCA 替换实验

| 注意力 | PatchTST ILI-ENG H=7 | PatchTST ILI-US2 H=7 | PatchTST ILI-US9 H=7 |
|--------|----------------------|----------------------|----------------------|
| Naïve (原始) | 2.3115 | 0.7097 | 0.4116 |
| ¬ Attn | 2.4723 | 0.7702 | 0.4585 |
| FED | 3.6158 | 0.9292 | 0.5614 |
| VDAB | 2.0799 | 0.5925 | 0.3820 |
| **MVCA** | **2.0054** | **0.5824** | **0.3705** |

- 用 MVCA 替换 vanilla attention 后，3 个基模型在 ILI 任务上平均 MAE 降低 **10.7%**
- PatchTST 受益最大（15.1%），因为其原本不建模变量间依赖

### 消融实验

| 消融变体 | ILI 平均 Δ% | WEA 平均 Δ% |
|---------|------------|------------|
| ¬ MVCA（完整模块） | +6.3% | +2.4% |
| ¬ 谱相干 | +5.2% | +1.7% |
| ¬ MLP 残差 | +4.1% | +1.5% |
| ¬ 联合嵌入 | +4.0% | +2.3% |
| ¬ Koopman | +3.2% | +1.5% |

### 关键发现
1. MVCA 是性能贡献最大的模块；随预测步长增加，谱相干的重要性显著上升（短期 +2% → 长期 +13.6%）
2. 保留时间顺序的模型（DeformTime, ModernTCN, Crossformer）比沿时间维度嵌入的模型（iTransformer, Samformer）表现更好
3. 不捕获变量间依赖的模型（DLinear, PatchTST）在 ILI 任务中甚至无法超过持续性基线
4. 当数据具有强季节性（如 ELEC）时，$\alpha=0$（仅用历史目标值）效果更好，说明外生变量并非总有帮助

## 亮点
- **即插即用的 MVCA**：可替换任何模型中的 vanilla attention，平均提升 10.7%
- **谱相干注意力的理论直觉**：基于信号处理的频域相关性度量，比 dot-product 更适合衡量时间序列变量间的依赖
- **Koopman 算子的稳定性保证**：QR 分解保持酉矩阵，酉变换不放大信号，避免崩溃
- **全面的评估体系**：12 个数据集、47 个任务、多测试季节、多种评估指标，远超通常 4-5 个数据集的标准

## 局限性
1. **可解释性不足**：无法直接归因预测结果到特定输入变量
2. **外生变量少 + 训练时间跨度短时**效果一般（如 ETT 数据集仅 6 个外生变量 + 2 年训练）
3. 可学习小波原子的形态依赖初始化，缺乏理论指导的最优原子数 $K$ 选择
4. 方法的有效性仅通过实验验证，缺乏理论收敛/泛化分析

## 相关工作
- **频域方法**：FEDformer（频率增强分解）、FiLM（频率域 MLP）、AdaWaveNet（小波季节分解）
- **变量间建模**：iTransformer、Crossformer（双维度注意力）、ModernTCN（时变卷积）、TimeXer（交叉注意力）、DeformTime（可变形注意力）
- **Koopman 方法**：Koopa（学习 Koopman 算子做时间演化）、Lange et al.（Fourier + Koopman）

## 评分
⭐⭐⭐⭐ — 谱相干注意力是新颖且有说服力的创新，MVCA 的即插即用性极强，实验评估体系远超同类工作（12 数据集 × 47 任务）。局限在于可解释性不足和在低维外生变量场景下优势不明显。AAAI Oral 实至名归。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting](../../ICML2025/time_series/hyperimts_hypergraph_neural_network_for_irregular_multivariate_time_series_forec.md)
- [\[ICML 2026\] Generalizing Multi-scale Time-Series Modeling with a Single Operator](../../ICML2026/time_series/generalizing_multi-scale_time-series_modeling_with_a_single_operator.md)
- [\[NeurIPS 2025\] A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting](../../NeurIPS2025/time_series/a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)
- [\[ICML 2026\] DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables](../../ICML2026/time_series/dag_a_dual_correlation_network_for_time_series_forecasting_with_exogenous_variab.md)
- [\[ICLR 2026\] Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](../../ICLR2026/time_series/routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)

</div>

<!-- RELATED:END -->
