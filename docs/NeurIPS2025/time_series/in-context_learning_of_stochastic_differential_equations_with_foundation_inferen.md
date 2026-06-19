---
title: >-
  [论文解读] In-Context Learning of Stochastic Differential Equations with Foundation Inference Models
description: >-
  [NeurIPS 2025][时间序列][随机微分方程] 提出FIM-SDE（基础推断模型），一个预训练的识别模型，能够从噪声时间序列数据中进行零样本(in-context)估计低维SDE的漂移和扩散函数，并通过快速微调进一步超越所有基线方法。 随机微分方程(SDE)描述了确定性流（由漂移函数控制）和随机波动（由扩散函数决定…
tags:
  - "NeurIPS 2025"
  - "时间序列"
  - "随机微分方程"
  - "上下文学习"
  - "基础推断模型"
  - "漂移函数估计"
  - "扩散函数估计"
---

# In-Context Learning of Stochastic Differential Equations with Foundation Inference Models

**会议**: NeurIPS 2025  
**arXiv**: [2502.19049](https://arxiv.org/abs/2502.19049)  
**作者**: Patrick Seifner, Kostadin Cvejoski, David Berghaus, Cesar Ojeda, Ramses J. Sanchez
**代码**: 有  
**领域**: 时间序列 / 随机微分方程  
**关键词**: 随机微分方程, 上下文学习, 基础推断模型, 漂移函数估计, 扩散函数估计

## 一句话总结

提出FIM-SDE（基础推断模型），一个预训练的识别模型，能够从噪声时间序列数据中进行零样本(in-context)估计低维SDE的漂移和扩散函数，并通过快速微调进一步超越所有基线方法。

## 研究背景与动机

随机微分方程(SDE)描述了确定性流（由漂移函数控制）和随机波动（由扩散函数决定）叠加的动力学系统：

$$dX_t = f(X_t, t)dt + g(X_t, t)dW_t$$

从观测数据中准确估计漂移函数 $f$ 和扩散函数 $g$ 是机器学习中的核心问题，在自然科学和社会科学中有广泛应用。然而，现有方法存在以下不足：

**依赖先验知识**：符号回归等方法需要预设函数形式

**训练复杂**：Neural SDE等方法需要针对每个数据集单独设计和训练

**泛化能力差**：现有模型通常不能跨SDE系统泛化

**缺乏基础模型**：SDE发现领域没有类似NLP/CV中的预训练基础模型

核心动机：能否训练一个基础推断模型，使其能够在零样本设置下（无需针对目标系统训练）准确估计任意SDE的漂移和扩散函数？

## 方法详解

### 整体框架

FIM-SDE是一个基于Transformer的基础推断模型，框架包含三个阶段：

1. **预训练阶段**：在大量合成SDE路径上以有监督方式训练
2. **上下文推断阶段**：给定新的观测序列，零样本估计目标SDE的函数
3. **微调阶段**（可选）：在目标数据集上快速适配

### 关键设计

#### 1. 摊销推断 (Amortized Inference)

FIM-SDE借鉴摊销推断的思想：不为每个新问题从头训练，而是预训练一个通用的"识别网络"，将观测数据映射到SDE参数空间。

核心优势：
- 推断时只需一次前向传播（而非迭代优化）
- 可自然处理不同长度和采样率的观测序列
- 天然支持上下文学习——通过看更多观测数据自动改善估计

#### 2. 神经算子 (Neural Operators)

利用神经算子的概念，FIM-SDE学习从观测路径到函数空间的映射：

$$\mathcal{F}: \{(t_i, X_{t_i})\}_{i=1}^N \mapsto (\hat{f}, \hat{g})$$

这使得模型能够输出函数级别的估计，而非仅仅是有限维度的参数。

#### 3. Transformer架构

模型架构的关键选择：
- **输入序列化**：将观测的SDE路径 $\{(t_i, X_{t_i})\}$ 编码为token序列
- **注意力机制**：捕获路径中的长程依赖关系
- **输出解码**：将Transformer的输出解码为在查询点上的漂移和扩散函数值
- **多路径聚合**：当有多条观测路径时，通过注意力机制自然聚合信息

#### 4. 预训练数据生成

核心创新之一是预训练数据集的构建：

1. **采样SDE系统**：从某个SDE函数空间（如高斯过程先验）中采样漂移和扩散函数
2. **数值模拟**：使用Euler-Maruyama等方法模拟SDE路径
3. **添加观测噪声**：模拟实际测量中的噪声
4. **离散化**：以随机间隔采样观测点

训练集规模大且多样，覆盖广泛的SDE动力学行为。

### 损失函数 / 训练策略

有监督预训练损失：

$$\mathcal{L} = \sum_{q \in Q} \left[ \|\hat{f}(q) - f^*(q)\|^2 + \|\hat{g}(q) - g^*(q)\|^2 \right]$$

其中 $Q$ 是一组查询点，$f^*$ 和 $g^*$ 是真实的漂移和扩散函数。

微调策略：
- 使用相同的损失函数
- 学习率通常设为预训练的1/10
- 少量epoch即可显著提升
- 可在无真实函数标签的情况下用重构损失替代

## 实验关键数据

### 主实验

#### 合成SDE系统的零样本估计

| SDE系统 | FIM-SDE (零样本) | 符号回归 | GP回归 | Neural SDE | FIM-SDE (微调) |
|---------|-----------------|---------|--------|-----------|---------------|
| 双阱动力学 | 接近匹配 | 需要先验 | 匹配 | 匹配 | **最优** |
| 弱扰动Lorenz | 接近匹配 | 困难 | 匹配 | 匹配 | **最优** |
| 几何布朗运动 | 接近匹配 | 匹配 | 匹配 | 匹配 | **最优** |
| Ornstein-Uhlenbeck | 匹配 | 匹配 | 匹配 | 匹配 | **最优** |
| 非线性扩散 | 接近匹配 | 困难 | 接近 | 接近 | **最优** |

注："匹配"指性能与最优基线相当；"最优"指超越所有基线。

#### 真实世界数据集

| 数据集 | 数据类型 | FIM-SDE (零样本) | FIM-SDE (微调) | 最优基线 |
|--------|---------|-----------------|---------------|---------|
| 股票价格 | 金融 | 接近基线 | **超越** | GP/Neural SDE |
| 油价波动 | 商品 | 接近基线 | **超越** | GP |
| 风速波动 | 气象 | 接近基线 | **超越** | Neural SDE |

关键观察：
- 零样本模式下，FIM-SDE匹配在目标数据集上单独训练的基线方法
- 微调后，FIM-SDE一致地超越所有基线

### 消融实验

#### 观测路径数量的影响

| 路径数 | 零样本MSE | 微调MSE | 改善幅度 |
|--------|----------|---------|---------|
| 1 | 较高 | 中等 | 大 |
| 5 | 中等 | 较低 | 中等 |
| 10 | 较低 | 低 | 小 |
| 50 | 低 | 最低 | 微小 |

观察：随着观测路径增多，零样本性能稳步提升——模型确实在进行上下文学习。

#### 预训练数据量的影响

| 预训练SDE数量 | 零样本泛化能力 | 微调收敛速度 |
|-------------|-------------|------------|
| 1K | 较差 | 中等 |
| 10K | 中等 | 较快 |
| 100K | 良好 | 快 |
| 1M | 最优 | 最快 |

#### 噪声水平的鲁棒性

| 观测噪声 $\sigma$ | FIM-SDE MSE | Neural SDE MSE | GP MSE |
|-------------------|------------|----------------|--------|
| 0.01 | 低 | 低 | 低 |
| 0.1 | 中等 | 较高 | 中等 |
| 0.5 | 较高 | 高 | 较高 |
| 1.0 | 高 | 很高 | 高 |

FIM-SDE在各噪声水平下表现均不低于最优基线。

### 关键发现

1. **零样本能力**：FIM-SDE无需任何目标系统的训练即可提供有意义的函数估计
2. **快速微调**：少量epoch微调即可使性能超越所有上在目标数据集上训练的基线
3. **真实世界有效性**：在金融、气象等真实数据上同样有效
4. **上下文学习现象**：更多观测路径 → 更好的估计，模型在推理时持续改善
5. **鲁棒性**：对观测噪声和不规则采样具有良好鲁棒性

## 亮点与洞察

1. **SDE发现的基础模型范式**：首次将基础模型/上下文学习的概念引入SDE函数估计，开创了新范式
2. **摊销推断的高效性**：预训练一次，多次使用，大幅降低了SDE发现的计算成本
3. **函数空间输出**：输出完整的漂移和扩散函数（而非参数），更加灵活
4. **跨域泛化**：同一个预训练模型能处理从物理到金融的不同领域SDE
5. **实用的微调机制**：在零样本已经不错的基础上，微调提供了额外的确定性提升
6. **之前版本标题变更**：v1标题为"Foundation Inference Models for SDEs: A Transformer-based Approach for Zero-shot Function Estimation"，反映了研究重点的演变

## 局限与展望

1. **维度限制**：目前仅适用于低维SDE，高维SDE（如偏微分方程）需要新方法
2. **预训练SDE分布的先验**：预训练数据的SDE分布可能与真实目标系统的分布不匹配
3. **理论保证缺失**：上下文学习的理论保证（如收敛率、泛化界）尚未建立
4. **跳跃扩散和非Markov过程**：当前仅处理标准SDE，不支持跳跃过程或记忆效应
5. **计算资源**：预训练阶段需要大量计算资源
6. **可解释性**：Transformer的黑箱性质使得估计结果难以物理解释

## 相关工作与启发

- **Neural SDEs** (Kidger et al., 2021)：为每个系统单独训练Neural SDE，本文的预训练方法避免了这一需求
- **SDE符号回归** (Brunton et al., 2016)：SINDy等方法需要预设函数形式
- **In-context learning** (Brown et al., 2020)：GPT系列展示的上下文学习能力，本文将其引入科学计算
- **Neural Operators** (Lu et al., 2021)：DeepONet等算子学习方法，为本文的函数空间映射提供了框架
- **Amortized inference** (Gershman & Goodman, 2014)：摊销推断的一般框架
- **Foundation models for science** (Bommasani et al., 2021)：科学领域基础模型的广泛趋势

## 评分

- **新颖性**: ★★★★★ — 将基础模型引入SDE发现是全新范式
- **理论深度**: ★★★☆☆ — 偏实证，理论分析较少
- **实验充分度**: ★★★★☆ — 合成+真实数据，多基线比较，消融实验充分
- **实用价值**: ★★★★★ — 零样本SDE发现对科学计算有重大实用价值
- **写作质量**: ★★★★☆ — 动机清晰，实验全面，结构合理

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Neural Stochastic Flows: Solver-Free Modelling and Inference for SDE Solutions](neural_stochastic_flows_solver-free_modelling_and_inference_for_sde_solutions.md)
- [\[NeurIPS 2025\] TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning](tirex_zero-shot_forecasting_across_long_and_short_horizons_with_enhanced_in-cont.md)
- [\[ICLR 2026\] Random Controlled Differential Equations](../../ICLR2026/time_series/random_controlled_differential_equations.md)
- [\[NeurIPS 2025\] Transformer Embeddings for Fast Microlensing Inference](transformer_embeddings_for_fast_microlensing_inference.md)
- [\[NeurIPS 2025\] Statistical Guarantees for High-Dimensional Stochastic Gradient Descent](statistical_guarantees_for_high-dimensional_stochastic_gradient_descent.md)

</div>

<!-- RELATED:END -->
