---
title: >-
  [论文解读] Transparent Networks for Multivariate Time Series
description: >-
  [AAAI 2026][时间序列][可解释模型] 提出 GATSM（Generalized Additive Time Series Model），一种透明的时间序列神经网络模型，通过共享权重的特征网络学习特征表示并用带掩码的多头注意力捕捉时序模式，在保持完全可解释性的同时达到与 Transformer 等黑箱模型可比的性能。
tags:
  - "AAAI 2026"
  - "时间序列"
  - "可解释模型"
  - "广义加性模型"
  - "透明网络"
  - "注意力机制"
---

# Transparent Networks for Multivariate Time Series

**会议**: AAAI 2026  
**arXiv**: [2410.10535](https://arxiv.org/abs/2410.10535)  
**代码**: [https://github.com/gim4855744/GATSM](https://github.com/gim4855744/GATSM)  
**领域**: 时间序列  
**关键词**: 可解释模型, 广义加性模型, 时间序列, 透明网络, 注意力机制

## 一句话总结

提出 GATSM（Generalized Additive Time Series Model），一种透明的时间序列神经网络模型，通过共享权重的特征网络学习特征表示并用带掩码的多头注意力捕捉时序模式，在保持完全可解释性的同时达到与 Transformer 等黑箱模型可比的性能。

## 研究背景与动机

### 问题背景

在医疗健康、欺诈检测等高风险领域，模型的**透明性（可解释性）** 至关重要。现有的可解释方法分为两类：
- **事后解释（Post-hoc XAI）**：如 LIME、SHAP，对已训练的黑箱模型进行解释。但可能产生不正确、不忠实的解释，无法提供输入特征的真实贡献
- **透明模型（Inherently Interpretable）**：模型结构本身就是可解释的，如广义加性模型（GAM）

### GAM 在时间序列上的困境

经典 GAM 的形式为 $g(\mathbb{E}(y|\mathbf{x})) = \sum_{i=1}^M f_i(x_i)$，每个特征有独立函数，可直接解读特征贡献。但将其应用于时间序列面临三个问题：

**无法处理时序数据**：传统表格 GAM 假设输入是静态特征向量，不能处理时序序列

**无法捕捉时序模式**：NATM 简单地在每个时间步使用独立函数 $f_{i,j}(x_{i,j})$，但各时间步之间没有交互，无法学习跨时间步的依赖关系

**固定长度限制**：NATM 需要固定长度输入，无法处理变长时间序列（如医院中患者住院时长不可预知）

### 核心贡献

作者定义了一种新的"时序 GAM"形式：

$$g(\mathbb{E}(y_t | \mathbf{X}_{:t})) = \sum_{i=1}^{t} \sum_{j=1}^{M} f_{i,j}(x_{i,j}, \mathbf{X}_{:t})$$

关键区别：函数 $f_{i,j}$ 可以接受整个序列历史 $\mathbf{X}_{:t}$ 作为额外输入，从而在保持加性结构的同时捕捉时序模式。GATSM 是首个实现这一形式的透明模型。

## 方法详解

### 整体框架

GATSM 由两个模块组成：
1. **Time-Sharing NBM（时间共享神经基模型）**：学习特征的非线性表示
2. **Masked MHA（带掩码的多头注意力）**：学习时序模式

### 关键设计

#### 1. **Time-Sharing NBM**

**问题**：如果为每个时间步和每个特征都分配独立函数，需要 $T \times M$ 个函数，参数爆炸。

**解决方案**：在所有时间步间共享基函数权重。

$$\tilde{x}_{i,j} = f_j(x_{i,j}) = \sum_{k=1}^{B} h_k(x_{i,j}) w_{j,k}^{nbm}$$

- $B = 100$ 个基函数 $h_k(\cdot)$（用 MLP 实现）
- 基函数在所有特征和时间步间共享，大幅减少参数
- 每个特征有独立的权重 $w_{j,k}^{nbm}$，确保特征特异性
- 参数量从 $T \times M$ 减少到 $B$

**设计动机**：NBM 的基策略（basis strategy）特别适合时间序列——不同时间步的同一特征可以共享非线性变换，同时通过特征特定权重保留区分能力。

#### 2. **Masked MHA（带掩码的多头注意力）**

采用 2 层注意力机制（来自 GAT），而非简单的 dot-product attention，以获得更强的表达能力。

**步骤 1**：转换特征表示并添加位置编码

$$\mathbf{v}_i = \tilde{\mathbf{x}}_i^\intercal \mathbf{Z} + \mathbf{pe}_i$$

- $\mathbf{Z} \in \mathbb{R}^{M \times D}$ 是可学习权重
- 采用正弦位置编码（而非可学习的），以支持变长序列

**步骤 2**：计算注意力分数

$$e_{k,i,j} = \sigma([\mathbf{v}_i | \mathbf{v}_j]^\intercal \mathbf{w}_k^{attn}) m_{i,j}$$

$$a_{k,i,j} = \frac{\exp(e_{k,i,j})}{\sum_{t=1}^{T} \exp(e_{k,i,t})}$$

- 使用因果掩码 $m_{i,j}$，确保时间步 $i$ 只能看到 $j \leq i$ 的信息
- 非线性激活 $\sigma(\cdot)$ 提升表达能力

#### 3. **推理与可解释性**

最终预测：

$$\hat{y}_t = \sum_{k=1}^{K} \mathbf{a}_{k,t}^\intercal \tilde{\mathbf{X}} \mathbf{w}_k^{out}$$

将其展开为标量形式：

$$= \sum_{u=1}^{t} \sum_{m=1}^{M} \underbrace{\sum_{k=1}^{K} \sum_{b=1}^{B} a_{k,t,u} h_b(x_{u,m}) w_{m,b}^{nbm} w_{k,m}^{out}}_{f_{u,m}(x_{u,m}, \mathbf{X}_{:t})}$$

这证明 GATSM 满足时序 GAM 的定义（Definition 3.1），并可以提取**三种可解释性**：

1. **时间步重要性**：$a_{k,t,u}$ 表示时间步 $u$ 在时间步 $t$ 的预测中的重要性
2. **时间无关的特征贡献**：$h_b(x_{u,m}) w_{m,b}^{nbm} w_{k,m}^{out}$ 表示特征 $m$ 的固有贡献
3. **时间依赖的特征贡献**：$a_{k,t,u} h_b(x_{u,m}) w_{m,b}^{nbm} w_{k,m}^{out}$ 表示特征 $m$ 在特定时间步的贡献

### 损失函数 / 训练策略

- 回归任务：MSE
- 二分类任务：Binary Cross-Entropy
- 多分类任务：Cross-Entropy
- 使用 AdamW 优化器
- 早停：验证损失 20 个 epoch 未下降即停止
- 超参数通过 Optuna 自动调优

## 实验关键数据

### 主实验

8 个公开时间序列数据集上的单步预测性能：

| 模型类型 | 模型 | Energy (R²↑) | Rainfall (R²↑) | AirQuality (R²↑) | Heartbeat (AUROC↑) | LSST (Acc↑) | NATOPS (Acc↑) | 平均排名 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 黑箱时序 | GRU | 0.435 | 0.089 | 0.701 | 0.694 | 0.629 | 0.931 | 4.500 |
| 黑箱时序 | Transformer | 0.263 | 0.098 | 0.711 | 0.690 | 0.679 | 0.967 | 4.125 |
| 透明表格 | NAM | 0.363 | 0.006 | 0.300 | 0.645 | 0.400 | 0.242 | 9.375 |
| 透明表格 | NBM | 0.330 | 0.007 | 0.301 | 0.716 | 0.388 | 0.189 | 9.250 |
| 透明时序 | NATM | 0.304 | 0.038 | 0.548 | 0.724 | 0.452 | 0.878 | 6.833 |
| **透明时序** | **GATSM** | **0.493** | **0.073** | **0.583** | **0.843** | **0.570** | **0.956** | **3.375** |

GATSM 达到了所有模型中**最佳平均排名**（3.375），超越了 Transformer（4.125），同时在透明模型中遥遥领先。

### 消融实验

#### 特征函数选择

| 特征函数 | Energy | Rainfall | AirQuality | Heartbeat | LSST | NATOPS |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Linear | 0.283 | 0.071 | 0.563 | 0.766 | — | — |
| NAM | — | — | — | — | — | — |
| **NBM** | **0.493** | **0.073** | **0.583** | **0.843** | **0.570** | **0.956** |

NBM 的基策略在 8 个数据集中的 6 个上取得最佳表现。

#### 时序模块设计

| 配置 | Energy | Heartbeat | LSST | NATOPS | 说明 |
|:---|:---:|:---:|:---:|:---:|:---|
| Base | 差 | 差 | 差 | 差 | 无时序模块 |
| Base + PE | 类似 Base | 类似 Base | 类似 Base | 类似 Base | 仅位置编码不够 |
| Base + MHA | 中等 | 中等 | 中等 | 中等 | 注意力有用 |
| **Base + PE + MHA** | **最佳** | **最佳** | **最佳** | **最佳** | PE + MHA 产生协同效应 |

位置编码和多头注意力需要**配合使用**才能有效捕捉时序模式。

### 关键发现

1. **GATSM 超越了 Transformer 的平均排名**：这是透明模型首次在综合排名上超越经典黑箱时序模型
2. **时序模式很重要**：在 Rainfall、AirQuality、LSST、NATOPS 等有显著时序模式的数据集上，时序模型显著优于表格模型
3. **变长时间序列处理**：GATSM 能够处理 Mortality、Sepsis 等变长数据集，而 NATM 不能
4. **医疗数据的特殊性**：Mortality 和 Sepsis 上时序模型 vs 表格模型差异不大，可能因为患者当前状态已包含历史信息
5. **可解释性的多层次**：GATSM 可以同时提供时间步重要性、全局特征贡献和局部时间依赖特征贡献

## 亮点与洞察

1. **理论定义的严谨性**：提出了时序 GAM 的形式化定义（Definition 3.1），并严格证明了 GATSM 满足该定义
2. **weight sharing 的精妙设计**：通过在时间步间共享基函数，在参数效率和表达能力之间取得了极好的平衡
3. **三种可解释性的提取**：不同层面的解释（时间步重要性、时间无关特征贡献、时间依赖特征贡献）在实际应用中非常有价值
4. **性能不妥协**：在保持完全透明的同时，性能甚至超越 Transformer，颠覆了"可解释性 vs 性能"的传统认知

## 局限与展望

1. **仅考虑一阶加性效应**：GAM 结构天然无法捕捉特征间的交互效应（如 GA²M 中的二阶项），这在某些数据集（AirQuality 等）上导致与黑箱模型的差距
2. **单步预测为主**：虽然讨论了多步预测，但实验主要在单步任务上进行，多步预测性能有待验证
3. **注意力头数有限**：多头注意力的表达能力受限于头数和隐层维度，对于非常复杂的时序模式可能不够
4. **未考虑缺失值的原生处理**：当前对缺失值采用简单的插补策略
5. **可扩展性**：在非常长的时间序列或高维特征场景下的效率未讨论

## 相关工作与启发

- **NBM**：GATSM 的特征网络直接基于 NBM 扩展，基策略对时间序列非常有效
- **NAM / NodeGAM / EBM**：其他透明表格模型，都无法处理时序数据
- **NATM**：唯一的先前透明时序模型，但不能捕捉时序模式且仅支持固定长度
- **GAT**：2 层注意力机制的来源，比 dot-product attention 更适合 GAM 场景
- 启发：将 GAM 的可解释性优势与现代深度学习组件结合，是一个值得进一步探索的方向

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首个能捕捉时序模式的透明 GAM，形式化定义严谨
- **实验充分度**: ⭐⭐⭐⭐ — 8个数据集、15个基线，消融完整，但主要在单步任务上
- **写作质量**: ⭐⭐⭐⭐⭐ — 逻辑清晰，理论推导完整，可解释性展示充分
- **价值**: ⭐⭐⭐⭐⭐ — 在可解释时序建模这一重要方向上做出了突破性贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting](revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)
- [\[AAAI 2026\] SELDON: Supernova Explosions Learned by Deep ODE Networks](seldon_supernova_explosions_learned_by_deep_ode_networks.md)
- [\[AAAI 2026\] Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification](counterfactual_explainable_ai_xai_method_for_deep_learning-based_multivariate_ti.md)
- [\[AAAI 2026\] Urban Incident Prediction with Graph Neural Networks: Integrating Government Ratings and Crowdsourced Reports](urban_incident_prediction_with_graph_neural_networks_integrating_government_rati.md)
- [\[AAAI 2026\] Mask the Redundancy: Evolving Masking Representation Learning for Multivariate Time-Series Clustering](mask_the_redundancy_evolving_masking_representation_learning_for_multivariate_ti.md)

</div>

<!-- RELATED:END -->
