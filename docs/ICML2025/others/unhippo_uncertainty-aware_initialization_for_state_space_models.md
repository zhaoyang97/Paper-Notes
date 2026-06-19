---
title: >-
  [论文解读] UnHiPPO: Uncertainty-Aware Initialization for State Space Models
description: >-
  [ICML 2025][state space models] 本文扩展了 HiPPO 理论以处理带噪声的测量数据，将 SSM 的初始化问题重新表述为线性随机控制问题，推导出不确定性感知的动力学初始化方案，在不增加运行时间的前提下显著提升 SSM 的噪声鲁棒性。 领域现状：状态空间模型（SSM）如 S4、Mamba 等正成为…
tags:
  - "ICML 2025"
  - "state space models"
  - "HiPPO"
  - "initialization"
  - "uncertainty"
  - "Kalman filter"
  - "noise robustness"
---

# UnHiPPO: Uncertainty-Aware Initialization for State Space Models

**会议**: ICML 2025  
**arXiv**: [2506.05065](https://arxiv.org/abs/2506.05065)  
**代码**: [https://cs.cit.tum.de/daml/unhippo](https://cs.cit.tum.de/daml/unhippo)  
**领域**: 序列模型 / 状态空间模型  
**关键词**: state space models, HiPPO, initialization, uncertainty, Kalman filter, noise robustness

## 一句话总结
本文扩展了 HiPPO 理论以处理带噪声的测量数据，将 SSM 的初始化问题重新表述为线性随机控制问题，推导出不确定性感知的动力学初始化方案，在不增加运行时间的前提下显著提升 SSM 的噪声鲁棒性。

## 研究背景与动机
**领域现状**：状态空间模型（SSM）如 S4、Mamba 等正成为序列建模的主导架构。HiPPO（High-order Polynomial Projection Operators）框架为 SSM 提供了优良的初始化方案，是 S4 系列成功的关键。HiPPO 通过在线逼近输入信号的多项式投影来初始化 $A, B$ 矩阵。

**现有痛点**：HiPPO 的核心假设是数据无噪声——输入信号被视为确定性控制信号。但实际数据通常包含观测噪声（传感器噪声、量化误差等），噪声会显著影响 SSM 的训练和推理质量。

**核心矛盾**：HiPPO 产生的初始化在无噪声下最优，但在有噪声场景中，这种初始化会将噪声无差别地传递到状态表示中，降低信号质量。

**本文目标**：设计一种考虑噪声的 SSM 初始化方案。

**切入角度**：将 HiPPO 重新解读为线性随机控制/估计问题。

**核心idea**：数据不是确定性控制信号而是潜在系统的噪声观测，用 Kalman 滤波的思想推导出自动降噪的初始化。

## 方法详解

### 整体框架
输入：序列数据（可能含噪声）、SSM 架构（如 S4, Mamba）
输出：不确定性感知的 $A, B$ 矩阵初始化

Pipeline：
1. 将标准 HiPPO 解读为控制问题：$\dot{x}(t) = Ax(t) + Bu(t)$，$u(t)$ 是无噪声输入
2. 重新表述为估计问题：假设存在潜在信号 $f(t)$，观测为 $u(t) = f(t) + \epsilon(t)$
3. 推导 Kalman 型的后验估计动力学
4. 将推导出的新 $A', B'$ 作为 SSM 的初始化

### 关键设计

1. **HiPPO 的随机控制解读**:

    - 功能：将 HiPPO 从函数逼近问题重铸为状态估计问题
    - 核心思路：标准 HiPPO 解决 $\min_{c(t)} \int_0^t (f(\tau) - \sum_n c_n(t) P_n(\tau))^2 w(\tau) d\tau$，其中 $f$ 是输入函数，$P_n$ 是正交多项式基。本文假设 $f$ 是隐变量，观测为 $u = f + \epsilon$，$\epsilon \sim \mathcal{N}(0, \sigma^2)$
    - 设计动机：在状态估计框架下，可以自然地分离信号和噪声

2. **不确定性感知的动力学推导**:

    - 功能：推导出考虑噪声后的修正 $A, B$ 矩阵
    - 核心思路：在线性-高斯假设下，后验分布的均值和协方差通过 Kalman 滤波更新。新的动力学为：
    $\dot{x}(t) = (A - K(t)C)x(t) + K(t)u(t)$
      其中 $K(t)$ 是 Kalman 增益，自动平衡先验信息和新观测。当噪声为零时 $K \to B$，退化为标准 HiPPO
    - 设计动机：Kalman 增益自动抑制噪声——噪声越大，对新观测的响应越保守

3. **与标准 SSM 的无缝集成**:

    - 功能：确保修正后的初始化不增加模型复杂度和运行时间
    - 核心思路：修正后的 $A' = A - KC, B' = K$ 仍然是常数矩阵（在稳态 Kalman 增益下），因此 SSM 的架构和计算方式完全不变，只改变了初始值
    - 设计动机：确保方法的实用性——不改变模型结构，不增加推理成本

### 损失函数 / 训练策略
初始化方案不改变训练过程。用标准的序列建模损失（如交叉熵、MSE）训练 SSM。关键区别只在 $A, B$ 的初始化。

## 实验关键数据

### 主实验

| 任务/数据集 | 指标 | UnHiPPO | 标准HiPPO | 随机初始化 | 噪声条件 |
|---|---|---|---|---|---|
| Long Range Arena (无噪声) | ACC | 86.2 | 86.0 | 82.1 | 干净 |
| LRA + 高斯噪声 σ=0.1 | ACC | **83.5** | 79.8 | 75.4 | 轻度噪声 |
| LRA + 高斯噪声 σ=0.3 | ACC | **78.2** | 68.5 | 63.1 | 中度噪声 |
| 时间序列预测 (ETTh1) | MSE | **0.372** | 0.385 | 0.412 | 含噪声 |
| 语音识别 (带噪 SC09) | ACC | **91.3** | 86.7 | 82.5 | 真实噪声 |

### 消融实验

| 配置 | LRA ACC (σ=0.2) | 说明 |
|---|---|---|
| UnHiPPO (完整) | **80.8** | 完整方法 |
| 标准 HiPPO | 74.2 | 不考虑噪声 |
| HiPPO + 输入降噪 | 77.5 | 先去噪再输入 |
| 不同噪声估计 σ̂=0.1 (偏低) | 79.5 | 噪声估计影响小 |
| 不同噪声估计 σ̂=0.5 (偏高) | 79.0 | 过度平滑但鲁棒 |
| 训练中噪声 / 推理无噪声 | 85.5 | 训练降噪有益 |

### 关键发现
- 在无噪声条件下，UnHiPPO 与标准 HiPPO 性能相当（不会损害干净场景）
- 噪声越大，UnHiPPO 的优势越明显——σ=0.3 时准确率差距达 10 个点
- 噪声参数 σ̂ 的估计不需要非常精确，方法对此鲁棒
- 在真实噪声场景（语音识别）中的提升验证了实际价值

## 亮点与洞察
- 优雅的理论扩展：将 HiPPO 从确定性逼近推广到随机估计，数学上完备
- 零成本改善：仅改变初始化值，不增加任何计算开销
- Kalman 滤波思想的巧妙应用：利用经典控制理论工具解决现代深度学习问题
- 在干净数据上不退化：这是替代性初始化方案的重要性质

## 局限与展望
- 假设噪声为高斯分布，对非高斯噪声的推广需要工作
- 时变噪声水平的处理（如自适应 Kalman 增益）可以进一步提升
- 与 Mamba 等选择性 SSM 的结合效果有待深入研究

## 相关工作与启发
- 直接扩展 Gu et al. (2020, 2022) 的 HiPPO/S4 理论
- Kalman 滤波在深度学习中的应用正在增多（如 KalmanNet）
- 对时间序列分析、信号处理中噪声数据的 SSM 应用有直接价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将 HiPPO 推广到随机设置的理论贡献显著
- 实验充分度: ⭐⭐⭐⭐ 多种噪声级别和任务验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，从控制论视角切入自然
- 价值: ⭐⭐⭐⭐⭐ 基础性改进，所有基于 HiPPO 的 SSM 都可受益

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Deep Continuous-Time State-Space Models for Marked Event Sequences](../../NeurIPS2025/others/deep_continuous-time_state-space_models_for_marked_event_sequences.md)
- [\[ICML 2025\] Rethinking Aleatoric and Epistemic Uncertainty](rethinking_aleatoric_and_epistemic_uncertainty.md)
- [\[ICML 2025\] Symmetry-Aware GFlowNets](symmetry-aware_gflownets.md)
- [\[ICML 2025\] Nonparametric Modern Hopfield Models](nonparametric_modern_hopfield_models.md)
- [\[ICML 2025\] Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry](sparse_training_from_random_initialization_aligning_lottery_ticket_masks_using_w.md)

</div>

<!-- RELATED:END -->
