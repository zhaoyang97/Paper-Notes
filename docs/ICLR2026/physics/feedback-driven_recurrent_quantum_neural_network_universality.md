---
title: >-
  [论文解读] Feedback-driven Recurrent Quantum Neural Network Universality
description: >-
  [ICLR2026][物理/科学计算][quantum reservoir computing] 本文首次为基于反馈的循环量子神经网络 (RQNN) 建立了定量逼近误差界和普适性证明，表明 RQNN 可在 qubit 数仅以 $\lceil\log_2(\varepsilon^{-1})\rceil$ 对数增长的条件下，以线性读出层逼近任意 fading memory 滤波器，且不受维度灾难影响。
tags:
  - "ICLR2026"
  - "物理/科学计算"
  - "quantum reservoir computing"
  - "recurrent quantum neural network"
  - "universal approximation"
  - "fading memory filter"
  - "NISQ"
---

# Feedback-driven Recurrent Quantum Neural Network Universality

**会议**: ICLR2026  
**arXiv**: [2506.16332](https://arxiv.org/abs/2506.16332)  
**代码**: 无  
**领域**: 物理学  
**关键词**: quantum reservoir computing, recurrent quantum neural network, universal approximation, fading memory filter, NISQ  

## 一句话总结

本文首次为基于反馈的循环量子神经网络 (RQNN) 建立了定量逼近误差界和普适性证明，表明 RQNN 可在 qubit 数仅以 $\lceil\log_2(\varepsilon^{-1})\rceil$ 对数增长的条件下，以线性读出层逼近任意 fading memory 滤波器，且不受维度灾难影响。

## 背景与动机

- **量子储层计算 (QRC)** 利用量子系统动力学处理时序数据，尤其适合 NISQ 设备；已有大量实证成功，但理论基础薄弱
- 经典循环神经网络 (RNN) 的万能逼近定理已有深入研究 (Hornik 1991, Barron 1993, Grigoryeva & Ortega 2018)，但 **量子 RNN 缺乏定量逼近界**
- 此前 QRC 的万能性证明依赖 **多项式读出层**（利用 Stone-Weierstrass 定理），但实际系统普遍使用 **线性读出层**，因其训练简单快速
- 反馈协议 (feedback protocol) 通过将输出状态反馈到下一时刻输入，使系统能用更少组件保留输入历史，支持实时计算，但其逼近能力此前缺乏理论保障

## 核心问题

1. 反馈驱动的 RQNN 能否以 **可控的量子资源**（qubit 数、电路大小）逼近一般性状态空间系统？
2. RQNN 族是否对 **任意因果、时不变、fading memory 滤波器** 具有万能逼近性质？
3. 能否在仅使用 **线性读出层** 的条件下保持万能性？

## 方法详解

### 整体框架

RQNN 把状态向量的每个分量交给一个并行量子电路计算，电路输出经反馈 $\hat{\bm{x}}_t = \bar{F}_R^{n,\bm{\theta}}(\hat{\bm{x}}_{t-1}, \bm{z}_t)$ 回灌为下一时刻的状态，从而把一个量子线路变成处理时序的循环系统。论文的核心不是设计新硬件，而是证明这套带反馈的循环量子网络能以可控的 qubit 数、配上一个线性读出层，逼近任意 fading memory 滤波器，并给出显式的误差衰减速率。

### 关键设计

**1. 余弦基量子读出：把电路测量结果写成可逼近的解析形式**

RQNN 的单个电路由初始化门 $\mathtt{V}$ 和参数化量子门 $\mathtt{U}$ 组成。$\mathtt{V}$ 先把 $|0\rangle^{\otimes \mathfrak{n}}$ 映射到均匀叠加态 $|\psi\rangle = \frac{1}{\sqrt{n}}\sum_{i=0}^{n-1}|i\rangle \otimes |00\rangle$，$\mathtt{U}$ 则是由 $n$ 个旋转块 $\bar{\mathtt{U}}^{(i)}$ 以块对角形式拼成的 uniformly controlled gate，每块把依赖状态 $\bm{x}$ 与输入 $\bm{z}$ 的编码门 $\mathtt{U}_1^{(i)}$ 和偏置门 $\mathtt{U}_2^{(i)}$ 张量积起来。测量目标 qubit 的概率后，状态映射可写成 $\bar{F}_{R,j}^{n,\bm{\theta}}(\bm{x},\bm{z}) = R - 2R[\mathbb{P}_1^{n,\bm{\theta}^j} + \mathbb{P}_2^{n,\bm{\theta}^j}]$，它恰好等价于一个余弦基展开 $\frac{1}{n}\sum_{i=1}^n R\cos(\gamma^{i,j})\cos(b^{i,j} + \bm{a}^{i,j}\cdot(\bm{x},\bm{z}))$。这一步是整套理论的支点：把量子测量结果显式写成 $n$ 个余弦特征的平均，逼近问题就转化为经典的随机特征逼近，可以套用 Barron 型分析。

**2. 对数级量子资源：用精度参数控制 qubit 与权重的增长**

块数 $n$ 是精度旋钮，电路只需作用在 $\mathfrak{n} = \lceil\log_2(2n)\rceil$ 个 qubit 上，所以 qubit 数随精度仅对数增长。要达到逼近误差 $\varepsilon$，整网需要 $\mathcal{O}(\varepsilon^{-2})$ 个可训练权重和 $\mathcal{O}(\lceil\log_2(\varepsilon^{-1})\rceil)$ 个 qubit。对数级的 qubit 需求正是这套架构对 NISQ 设备友好的关键——同样的精度下，经典储层往往要付出多项式级的资源。

**3. 导数可控的反馈误差传播：让循环不放大逼近误差**

反馈结构最难处理的地方在于，单步逼近的微小误差会沿时间轴累积。论文对满足 Barron 型可积条件、收缩系数 $\lambda < 1$ 的状态映射 $F$ 给出定理 4.6 的均匀逼近界 $\sup_{\bm{z}}\sup_t \|U^F(\bm{z})_t - \bar{U}(\bm{z})_t\| \leq \frac{1}{1-\lambda}\frac{\sqrt{N}\max_j C_j^\infty}{\sqrt{n}}$。误差以 $1/\sqrt{n}$ 衰减，且分子里只出现 $\sqrt{N}$ 而非 $N$、$d$ 的指数项，因此**与输入维度 $d$ 和状态维度 $N$ 无关**，避开了维度灾难。能做到这一点靠的是 Proposition 4.4：QNN 不仅逼近目标函数本身，还同时逼近其导数，于是反馈回路里由 Jacobian 控制的误差放大被压住，$\frac{1}{1-\lambda}$ 这个收缩因子才得以封顶整条时间链上的累积误差。

**4. 线性读出即万能：去掉多项式读出层的实现负担**

此前 QRC 的万能性证明依赖多项式读出（借 Stone-Weierstrass），训练复杂、实验难落地。定理 4.8 证明只用线性读出 $W$ 就够：对**任意因果、时不变、fading memory 滤波器** $U$，存在 RQNN 参数与线性 $W$ 使 $\sup_{\bm{z}}\sup_t \|U(\bm{z})_t - \bar{U}_W(\bm{z})_t\| \leq \varepsilon$。这一步既不需要 Barron 可积条件、也不需要收缩性假设，代价是额外引入线性预处理矩阵 $P_j$ 并对记忆做有限步分区，以保证 echo state property（系统对足够久远的初始状态不敏感）。整套证明走的是 internal approximation approach：先建立 QNN 对静态函数及其导数的逼近界，再用导数界压住反馈回路中的误差累积，最后从状态映射逼近推出滤波器逼近。

## 实验关键数据

本文为纯理论工作，无数值实验。核心定量结果为逼近误差界中的常数与收敛速率。

## 亮点

- **首个 RQNN 定量逼近界**：填补了量子 RNN 逼近理论的空白
- **无维度灾难**：误差率 $1/\sqrt{n}$ 与输入/状态维度无关，qubit 数仅对数增长
- **线性读出即万能**：不需要多项式读出层，大幅降低实验实现难度
- **比经典 RNN 条件更弱**：所需 Sobolev 光滑性条件 $s > \frac{N+d}{2}+4$，弱于经典 RNN 所需的 $s > N+d+3$
- **与硬件兼容**：基于 uniformly controlled gates 的电路已有高效分解方案和 Rydberg 原子实现

## 局限与展望

- **仅限理论分析**：缺少数值验证，无法评估实际 NISQ 设备上的表现
- **Barron 型条件限制**：定量界仅适用于傅里叶变换充分可积的函数，对粗糙或非收缩动力学尚无误差率
- **Barren plateau 问题未讨论**：变分量子电路训练的梯度消失问题可能影响实际可训练性
- **Monte Carlo 采样误差**：量子测量的有限采样误差仅在附录中简要讨论，未纳入主逼近界
- **未与随机储层比较**：结论仅适用于全参数可训练的变分设置，对参数部分随机的真正 QRC 尚未推广

## 与相关工作的对比

| 方法 | 线性读出万能性 | 定量误差界 | 无维度灾难 | 反馈/时序 |
|------|:---:|:---:|:---:|:---:|
| 经典 ESN (Grigoryeva & Ortega 2018) | ✗ | ✗ | - | ✓ |
| 经典 RNN (Gonon et al. 2023) | ✓ | ✓ | ✓ | ✓ |
| 前馈 QNN (Gonon & Jacquier 2025) | - | ✓ | ✓ | ✗ |
| QRC 多项式读出 (Sannia et al. 2024) | ✗ | ✗ | - | ✓ |
| **本文 RQNN** | **✓** | **✓** | **✓** | **✓** |

相比经典 RNN，RQNN 对目标函数的光滑性要求更低；相比前馈 QNN，本文处理了反馈回路带来的额外分析难度；相比已有 QRC 万能性结果，本文首次实现线性读出 + 定量界。

## 启发与关联

- 为量子储层计算提供了坚实的理论基础，未来可结合泛化误差界 (generalization bound) 建立完整的学习理论
- RQNN 的余弦基展开形式 (Proposition 4.1) 暗示了与经典随机特征方法的深层联系
- 线性预处理 + 有限记忆分区保证 echo state property 的技巧可能对设计实用 QRC 架构有指导意义
- Barren plateau 与表达能力之间的权衡是落地的关键瓶颈，值得后续深入研究

## 评分

- 新颖性: ⭐⭐⭐⭐☆ — 首次为反馈驱动 RQNN 建立完整逼近理论
- 实验充分度: ⭐⭐☆☆☆ — 纯理论工作，无实验验证
- 写作质量: ⭐⭐⭐⭐☆ — 结构清晰，主要结果陈述精确
- 价值: ⭐⭐⭐⭐☆ — 为量子机器学习时序任务奠定理论基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Test-Time Accuracy-Cost Control in Neural Simulators via Recurrent-Depth](test-time_accuracy-cost_control_in_neural_simulators_via_recurrent-depth.md)
- [\[ICLR 2026\] Accelerating Inference for Multilayer Neural Networks with Quantum Computers](accelerating_inference_for_multilayer_neural_networks_with_quantum_computers.md)
- [\[ICLR 2026\] A Function-Centric Graph Neural Network Approach for Predicting Electron Densities](a_function-centric_graph_neural_network_approach_for_predicting_electron_densiti.md)
- [\[NeurIPS 2025\] Neural Network for Simulating Radio Emission from Extensive Air Showers](../../NeurIPS2025/physics/neural_network_for_simulating_radio_emission_from_extensive_air_showers.md)
- [\[NeurIPS 2025\] Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding](../../NeurIPS2025/physics/physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)

</div>

<!-- RELATED:END -->
