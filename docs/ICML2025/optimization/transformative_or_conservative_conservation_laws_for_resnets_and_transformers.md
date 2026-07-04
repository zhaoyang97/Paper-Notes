---
title: >-
  [论文解读] Transformative or Conservative? Conservation Laws for ResNets and Transformers
description: >-
  [ICML2025][优化/理论][conservation laws] 系统推导并证明了卷积 ResNet 和 Transformer 等现代架构在梯度流训练动态下的守恒律，揭示残差连接不改变守恒律、块级守恒律等价于孤立块的守恒律，并证明离散 SGD 下守恒误差为 $O(\text{step-size}^2)$。
tags:
  - "ICML2025"
  - "优化/理论"
  - "conservation laws"
  - "梯度流"
  - "ResNet"
  - "Transformer"
  - "训练动态"
  - "隐式偏差"
  - "Lie algebra"
---

# Transformative or Conservative? Conservation Laws for ResNets and Transformers

**会议**: ICML2025  
**arXiv**: [2506.06194](https://arxiv.org/abs/2506.06194)  
**代码**: 待确认  
**领域**: 优化  
**关键词**: conservation laws, 梯度流, ResNet, Transformer, 训练动态, 隐式偏差, Lie algebra

## 一句话总结
系统推导并证明了卷积 ResNet 和 Transformer 等现代架构在梯度流训练动态下的守恒律，揭示残差连接不改变守恒律、块级守恒律等价于孤立块的守恒律，并证明离散 SGD 下守恒误差为 $O(\text{step-size}^2)$。

## 研究背景与动机

**现状**：守恒律（训练过程中保持不变的量）是理解神经网络训练动态的重要工具。对于浅层 ReLU 和线性网络，已知的守恒律形如 $\|u_k\|^2 - \|v_k\|^2 = \text{const}$（"平衡性条件"），且 Marcotte et al. (2023) 证明了其完备性。

**痛点**：
1. 现有理论仅覆盖浅层的 ReLU/线性网络，**卷积网络、注意力层、深层 ResNet 和 Transformer 的守恒律完全未知**
2. 守恒律揭示隐式偏差（从初始化到最终解保持的性质）和收敛性，对现代架构的缺失限制了理论理解
3. 梯度流下的守恒律在实际离散 SGD 训练中是否仍然近似成立尚不清楚

**Idea**：利用 Lie 代数框架和重参数化技术，系统化地推导卷积层、注意力层、残差块等现代架构基本构件的守恒律，再通过"块级守恒律"的概念将分析推广到深层网络。

## 方法详解

### 理论框架

考虑梯度流训练动态（带可选 weight decay）：

$$\dot{\theta}(t) + \lambda(t)\theta(t) = -\nabla L_Z(\theta(t))$$

**Structure Theorem（Theorem 2.1）**：带 weight decay 的守恒律可由不带 weight decay 的守恒律完全确定：$h(t,\theta) = H(\theta \exp(\int_0^t \lambda(s)ds))$，从而将分析简化为时间无关的守恒函数。

**守恒律的刻画（Proposition 2.3）**：光滑函数 $h(\theta)$ 是守恒律当且仅当 $\nabla h(\theta) \perp \mathcal{W}_\theta^{g,\ell}$，其中 $\mathcal{W}_\theta^{g,\ell}$ 是梯度在参数空间中的张成子空间。

**守恒律计数（Theorem 2.11）**：独立守恒律的数目精确等于 $D - k$，其中 $D$ 是参数维度，$k$ 是 Lie 代数 $\text{Lie}(\mathbb{W}^{g,\ell})(\theta)$ 的维度。

### 基本构件的守恒律

**1. 残差连接不改变守恒律（Proposition 3.2）**：
$\tilde{g}(\theta, x) = x + g(\theta, x)$ 与 $g(\theta, x)$ 具有完全相同的守恒律。证明极其简洁：$\partial_\theta g = \partial_\theta \tilde{g}$。

**2. 多通道卷积 ReLU 网络（Theorem 3.6）**：
$c_1$ 个隐藏通道的网络恰好有 $c_1$ 个独立守恒律：

$$h_j(\theta) = \sum_{k=1}^{c_2}\|u_{k,j}\|^2 - \sum_{i=1}^{c_0}\|v_{j,i}\|^2, \quad 1 \leq j \leq c_1$$

**3. 单头注意力层（Corollary 3.9）**：
对于 $g(\theta,x) = \text{softmax}(XQ^\top KX^\top)XV^\top O$，所有守恒律均为以下量的函数：
$$QQ^\top - KK^\top, \quad VV^\top - OO^\top$$

**4. 多头注意力（Corollary 3.10）**：
各头独立守恒：$Q_h Q_h^\top - K_h K_h^\top$ 和 $V_h V_h^\top - O_h O_h^\top$（完备性留为 open problem）。

**5. Cross-Entropy 分类层（Proposition 3.11）**：
Softmax 层有 $m$ 个守恒律：$h_j(\theta) = \sum_i \theta_{i,j}$（各列权重之和守恒）。

### 深层网络的块级守恒律

**核心定理（Theorem 4.6）**：深层网络 $g_\theta$ 中仅依赖于第 $l$ 层参数 $\theta_l$ 的守恒律，恰好等价于该层浅层网络 $g_{\theta_l}^l$ 的守恒律（关于欧氏损失）。

**跨残差连接的块（Theorem 4.7）**：跨越一个 skip connection 的相邻参数对 $(V^{l+1}, U^l)$ 不存在非平凡守恒律——残差连接"打破"了跨块守恒。

### 离散 SGD 近似守恒（Proposition 5.1）

SGD 下守恒误差的上界：
$$\mathbb{E}|h(\theta_k) - h(\theta_0)| \leq \frac{C_h C_L}{2}\sum_{i=0}^{k-1}\tau_i^2$$

常数步长 $\tau$ 下为 $O(\tau^2 k)$，衰减步长 $\tau_k = \tau_0/(k+1)$ 下保持有界 $O(\tau_0^2)$。

### Adam 流的守恒律分析

简化版 Adam（sign gradient descent）的守恒律空间为 $\text{span}\{\text{sign}(\nabla L_Z(\theta))\}$，与梯度流根本不同。对二层线性网络，数值发现（除 $n=m=r=1$ 外）不存在守恒律。

## 实验关键数据

### ResNet-18 / CIFAR-10 实验

| 设置 | 观测量 | 结论 |
|------|--------|------|
| SGD, 学习率 $\in [10^{-3}, 5\times10^{-3}]$, 无动量/WD | 第一残差块 $\sum_j h_j(\theta_T)$ 的变化 | 守恒误差斜率与 $\tau^2$ 成正比 |
| 10 个随机种子 × 多组学习率 | 理论斜率 $C\tau^2$ vs 实际 | 理论预测与实验吻合 |
| 50 步训练中守恒误差 | 相对误差 $|h(\theta_k)-h(\theta_0)|/|h(\theta_0)|$ | 在合理学习率下守恒良好 |

### Transformer / IMDb 实验

- 在 IMDb 情感分析上训练 Transformer，跟踪第一层第一个注意力头的 $\|QQ^\top - KK^\top\|_F$
- 守恒误差同样遵循 $O(\text{step-size}^2)$ 规律
- 有无 masking 对守恒行为无影响

### 数值验证：深层网络无额外守恒律

- 对 $q=2$ 残差块的 ResNet，在 $m > 1$ 时数值计算确认无超出块级守恒律的额外守恒律
- $m=1$ 时存在额外守恒律（Example 4.8：两块 ReLU 网络在特定符号条件下有 3 个而非 2 个守恒律）

## 亮点与洞察

1. **理论优美性**：用 Lie 代数框架统一处理各种架构的守恒律，证明简洁有力
2. **Proposition 3.2 的简洁**：残差连接不影响守恒律的证明仅一行（$\partial_\theta g = \partial_\theta \tilde{g}$），但结论意义深远
3. **块级分析的可组合性**：Theorem 4.6 将深层网络分析还原为浅层构件分析，大幅降低复杂度
4. **Theorem 4.7 的负结果**：跨残差连接无守恒律，说明 skip connection 从优化动力学角度"解耦"了相邻层
5. **连续到离散的桥梁**：Proposition 5.1 量化了 SGD 下的守恒破坏程度，增强了理论的实用价值
6. **Weight decay 的 Structure Theorem**：优雅地证明了 WD 下的守恒律在最优点处趋于零（对应已知的平衡性）

## 局限与展望

1. **未覆盖 LayerNorm**：Transformer 中的归一化层被忽略，这是实际模型中不可或缺的组件
2. **多头注意力完备性**：Corollary 3.10 中多头守恒律的完备性未证明，留为 open problem
3. **Max-Pooling 缺失**：CNN 中常用的 max-pooling 未纳入分析
4. **SGD 误差界的常数**：$C_h, C_L$ 在实践中难以显式确定，限制了定量预测能力
5. **Adam 优化器**：仅分析了简化版 Adam，全版本 Adam 的守恒律未覆盖
6. **$m=1$ 特殊情况**：揭示了额外守恒律的存在，但通用条件的刻画未完成

## 相关工作与启发

- **Marcotte et al. (2023)**：浅层 ReLU/线性网络守恒律的完备性分析框架，本文的直接基础
- **Du et al. (2018)**：单通道卷积网络的守恒律，本文推广到多通道并证明完备
- **Marion et al. (2023)**：ResNet 训练解对应 Neural ODE 离散化的隐式偏差
- **Vasudeva et al. (2024)**：自注意力层梯度下降收敛到 hard-margin SVM 解
- **Noether 定理类比**：守恒律与网络不变性（隐藏神经元重缩放）的内在联系

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (首次系统推导 ResNet/Transformer 的守恒律并证明完备性)
- 实验充分度: ⭐⭐⭐⭐ (理论验证充分，但实际训练场景覆盖有限)
- 写作质量: ⭐⭐⭐⭐⭐ (理论框架清晰，层次递进，证明组织优秀)
- 价值: ⭐⭐⭐⭐⭐ (为深度学习优化理论提供了基础性工具)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Conservation Laws for Modern Neural Architectures](../../ICML2026/optimization/conservation_laws_for_modern_neural_architectures.md)
- [\[ICML 2025\] How Transformers Learn Regular Language Recognition: A Theoretical Study on Training Dynamics and Implicit Bias](how_transformers_learn_regular_language_recognition_a_theoretical_study_on_train.md)
- [\[ICML 2025\] Can Transformers Learn Full Bayesian Inference In Context?](can_transformers_learn_full_bayesian_inference_in_context.md)
- [\[NeurIPS 2025\] Multi-head Transformers Provably Learn Symbolic Multi-step Reasoning via Gradient Descent](../../NeurIPS2025/optimization/multi-head_transformers_provably_learn_symbolic_multi-step_reasoning_via_gradien.md)
- [\[ICML 2025\] In-Context Linear Regression Demystified: Training Dynamics and Mechanistic Interpretability of Multi-Head Softmax Attention](in-context_linear_regression_demystified_training_dynamics_and_mechanistic_inter.md)

</div>

<!-- RELATED:END -->
