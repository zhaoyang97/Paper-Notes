---
title: >-
  [论文解读] How Transformers Learn Regular Language Recognition: A Theoretical Study on Training Dynamics and Implicit Bias
description: >-
  [ICML2025][优化/理论][Transformer] 从理论上刻画了一层 Transformer 学习 "even pairs" 和 "parity check" 两类正则语言识别任务时的两阶段训练动力学，证明了线性层在梯度下降下隐式收敛到最大间隔超平面，并揭示了 CoT 在解决 parity 问题中的关键作用。
tags:
  - "ICML2025"
  - "优化/理论"
  - "Transformer"
  - "训练动力学"
  - "隐式偏置"
  - "正则语言识别"
  - "Chain-of-Thought"
  - "最大间隔"
---

# How Transformers Learn Regular Language Recognition: A Theoretical Study on Training Dynamics and Implicit Bias

**会议**: ICML2025  
**arXiv**: [2505.00926](https://arxiv.org/abs/2505.00926)  
**代码**: 无  
**领域**: 优化  
**关键词**: Transformer理论, 训练动力学, 隐式偏置, 正则语言识别, Chain-of-Thought, 最大间隔

## 一句话总结

从理论上刻画了一层 Transformer 学习 "even pairs" 和 "parity check" 两类正则语言识别任务时的两阶段训练动力学，证明了线性层在梯度下降下隐式收敛到最大间隔超平面，并揭示了 CoT 在解决 parity 问题中的关键作用。

## 研究背景与动机

- **形式语言识别** 是 NLP 的基础基准任务，被广泛用于衡量 LLM 的推理能力，也是理解 Transformer 内部机制的重要切入点
- 已有大量工作研究 Transformer 在形式语言上的**表达能力**和**可学性**，但对其**训练动力学**（参数如何在梯度下降下演化）的理论分析几乎空白
- 具体关注两个代表性任务：
    - **Even pairs**：判断二元序列中 "ab" 和 "ba" 子序列总数是否为偶数（等价于首尾 token 是否相等）
    - **Parity check**：判断序列中 "b" 的个数是否为偶数
- 已有 parity check 理论工作（Kim & Suzuki 2024b; Wen et al. 2024）仅分析单独的注意力层或有限步训练，缺乏对注意力层+线性层**联合训练**的完整收敛性分析

## 方法详解

### 模型架构

采用**一层 Transformer**：注意力层 + 线性层，形式化为：

$$\mathtt{T}_\theta(X) = u^\top \sum_{\ell=1}^{L} x_\ell \varphi_\ell$$

其中 $\varphi_\ell = [\phi(X^\top W x_L / \lambda)]_\ell$ 是 softmax 注意力权重，$\lambda$ 为缩放因子；可训练参数 $\theta = (u, W)$，$u$ 是线性层，$W$ 是注意力层。

### 训练目标

采用 logistic 损失进行二分类：

$$\mathcal{L}(u,W) = \sum_{L=1}^{L_{\max}} \frac{1}{|I_L|} \sum_{n \in I_L} \log\left(1 + \exp(-y_n \mathtt{T}_\theta(X^{(n)}))\right)$$

使用两阶段梯度下降：Phase 1 中注意力层学习率额外乘以 $\lambda$，Phase 2 恢复标准 GD。

### 关键理论结果：Even Pairs 的两阶段动力学

**Phase 1（快速增长阶段）**：

- **线性层**：第一个 token 分数 $\langle u_t, E_1^w \rangle = \Theta(\eta t)$ 快速增长；非首位 token 分数 $\langle u_t, E_\ell^w \rangle = -\Theta(\eta^2 t^2)$ 快速下降
- **注意力层**：正样本中首 token 获得最高注意力；负样本中第二个 token 获得最高注意力
- Phase 1 结束时，注意力层输出满足**可分性**（Proposition 4.3）

**Phase 2（间隔最大化阶段）**：

- 注意力层几乎不变：$\|W_t - W_{t_0}\| \leq O(1)$
- 线性层范数对数增长：$\|u_t\| \geq \Omega(\log t)$
- 线性层方向收敛到 **最大间隔超平面** $u_{EP}^*$（Theorem 4.4）
- 损失以 $O(L_{\max}\|u_{EP}^*\|^2 / (\eta\sqrt{t}))$ 速率收敛（Theorem 4.5）

### Parity Check 的两种 CoT 方法

**方法一：截断 CoT 推理（零样本）**

- 利用 even pairs ≈ 判断首尾 token 是否相等的等价关系
- 已训练好的 even pairs Transformer 通过迭代式截断 CoT（每步比较首尾、追加预测、移除首 token）直接解 parity，**无需额外训练**

**方法二：Teacher Forcing 下的 CoT 训练**

- 训练损失 = CoT 损失 + Even Pairs 正则化损失：$\mathcal{L}_{Parity} = \mathcal{L}_{CoT} + \mathcal{L}_{Reg}$
- Even pairs 损失作为正则化稳定训练（直接训练 CoT 会导致梯度消失）
- 同样展现两阶段动力学，线性层收敛到 CoT 对应的最大间隔解 $u_{CoT}^*$

### 关键分析技术

- 采用**高阶 Taylor 展开**精确分析两层梯度的耦合效应
- 利用**隐式偏置**理论刻画 Phase 2 的收敛方向
- 精心设计缩放因子 $\lambda$，在 Phase 2 抑制注意力层更新，确保训练稳定

## 实验关键数据

### 实验设置

| 参数 | 值 |
|------|------|
| $L_{\max}$ | 6 |
| $L_0$（parity） | 4 |
| 学习率 $\eta$ | 0.1 |
| Phase 1 步数 $t_0$ | 100 |
| 缩放因子 $\lambda$ | 2 |
| 设备 | i5-12400F + 16GB |

### 主要实验发现

| 现象 | Even Pairs | Parity Check |
|------|-----------|-------------|
| 损失快速衰减 | ✓ | ✓ |
| 首 token 分数增长 | ✓ | ✓ |
| 非首位 token 分数下降 | ✓ | ✓ |
| 正样本首 token 注意力更高 | ✓ | ✓ |
| Phase 2 后 $\|u_t\|$ 对数增长 | ✓（$t_2 \approx 600$）| ✓（$t_2 \approx 600$）|

### 消融实验

- 不同 $\lambda$ 配置下验证了两阶段训练动力学的一致性
- Phase 1 和 Phase 2 的边界与理论预测吻合

## 亮点与洞察

1. **首次为 even pairs 问题建立完整的训练动力学理论**，填补了该领域空白
2. **联合训练分析**：不同于仅分析注意力层的已有工作，本文同时分析了注意力层和线性层的耦合训练过程
3. **令人惊讶的零样本 CoT 结果**：在 even pairs 上训练的 Transformer 可以零样本解决 parity check，揭示了两个任务之间的深层联系
4. **隐式偏置 → 最大间隔**：将分类任务中经典的隐式偏置理论扩展到具有独特结构的语言识别任务
5. **缩放因子的关键角色**：$\lambda$ 不仅稳定训练，更是控制两阶段转换的核心机制

## 局限与展望

- 仅分析了**一层 Transformer**，多层情况未涉及
- 序列最大长度 $L_{\max}=6$ 的实验规模较小，未验证对长序列的扩展性
- 理论分析依赖正交 embedding 假设，实际 token embedding 通常非正交
- 未讨论更复杂的上下文无关语言或更一般的自动机任务
- Phase 1 到 Phase 2 的过渡条件（$t_0$ 的选择）依赖超参数设定

## 相关工作与启发

- **训练动力学**：Tarzanagh et al. (2023) 证明注意力训练等价于 SVM；Vasudeva et al. (2024) 建立了收敛速率
- **CoT 理论**：Kim & Suzuki (2024b) 分析了 CoT 下注意力模型学 parity；Wen et al. (2024) 研究了样本复杂度但未建立完整收敛
- **隐式偏置**：Soudry et al. (2018) 开创了分类中 GD 隐式偏置理论；Huang et al. (2024a) 扩展到 NTP 任务
- 本文为 Transformer 理论分析提供了新的分析工具（高阶 Taylor + 隐式偏置组合），可推广到其他结构化学习任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次完整刻画 even pairs 训练动力学，零样本 CoT 结果新颖
- 实验充分度: ⭐⭐⭐ 合成实验验证理论，但规模偏小
- 写作质量: ⭐⭐⭐⭐ 理论清晰，两阶段结构条理分明
- 价值: ⭐⭐⭐⭐ 为理解 Transformer 训练机制和 CoT 原理提供重要理论基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Can Transformers Learn Full Bayesian Inference In Context?](can_transformers_learn_full_bayesian_inference_in_context.md)
- [\[NeurIPS 2025\] Multi-head Transformers Provably Learn Symbolic Multi-step Reasoning via Gradient Descent](../../NeurIPS2025/optimization/multi-head_transformers_provably_learn_symbolic_multi-step_reasoning_via_gradien.md)
- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](../../NeurIPS2025/optimization/the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [\[NeurIPS 2025\] Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study](../../NeurIPS2025/optimization/do_neural_networks_need_gradient_descent_to_generalize_a_theoretical_study.md)
- [\[ICML 2025\] Training Dynamics of In-Context Learning in Linear Attention](training_dynamics_of_in-context_learning_in_linear_attention.md)

</div>

<!-- RELATED:END -->
