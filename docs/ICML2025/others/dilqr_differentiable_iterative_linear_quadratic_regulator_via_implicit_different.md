---
title: >-
  [论文解读] DiLQR: Differentiable Iterative Linear Quadratic Regulator via Implicit Differentiation
description: >-
  [ICML 2025][可微分控制] 本文提出 DiLQR 框架，通过在 iLQR 控制器的不动点上施加隐式微分，得到解析梯度解，将反向传播的计算复杂度从随迭代数线性增长降为 $O(1)$ 常数，实现最高 128× 加速，同时学习性能比传统神经网络策略提升 $10^6$ 倍。 领域现状：可微分控制是结合模型无关（model-…
tags:
  - "ICML 2025"
  - "可微分控制"
  - "iLQR"
  - "隐式微分"
  - "不动点微分"
  - "模仿学习"
---

# DiLQR: Differentiable Iterative Linear Quadratic Regulator via Implicit Differentiation

**会议**: ICML 2025  
**arXiv**: [2506.17473](https://arxiv.org/abs/2506.17473)  
**代码**: [https://sites.google.com/view/dilqr/](https://sites.google.com/view/dilqr/) (项目主页)  
**领域**: LLM评测  
**关键词**: 可微分控制, iLQR, 隐式微分, 不动点微分, 模仿学习

## 一句话总结

本文提出 DiLQR 框架，通过在 iLQR 控制器的不动点上施加隐式微分，得到解析梯度解，将反向传播的计算复杂度从随迭代数线性增长降为 $O(1)$ 常数，实现最高 128× 加速，同时学习性能比传统神经网络策略提升 $10^6$ 倍。

## 研究背景与动机

**领域现状**：可微分控制是结合模型无关（model-free）灵活性和模型驱动（model-based）高效性的新范式。迭代线性二次调节器（iLQR）作为强大的数值控制器，在轨迹优化中广泛使用，但其可微分版本的发展滞后于 LQR。

**现有痛点**：要将 iLQR 作为可训练的神经网络模块，朴素的自动微分（AutoDiff）方法需要通过整个展开的迭代链进行反向传播。前向传播迭代数百次求解 LQR 优化问题，反向传播必须遍历所有这些层，导致内存占用和计算时间随迭代次数和时间步长度线性增长，严重限制了可扩展性。

**核心矛盾**：iLQR 的前向计算需要多次迭代才能收敛到最优轨迹，但 AutoDiff 将前向和反向传播耦合在一起——迭代次数越多、horizon 越长，反向传播就越慢越耗内存。DiffMPC 虽然提出了解析梯度方法，但它将 iLQR 最后一层的输入视为常数而非学习参数的函数，导致梯度不精确。

**本文目标** 如何高效且精确地计算 iLQR 控制器关于可学习参数 $\theta$ 的梯度，使其可以作为端到端学习框架中的可微分模块？

**切入角度**：iLQR 收敛后的轨迹是一个不动点——输入自身就是输出。在不动点上可以用隐式函数定理直接求解 $\partial \tau^*/\partial \theta$，无需展开迭代过程。这将反向传播的计算复杂度完全解耦于前向迭代次数。

**核心 idea**：通过在 iLQR 不动点上施加隐式微分，得到精确的解析梯度解，将反向传播复杂度从 $O(\text{迭代数})$ 降为 $O(1)$。

## 方法详解

### 整体框架

DiLQR 将 iLQR 封装为可微分模块。前向传播：正常运行 iLQR 迭代求解最优轨迹 $\tau^*$。反向传播：不展开迭代链，而是利用不动点条件 $\tau^* = \text{iLQR}(\tau^*, \theta)$ 直接通过隐式微分计算 $\partial \tau^*/\partial \theta$。这个模块可以嵌入更大的神经网络中（如与视觉编码器组合）实现端到端学习。

### 关键设计

1. **不动点隐式微分（Fixed-Point Implicit Differentiation）**:

    - 功能：在不展开迭代过程的前提下，精确计算最优轨迹关于参数的梯度
    - 核心思路：在不动点 $X^* = F(X^*, U^*, \theta)$, $U^* = G(X^*, U^*, \theta)$ 上对 $\theta$ 全微分，得到线性方程组 $(I - F_X)\nabla_\theta X^* - F_U \nabla_\theta U^* = F_\theta$。求解得到解析式：$\nabla_\theta X^* = M(F_\theta + F_U(K - G_X M F_U)^{-1}(G_X M F_\theta - G_\theta))$，其中 $M = (I - F_X)^{-1}$，$K = I - G_U$
    - 设计动机：相比 DiffMPC 将不动点轨迹视为常数只取偏导 $\partial A^i / \partial \theta$，DiLQR 正确处理了全导数 $\nabla_\theta A^i = \partial A^i / \partial \theta + \partial A^i / \partial \tau^i \cdot \partial \tau^i / \partial \theta$（方框中的项），使梯度更精确

2. **前向梯度传播算法（Forward Algorithm）**:

    - 功能：高效计算线性化动力学矩阵 $D_t$ 关于参数 $\theta$ 的导数
    - 核心思路：利用时间步之间的递推关系，$\nabla_\theta x_t$ 可以从 $\nabla_\theta x_{t-1}$ 递推得到：$\nabla_\theta x_t = \partial x_t / \partial \theta + [\partial x_t / \partial x_{t-1} + \partial x_t / \partial u_{t-1} \cdot \partial u_{t-1} / \partial x_{t-1}] \nabla_\theta x_{t-1}$。各时间步的偏导数可以预先解析计算，运行时只需代入数值
    - 设计动机：PyTorch 的 torch.autograd.jacobian 不复用时间步之间的梯度信息，导致大量重复计算。前向算法利用递推关系将计算量降低数十倍

3. **稀疏性利用与并行化（Sparsity & Parallelization）**:

    - 功能：利用问题的结构稀疏性和计算的独立性来加速
    - 核心思路：(a) 稀疏性：$\partial D / \partial X$ 是块对角矩阵（$\partial D_t / \partial x_{t'}=0$ 当 $t' \neq t$），无需实例化完整矩阵，只用对角块；(b) 并行化：构造二值损失函数的 batch 来并行计算 $\partial H_{i,j} / \partial D$，将 $L_{i,j}$ 设为 1 其余设为 0，各元素的计算完全独立可并行
    - 设计动机：隐式微分的解析解涉及大雅可比矩阵运算，如果不利用问题结构，计算量仍然很大

### 损失函数 / 训练策略

模仿学习损失：$L(\tau(\theta))$，最小化预测轨迹与专家轨迹的差距。总梯度通过链式法则 $\nabla_\theta(L \circ \tau)(\theta) = \nabla_\tau L(\tau(\theta)) \cdot \partial \tau / \partial \theta$ 计算，其中 $\nabla_\tau L$ 由自动微分提供，$\partial \tau / \partial \theta$ 由 DiLQR 的解析解提供。

## 实验关键数据

### 主实验（计算效率）

| Horizon | iLQR迭代数 | AutoDiff 时间 (s) | DiLQR 时间 (s) | 加速比 |
|---------|----------|------------------|---------------|-------|
| 10 | 50 | 1.41 | 0.067 | 21× |
| 10 | 300 | 8.57 | 0.067 | **128×** |
| 30 | 50 | ~4 | ~0.2 | ~20× |
| 30 | 300 | ~25 | ~0.2 | ~125× |

### 模仿学习性能

| 任务 | 方法 | 模仿损失 | 相对NN提升 |
|------|------|---------|----------|
| Pendulum | NN (LSTM) | ~1e-1 | 基线 |
| Pendulum | DiLQR.dx | ~1e-7 | **10^6×** |
| Cartpole | NN (LSTM) | ~1e-1 | 基线 |
| Cartpole | DiLQR.dx | ~1e-5 | **10^4×** |

### 消融实验

| 配置 | Horizon=10 时间 | Horizon=30 时间 | 说明 |
|------|----------------|----------------|------|
| Full DiLQR | 最快 | 最快 | 所有优化开启 |
| w/o Forward Algorithm | 显著增加 | 急剧增加 | 前向算法贡献最大 |
| w/o Parallelization | 进一步增加 | 长 horizon 更慢 | 并行化在长 horizon 关键 |
| w/o Sparsity | 略有增加 | 略有增加 | 稀疏利用的影响较小 |

### 模型损失与物理一致性

| 指标 | DiLQR | DiffMPC | 说明 |
|------|-------|---------|------|
| dcost 模型损失 | 降低 32% | 基线 | 成本函数参数恢复更准确 |
| dx 模型损失 (train=50) | 最终更低 41% | 较早平稳 | 动力学参数学习更持续优化 |
| 负值参数比例 (train=100) | 2.76% | 16.85% | 物理一致性远优于 DiffMPC |
| 负值参数比例 (train=50) | 7.23% | 17.82% | 即使数据少也保持物理合理性 |

### 关键发现

- DiLQR 的反向传播时间关于迭代次数完全恒定（水平线 vs AutoDiff 的线性增长），最低 21× 最高 128× 加速
- 在 dx 模式下，DiLQR 的模仿损失比 LSTM 策略低 $10^6$ 个数量级，说明结构化控制器嵌入学习框架的巨大优势
- 物理一致性显著优于 DiffMPC（负值参数 2.76% vs 16.85%），说明精确梯度带来的不仅是数值精度，还有语义合理性
- 前向算法是计算效率提升的最大贡献者，尤其在长 horizon 设置下

## 亮点与洞察

- **隐式微分解耦前向与反向传播**：这是本文最核心的贡献——将 iLQR 的反向传播从"展开所有迭代"变为"在不动点求解线性方程组"，复杂度从 $O(N_{\text{iter}})$ 降为 $O(1)$；这个思路可以推广到任何基于不动点迭代的可微分计算模块
- **精确梯度 vs 近似梯度的实际影响**：DiffMPC 忽略不动点对参数的依赖导致模型损失高 32% 和大量非物理参数（16.85% 负值），说明在控制领域梯度精度直接影响学习质量
- **视觉端到端控制的模块化演示**：将 DiLQR 嵌入编码器-解码器架构实现从像素输入的端到端控制，只给一张真实图片就能"想象"后续轨迹图像，展示了模块化设计的组合能力

## 局限与展望

- 实验仅在 CartPole 和 Inverted Pendulum 两个经典控制任务上验证，这些是相对简单的低维系统，在高维机器人控制等复杂任务上的表现有待检验
- 方法假设 iLQR 能收敛到不动点，对于某些非凸问题 iLQR 可能不收敛或收敛到局部最优，此时隐式微分的假设可能不满足
- 需要系统动力学的一阶和二阶导数，对于用神经网络拟合的动力学模型，二阶导数的计算可能成为新的瓶颈
- 视觉控制实验只是概念验证级别，与 DiffTORI 等更完整的感知控制框架相比还有较大差距

## 相关工作与启发

- **vs DiffMPC (Amos et al., 2018)**: DiffMPC 是可微 LQR 的先驱，但它将不动点轨迹视为常数计算偏导数；DiLQR 通过隐式微分计算全导数，修正了这个近似，模型损失降低 32%
- **vs SafePDP / IDOC**: 基于 Pontryagin 最大值原理的可微分方法，收敛速度比 iLQR 慢（iLQR 有 1.5 阶收敛率）；在使用 iLQR 生成的专家轨迹时，DiLQR 性能明显优于两者
- **vs Deep Equilibrium Models (DEQ)**: 概念上类似——DEQ 对深度网络的不动点做隐式微分，DiLQR 对 iLQR 的不动点做隐式微分；但 iLQR 的自指结构（$x^* = f_{x^*}(x^*)$）比一般不动点更复杂

## 评分

- 新颖性: ⭐⭐⭐⭐ 将隐式微分应用于 iLQR 不动点的思路新颖，解析解的推导完整严谨
- 实验充分度: ⭐⭐⭐ 实验场景较简单（Pendulum/CartPole），但计算效率的对比非常充分
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导清晰完整，方法论对比（vs DiffMPC）精确到具体公式项
- 价值: ⭐⭐⭐⭐ 为可微分控制提供了高效精确的基础工具，128× 加速的实际意义重大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Sampling from Binary Quadratic Distributions via Stochastic Localization](sampling_from_binary_quadratic_distributions_via_stochastic_localization.md)
- [\[ICML 2025\] AutoAL: Automated Active Learning with Differentiable Query Strategy Search](autoal_automated_active_learning_with_differentiable_query_strategy_search.md)
- [\[ACL 2025\] Predicting Implicit Arguments in Procedural Video Instructions](../../ACL2025/others/implicit_arguments_video_instructions.md)
- [\[ACL 2025\] Implicit Reasoning in Transformers is Reasoning through Shortcuts](../../ACL2025/others/implicit_reasoning_in_transformers_is_reasoning_through_shortcuts.md)
- [\[NeurIPS 2025\] Exact Learning of Arithmetic with Differentiable Agents](../../NeurIPS2025/others/exact_learning_of_arithmetic_with_differentiable_agents.md)

</div>

<!-- RELATED:END -->
