---
title: >-
  [论文解读] From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback
description: >-
  [AAAI 2026][Decision-Focused Learning] 本文首次提出递归决策聚焦学习（R-DFL）框架，通过在预测模块与优化模块之间引入双向反馈回路，突破了传统顺序式 DFL 的单向信息流限制，并设计了显式展开和隐式微分两种梯度传播方法，在报童问题和二部匹配问题上显著提升了最终决策质量。
tags:
  - "AAAI 2026"
  - "Decision-Focused Learning"
  - "Predict-then-Optimize"
  - "递归学习"
  - "隐式微分"
  - "双向反馈"
---

# From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback

**会议**: AAAI 2026  
**arXiv**: [2511.08035](https://arxiv.org/abs/2511.08035)  
**代码**: 无  
**领域**: 其他  
**关键词**: Decision-Focused Learning, Predict-then-Optimize, 递归学习, 隐式微分, 双向反馈

## 一句话总结

本文首次提出递归决策聚焦学习（R-DFL）框架，通过在预测模块与优化模块之间引入双向反馈回路，突破了传统顺序式 DFL 的单向信息流限制，并设计了显式展开和隐式微分两种梯度传播方法，在报童问题和二部匹配问题上显著提升了最终决策质量。

## 研究背景与动机

决策聚焦学习（Decision-Focused Learning, DFL）是近年来运筹优化领域的重要进展。传统的"先预测再优化"（Predict-then-Optimize, PTO）管线分两步走：先用机器学习预测不确定参数，再基于预测结果求解优化问题。但 PTO 的核心矛盾在于——它最小化的是中间预测误差，而非最终决策质量，导致预测精确但决策次优的问题。

DFL 通过将优化层嵌入深度学习架构中，直接端到端地最小化决策后悔（decision regret），取得了比 PTO 更好的效果。然而，现有 DFL 框架（包括 OptNet、CvxpyLayers、PyEPO 等）本质上仍是**顺序结构**（Sequential DFL, S-DFL）：预测结果单向流向优化模块，优化结果不会反馈给预测模块。

在许多真实场景中，决策和预测之间存在**双向耦合**。以网约车匹配为例：平台提出司机-乘客匹配方案后，用户的接受/拒绝反馈应当反过来修正后续匹配决策。类似的闭环交互广泛存在于动态定价、任务分配等多方博弈系统中。S-DFL 无法捕获这种递归交互，导致训练不稳定和决策次优。

本文的核心 idea 是：**将 DFL 从顺序结构扩展为递归结构**，让优化结果作为反馈信号回传到预测模块，形成预测→优化→反馈→预测的闭环。这一思路自然且直觉性强，但技术上面临一个关键挑战——递归结构产生了有向有环图（cyclic graph），违反了深度学习中 DAG 的基本假设，标准反向传播无法直接应用。

## 方法详解

### 整体框架

R-DFL 由两个核心模块组成：参数化预测模型 $\mathcal{F}_\theta$ 和凸优化模型 $\mathcal{G}$。与 S-DFL 不同的是，R-DFL 的预测模型接收两路输入——特征向量 $\boldsymbol{v}$ 和优化结果 $\boldsymbol{x}$，输出预测参数 $\hat{\boldsymbol{c}} = \mathcal{F}_\theta(\boldsymbol{x}, \boldsymbol{v})$。优化模型基于预测参数生成最优决策 $\boldsymbol{x}^* = \arg\min_{\boldsymbol{x} \in \mathcal{A}} g(\boldsymbol{x}; \hat{\boldsymbol{c}})$。决策结果又反馈给预测模型，形成闭环。

为了在这种循环结构中实现梯度传播，作者提出了两种微分方案：显式展开法和隐式微分法。

### 关键设计

1. **显式展开方法（Explicit Unrolling）**:

    - 功能：将预测-优化的循环交互展开为 $K$ 层序列，每层执行一次 $\hat{\boldsymbol{c}}_i = \mathcal{F}_\theta(\boldsymbol{x}_{i-1}, \boldsymbol{v})$ 和 $\boldsymbol{x}_i = \mathcal{G}(\hat{\boldsymbol{c}}_i)$
    - 核心思路：定义复合层 $\Phi_\theta = \mathcal{G} \circ \mathcal{F}_\theta$，将 $K$ 次迭代展开为显式计算图，通过标准自动微分框架反向传播梯度。总梯度为 $\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_K} \sum_{i=1}^{K} \left(\prod_{j=i+1}^{K} J_{\Phi_\theta}|_{\boldsymbol{x}_{j-1}}\right) \frac{\partial \Phi_\theta(\boldsymbol{x}_{i-1})}{\partial \theta}$
    - 设计动机：实现直观，可直接利用 PyTorch 的自动微分机制。但计算开销随展开深度 $K$ 线性增长，且存在梯度消失/爆炸风险

2. **隐式微分方法（Implicit Differentiation）**:

    - 功能：将递归系统建模为不动点问题 $\boldsymbol{x}^* = \Phi_\theta(\boldsymbol{x}^*)$，在均衡点处直接计算梯度
    - 核心思路：前向传播通过 RootFind 求解不动点方程 $\mathcal{H}_\theta = \boldsymbol{x}^* - \Phi_\theta(\boldsymbol{x}^*) \to 0$，反向传播利用隐函数定理直接给出梯度 $\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}^*}(I - J_{\Phi_\theta}|_{\boldsymbol{x}^*})^{-1} \frac{\partial \Phi_\theta(\boldsymbol{x}^*)}{\partial \theta}$
    - 设计动机：通过逆 Jacobian 矩阵 $(I - J_{\Phi_\theta}|_{\boldsymbol{x}^*})^{-1}$ 隐式编码了完整展开结构的信息，无需存储中间状态，内存和计算效率显著优于显式方法。但需要手工推导梯度表达式

3. **梯度等价性证明**:

    - 功能：证明当展开步数 $K \to \infty$ 时，显式展开法和隐式微分法给出的梯度完全一致
    - 核心思路：利用 Neumann 级数展开 $(I - J)^{-1} = \sum_{k=0}^{\infty} J^k$（当谱半径 $\rho(J) < 1$ 时），建立两种方法的数学等价关系
    - 设计动机：为实践者提供理论保障——无论选择哪种微分方法，最终精度是一致的，选择仅影响计算效率

### 损失函数 / 训练策略

损失函数采用决策后悔（decision regret）：$\mathcal{R}(\hat{\boldsymbol{c}}, \boldsymbol{c}) = g(\boldsymbol{x}^*(\hat{\boldsymbol{c}}), \boldsymbol{c}) - g(\boldsymbol{x}^*(\boldsymbol{c}), \boldsymbol{c})$，衡量基于预测参数所做决策与基于真实参数所做最优决策之间的目标函数值差距。训练中，所有展开层共享同一组参数 $\theta$，通过梯度下降更新预测模型。

## 实验关键数据

### 主实验

在两个基准问题上（报童问题-合成数据、二部匹配问题-NYC真实数据）比较了四种方法，涵盖三种规模（Small/Mid/Large）。

| 数据集 | 指标 | R-DFL-I | R-DFL-U | S-DFL | PTO |
|--------|------|---------|---------|-------|-----|
| Newsvendor-Small | RMSE↓ | **8.831** | 8.983 | 12.245 | 12.771 |
| Newsvendor-Mid | RMSE↓ | **9.106** | 9.173 | 12.536 | 12.747 |
| Newsvendor-Large | RMSE↓ | **9.327** | 9.343 | 12.649 | 12.684 |
| Matching-Small | RMSE↓ | 0.398 | **0.396** | 0.408 | 0.412 |
| Matching-Mid | RMSE↓ | **0.220** | 0.222 | 0.231 | 0.232 |
| Matching-Large | RMSE↓ | 0.171 | **0.170** | 0.187 | 0.190 |

R-DFL 相比 S-DFL 在报童问题上 RMSE 降低约 26-28%，在匹配问题上降低约 5-9%。

### 消融实验

不同预测模型（LSTM/RNN/Transformer）在大规模问题上的对比：

| 配置 | Newsvendor RMSE | Matching RMSE | 说明 |
|------|----------------|---------------|------|
| R-DFL-I + LSTM | 10.104 | 0.174 | 隐式法，各预测模型均优于S-DFL |
| R-DFL-U + LSTM | 10.112 | 0.176 | 显式法，结果与隐式法接近 |
| S-DFL + LSTM | 12.693 | 0.230 | 顺序基线 |
| R-DFL-I + Transformer | 11.332 | **0.166** | Transformer在匹配问题上最优 |
| PTO + RNN | 12.842 | 0.185 | 传统两阶段基线 |

展开层数敏感性分析显示：增加展开层数对精度提升边际递减，但显式方法时间开销线性增长，隐式方法时间基本稳定。

### 关键发现
- R-DFL 两种微分方法在精度上高度一致（QQ 图显示强线性对齐），验证了梯度等价性定理
- 隐式方法在大规模问题上训练速度比显式方法快约 1.5 倍（2704s vs 1867s/epoch）
- R-DFL 的优势与预测模型架构无关，在 LSTM、RNN、Transformer 上均成立

## 亮点与洞察
- 首次将 DFL 从顺序范式推广到递归范式，概念简洁但意义重大。闭环决策制定是真实世界中更普遍的范式
- 显式展开与隐式微分的梯度等价性证明优雅且实用，为选择微分方法提供了理论支撑
- 将 Deep Equilibrium Model（DEQ）的思想引入运筹优化中的 DFL 领域，形成了跨领域的方法论迁移
- 实验设计考虑了不同问题规模和不同预测模型，验证了框架的通用性

## 局限与展望
- 目前仅支持凸优化问题（目标函数可微凸、约束为线性），难以直接处理整数规划、组合优化等离散优化问题
- 不动点收敛假设（PL 条件、谱半径 < 1）在实践中可能不总是满足
- 实验规模仍然偏小（最大 900 决策变量），大规模工业级问题上的表现有待验证
- 未考虑随机递归环境，即反馈信号本身带有噪声的情况

## 相关工作与启发
- 与 Deep Equilibrium Models（Bai et al., 2019）思路类似，都是用不动点方法替代逐层展开，但 R-DFL 将其用于"学习+优化"的混合系统而非纯神经网络
- DFL 领域的代表性工作包括 OptNet、SPO+、PyEPO 等，本文是首个系统性地建模双向交互的 DFL 框架
- 方法论上可以启发 RL 中的 model-based planning、博弈论中的 Stackelberg 博弈求解等场景

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Decision Tree Learning on Product Spaces](../../ICML2026/others/decision_tree_learning_on_product_spaces.md)
- [\[ICML 2026\] Sequential Group Composition: A Window into the Mechanics of Deep Learning](../../ICML2026/others/sequential_group_composition_a_window_into_the_mechanics_of_deep_learning.md)
- [\[ICLR 2026\] Active Learning for Decision Trees with Provable Guarantees](../../ICLR2026/others/active_learning_for_decision_trees_with_provable_guarantees.md)
- [\[ECCV 2024\] Bidirectional Uncertainty-Based Active Learning for Open-Set Annotation](../../ECCV2024/others/bidirectional_uncertainty-based_active_learning_for_open-set_annotation.md)
- [\[ACL 2025\] Learning to Reason from Feedback at Test-Time](../../ACL2025/others/learning_to_reason_from_feedback_at_test-time.md)

</div>

<!-- RELATED:END -->
