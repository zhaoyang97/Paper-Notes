---
title: >-
  [论文解读] Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data
description: >-
  [ICLR 2026][可解释性][可解释ML] 受行为科学中效用最大化范式启发，提出 Behavior Learning (BL) 框架，将数据建模为由可解释的模块化效用最大化问题（UMP）层次组合所诱导的 Gibbs 分布，在预测性能、内在可解释性和参数可辨识性三者之间实现了统一。 领域现状：可解释机器学习一直面临"性能…
tags:
  - "ICLR 2026"
  - "可解释性"
  - "可解释ML"
  - "效用最大化"
  - "层次优化结构"
  - "可辨识性"
  - "Gibbs分布"
---

# Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data

**会议**: ICLR 2026  
**arXiv**: [2602.20152](https://arxiv.org/abs/2602.20152)  
**代码**: [https://github.com/MoonYLiang/Behavior-Learning](https://github.com/MoonYLiang/Behavior-Learning) (pip install blnetwork)  
**领域**: 可解释性  
**关键词**: 可解释ML, 效用最大化, 层次优化结构, 可辨识性, Gibbs分布  

## 一句话总结
受行为科学中效用最大化范式启发，提出 Behavior Learning (BL) 框架，将数据建模为由可解释的模块化效用最大化问题（UMP）层次组合所诱导的 Gibbs 分布，在预测性能、内在可解释性和参数可辨识性三者之间实现了统一。

## 研究背景与动机

**领域现状**：可解释机器学习一直面临"性能-可解释性权衡"——深度神经网络预测强但不透明，线性模型/决策树透明但无法捕捉复杂非线性模式。现有方法如 EBM、NAM 等试图缓解这一矛盾，但在科学建模场景中仍有不足。

**现有痛点**：
   - **缺乏科学理论对齐**：大多数可解释方法只是对已有ML方法的扩展，而非从科学原理（如优化问题、微分方程）出发设计，导致难以从模型中提取科学知识。
   - **解释不唯一**：绝大多数模型不具备可辨识性（identifiability），即同一预测行为可能对应多组不同的参数/解释，无法保证所恢复的"科学知识"是真实的。

**核心矛盾**：如何设计一个既有强预测能力、又内在可解释、同时可辨识的机器学习框架？

**切入角度**：从行为科学的核心范式——效用最大化（UMP）出发。UMP 假设人类行为来源于在约束下最大化效用的优化过程，且任何优化问题都可等价写成 UMP（Theorem 2.2），因此以 UMP 为基础构建的框架具有通用性。

**核心idea**：将模型参数化为由多个可解释UMP模块层次组合而成的"复合效用函数"，通过 Gibbs 分布建模数据的条件分布，使得每个模块可以被解读为一个符号化的优化问题。

## 方法详解

### 整体框架
BL 要解决的是"性能-可解释性-可辨识性"三者难以兼得的问题：输入是特征 $\mathbf{x}$ 和响应 $\mathbf{y}$，输出是条件概率 $p(\mathbf{y}|\mathbf{x})$。它的做法是先把数据建模为一个"复合效用函数"$\mathrm{BL}(\mathbf{x},\mathbf{y})$，再用 Gibbs 分布 $p_\tau(\mathbf{y}|\mathbf{x}) \propto \exp(\mathrm{BL}(\mathbf{x},\mathbf{y})/\tau)$ 诱导出数据分布（温度 $\tau \to 0$ 时退化为确定性最优响应）。关键在于这个复合效用函数并非黑盒，而是由多个可解释的效用最大化（UMP）模块层次堆叠而成——这正是"从数据中学习层次优化结构"的含义：每个模块对应一个可读回的符号化优化问题，堆叠的深度则控制表达力。

### 关键设计

**1. 基础模块 $\mathcal{B}(\mathbf{x}, \mathbf{y})$：把一个带约束的效用最大化问题压成一层可读的罚函数**

每个 block 都对应一个效用最大化问题（UMP），但带约束的优化不好直接嵌进网络。BL 基于 Theorem 2.1 的局部精确罚重构，把带约束的 UMP 等价改写成无约束的罚函数形式：

$$\mathcal{B} = \lambda_0^\top \tanh(\mathbf{p}_u) - \lambda_1^\top \mathrm{ReLU}(\mathbf{p}_c) - \lambda_2^\top |\mathbf{p}_t|$$

其中 $\mathbf{p}_u, \mathbf{p}_c, \mathbf{p}_t$ 是多项式特征映射。三个项各自带经济学含义：$\tanh$ 表示"边际效用递减"的偏好，$\mathrm{ReLU}$ 惩罚不等式约束（预算）的违反，$|\cdot|$ 度量等式约束（信念）的偏差。正因为每个 block 都能被读回成一个符号化的 UMP，它的透明度可比拟线性回归——参数不是黑盒权重，而是可解读的偏好、预算、信念系数。

**2. 三种架构变体：用层数在可解释性和表达力之间滑动**

在基础 block 之上，BL 用层数控制"可解释性—表达力"的取舍。**BL(Single)** 只用单个 $\mathcal{B}$ block，可解释性最强，直接对应一个 UMP；**BL(Shallow)** 是 1-2 层的浅层组合，把多个并行 block 的输出送入下一层、最后经仿射变换输出；**BL(Deep)** 则堆叠两层以上，用来建模层次化的优化结构（如"微观偏好→宏观权衡→整体决策"）。三者统一写成 $\mathrm{BL}(\mathbf{x},\mathbf{y}) = \mathbf{W}_L \cdot \mathbb{B}_L(\cdots\mathbb{B}_2(\mathbb{B}_1(\mathbf{x},\mathbf{y}))\cdots)$，层数越深表达力越强，但单个模块的符号语义越难直接读出。

**3. 可辨识变体 IBL：换掉两个算子，让"解释唯一"在数学上成立**

BL 虽然可解释，但同一预测行为可能对应多组参数，解释并不唯一。IBL 在 block 上施加更严的结构约束（要求 $\phi^{\rm id}$ 严格递增、$\psi^{\rm id}$ 对称且严格递增），实例化为：

$$\mathcal{B}^{\rm id} = \lambda_0^\top \tanh(\mathbf{p}_u) - \lambda_1^\top \mathrm{softplus}(\mathbf{p}_c) - \lambda_2^\top (\mathbf{p}_t)^{\odot 2}$$

也就是用 softplus 替掉 ReLU（保证光滑且严格单调）、用平方替掉绝对值（保证对称且光滑）。这两处替换让参数在商空间上被唯一确定，从而模型恢复的"科学知识"在数学上可信，而不是众多等价解中随便一个。难得的是这套约束并不牺牲表达力：Theorem 2.3 保证 BL/IBL 在足够容量下仍能在 KL 散度意义下逼近任意连续条件分布；在此基础上 IBL 进一步拿到可辨识性（Thm 2.4-2.5，参数在商空间上唯一确定）、（万能）一致性（Thm 2.6-2.7，样本量趋于无穷时收敛到真实参数，模型错误设定时仍收敛到真实分布），以及渐近正态性与渐近有效性（方差达到有效信息下界）。这是"可解释、可辨识、又不丢逼近能力"能同时成立的理论支撑。

### 损失函数 / 训练策略
混合离散和连续响应的训练目标：
- 离散部分：直接使用交叉熵损失
- 连续部分：使用去噪分数匹配（denoising score matching），将 BL 看作能量函数
- 总目标：$\mathcal{L}(\theta) = \gamma_d \mathbb{E}[-\log p_\tau(\mathbf{y}^{\rm disc}|\mathbf{x})] + \gamma_c \mathbb{E}\|\nabla_{\tilde{\mathbf{y}}} \log p_\tau - \sigma^{-2}(\tilde{\mathbf{y}} - \mathbf{y})\|^2$

## 实验关键数据

### 主实验：10个数据集上的预测性能
在涵盖不同样本量、特征维度、科学领域的10个数据集上，与10种 baseline（MLP、XGBoost、LightGBM、决策树、贝叶斯方法等）对比。

| 模型 | F1-Macro排名 | AUC排名 | 可解释性 |
|------|-------------|---------|---------|
| BL(Shallow) | 第2-3名 | 第一梯队 | ✓ 内在可解释 |
| BL(Single) | 第3-4名 | 第一梯队 | ✓ 最强可解释 |
| MLP | 低于BL(Shallow) | 第一梯队 | ✗ |
| XGBoost | 第1-2名 | 第一梯队 | ✗ |
| 决策树 | 最末 | 最末 | ✓ |

### 消融/分析实验：高维输入上的BL vs E-MLP

| 数据集 | 模型 | 准确率(%) | OOD AUROC(%) |
|--------|------|----------|-------------|
| MNIST | BL(d=3) | 97.93 | **92.92** |
| MNIST | E-MLP(d=3) | 98.14 | 87.76 |
| Fashion-MNIST | BL(d=2) | 88.96 | **89.87** |
| Fashion-MNIST | E-MLP(d=2) | 88.88 | 84.61 |
| AG News | BL(d=1) | **89.52** | **66.18** |
| AG News | E-MLP(d=1) | 88.74 | 59.24 |

BL 在相近参数量下，ID准确率与 E-MLP 持平，但 OOD 检测显著更优，且 ECE/NLL 校准指标更好（如 Fashion-MNIST NLL: BL=0.36 vs E-MLP=0.74）。

### 关键发现
- BL(Shallow) 超越 MLP，说明可解释性不以牺牲性能为代价
- 在 OOD 检测上 BL 优势明显，表明优化结构建模有助于不确定性估计
- BL 在 Boston Housing 数据集上恢复的偏好模式和权衡结构与经济学文献中的已有发现一致（Table 11），表明 BL 可以从数据中重建科学知识
- BL(Deep) [5,3,1] 架构恢复了"5种微观偏好→3种宏观权衡→1个代表性买家"的层次结构

## 亮点与洞察
- **科学理论与ML的优雅融合**：以行为科学的UMP为根基，使得每个网络模块都有明确的经济学/优化含义（效用、约束、信念），而不是事后解释。这种"理论先行"的可解释ML设计范式非常新颖。
- **层次优化结构的自动发现**：BL(Deep) 不仅预测准确，还能自动发现数据中蕴含的层次化优化结构。在 Boston Housing 上，从原始特征到微观偏好到宏观权衡的"粗粒化"过程，与统计物理中的重整化思想相吻合。
- **可辨识性保证**：IBL 通过对罚函数施加光滑单调约束，在不损失万能逼近能力的前提下实现了参数可辨识，这在可解释ML中非常稀缺。这意味着模型学到的"解释"在数学上是唯一的。
- **Pareto前沿下移**：BL 在参数量和运行时间与 E-MLP 相当的情况下，实现了类似预测性能+内在可解释性+更好的OOD检测，推动了性能-可解释性Pareto前沿。

## 局限与展望
- **多项式特征映射的可扩展性**：BL(Single)使用二次多项式基，在高维特征空间中特征数量会爆炸式增长，深层变体通过仿射映射规避此问题但牺牲了符号可解释性的精细度
- **UMP假设的适用范围**：并非所有数据生成过程都可以用"效用最大化"来概括（如某些物理过程可能更适合用微分方程描述），虽然理论上任何优化问题都可等价为UMP，但在实际建模中这种等价可能远离直觉
- **计算效率**：使用 denoising score matching 训练连续响应部分，训练时间略高于标准MLP
- **深层BL的解释复杂性**：随着层数增加，虽然理论上有层次化优化结构的解释，但实际上理解多层嵌套的UMP组合并不容易，需要领域知识

## 相关工作与启发
- **vs EBM (Nori et al., 2019)**：EBM 基于广义加性模型，逐特征可解释但不建模特征交互；BL 通过多项式映射和层次组合自然建模交互，且具有优化结构的科学解释
- **vs 能量模型 (LeCun et al., 2006)**：BL 可视为一种结构化能量模型，但BL的每个模块都有明确的优化语义（效用/约束/信念），而传统能量模型是黑盒
- **vs 逆优化 (Ahuja & Orlin, 2001)**：传统逆优化通常假设优化问题的形式已知而参数未知；BL 则同时从数据中学习优化的结构和参数，且支持层次化组合

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从行为科学UMP出发构建可解释ML框架的思路极具原创性
- 实验充分度: ⭐⭐⭐⭐ 10个标准数据集+case study+高维实验覆盖全面，但缺少大规模真实科学发现任务的验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论严密、结构清晰、图示优秀
- 价值: ⭐⭐⭐⭐ 为可解释ML提供了新范式，但实际科学应用的影响力有待观察

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Behavior Learning (BL)](behavior_learning_bl.md)
- [\[ICLR 2026\] LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data](lore_jointly_learning_the_intrinsic_dimensionality_and_relative_similarity_struc.md)
- [\[ICLR 2026\] Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)
- [\[ICLR 2026\] Tokenizing Single-Channel EEG with Time-Frequency Motif Learning](tokenizing_single-channel_eeg_with_time-frequency_motif_learning.md)
- [\[ICLR 2026\] Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer](towards_understanding_subliminal_learning_when_and_how_hidden_biases_transfer.md)

</div>

<!-- RELATED:END -->
