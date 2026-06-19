---
title: >-
  [论文解读] No Soundness in the Real World: On the Challenges of the Verification of Deployed Neural Networks
description: >-
  [ICML 2025][LLM推理][神经网络验证] 本文证明所有当前最先进的神经网络验证器都只提供"理论健全性"（约束全精度输出）而非"实际健全性"（约束部署环境中的浮点输出），并通过构造环境敏感的对抗性后门网络，实证验证了所有测试验证器均可被欺骗。 神经网络验证的目标 神经网络验证的目标是提供数学证明…
tags:
  - "ICML 2025"
  - "LLM推理"
  - "神经网络验证"
  - "浮点运算"
  - "soundness"
  - "部署环境"
  - "对抗性后门"
  - "区间分析"
---

# No Soundness in the Real World: On the Challenges of the Verification of Deployed Neural Networks

**会议**: ICML 2025  
**arXiv**: [2506.01054](https://arxiv.org/abs/2506.01054)  
**代码**: [https://github.com/szasza1/no_soundness](https://github.com/szasza1/no_soundness)  
**领域**: 音频语音  
**关键词**: 神经网络验证, 浮点运算, soundness, 部署环境, 对抗性后门, 区间分析

## 一句话总结

本文证明所有当前最先进的神经网络验证器都只提供"理论健全性"（约束全精度输出）而非"实际健全性"（约束部署环境中的浮点输出），并通过构造环境敏感的对抗性后门网络，实证验证了所有测试验证器均可被欺骗。

## 研究背景与动机

### 神经网络验证的目标

神经网络验证的目标是提供数学证明，保证网络具有某些安全属性（如对抗鲁棒性）。对于分类网络，核心验证任务是：给定输入 $x^*$ 及其 $\epsilon$-邻域 $D_{\epsilon,p}(x^*)$，证明邻域内所有输入都被分配到同一类别。

### 理论模型 vs 部署模型

当前所有验证方法都针对网络的**理论模型**——即使用全精度实数运算的数学抽象。然而实际部署中：
- 硬件使用**浮点运算**（IEEE 754），存在舍入误差
- 浮点运算是**非结合性的**：相同加法运算，不同的求和顺序可能产生不同结果
- 并行化、不同框架（PyTorch vs Flux）、不同硬件（CPU vs GPU）、不同batch size都会改变运算的表达式树
- 部署环境可能是**随机性的**（因并行计算导致运算顺序不确定）

### 核心洞察

$$\text{理论健全性} \neq \text{实际健全性}$$

即：正确约束全精度输出的验证器，不一定能正确约束浮点部署时的真实输出。这意味着攻击者可以设计出一种网络后门，在理论模型验证中完全隐形，但在特定部署环境中激活恶意行为。

### 为什么这个问题重要？

- 验证器被用于安全关键应用（自动驾驶、医疗诊断），如果验证结果不可靠，后果严重
- 当前所有主流验证器（β-CROWN、MIPVerify、DeepPoly等）都存在此问题
- 问题不是浮点精度误差，而是浮点运算的**非结合性**和**环境依赖性**这一根本性质

## 方法详解

### 整体框架

论文的论证框架分为三个层次：

1. **形式化建模**：定义部署网络 $r(x;\theta,\mathcal{E})$，区别于理论网络 $f(x;\theta)$
2. **理论证明**：证明基于区间传播（IBP）和符号传播的验证方法都无法保证实际健全性
3. **实证攻击**：构造对抗性后门网络，验证所有主流验证器均被欺骗

### 关键设计一：部署验证问题的形式化

**理论验证问题**（已有工作解决的）：

$$\forall x \in D_{\epsilon,p}(x^*),\ f(x;\theta) \geq 0$$

**部署验证问题**（本文提出需要解决的）：

$$\forall x \in D_{\epsilon,p,\mathcal{E}}(x^*)\ \forall z \in r(x;\theta,\mathcal{E}),\ z \geq 0$$

两者的关键差异：
- **函数不同**：$r \neq f$，部署版本因浮点舍入产生不同输出
- **域不同**：$D_{\epsilon,p,\mathcal{E}} = D_{\epsilon,p} \cap X$，$X$ 取决于环境的数值表示
- **属性不同**：$P_\mathcal{E} \neq P \cap X$，因为某些 $f$ 的负输出可能在 $r$ 中变为零
- **随机性**：在随机环境中，$r(x;\theta,\mathcal{E})$ 可能是一个**集合**（所有概率 > 0 的输出向量）

### 关键设计二：理论证明——区间传播的不健全性

论文聚焦于最简单的函数类——**输入变量之和**，证明即使对这类简单函数，验证也不能保证实际健全性。

**命题 6.1（正面结果）**：如果已知部署环境的确定性表达式树，按该树执行IBP，则IBP是实际健全的。

**命题 6.2（核心负面结果）**：对于任何允许所有合法表达式树、使用IEEE 754浮点的环境 $\mathcal{E}$，存在表达式树 $o$ 和输入 $x$，使得IBP的区间 $[a,b]_o$ 不能覆盖部署的真实最小/最大输出。

**命题 6.3（排除简单修复）**：按递减顺序或按绝对值递减顺序求和都**不能**保证得到最小输出。

**推论**：找到能最小化/最大化浮点输出的表达式树很可能是NP-hard问题（基于Kao & Wang 2000的类似结论）。

### 关键设计三：符号传播方法同样不健全

- **多面体方法**（DeepPoly、CROWN）：在计算输入变量之和时退化为IBP，同样不实际健全
- **Zonotope方法**（DeepZ、RefineZono）：使用仿射算术的扩宽技术，区间比IBP更宽，但**仍然不能保证覆盖所有可能输出**（命题 6.4、6.5）

### 关键设计四：对抗性后门网络

论文设计了**检测器神经元**（detector neuron），利用浮点运算特性来检测部署环境：

#### 精度检测器

利用特殊数 $\omega$（最小的满足下一个可表示数是 $\omega+2$ 的正数）：
- binary32格式：$\omega = 2^{24}$
- binary64格式：$\omega = 2^{53}$

检测器 $\omega + 1 - \omega$ 在目标精度下输出0，在更高精度下输出1。

#### 表达式树检测器（三种难度递增）

| 检测器 | 原理 | 对验证器的难度 |
|--------|------|---------------|
| Order1 | $(2h_1+1)h_2$ 个求和项交替排列正负$\omega/h_1$ | 简单：默认顺序的IBP即可覆盖 |
| Order2 | $h+2$ 个求和项 $2/h + ... + \omega - \omega$ | 中等：默认顺序IBP不覆盖0 |
| Order3 | $h+2$ 个求和项 $1 + ... + 1 + \omega - \omega$ | 困难：极少表达式树覆盖0 |

后门嵌入方法：采用 Zombori et al. (2021) 的方案，将检测器神经元插入到正常MNIST分类网络中。当检测器输出触发值时，网络行为发生任意改变（如翻转分类结果）。

### 理论分析

本文的理论贡献可总结为以下层次逻辑：

1. **浮点非结合性 → 运算顺序影响结果**：同一数学表达式，不同运算树可能产生不同浮点输出
2. **部署环境 → 确定运算顺序**：不同的硬件/软件/框架/batch size选择不同的运算树
3. **IBP/符号传播只跟踪一棵运算树**：验证器的区间可能不覆盖其他运算树的输出
4. **攻击者可以利用这个gap**：设计在特定运算树下激活、在验证器运算树下隐形的后门
5. **根本困难**：穷举所有可能运算树是NP-hard的

## 实验关键数据

### 实验设置

- **宿主网络**：Wong & Kolter (2018) 的MNIST网络，2层卷积（16和32滤波器，stride 2）+ 100神经元全连接，ReLU激活，在 $l_\infty$ 半径0.1下对抗训练
- **五种对抗网络**：2个精度攻击 + 3个表达式树攻击（Order1/2/3）
- **测试验证器**：MIPVerify、MN-BAB、β-CROWN BaB、GCP-CROWN、DeepPoly、RefinePoly、DeepZ、RefineZono（共8种）
- **验证对象**：MNIST测试集前100个样本

### 后门网络有效性

| 攻击类型 | 32-bit精度 | 64-bit精度 |
|---------|-----------|-----------|
| 精度攻击（32-bit对抗） | 0.11% | 98.11% |
| 精度攻击（64-bit对抗） | 98.11% | 0.11% |

精度攻击完美生效：在目标精度下准确率降至0.11%（近乎完全翻转分类），而在非目标精度下保持98.11%正常准确率。

### 表达式树攻击在不同环境下的表现

| 攻击 | Batch | PyTorch CPU | PyTorch GPU | Flux CPU | Flux GPU |
|------|-------|-------------|-------------|----------|----------|
| Order1 | 1 | 0.11% | 0.11% | 98.11% | 0.11% |
| Order2 | 1 | 98.11% | 98.11% | 98.11% | 0.11% |
| Order3 | 1 | 98.11% | 98.11% | 98.11% | 98.11% |
| Order1 | 10 | 0.11% | 98.11% | 0.11% | 98.11% |
| Order2 | 10 | 98.11% | 98.11% | 98.11% | 98.11% |
| Order3 | 10 | 98.11% | 98.11% | 98.11% | 98.11% |

Order1在多个环境中被激活（0.11%），Order2在Flux GPU/batch=1时激活，Order3设计为最难检测，几乎无环境激活。

### 验证器被攻击结果

| 验证器 | 验证环境 | 传播方式 | Precision攻击 | Order1 | Order2 | Order3 |
|--------|---------|---------|--------------|--------|--------|--------|
| MIPVerify | 64-bit CPU | IBP | **unsound** | sound | **unsound** | **unsound** |
| MN-BAB | 64-bit GPU | Polyhedra | **unsound** | sound | **unsound** | **unsound** |
| β-CROWN (32-bit) | 32-bit CPU | Polyhedra | **unsound** | sound | **unsound** | **unsound** |
| β-CROWN (64-bit CPU) | 64-bit CPU | Polyhedra | **unsound** | sound | **unsound** | **unsound** |
| β-CROWN (64-bit GPU) | 64-bit GPU | Polyhedra | **unsound** | sound | **unsound** | **unsound** |
| GCP-CROWN | 64-bit CPU | Polyhedra | **unsound** | sound | **unsound** | **unsound** |
| DeepPoly | 64-bit CPU | Polyhedra | **unsound** | sound | sound | **unsound** |
| DeepZ | 64-bit CPU | Zonotope | **unsound** | sound | sound | **unsound** |

**核心发现**：所有8种验证器对Precision攻击和Order3攻击均**unsound**——无法检测后门，错误地认证网络是安全的。

## 亮点与洞察

1. **概念突破**：首次清晰区分"理论健全性"和"实际健全性"，揭示所有现有验证器都存在的根本性盲区
2. **攻击可利用性强**：构造的后门网络在正常验证下完全隐形，但在特定部署环境中可完全翻转分类结果（准确率从98%降至0.1%）
3. **问题的根源不在精度**：即使验证器使用无限精度的全精度运算，也无法解决此问题——因为问题在于验证的是"理论模型"而非"部署模型"
4. **攻击难度递增设计精妙**：Order1→Order2→Order3逐步增加检测难度，系统性地暴露验证器的弱点
5. **固定点运算的启示**：论文指出fixed-point arithmetic（定点运算）因量化舍入误差可控，可能是实现实际健全验证的替代路径
6. **实际部署安全启示**：在安全关键系统中，验证必须将部署环境（硬件、精度、运算库、并行策略）纳入考量

## 局限性

1. **未提出解决方案**：论文只暴露问题，未提出任何实现实际健全性的验证方法
2. **攻击场景假设较强**：攻击者需要了解部署环境的浮点特性（精度、运算顺序），在黑盒部署场景中攻击难度增大
3. **仅测试ReLU网络**：实验只使用ReLU激活的MNIST网络，未覆盖更复杂的架构（Transformer、BatchNorm等）或更大规模网络
4. **计算复杂度未完全解决**：猜测找最优表达式树是NP-hard但未给出严格证明（仅引用类似问题的已知结论）
5. **实际威胁评估不足**：真实攻击场景中，攻击者能否在不知道确切部署细节的情况下构造有效后门？论文未深入讨论
6. **后门嵌入方式依赖现有方法**：使用Zombori et al. (2021)的嵌入方案，并非完全独立的攻击体系

## 相关工作

- **验证方法分类**：sound方法（IBP、DeepPoly、CROWN、DeepZ）基于bound propagation；sound+complete方法（MIPVerify、Reluplex）基于SMT/MILP；BaB框架（β-CROWN）融合两者
- **浮点数值漏洞**：Zombori et al. (2021) 利用MILP求解器的数值误差攻击验证器（但只影响使用优化的验证器）；Jia & Rinard (2021) 利用验证与部署不同精度的问题
- **部署环境差异**：Schlögl et al. (2023) 研究不同平台对部署网络输出的影响；Shanmugavelu et al. (2024/2025) 展示仅通过置换运算顺序即可生成对抗样本
- **实现差距**：Cordeiro et al. (2025) 提出"实现差距"（implementation gap）概念，将此问题定位为编程语言挑战
- **定点运算替代**：Lohar et al. (2023) 研究定点量化下的sound验证，舍入误差可控

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|-----------|------|
| 新颖性 | 9 | 首次系统性地揭示理论/实际健全性的根本差距 |
| 理论深度 | 8 | 严格的形式化证明，但NP-hard猜想未完全证实 |
| 实验充分性 | 8 | 覆盖8种主流验证器和多种部署环境，说服力强 |
| 实用价值 | 7 | 揭示重要安全问题，但未提供解决方案 |
| 写作质量 | 8 | 逻辑清晰，直觉解释与形式化证明结合得当 |
| 综合评分 | **8.0** | 在神经网络验证领域提出了根本性的开放问题 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios](../../NeurIPS2025/llm_reasoning/time_a_multilevel_benchmark_for_temporal_reasoning_of_llms_i.md)
- [\[NeurIPS 2025\] Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerating Neural Network Verification](../../NeurIPS2025/llm_reasoning/clip-and-verify_linear_constraint-driven_domain_clipping_for_accelerating_neural.md)
- [\[ACL 2025\] MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](../../ACL2025/llm_reasoning/mm-verify_enhancing_multimodal_reasoning_with_chain-of-thought_verification.md)
- [\[AAAI 2026\] Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs](../../AAAI2026/llm_reasoning/graph_of_verification_structured_verification_of_llm_reasoning_with_directed_acy.md)
- [\[NeurIPS 2025\] Two-Stage Learning of Stabilizing Neural Controllers via Zubov Sampling and Iterative Domain Expansion](../../NeurIPS2025/llm_reasoning/two-stage_learning_of_stabilizing_neural_controllers_via_zubov_sampling_and_iter.md)

</div>

<!-- RELATED:END -->
