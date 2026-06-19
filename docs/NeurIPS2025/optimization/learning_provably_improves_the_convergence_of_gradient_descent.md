---
title: >-
  [论文解读] Learning Provably Improves the Convergence of Gradient Descent
description: >-
  [NeurIPS 2025][优化/理论][Learn to Optimize] 首次严格证明了基于unrolling的Learn to Optimize (L2O)框架（Math-L2O）的训练收敛性，利用NTK理论建立了线性收敛速率，并提出确定性初始化策略确保L2O可证明地改善梯度下降算法的收敛性能，实验验证相比标准GD提升超50%的最优性。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "Learn to Optimize"
  - "梯度下降"
  - "Neural Tangent Kernel"
  - "收敛性证明"
  - "初始化策略"
---

# Learning Provably Improves the Convergence of Gradient Descent

**会议**: NeurIPS 2025  
**arXiv**: [2501.18092](https://arxiv.org/abs/2501.18092)  
**作者**: Qingyu Song (Xiamen University), Wei Lin, Hong Xu (CUHK)  
**代码**: [GitHub](https://github.com/NetX-lab/MathL2OProof-Official)  
**领域**: 优化  
**关键词**: Learn to Optimize, 梯度下降, Neural Tangent Kernel, 收敛性证明, 初始化策略  

## 一句话总结

首次严格证明了基于unrolling的Learn to Optimize (L2O)框架（Math-L2O）的训练收敛性，利用NTK理论建立了线性收敛速率，并提出确定性初始化策略确保L2O可证明地改善梯度下降算法的收敛性能，实验验证相比标准GD提升超50%的最优性。

## 研究背景与动机

### 问题背景
Learn to Optimize (L2O) 利用深度神经网络学习优化算法的超参数（如步长），在凸优化（LASSO、逻辑回归）和非凸优化（MIMO求和率最大化）等问题上取得了成功。其中"白盒"方法（如Math-L2O）通过将NN嵌入传统优化算法的框架中，兼具可解释性和实用性。

### 已有工作的不足
- **LISTA-CPSS**：虽然构造性地证明了线性收敛率，但其理论保证依赖于极其严格的假设（迭代值符号一致性、参数矩阵列约束），实验表明这些条件在训练后频繁被违反
- **Math-L2O**：仅推导了收敛的必要条件，未阐明训练过程本身如何保证收敛；后续工作依赖于L2O模型必须模仿标准GD行为这一强假设
- **两者共同问题**：在长优化步数下均存在梯度爆炸导致的训练不稳定——低学习率收敛慢，高学习率训练崩溃

### 核心动机
填补两个关键理论空白：(1) unrolling-based L2O模型的训练收敛性证明；(2) NN训练收敛（参数优化维度）与L2O求解收敛（迭代求解维度）之间的精确关联。

## 方法详解

### 问题建模
- **优化目标**：$\min_{x} f(x) = \frac{1}{2}\|\mathbf{M}x - y\|_2^2$，其中$\mathbf{M} \in \mathbb{R}^{b \times d}$满行秩，$f$为$\beta$-光滑
- **Math-L2O更新**：$X_t = X_{t-1} - \frac{1}{\beta} P_t \odot \nabla F(X_{t-1})$，其中$P_t = g_W(X_{t-1}, \nabla F(X_{t-1}))$为NN生成的逐坐标步长向量
- **NN架构**：$L$层网络，内层使用ReLU激活，输出层使用$2\sigma$（Sigmoid乘2）确保$0 < P_t < 2$
- **训练损失**：$F(W) = \frac{1}{2}\|\mathbf{M}X_T - Y\|_2^2$，通过BPTT（Back-Propagation-Through-Time）优化

### 核心理论贡献1：L2O与GD的对齐
建立训练收敛率$r_k$与GD收敛率的显式关联：

$$F(X_T^k) \leq r_k \cdot \frac{\beta}{T}\|X_0^k - X^*\|_2^2$$

其中$r_k < 1$为训练迭代$k$的收敛因子，$\frac{\beta}{T}$为GD的子线性收敛率。这意味着L2O至少与GD一样快，且训练可进一步加速。

### 核心理论贡献2：线性训练收敛率（Theorem 4.3）
利用NTK理论证明Math-L2O训练的线性收敛：

$$F(W^k) \leq \left(1 - 4\eta \frac{\beta_0^2}{\beta^2} \delta_4 \alpha_0^2\right)^k F(W^0)$$

- $\alpha_0 = \sigma_{\min}(G_{L-1,T}^0)$：倒数第二层输出的最小奇异值
- $\delta_4 = \sigma(\delta_3 \Theta_L)(1 - \sigma(\delta_3 \Theta_L)) > 0$
- 只需$\mathcal{O}(Nd)$宽度的NN（远低于经典NTK要求的无穷宽）

**关键引理**：
- **Lemma 4.1（梯度界）**：训练梯度范数以目标函数值为上界，使得梯度随训练下降
- **Lemma 4.2（半光滑性）**：Math-L2O输出对参数扰动的响应有界，系数为$\mathcal{O}(e^{LT})$，与深度架构的已知结果一致

### 核心理论贡献3：确定性初始化策略
1. **对齐初始化**：前$L-1$层从标准高斯分布采样后QR分解并非负化，最后一层$W_L^0 = \mathbf{0}$；此时输出$P_T = \mathbf{I}$，等价于标准GD（步长$1/\beta$）
2. **奇异值增强**：引入扩展系数$e \geq 1$，将初始参数放大为$\{eW_1^0, \ldots, eW_{L-1}^0\}$，使$\alpha_0$放大为$e^{L-1}\alpha_0$
3. **四条引理**给出$e$的最低要求：
    - Lemma 5.1: $e = \Omega(T^{1/(L-1)})$
    - Lemma 5.2: $e = \Omega(T^{(3T+6)/(TL-T-4L+6)})$（最严格，随$T$指数增长）
    - Lemma 5.3: $e = \Omega(T^{4/(L-1)})$
    - Lemma 5.4: $e = \Omega(T^{5/(L-1)} L^{1/(L-1)})$

增加网络深度$L$可减轻对$e$的需求。

## 实验关键数据

### 实验1：训练收敛性与鲁棒性

实验环境：Python 3.9, PyTorch 1.12.0, Ubuntu 20.04, 128GB RAM, 2x NVIDIA RTX 3090。数据：10个高斯随机问题实例，$X \in \mathbb{R}^{5120 \times 1}$, $Y \in \mathbb{R}^{4000 \times 1}$。网络：$L=3$层，$T=100$步，宽度5120。

| 实验设定 | 方法 | 结果 |
|---------|------|------|
| $T=100$, 多种学习率 | 本文L2O | 所有学习率（$10^{-3}$到$10^{-7}$）均稳定收敛 |
| $T=100$, 多种学习率 | Math-L2O | 低LR收敛慢，高LR训练崩溃 |
| $T=100$, 多种学习率 | LISTA-CPSS | 低LR收敛慢，高LR数值溢出 |
| 多种优化步数+LR组合 | 本文L2O | 全部配置一致收敛，超越GD基线 |

关键发现：本文方法在$\eta \in [10^{-3}, 10^{-7}]$范围内均达到相似收敛速率，验证了Theorem 4.3的线性收敛保证。相比之下，SOTA方法在100步前即因梯度爆炸而中断。

### 实验2：消融实验——学习率与扩展系数的交互

实验设定：$T=20$, $X \in \mathbb{R}^{32 \times 32}$, $Y \in \mathbb{R}^{32 \times 20}$, 网络宽度1024。性能指标为相对改进率$\frac{\text{obj}_{\text{GD}} - \text{obj}_{\text{L2O}}}{\text{obj}_{\text{GD}}}$。

| 固定参数 | 变化参数 | 关键观察 |
|---------|---------|---------|
| $e=50$ | $\eta \in \{10^{-3}, \ldots, 10^{-7}\}$ | $\eta=10^{-3}$导致不稳定/发散；$\eta \leq 10^{-4}$均收敛；更大的$\eta$收敛更快 |
| $\eta=10^{-7}$ | $e$取不同值 | 相对改进率随$e$增大而单调上升 |

实验精确验证了Corollary C.1：$e$和$\eta$存在反比关系，更大的$e$需要更小的$\eta$来保证收敛。学习率存在一个操作上界，低于该上界时更大的$\eta$收敛更快，符合Theorem 4.3的理论预测。

**综合性能**：本文L2O框架相比标准GD可达超过50%的最优性改进，且对超参数具有良好鲁棒性，优于LISTA-CPSS、Math-L2O和Adam优化器。

## 亮点

- **开创性理论贡献**：首次严格证明unrolling-based L2O的训练收敛性，建立了训练收敛率与目标优化收敛率的显式桥梁
- **实用的确定性初始化**：零初始化最后一层确保L2O初始行为等价于标准GD，避免了梯度爆炸；QR分解+非负化保证$\alpha_0 > 0$
- **NTK理论的新应用**：将NTK从标准ReLU网络扩展到具有循环结构和Sigmoid输出的Math-L2O架构，宽度需求仅$\mathcal{O}(Nd)$
- **理论-实验一致性强**：消融实验精确验证了$e$-$\eta$反比关系、学习率上界存在性等理论预测

## 局限与展望

- **仅限二次规划**：理论分析局限于$\frac{1}{2}\|\mathbf{M}x - y\|_2^2$形式的目标函数，未扩展到一般凸或非凸问题
- **合成数据验证**：实验仅在随机生成的高斯数据上进行，缺乏真实优化问题（如LASSO、压缩感知、信号恢复）的验证
- **学习率衰减较快**：$\eta$随$L$和$T$指数衰减，虽然理论允许但可能限制大规模问题的实用性
- **扩展系数$e$的选取**：$e$随$T$指数增长（Lemma 5.2），当优化步数很大时可能导致数值问题
- **坐标独立假设**：coordinate-wise架构将各维度独立处理，无法捕获维度间的耦合关系
- **未涉及随机优化**：分析基于全批量GD，未扩展到SGD等随机方法

## 与相关工作的对比

- **LISTA-CPSS (Chen et al. 2018)**：构造性证明线性收敛，但依赖推理时几乎不满足的条件（符号一致性、参数矩阵约束）；本文通过NTK直接证明训练收敛
- **Math-L2O (Liu et al. 2023)**：仅给出收敛必要条件，未证明训练收敛；本文证明其线性训练收敛率
- **Song et al. 2024**：分析Math-L2O推理时收敛，但假设L2O已学到模仿GD的行为；本文消除此假设
- **NTK理论 (Jacot 2018, Du 2018, Allen-Zhu 2019)**：本文将NTK从前馈网络和标准RNN扩展到具有循环+梯度结构的L2O模型，处理了参数高阶多项式依赖的新挑战
- **Nguyen et al. 2021**：提供了ReLU-Net的训练收敛分析；本文继承其松弛策略但面临更复杂的$\mathcal{O}(e^{LT})$半光滑系数

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次严格证明L2O的训练收敛性，理论贡献重要且原创
- 实验充分度: ⭐⭐⭐ — 仅在合成数据上验证，缺乏真实应用场景实验
- 写作质量: ⭐⭐⭐⭐ — 理论推导严谨，结构清晰，但大量符号定义增加阅读难度
- 价值: ⭐⭐⭐⭐ — 填补L2O收敛理论空白，但二次规划限制降低了通用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Multi-head Transformers Provably Learn Symbolic Multi-step Reasoning via Gradient Descent](multi-head_transformers_provably_learn_symbolic_multi-step_reasoning_via_gradien.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)
- [\[NeurIPS 2025\] OrthoGrad Improves Neural Calibration](orthograd_improves_neural_calibration.md)
- [\[NeurIPS 2025\] Gradient Descent as Loss Landscape Navigation: a Normative Framework for Deriving Learning Rules](gradient_descent_as_loss_landscape_navigation_a_normative_framework_for_deriving.md)
- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](../../ICML2026/optimization/on_the_convergence_rate_of_lora_gradient_descent.md)

</div>

<!-- RELATED:END -->
