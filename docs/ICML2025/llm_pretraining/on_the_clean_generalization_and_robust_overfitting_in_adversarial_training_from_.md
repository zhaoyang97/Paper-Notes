---
title: >-
  [论文解读] On the Clean Generalization and Robust Overfitting in Adversarial Training from Two Theoretical Views: Representation Complexity and Training Dynamics
description: >-
  [ICML2025][预训练][对抗训练] 本文从**表示复杂度**和**训练动态**两个视角，理论解释了对抗训练中"干净泛化与鲁棒过拟合共存"(CGRO)现象：CGRO分类器仅需额外 $\tilde{O}(ND)$ 参数即可通过鲁棒记忆实现，而真正的鲁棒泛化在最坏情况下需要指数级模型容量；在结构化数据上，对抗训练的三阶段相变过程会使网络部分学习真特征、完全记忆噪声，从而可证地收敛到CGRO状态。
tags:
  - "ICML2025"
  - "预训练"
  - "对抗训练"
  - "鲁棒过拟合"
  - "干净泛化"
  - "表示复杂度"
  - "训练动态"
  - "特征学习"
---

# On the Clean Generalization and Robust Overfitting in Adversarial Training from Two Theoretical Views: Representation Complexity and Training Dynamics

**会议**: ICML2025  
**arXiv**: [2306.01271](https://arxiv.org/abs/2306.01271)  
**代码**: 无  
**领域**: LLM预训练  
**关键词**: 对抗训练, 鲁棒过拟合, 干净泛化, 表示复杂度, 训练动态, 特征学习

## 一句话总结

本文从**表示复杂度**和**训练动态**两个视角，理论解释了对抗训练中"干净泛化与鲁棒过拟合共存"(CGRO)现象：CGRO分类器仅需额外 $\tilde{O}(ND)$ 参数即可通过鲁棒记忆实现，而真正的鲁棒泛化在最坏情况下需要指数级模型容量；在结构化数据上，对抗训练的三阶段相变过程会使网络部分学习真特征、完全记忆噪声，从而可证地收敛到CGRO状态。

## 研究背景与动机

对抗训练（Adversarial Training）是提升模型对抗鲁棒性的主流方法，但实践中观察到一个矛盾现象：
- **干净泛化**：经过对抗训练的模型在干净测试数据上仍保持很高精度（如CIFAR10上>80%）
- **鲁棒过拟合**：鲁棒训练误差可降至接近0，但鲁棒测试误差居高不下（如CIFAR10上仅~50%），存在显著的鲁棒泛化差距

这种"Clean Generalization and Robust Overfitting"（CGRO）现象既不同于标准的 benign overfitting（干净测试也能泛化），也不能用简单的"鲁棒-精度权衡"来解释。已有工作虽从样本复杂度、Lipschitz稳定性等角度给出了部分解释，但存在两个关键缺口：

**机制不清**：未阐明对抗训练中什么机制直接导致了鲁棒过拟合

**忽视干净泛化**：大多数理论仅关注鲁棒性下降，忽略了干净测试精度依然很高这一事实

本文核心问题：**什么底层机制导致了对抗训练过程中干净泛化与鲁棒过拟合并存？**

## 方法详解

### 问题形式化

考虑二分类设定 $(X, y) \sim \mathcal{D}$，$y \in \{-1, 1\}$，分类器 $f: \mathcal{X} \to \mathbb{R}$。定义三个关键指标：
- **干净测试误差**：$\mathcal{L}_\mathcal{D}(f) = \mathbb{P}[\text{sgn}(f(X)) \neq y]$
- **鲁棒测试误差**：$\mathcal{L}_\mathcal{D}^{p,\delta}(f) = \mathbb{E}[\max_{\|X'-X\|_p \le \delta} \mathbb{I}\{\text{sgn}(f(X')) \neq y\}]$
- **鲁棒训练误差**：对训练集 $\mathcal{S}$ 的鲁棒误差

**CGRO分类器定义**：满足 $\mathcal{L}_\mathcal{D}(f) = o(1)$（干净泛化好）、$\mathcal{L}_\mathcal{S}^{p,\delta}(f) = o(1)$（鲁棒训练好）、但 $\mathcal{L}_\mathcal{D}^{p,\delta}(f) = \Omega(1)$（鲁棒泛化差）的分类器。

### 视角一：表示复杂度分析

在三个合理假设下（数据有界、类间分离>2δ、存在多项式大小的干净分类器），作者构造了一个关键的CGRO分类器：

$$f_\mathcal{S}(X) = \underbrace{f_{\text{clean}}(X)(1 - \mathbb{I}\{X \in \cup_i \mathbb{B}_p(X_i, \delta)\})}_{\text{对未见数据用干净分类}} + \underbrace{\sum_{i=1}^N y_i \mathbb{I}\{X \in \mathbb{B}_p(X_i, \delta)\}}_{\text{对训练数据做鲁棒记忆}}$$

**核心思想**：在训练数据点的 $\delta$-邻域内，直接记忆正确标签（鲁棒记忆）；在邻域外，使用干净分类器预测。

**定理4.4（CGRO多项式上界）**：该CGRO分类器可用 $\text{poly}(D) + \tilde{O}(ND)$ 参数的ReLU网络表示。关键是用ReLU网络近似距离函数 $\|X - X_i\|_p$ 和指示函数。

**定理4.7（鲁棒分类器指数下界）**：存在满足上述假设的分布 $\mathcal{D}$，使得任何参数量 $\le \Omega(\exp(D))$ 的ReLU网络都无法实现鲁棒泛化。

**关键不等式**：
$$\underbrace{\text{Clean Classifier}}_{\text{poly}(D)} \lesssim \underbrace{\text{CGRO Classifier}}_{\text{poly}(D) + \tilde{O}(ND)} \ll \underbrace{\text{Robust Classifier}}_{\Omega(\exp(D))}$$

这揭示了CGRO是一个"容量陷阱"——模型容量足以CGRO但远不及鲁棒泛化所需。

### 视角二：训练动态分析

在结构化数据（Patch Data）上分析对抗训练的学习过程：
- **数据结构**：$X = (X[1], \ldots, X[P])$，含一个信号patch $X[\text{signal}] = \alpha y w^*$ 和 $P-1$ 个噪声patch $X[j] \sim \mathcal{N}(0, (I_d - w^* w^{*\top})\sigma_p^2)$
- **模型**：两层卷积网络，宽度 $m = \Theta(N)$，使用 $\text{ReLU}^q$ 激活
- **训练**：梯度下降最小化对抗训练损失（logistic loss + 鲁棒正则化）

**定理5.9（三阶段相变）**：经 $T = \Omega(\text{poly}(d))$ 轮对抗训练后，网络：
1. **部分学习真特征**：$\mathcal{U}^{(T)} = \Theta(\alpha^{-q})$
2. **完全记忆噪声特征**：$\forall i, \mathcal{V}_i^{(T)} = \Theta(1)$

三阶段过程：
- **Phase I**（信号增长期）：信号分量 $u^{(t)}$ 二次增长，达到 $\tilde{\Omega}(\alpha^{-1})$ 量级，模型学到部分真特征
- **Phase II**（信号停滞期）：信号增长被噪声分量主导，增长趋于停止
- **Phase III**（噪声记忆期）：噪声分量 $v_{i,j}^{(t)}$ 二次增长至 $\Omega(1)$，模型通过记忆样本级噪声实现鲁棒训练精度

**物理直觉**：对抗训练迫使模型在训练点 $\delta$-邻域内正确分类。由于真特征方向的对抗扰动几乎可以完全翻转信号（$\delta \approx \alpha$），模型无法仅靠真特征抵抗扰动，转而记忆每个训练样本的噪声模式。这对训练数据有效但无法泛化——因为测试数据的噪声是全新的。

## 实验关键数据

### 主实验：不同模型大小的对抗训练表现

| 数据集 | 模型大小 | 干净测试Acc | 鲁棒测试Acc | 鲁棒训练Acc |
|--------|---------|------------|------------|------------|
| MNIST  | ×1      | 11.35      | 11.35      | 11.70      |
| MNIST  | ×8      | 11.35      | 11.35      | 11.70      |
| MNIST  | ×12     | 95.06      | 77.96      | 99.30      |
| MNIST  | ×16     | 94.85      | 83.43      | 99.50      |
| CIFAR10| ×1 (WRN)| 82.56      | 43.39      | 64.19      |
| CIFAR10| ×5      | 85.83      | 46.25      | 97.37      |
| CIFAR10| ×10     | 86.05      | 50.08      | 99.57      |

**发现**：模型变大时，鲁棒训练精度先上升；鲁棒泛化差距先增大后缓慢缩小；小模型(MNIST ×1/×8)对抗训练退化为平凡解，验证了表示复杂度理论。

### 合成数据实验

| 指标 | 训练 | 测试 |
|------|------|------|
| 干净Acc | 100.0 | 98.5 |
| 鲁棒Acc | 100.0 | 17.5 |

训练动态图（Figure 2c）清晰展示三阶段相变：信号分量先快速增长后停滞，噪声记忆逐步攀升至 $\Theta(1)$，与理论预测完全一致。

## 亮点与洞察

1. **CGRO概念的形式化**：首次明确定义了"干净泛化+鲁棒过拟合"共存现象，区别于单纯的鲁棒-精度权衡
2. **鲁棒记忆机制**：揭示了CGRO的核心机制——模型在训练点邻域内通过记忆噪声实现鲁棒训练精度，但这种记忆无法泛化
3. **复杂度层次分明**：建立了Clean ≲ CGRO ≪ Robust的表示复杂度层次，解释了为什么对抗训练自然收敛到CGRO而非鲁棒泛化
4. **三阶段分析技术**：将复杂的对抗训练动态解耦为三个可分析的阶段，是特征学习理论在对抗鲁棒性领域的首次应用
5. **理论-实验一致性**：合成数据实验精确复现了理论预测的三阶段相变

## 局限与展望

1. **数据假设较强**：Patch结构化数据与真实图像数据差距较大，单信号patch假设过于简化
2. **激活函数限制**：理论分析使用 $\text{ReLU}^q$（$q \ge 2$）而非标准ReLU，实践中并不常用
3. **扰动范围约束**：对抗扰动被限制在信号方向 $\text{span}(w^*)$ 上，简化了优化分析但偏离了真实的全方向扰动
4. **仅分析两层网络**：现代对抗训练使用深层网络（如WideResNet），两层CNN的分析能否推广存疑
5. **$\delta \approx \alpha$ 的设定**：鲁棒半径几乎等于信号幅度，是一个极端情形，实际中 $\delta/\alpha$ 的比值对CGRO的影响未讨论
6. **缺少缓解策略**：仅解释了CGRO现象，未提出如何利用该理论缓解鲁棒过拟合

## 相关工作与启发

- **鲁棒过拟合实证**：Rice et al. (2020) 首次系统观察了对抗训练中的鲁棒过拟合
- **样本复杂度**：Schmidt et al. (2018) 证明鲁棒泛化需要更多数据
- **表示复杂度**：Li et al. (2022) 证明鲁棒分类需指数级模型，本文推广到非线性可分设定
- **特征学习理论**：Allen-Zhu & Li (2020, 2022)、Jelassi & Li (2022) 的patch数据范式，本文首次将其引入对抗训练分析
- **记忆效应**：Dong et al. (2021)、Xu et al. (2021) 探索对抗训练中的记忆效应，本文进一步区分了干净泛化的保持

**启发**：该工作提示，缓解鲁棒过拟合可能需要(1)增大模型容量以逼近指数级需求，或(2)引入正则化抑制噪声记忆，或(3)设计数据增强使模型不依赖样本级噪声。

## 评分

- 新颖性: ⭐⭐⭐⭐ — CGRO形式化定义和双视角分析框架具有原创性
- 实验充分度: ⭐⭐⭐ — 合成数据验证充分，真实数据实验偏简单
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，证明思路展示得当
- 价值: ⭐⭐⭐⭐ — 对理解对抗训练本质有重要理论贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Disaggregation Reveals Hidden Training Dynamics: The Case of Agreement Attraction](../../NeurIPS2025/llm_pretraining/disaggregation_reveals_hidden_training_dynamics_the_case_of_agreement_attraction.md)
- [\[ICCV 2025\] ConstStyle: Robust Domain Generalization with Unified Style Transformation](../../ICCV2025/llm_pretraining/conststyle_robust_domain_generalization_with_unified_style_transformation.md)
- [\[ICLR 2026\] Intrinsic Training Dynamics of Deep Neural Networks](../../ICLR2026/llm_pretraining/intrinsic_training_dynamics_of_deep_neural_networks.md)
- [\[ACL 2025\] Adversarial Tokenization](../../ACL2025/llm_pretraining/adversarial_tokenization.md)
- [\[ECCV 2024\] PreLAR: World Model Pre-training with Learnable Action Representation](../../ECCV2024/llm_pretraining/prelar_world_model_pre-training_with_learnable_action_representation.md)

</div>

<!-- RELATED:END -->
