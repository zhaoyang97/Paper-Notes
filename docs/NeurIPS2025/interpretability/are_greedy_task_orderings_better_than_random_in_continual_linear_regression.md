---
title: >-
  [论文解读] Are Greedy Task Orderings Better Than Random in Continual Linear Regression?
description: >-
  [NeurIPS 2025][可解释性][持续学习] 本文系统分析了持续线性回归中贪心任务排序（最大化连续任务间不相似度）与随机排序的收敛性差异，揭示了贪心排序在高秩设定下可媲美随机排序，但在一般秩设定下单遍贪心可能灾难性失败，而允许重复的贪心排序收敛速率为 $\mathcal{O}(1/\sqrt[3]{k})$。
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "持续学习"
  - "任务排序"
  - "线性回归"
  - "Kaczmarz方法"
  - "贪心策略"
---

# Are Greedy Task Orderings Better Than Random in Continual Linear Regression?

**会议**: NeurIPS 2025  
**arXiv**: [2510.19941](https://arxiv.org/abs/2510.19941)  
**代码**: 无  
**领域**: 持续学习 / 优化理论  
**关键词**: 持续学习, 任务排序, 线性回归, Kaczmarz方法, 贪心策略

## 一句话总结

本文系统分析了持续线性回归中贪心任务排序（最大化连续任务间不相似度）与随机排序的收敛性差异，揭示了贪心排序在高秩设定下可媲美随机排序，但在一般秩设定下单遍贪心可能灾难性失败，而允许重复的贪心排序收敛速率为 $\mathcal{O}(1/\sqrt[3]{k})$。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：持续学习（Continual Learning）中，模型需要按顺序学习多个任务，任务的呈现顺序对最终性能有显著影响。先前研究主要关注随机排序（random ordering），包括有放回和无放回两种形式，已证明其平均损失在 $k$ 次迭代后以 $\mathcal{O}(1/\sqrt{k})$ 收敛。然而，贪心排序——即每次选择与当前任务最不相似的下一个任务——虽然直觉上更优，但缺乏严格理论分析。

本文的核心动机是：

**贪心排序是否总是优于随机排序？** 直觉告诉我们多样化任务序列应该更好，但理论保证缺失

**允许任务重复与否对贪心排序有何影响？** 这涉及实际应用中的数据可用性约束

**Kaczmarz方法的理论工具如何迁移到持续学习分析？** 这是一个跨领域的方法论贡献

## 方法详解

### 整体框架

本文将持续线性回归问题形式化为 Kaczmarz 方法的变体。给定 $T$ 个任务，每个任务 $t$ 对应一个线性系统 $A_t x = b_t$，模型按某种顺序依次在各任务上执行梯度下降。

**不相似度度量**：定义两个连续任务之间的不相似度为其数据子空间之间的主角度（principal angles）的函数。贪心排序即在每一步选择与当前任务子空间主角度最大的下一个任务。

### 关键设计

**贪心排序的形式化**：
- **单遍贪心（Single-pass Greedy）**：每个任务恰好使用一次，贪心地按最大不相似度排序
- **允许重复的贪心（Greedy with Repetition）**：任务可以被多次选择，每步贪心选择与上一任务最不同的任务

**联合可实现性假设（Joint Realizability）**：假设存在一个全局最优解 $x^*$ 同时满足所有任务，即 $A_t x^* = b_t, \forall t$。这是一个合理的简化假设，在线性回归中对应过参数化设定。

**几何直觉**：作者利用子空间之间的几何关系（如正交性、主角度），建立了贪心排序选择行为的代数表达式，并给出了直观的几何解释——贪心排序倾向于在参数空间中沿正交方向交替更新。

### 损失函数 / 训练策略

模型使用均方误差损失，每个任务上执行一步梯度投影：

$$x_{k+1} = x_k - A_{i_k}^\dagger (A_{i_k} x_k - b_{i_k})$$

其中 $i_k$ 是第 $k$ 步选择的任务索引，$A_{i_k}^\dagger$ 是伪逆。

## 实验关键数据

### 主实验

**合成数据实验**：在随机生成的线性回归任务上比较不同排序策略。

| 排序策略 | 50次迭代平均损失 | 100次迭代平均损失 | 收敛趋势 |
|---------|----------------|-----------------|---------|
| 随机（有放回） | ~0.15 | ~0.08 | $\mathcal{O}(1/\sqrt{k})$ |
| 随机（无放回） | ~0.12 | ~0.06 | $\mathcal{O}(1/\sqrt{k})$ |
| 贪心（允许重复） | ~0.08 | ~0.04 | 实际更快 |
| 贪心（单遍） | 发散/不稳定 | - | 可能灾难性失败 |

**CIFAR-100 线性探针实验**：使用预训练特征进行线性分类，验证贪心排序在实际任务上的表现。

| 排序策略 | 10轮后平均准确率 | 50轮后平均准确率 | 最终准确率 |
|---------|---------------|----------------|----------|
| 随机排序 | 45.2% | 62.1% | 68.3% |
| 贪心排序（允许重复） | 52.8% | 67.5% | 72.1% |
| 贪心排序（单遍） | 38.6% | 不稳定 | 波动大 |

### 消融实验

- **秩的影响**：在高秩设定（所有任务数据矩阵满秩）下，贪心和随机排序的理论上界相同；在低秩设定下差异显现
- **任务数量的影响**：任务数增多时，贪心排序的优势更明显，但单遍贪心的失败风险也增大
- **重复次数的影响**：允许每个任务重复 2-3 次即可获得大部分收益

### 关键发现

1. **高秩设定**：贪心排序的损失上界与随机排序类同，均为 $\mathcal{O}(1/\sqrt{k})$
2. **一般秩设定**：单遍贪心可能灾难性失败（损失不收敛），而允许重复的贪心收敛速率为 $\mathcal{O}(1/\sqrt[3]{k})$
3. **实证优势**：尽管理论上界较松，贪心排序在实验中一致优于随机排序

## 亮点与洞察

- **方法论创新**：将 Kaczmarz 方法文献中的工具系统地引入持续学习理论分析，建立了两个领域的桥梁
- **反直觉发现**：单遍贪心排序可能比随机排序更差，揭示了"最大多样性"并非总是有益
- **实用指导**：允许任务重复是贪心排序成功的关键条件，这对课程学习（curriculum learning）和数据回放策略有直接启示
- **理论与实验的落差**：理论上界 $\mathcal{O}(1/\sqrt[3]{k})$ 慢于随机的 $\mathcal{O}(1/\sqrt{k})$，但实验中贪心更快，说明上界可能不紧

## 局限与展望

1. **限定于线性回归**：结论能否推广到非线性模型（如神经网络）尚不确定
2. **联合可实现性假设过强**：实际持续学习中任务间常存在冲突
3. **理论上界可能不紧**：实验表现远好于理论预测，暗示分析技术有改进空间
4. **贪心排序的计算成本**：计算所有任务对之间的不相似度在任务数量大时开销显著
5. **未考虑任务到达的在线设定**：实际中任务可能按流式到达，无法全局贪心选择

## 相关工作与启发

- **Kaczmarz方法**：经典的线性方程组迭代求解方法，随机化版本（Strohmer & Vershynin, 2009）是本文的重要理论基础
- **持续学习中的课程设计**：如何选择任务顺序是持续学习的核心问题之一，本文提供了严格的理论视角
- **灾难性遗忘**：贪心排序通过最大化任务多样性间接缓解遗忘，但单遍失败的发现提示多样性需与重复机制配合

## 评分

- 理论深度: ⭐⭐⭐⭐⭐
- 实验充分性: ⭐⭐⭐⭐
- 创新性: ⭐⭐⭐⭐
- 实用性: ⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] The Knowledge Microscope: Features as Better Analytical Lenses than Neurons](../../ACL2025/interpretability/the_knowledge_microscope_features_as_better_analytical_lenses_than_neurons.md)
- [\[NeurIPS 2025\] Towards Scaling Laws for Symbolic Regression](towards_scaling_laws_for_symbolic_regression.md)
- [\[NeurIPS 2025\] Do Different Prompting Methods Yield a Common Task Representation?](do_different_prompting_methods_yield_a_common_task_representation_in_language_mo.md)
- [\[NeurIPS 2025\] Emergence of Linear Truth Encodings in Language Models](emergence_of_linear_truth_encodings_in_language_models.md)
- [\[NeurIPS 2025\] Better Estimation of the Kullback-Leibler Divergence Between Language Models](better_estimation_of_the_kullback--leibler_divergence_between_language_models.md)

</div>

<!-- RELATED:END -->
