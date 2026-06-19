---
title: >-
  [论文解读] Non-Clairvoyant Scheduling with Progress Bars
description: >-
  [NeurIPS 2025][在线调度算法 / 学习增强算法 / 竞争分析][非透视调度] 引入"进度条"信息模型作为透视与非透视调度之间的插值框架，针对对抗性和随机性进度条分别设计了具有最优一致性-鲁棒性权衡的调度算法，同时推进了学习增强调度的理论前沿。 经典单机调度问题：$n$个作业各有处理时间$p_j$…
tags:
  - "NeurIPS 2025"
  - "在线调度算法 / 学习增强算法 / 竞争分析"
  - "非透视调度"
  - "进度条"
  - "竞争比"
  - "学习增强算法"
  - "探索-利用权衡"
---

# Non-Clairvoyant Scheduling with Progress Bars

**会议**: NeurIPS 2025  
**arXiv**: [2509.19662](https://arxiv.org/abs/2509.19662)  
**代码**: 暂无  
**领域**: 在线调度算法 / 学习增强算法 / 竞争分析  
**关键词**: 非透视调度, 进度条, 竞争比, 学习增强算法, 探索-利用权衡

## 一句话总结

引入"进度条"信息模型作为透视与非透视调度之间的插值框架，针对对抗性和随机性进度条分别设计了具有最优一致性-鲁棒性权衡的调度算法，同时推进了学习增强调度的理论前沿。

## 研究背景与动机

经典单机调度问题：$n$个作业各有处理时间$p_j$，目标是最小化总完成时间$\sum C_j$。透视(clairvoyant)设定下最优策略是SPT(最短处理时间优先)；非透视(non-clairvoyant)设定下最优策略是Round-Robin，竞争比为2（已知最优）。这两个极端之间信息模型的差距巨大。

学习增强算法(algorithms with predictions)尝试用预测信息弥补这一差距，但现有模型给出的是**静态的先验预测**（基于作业ID等），无法随执行动态改善。这在profiling等场景中不自然——随着作业执行，我们对其处理时间的估计应当越来越准确。

本文核心提问：**是否存在一种信息模型，允许算法在执行过程中动态改善其决策？** 由此引入进度条模型——在处理作业时，调度器收到已完成百分比的估计(可能不准确)。这自然地在透视($\varphi_j: x \mapsto x$)和非透视($\varphi_j: x \mapsto \mathbb{1}(x=1)$)之间插值。

## 方法详解

### 整体框架

按信息模型复杂度递进：(1) 单信号进度条(granularity $g=1$)——作业在处理$\alpha$比例后发出一次信号；(2) 不可信进度条(granularity $g$)——多级信号+算法组合策略；(3) 随机进度条——Poisson模型下的探索-利用。

### 关键设计

1. **单信号——可信信号 (Section 2.1)**: 每个作业在精确处理$\alpha$比例后发出信号。算法：Round-Robin直到某作业发出信号，然后优先执行该作业至完成。**Theorem**: 竞争比为$1+\alpha$，对随机化算法也最优。设计动机：信号将调度问题分为两阶段——发信号前全部并行(无法区分)，发信号后切换到优先执行(利用新信息)。

2. **单信号——不可信信号 (Algorithm 1)**: 信号可能在$\beta_j \neq \alpha$时发出。核心困难：直接信任信号虽然$(1+\alpha)$-一致但缺乏鲁棒性(Lemma)，且朴素鲁棒化(时间共享)无法达到完美一致性。**Algorithm 1**引入参数$\rho \in (0,1]$：收到信号后优先执行$(1/\alpha\rho - 1) \cdot e_j(t)$时间单位(其中$e_j(t)$是已处理时间)。如果未完成则切换到SETF(最短已执行时间优先)直到追赶上。**Theorem**: $(1+\alpha)$-一致、$(1+1/\rho\alpha)$-鲁棒，且平滑性界为 $\text{ALG} \leq (1+\alpha)\text{OPT} + \frac{2n}{\rho(1-\rho)\alpha^2}\sum_i |\beta_i - \alpha|p_i$。$\rho$控制了**平滑性-鲁棒性权衡**（而非通常的一致性-鲁棒性权衡）。

3. **算法组合策略 (Algorithm 2, Section 3)**: 给定$g$个调度算法$A^{(1)},\ldots,A^{(g)}$（各自可能使用不同预测），如何选择最优的？方法：随机采样$m$对作业运行至完成，计算每个算法的经验延迟，选择最小延迟的算法调度剩余作业。**Theorem**: $\mathbb{E}[\text{ALG}] \leq \min_h A^{(h)} + \frac{9}{4}n^{5/3}(\log g)^{1/3} \max_i p_i$。当$\max_i p_i = o(n^{-5/3}\text{OPT})$时，regret项为$o(\text{OPT})$。这推广了Eliás et al. (2024)的排列预测组合到任意可计算延迟的算法。

4. **随机进度条 (Algorithm 3, Section 4)**: 进度条跳跃按Poisson过程产生，形成多臂老虎机类似结构。**重复探索-提交算法**：Round-Robin直到某作业进度达到$k/(g+1)$，提交(执行至完成)后继续Round-Robin。**Theorem**: 选$k = \lceil(g/2)^{2/3}\rceil + 1$时竞争比为$1 + O(g^{-1/3})$。**下界**$1 + \Omega(g^{-1/3})$，证明算法渐近最优。下界通过构造两个相近处理时间的作业，利用Poisson过程的不可区分性论证。

### 学习增强调度的推论

将预测$\pi_j$转化为信号$\beta_j = \alpha\pi_j/p_j$，算法1直接适用。这给出了$(1+\alpha)$-一致、$(1+1/\rho\alpha)$-鲁棒的算法，改进了此前最优的$(1/\lambda, 2/(1-\lambda))$权衡(Purohit et al. 2018)。组合策略(Algorithm 2)更进一步实现了$(1+o(1))$-一致、$(2+o(1))$-鲁棒（在一定实例类别上）。

## 实验关键数据

### 主实验（n=500，Pareto(1.1)分布，σ为噪声水平）

| 配置 | 竞争比(σ=0) | 竞争比(σ=1) | 竞争比(σ=5) | 说明 |
|------|-----------|-----------|-----------|------|
| Alg.1, ρ→0 (平滑) | 1+α≈1.5 | ~1.6 | ~1.8 | 平滑退化 |
| Alg.1, ρ=0.5 | 1+α≈1.5 | ~1.55 | ~2.0 | 中间权衡 |
| Alg.1, ρ=1 (鲁棒) | 1+α≈1.5 | ~1.5 | ~3.0 | 无误差时最优但脆性明显 |
| Time sharing (基线) | ~1.6 | ~1.65 | ~2.0 | 一致性差但鲁棒 |
| Combining (Alg.2) | ~1.5 | ~1.55 | ~1.8 | 无需调参，性能平衡 |

### 消融实验（随机进度条）

| 粒度g | Alg.3 (k=Θ(g^{2/3})) | Alg.3 (k=1) | Round-Robin | 通用ETC |
|-------|----------------------|-------------|-------------|---------|
| 10 | ~1.35 | ~1.55 | 2.0 | ~1.65 |
| 50 | ~1.15 | ~1.40 | 2.0 | ~1.55 |
| 200 | ~1.08 | ~1.25 | 2.0 | ~1.30 |

### 关键发现

- **ρ=1时存在脆性(brittleness)**：Algorithm 1在ρ=1时对微小信号误差就会跳到竞争比2(Proposition A.3)，验证了平滑性的重要性。
- **组合算法的实际优势**：无需参数调优，在不同噪声水平下均表现平衡，gap随$n$增大更显著。
- **随机进度条的$g^{-1/3}$收敛率**：实验验证了竞争比随$g$增大趋近于1，且$k=\Theta(g^{2/3})$显著优于$k=1$和通用ETC。

## 亮点与洞察

- 进度条模型提供了透视/非透视之间的自然连续体，具有直觉吸引力和实际意义
- 首次实现了$(1+\alpha)$-一致同时利用**预测数值**(而非仅排列)进行鲁棒化，建立了数值预测和排列预测之间的分离
- 组合策略的一般性令人印象深刻——可组合任何具有可计算延迟的调度算法
- $1+\Theta(g^{-1/3})$的紧致竞争比利用了Poisson过程的统计不可区分性，类似于bandit下界的论证

## 局限与展望

- 一致性-鲁棒性权衡下界$1+\Omega(1/\sqrt{\alpha})$与上界$1+1/\alpha$之间仍有$\Theta(\sqrt{\alpha})$的gap
- 组合策略的regret项$O(n^{5/3}\max p_i)$在$\max p_i$较大时可能主导
- 随机进度条模型在$g \leq 12$时无法改进竞争比2
- 未考虑多机并行调度的一般情形(Appendix E有初步扩展)

## 相关工作与启发

- 与Benomar & Perchet (2024)的部分预测非透视调度互补——该工作在不同时刻获取预测，本文在执行过程中连续获取
- 组合策略推广了Eliás et al. (2024)的排列预测组合，可用于更广泛的学习增强在线问题
- 随机进度条与mortal bandits(Chakrabarti et al. 2008)的联系可能启发更多bandit理论工具的应用

## 核心结果速查

- 可信单信号: 竞争比 $1+\alpha$（最优）
- 不可信单信号(Alg.1): $(1+\alpha)$-一致, $(1+1/\rho\alpha)$-鲁棒
- 组合策略(Alg.2): regret $O(n^{5/3}(\log g)^{1/3}\max p_i)$
- 随机进度条(Alg.3): 竞争比 $1+O(g^{-1/3})$, 下界 $1+\Omega(g^{-1/3})$
- 学习增强推论: 改进了$(1/\lambda, 2/(1-\lambda))$到$(2/(1+\lambda), 2/(1-\lambda))$的权衡

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 进度条模型是非常自然且富有洞察力的新信息模型；紧致的$g^{-1/3}$率令人满意
- 实验充分度: ⭐⭐⭐⭐ 合成实验充分验证了理论预测，实际应用场景评估可更多
- 写作质量: ⭐⭐⭐⭐⭐ 模型定义清晰，结果层次递进，证明思路解释充分
- 价值: ⭐⭐⭐⭐⭐ 对学习增强调度理论有实质推进，改进了最优一致性-鲁棒性权衡

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] A Switching Framework for Online Interval Scheduling with Predictions](../../AAAI2026/learning_theory/a_switching_framework_for_online_interval_scheduling_with_pr.md)
- [\[ICLR 2026\] ATLAS: Alibaba Dataset and Benchmark for Learning-Augmented Scheduling](../../ICLR2026/learning_theory/atlas_alibaba_dataset_and_benchmark_for_learning-augmented_scheduling.md)
- [\[ICLR 2026\] Scalable Random Wavelet Features: Efficient Non-Stationary Kernel Approximation with Convergence Guarantees](../../ICLR2026/learning_theory/scalable_random_wavelet_features_efficient_non-stationary_kernel_approximation_w.md)
- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](learning-augmented_online_bipartite_fractional_matching.md)
- [\[NeurIPS 2025\] Optimism Without Regularization: Constant Regret in Zero-Sum Games](optimism_without_regularization_constant_regret_in_zero-sum_games.md)

</div>

<!-- RELATED:END -->
