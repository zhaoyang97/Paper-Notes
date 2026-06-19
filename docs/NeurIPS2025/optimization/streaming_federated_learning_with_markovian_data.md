---
title: >-
  [论文解读] Streaming Federated Learning with Markovian Data
description: >-
  [NeurIPS 2025][优化/理论][联邦学习] 首次严格分析了非凸目标函数下具有马尔可夫数据流的流式联邦学习，证明 Minibatch SGD、Local SGD 和 Local SGD-M 均能实现与客户端数成反比的样本复杂度（线性加速），且 Local SGD-M 无需异质性假设即可匹配 Minibatch SGD 的通信复杂度。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "联邦学习"
  - "马尔可夫数据流"
  - "随机梯度下降"
  - "非凸优化"
  - "客户端漂移"
---

# Streaming Federated Learning with Markovian Data

**会议**: NeurIPS 2025  
**arXiv**: [2503.18807](https://arxiv.org/abs/2503.18807)  
**代码**: 无  
**领域**: 优化  
**关键词**: 联邦学习, 马尔可夫数据流, 随机梯度下降, 非凸优化, 客户端漂移

## 一句话总结
首次严格分析了非凸目标函数下具有马尔可夫数据流的流式联邦学习，证明 Minibatch SGD、Local SGD 和 Local SGD-M 均能实现与客户端数成反比的样本复杂度（线性加速），且 Local SGD-M 无需异质性假设即可匹配 Minibatch SGD 的通信复杂度。

## 研究背景与动机
传统联邦学习（FL）假设客户端拥有预采集的数据集，但在健康监测、环境监控、机器人控制等场景中，客户端持续收集数据，形成数据流。这些数据流的特点是：

**马尔可夫依赖性**：由物理/生物系统产生的数据自然具有时间相关性，而非 i.i.d. 采样

**内存约束**：客户端只能存储有限数量的当前样本，无法控制采样过程

**非平稳性**：马尔可夫链的初始分布与稳态分布不同，导致早期样本偏差

**核心问题**：FL 在马尔可夫数据流下是否仍能支持协作学习？具体来说，每个客户端的样本复杂度能否随客户端数 $M$ 线性减少？

**现有工作不足**：
- 集中式设置下的马尔可夫 SGD 已有研究，但联邦场景聚焦于联邦强化学习（FRL）且限于凸目标或线性函数逼近
- FL 中的流式数据研究要么假设对抗环境，要么假设 i.i.d. 数据流

**本文贡献**：在非凸光滑目标函数下，严格刻画马尔可夫采样对 FL 收敛率的影响，并证明动量可以在无异质性假设下消除客户端漂移。

## 方法详解

### 整体框架
考虑 $M$ 个客户端协作最小化 $\min_{w} \frac{1}{M}\sum_{m=1}^M F_m(w)$，其中每个 $F_m(w) = \mathbb{E}_{x \sim \pi_m}[f_m(w;x)]$。数据流 $X_m = (x_t^m)_{t \in \mathbb{N}}$ 建模为时间齐次马尔可夫链，收敛到唯一稳态分布 $\pi_m$。

### 关键设计

1. **Minibatch SGD 分析**（问题类 $\mathcal{F}_1$）：

    - 仅需全局平滑性（Assumption 4.1）+ 有界梯度噪声（Assumption 4.3）
    - 收敛上界：$\mathbb{E}[\|\nabla F(\hat{w}_T)\|^2] \leq \mathcal{O}\left(\frac{\Delta_0}{\gamma T} + \frac{C_\infty \sigma^2}{\nu_{ps} MK}\right)$
    - 第二项反映马尔可夫数据的代价：梯度噪声被谱隙 $\nu_{ps}$ 的倒数放大，且**无法通过调小学习率消除**
    - 样本复杂度 $K = \mathcal{O}(C_\infty \sigma^2 / (\nu_{ps} M\epsilon^2))$：与 $1/M$ 成正比 → **线性加速✓**

2. **Local SGD 分析**（问题类 $\mathcal{F}_3$，需要异质性假设）：

    - 额外需要逐样本平滑性（Assumption 4.2）+ 有界梯度散度 BGD（Assumption 4.4）
    - 多出客户端漂移项 $\frac{LK\eta(\theta^2+\sigma^2)}{\delta^2}$，可通过小学习率控制
    - 通信复杂度 $T = \mathcal{O}\left(\frac{L\Delta_0}{\epsilon^2}(\delta^2 \vee \frac{\theta^2+\sigma^2}{\delta^2\epsilon^2})\right)$
    - 只有在低异质性、低噪声条件下才能达到最优通信复杂度

3. **Local SGD with Momentum（Local SGD-M）**：

    - 核心更新：$v_t^{(m,k)} = \beta \nabla f_m(w_t^{(m,k)}; x_t^{(m,k)}) + (1-\beta) v_t$
    - 将梯度估计与上一轮聚合更新做凸组合
    - **关键优势**：无需 BGD 异质性假设，在问题类 $\mathcal{F}_2$ 上即可工作
    - 收敛上界：$\mathbb{E}[\|\nabla F(\hat{w}_T)\|^2] \leq \mathcal{O}\left(\frac{L\Delta_0}{\beta T} + \frac{C_\infty \sigma^2}{\nu_{ps} MK}\right)$
    - 通信复杂度 $T = \mathcal{O}(L\Delta_0/(\beta\epsilon^2))$：仅多一个常数 $1/\beta$，达到 i.i.d. 下界

4. **更宽松的马尔可夫假设**：

    - 不要求常用的一致几何遍历性（指数快速收敛到稳态），仅需转移核对稳态测度的绝对连续性（Assumption 3.2）
    - 通过简单的测度变换处理非平稳性，而非依赖快速混合
    - 结果更具广泛适用性

### 损失函数 / 训练策略
- 非凸光滑损失函数 + 全局有界下方
- 服务器聚合采用简单平均

## 实验关键数据

### 主实验

| 方法 | K=10 表现 | K=50 表现 | K=100 表现 | 说明 |
|------|----------|----------|-----------|------|
| Minibatch SGD | 持续改善 | 持续改善 | 持续改善 | 随K增大梯度范数持续下降 |
| Local SGD-M | 持续改善 | 持续改善 | 持续改善 | 与 Minibatch 趋势一致 |
| Local SGD | 改善 | 边际改善 | 几乎不变 | 异质性成瓶颈 |
| SCAFFOLD | 改善 | 几乎不变 | 几乎不变 | 马尔可夫数据下优势不明显 |

实验使用北京多站空气质量数据集（12个站点，2013-2017），预测 PM2.5 浓度。

### 消融实验（客户端数影响）

| 方法 | 60客户端 | 120客户端 | 240客户端 | 说明 |
|------|---------|----------|----------|------|
| Minibatch SGD | 梯度范数较高 | 中等 | 最低 | 线性加速验证 |
| Local SGD | 同上趋势 | 同上 | 同上 | 协作确实有益 |
| Local SGD-M | 同上趋势 | 同上 | 同上 | 不受异质性影响 |

### 关键发现
- 更多客户端 → 所有三种方法的梯度范数都下降，验证了协作在马尔可夫数据下确实有益
- Local SGD 在 K 增大时性能饱和（受异质性限制），而 Minibatch SGD 和 Local SGD-M 持续受益
- SCAFFOLD（i.i.d. 下消除异质性的经典算法）在马尔可夫数据下效果不佳
- Local SGD-M 用动量代替控制变量来消除客户端漂移，更简单且效果更好

## 亮点与洞察
- **理论清洁**：分层定义三个问题类 $\mathcal{F}_3 \subset \mathcal{F}_2 \subset \mathcal{F}_1$，对应三种算法的假设需求，层次分明
- **动量的新作用**：动量不仅是加速手段，更是消除客户端漂移的机制——通过与前一轮全局方向做凸组合，自然约束了局部更新的偏离
- **宽松假设的实际意义**：不要求一致几何遍历性（现实中很难验证），使结果可应用于更广泛的物理/生物系统

## 局限与展望
- 仅考虑独立客户端（$X_m$ 与 $X_{m'}$ 独立），未处理空间相关数据流
- 马尔可夫数据的样本复杂度仍高于 i.i.d.，被 $C_\infty / \nu_{ps}$ 因子放大
- 未考虑部分参与（partial participation）场景
- 动量 + 控制变量在马尔可夫数据下的组合效果未分析

## 相关工作与启发
- **vs SCAFFLSA (Wang et al.)**: SCAFFLSA 用控制变量消除异质性但需要更高的每轮样本数和异质性假设，Local SGD-M 更简洁
- **vs Deng et al. (2024)**: 联邦强化学习中建立线性加速但限于线性函数逼近和凸目标，本文扩展到非凸
- **vs Sun et al. (2018)**: 集中式马尔可夫 SGD 不涉及通信复杂度问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 马尔可夫数据流 × 联邦学习 × 非凸目标的组合是新的理论贡献
- 实验充分度: ⭐⭐⭐ 实验规模较小（线性回归+非凸正则化），缺乏深度学习实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论表述严谨清晰，假设和结果的关系梳理得非常好
- 价值: ⭐⭐⭐⭐ 填补了流式FL理论的空白，对健康监测/环境监控等应用有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[NeurIPS 2025\] Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients](efficient_federated_learning_against_byzantine_attacks_and_data_heterogeneity_vi.md)
- [\[NeurIPS 2025\] Optimality and NP-Hardness of Transformers in Learning Markovian Dynamical Functions](optimality_and_np-hardness_of_transformers_in_learning_markovian_dynamical_funct.md)
- [\[NeurIPS 2025\] Multiplayer Federated Learning: Reaching Equilibrium with Less Communication](multiplayer_federated_learning_reaching_equilibrium_with_less_communication.md)
- [\[NeurIPS 2025\] Personalized Subgraph Federated Learning with Differentiable Auxiliary Projections](personalized_subgraph_federated_learning_with_differentiable_auxiliary_projectio.md)

</div>

<!-- RELATED:END -->
