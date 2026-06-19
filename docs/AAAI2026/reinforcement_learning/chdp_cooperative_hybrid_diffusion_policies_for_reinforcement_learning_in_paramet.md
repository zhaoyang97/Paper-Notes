---
title: >-
  [论文解读] CHDP: Cooperative Hybrid Diffusion Policies for RL in Parametric Environments
description: >-
  [AAAI 2026][强化学习][混合动作空间] 将混合动作空间问题建模为两个agent的全合作博弈，分别用离散和连续扩散策略生成动作，通过顺序更新和Q引导码本解决策略冲突与高维可扩展性问题，成功率最高提升19.3%。 混合动作空间（Hybrid Action Space）同时包含离散选择和连续参数，广泛存在于机器人控制和…
tags:
  - "AAAI 2026"
  - "强化学习"
  - "混合动作空间"
  - "扩散策略"
  - "多智能体合作"
  - "向量量化码本"
  - "参数化动作MDP"
---

# CHDP: Cooperative Hybrid Diffusion Policies for RL in Parametric Environments

**会议**: AAAI 2026  
**arXiv**: [2601.05675](https://arxiv.org/abs/2601.05675)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 混合动作空间, 扩散策略, 多智能体合作, 向量量化码本, 参数化动作MDP

## 一句话总结

将混合动作空间问题建模为两个agent的全合作博弈，分别用离散和连续扩散策略生成动作，通过顺序更新和Q引导码本解决策略冲突与高维可扩展性问题，成功率最高提升19.3%。

## 研究背景与动机

混合动作空间（Hybrid Action Space）同时包含离散选择和连续参数，广泛存在于机器人控制和游戏AI中。例如足球射门需先选左脚/右脚（离散），再确定力度和角度（连续）。现有方法面临两大挑战：

**策略表达力不足**：多数方法基于单模态架构（高斯或确定性策略），无法捕捉多模态分布。当多组动作对同样有效时，策略要么折中平均，要么坍缩到单一模式

**可扩展性差**：高维离散空间中组合爆炸导致探索效率极低。n个执行器的开关组合为 $2^n$，当n=10时有1024种离散动作

HyAR虽提供了可扩展潜空间框架但受限于确定性策略的表达力；HyDo使用扩散策略但继承了PDQN的可扩展性瓶颈。没有方法能同时解决两个问题。

## 方法详解

### 整体框架

CHDP将混合动作空间建模为两个agent间的全合作博弈，形式化为PAMDP。框架包含三个核心组件：

- **离散agent**：扩散策略 $\pi_{\theta_d}$ 生成潜表示并通过码本量化为离散动作
- **连续agent**：扩散策略 $\pi_{\theta_c}$ 以离散码字为条件生成连续参数
- **Q引导码本**：可学习码本 $\mathbf{E}_\zeta \in \mathbb{R}^{K \times d_e}$ 将高维离散空间嵌入低维潜空间

推理流程为顺序协作：(1) 离散策略从噪声出发经反向扩散生成潜表示 $e$；(2) VQ量化映射到码本最近邻，获取离散动作索引 $k$ 和码字 $e_k$；(3) 连续策略以 $(s, e_k)$ 为条件扩散生成 $a^c$。

### 关键设计

**1. 双扩散策略的多模态表达**

离散策略采样：$e_{i-1} = \frac{1}{\sqrt{\alpha_i}}\left(e_i - \frac{1-\alpha_i}{\sqrt{1-\bar{\alpha}_i}}\epsilon_{\theta_d}(e_i, s, i)\right) + \sqrt{\beta_i}z$

连续策略额外以码字 $e_k$ 为条件：$a_{i-1}^c = \frac{1}{\sqrt{\alpha_i}}\left(a_i^c - \frac{1-\alpha_i}{\sqrt{1-\bar{\alpha}_i}}\epsilon_{\theta_c}(a_i^c, s, e_k, i)\right) + \sqrt{\beta_i}z$

条件关系显式建模了离散-连续动作间的依赖。

**2. 顺序更新机制**

解决同时更新两个策略时的优化冲突：

- **Step 1**：先更新离散策略，$\mathcal{L}(\theta_d) = \mathcal{L}_d(\theta_d) - \alpha \cdot \mathbb{E}[Q_\phi(s, e, a^c)]$，其中 $a^c$ 从replay buffer采样为固定目标
- **Step 2**：冻结离散策略，用更新后的 $\pi'_{\theta_d}$ 生成新潜表示，联合优化连续策略和码本。stop-gradient阻止梯度回传到离散策略

**3. Q引导码本**

受VQ-VAE启发但不使用重建目标——码字价值由下游Q值决定。量化操作 $k = \arg\min_k \|e - e_k\|^2$。梯度流刻意不对称：

- 码本通过连续策略Q值改善项获得梯度，使码字向支持高价值动作方向移动
- 离散策略通过相同Q函数但独立路径优化（连续动作从buffer采样，切断梯度）
- 两者被同一Q值度量隐式对齐到共享潜空间

### 损失函数 / 训练策略

采用DQL框架，每个策略的损失为 $\mathcal{L}(\theta) = \mathcal{L}_d(\theta) + \alpha \cdot \mathcal{L}_q(\theta)$，$\alpha = \eta / \mathbb{E}[|Q_\phi(s,a)|]$ 平衡两项。

Critic使用Double Q-learning：$y_t = r_t + \gamma \min_{j=1,2} Q'_{\phi'_j}(s_{t+1}, e_{t+1}, a_{t+1}^c)$

## 实验关键数据

### 主实验

8个PAMDP基准上的成功率对比（%）：

| 环境 | HPPO | PATD3 | PDQN-TD3 | HyAR-TD3 | **CHDP** |
|------|------|-------|----------|----------|----------|
| Goal | 0.0 | 0.0 | 71.4 | 77.3 | **80.9** |
| Hard Goal | 0.0 | 43.0 | 0.0 | 60.2 | **79.5** |
| Platform | 66.3 | 95.1 | 96.7 | 96.6 | **99.7** |
| Catch Point | 55.7 | 86.7 | 89.8 | 86.6 | **93.8** |
| Hard Move(n=4) | 3.3 | 63.9 | 79.7 | 91.4 | **94.2** |
| Hard Move(n=6) | 2.5 | 9.8 | 31.1 | 92.3 | **93.9** |
| Hard Move(n=8) | 2.3 | 4.6 | 6.6 | 88.3 | **90.6** |
| Hard Move(n=10) | 3.4 | 10.3 | 3.3 | 69.0 | **79.8** |

全部SOTA。Hard Goal比HyAR提升19.3个百分点；Hard Move(n=10)提升10.8个百分点。

### 消融实验

| 变体 | Hard Goal | Hard Move(n=6) |
|------|-----------|----------------|
| CHDP (Full) | **75.9±3.7** | **93.9±1.0** |
| w/o 扩散策略 | 51.3±10.2 | 45.1±19.5 |
| w/o 码本 | 71.0±6.0 | 11.1±6.9 |
| w/o 顺序更新 | 32.8±4.3 | 89.4±3.5 |
| w/o 两者 | 31.5±16.4 | 10.7±5.1 |

三个组件各自不可或缺，且对应解决不同挑战。

### 关键发现

- HyAR确定性策略在Hard Move(n=6)坍缩为单一模式（100%频率选同一方向，方差=0）
- CHDP发现至少3种策略（频率79%/17%/4%），包括反直觉的方向反转策略
- 学习曲线显示CHDP收敛更快，样本效率更高

## 亮点与洞察

1. 将混合动作空间转化为全合作博弈是一个优雅的问题抽象
2. 定性实验直观证明扩散策略的多模态优势——发现确定性策略无法到达的反直觉解
3. 码本用下游Q值而非重建目标引导，使离散表示天然任务对齐

## 局限与展望

1. 仅在PAMDP基准上验证，缺乏更复杂真实场景（机器人操作等）实验
2. 扩散模型多步采样的推理延迟可能限制实时性要求高的场景
3. 码本大小固定为K，未探索自适应机制
4. 能否用梯度投影等方法替代顺序更新降低复杂度值得研究

## 相关工作与启发

- **HyAR**：可扩展潜空间框架但受限于确定性策略；CHDP码本设计更优雅
- **DQL**：将扩散模型引入RL的先驱，CHDP是其在混合空间的自然扩展
- **VQ-VAE**：码本灵感来源，但CHDP用Q值替代重建目标是关键创新
- **HARL**：顺序更新机制的理论基础

## 评分

- 新颖性: ⭐⭐⭐⭐ （合作博弈+扩散策略+Q引导码本组合新颖）
- 技术深度: ⭐⭐⭐⭐ （梯度流设计、多组件协调扎实）
- 实验充分性: ⭐⭐⭐⭐ （消融完整，定性分析有说服力）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，图示直观）
- 实用价值: ⭐⭐⭐ （基准偏简单，实际应用缺乏验证）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Learning Unmasking Policies for Diffusion Language Models](../../ICML2026/reinforcement_learning/learning_unmasking_policies_for_diffusion_language_models.md)
- [\[AAAI 2026\] Formal Verification of Diffusion Auctions](formal_verification_of_diffusion_auctions.md)
- [\[AAAI 2026\] One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow](one-step_generative_policies_with_q-learning_a_reformulation_of_meanflow.md)
- [\[AAAI 2026\] Explaining Decentralized Multi-Agent Reinforcement Learning Policies](explaining_decentralized_multi-agent_reinforcement_learning_policies.md)
- [\[NeurIPS 2025\] Adaptive Cooperative Transmission Design for URLLC via Deep RL](../../NeurIPS2025/reinforcement_learning/adaptive_cooperative_transmission_design_for_ultra-reliable_low-latency_communic.md)

</div>

<!-- RELATED:END -->
