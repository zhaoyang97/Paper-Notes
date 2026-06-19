---
title: >-
  [论文解读] DeepProofLog: Efficient Proving in Deep Stochastic Logic Programs
description: >-
  [AAAI 2026 Oral][强化学习][神经符号AI] 提出DeepProofLog（DPrL），一种基于随机逻辑程序的神经符号系统，通过在每个证明步骤引入神经网络参数化，并建立SLD解析过程与MDP的形式化映射，使得动态规划和强化学习技术可用于高效推理与学习，显著提升了神经符号系统的可扩展性。
tags:
  - "AAAI 2026 Oral"
  - "强化学习"
  - "神经符号AI"
  - "随机逻辑程序"
  - "马尔可夫决策过程"
  - "动态规划"
  - "策略梯度"
---

# DeepProofLog: Efficient Proving in Deep Stochastic Logic Programs

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.08581](https://arxiv.org/abs/2511.08581)  
**代码**: [DeepProofLog/DPrL-AAAI](https://github.com/DeepProofLog/DPrL-AAAI)  
**领域**: 强化学习  
**关键词**: 神经符号AI, 随机逻辑程序, 马尔可夫决策过程, 动态规划, 策略梯度

## 一句话总结

提出DeepProofLog（DPrL），一种基于随机逻辑程序的神经符号系统，通过在每个证明步骤引入神经网络参数化，并建立SLD解析过程与MDP的形式化映射，使得动态规划和强化学习技术可用于高效推理与学习，显著提升了神经符号系统的可扩展性。

## 研究背景与动机

神经符号AI（NeSy）结合神经网络和符号推理以提升准确性、可解释性和泛化能力。主流NeSy系统通常分两步推理：(1) 符号证明找到查询的所有证明；(2) 在证明上做概率推理计算查询成立的概率。然而这对应加权模型计数问题，是#P-hard的。

现有两类加速方案都有局限：

**采样/变分方法**（如Scallop、A-NeSI）：近似概率推理阶段但不解决证明搜索的计算开销

**特定任务的解决方案**：需要大量人工适配，不够通用

更根本的问题是：现有方法不处理**寻找证明本身**的计算代价，在大规模关系推理场景下仍然不可行。

此外，两种已有的随机逻辑程序系统有表达力限制：
- **SLP**（标准随机逻辑程序）：子句概率是固定标量，不依赖查询上下文
- **DeepStochLog**：用神经网络条件化子句概率，但要求固定输出域，且仅以部分目标为条件

## 方法详解

### 整体框架

DPrL的核心转变：**不参数化逻辑程序本身（如子句权重），而是参数化解析过程**。具体地，在SLD解析的每一步，用神经网络计算转移概率——给定当前目标 $G_i$，评估所有可能的下一步目标 $G_{i+1}$ 的概率：

$$p_\lambda(G_{i+1} | G_i) = \frac{\exp(f_\lambda(G_i, G_{i+1}))}{\sum_{G \in \mathcal{N}(G_i)} \exp(f_\lambda(G_i, G))}$$

其中 $\mathcal{N}(G)$ 是解析最左原子后所有可能的下一步目标集合，$f_\lambda$ 计算当前目标和候选目标嵌入的兼容性得分。

**与MDP的形式化映射**是本文最核心的理论贡献：

- **状态** $\mathcal{S}$：初始查询 ∪ 中间子目标 ∪ {True, False}
- **动作** $\mathcal{A}(G)$：所有可用的(子句, 统一替换)对
- **转移函数**：确定性，$P(G'|G_i, C, \theta) = \mathbf{1}\{G' = \text{res}(G_i, C, \theta)\}$
- **奖励**：仅在终端状态给出，正查询证明成功+1，负查询证明成功-1
- **策略**：$\pi_\lambda(a|G_i) = p_\lambda(G_{i+1}|G_i)$，即解析转移模型

### 关键设计

**1. 层次化目标表示**

为使符号目标可被神经网络处理，构建三层嵌入：

- **（子）符号嵌入**：所有实体（谓词、常量、变量、图像等）共享可学习嵌入 $\mathbf{e}_v \in \mathbb{R}^d$
- **原子嵌入**：$\mathbf{e}_A = f_{\text{atom}}(\mathbf{e}_r, \mathbf{e}_{t_1}, \ldots, \mathbf{e}_{t_n})$，可用任意神经网络或知识图谱嵌入模型
- **目标嵌入**：$\mathbf{e}_G = f_{\text{agg}}(\mathbf{e}_{A_1}, \ldots, \mathbf{e}_{A_n})$，聚合所有原子的嵌入
- **兼容性得分**：$f_\lambda(G, G') = \mathbf{e}_G \cdot \mathbf{e}_{G'}$（内积）

这种设计使策略以**完整当前目标**为条件（而非仅部分子目标），能做出更知情的决策。

**2. 精确推理 — 动态规划**

当符号目标空间可管理时，直接用DP精确求解MDP的值函数：

$$V^{\pi_\lambda}(G, y) = \sum_{a \in \mathcal{A}(G)} \pi_\lambda(\text{res}(G,a), G) \cdot V^{\pi_\lambda}(\text{res}(G,a), y)$$

边界条件 $V^{\pi_\lambda}(G,y) = R(G,y)$ 对终端目标成立。

关键定理证明：最大化MDP的期望回报 $J(\pi_\lambda)$ 等价于最小化NeSy的标准学习目标 $\mathcal{L}(\lambda)$：

$$J(\pi_\lambda) = \sum_{i=1}^N (2y^{(i)} - 1) \cdot p_{\text{success}}^{\pi_\lambda}(q^{(i)})$$

**3. 近似推理 — 策略梯度**

当目标空间不可枚举时，用PPO（近端策略优化）近似求解：
- 用标准clipped surrogate objective优化神经策略
- 同架构的值函数网络估计回报、降低方差
- 逻辑程序的结构自然约束有效动作集，减少采样方差

**4. 万能近似性（Proposition 1）**

证明DPrL可以以任意精度近似SLD推导上的任意概率分布，保证了模型的表达能力。

### 损失函数 / 训练策略

学习目标：

$$\mathcal{L}(\lambda) = \sum_{(q^{(i)}, y^{(i)}) \in \mathcal{D}} \ell(p_{\text{success}}^\lambda(q^{(i)}), y^{(i)})$$

其中 $\ell(p, y) = (1 - 2y) \cdot p$ 是线性损失，使正查询的成功概率最大化，负查询最小化。

DP模式：通过自动微分直接优化值函数。PG模式：PPO + 值函数基线。

## 实验关键数据

### MNIST加法（可扩展性测试）

预测两个N位MNIST数字序列之和，仅提供最终和的监督。

| N | 参考精度 | DeepStochLog | DeepSoftLog | **DPrL(DP)** | Scallop | A-NeSI | **DPrL(PG)** |
|---|---------|-------------|-------------|-------------|---------|--------|-------------|
| 4 | 92.9% | 92.7% | 93.5% | **94.0%** | 92.3% | 92.6% | 93.5% |
| 15 | 75.8% | T/O | 77.1% | **80.8%** | T/O | 75.9% | 76.9% |
| 100 | 15.4% | T/O | T/O | **37.8%** | T/O | T/O | 0.0% |
| 200 | 2.4% | T/O | T/O | **23.2%** | T/O | T/O | 0.0% |
| 500 | 0.009% | T/O | T/O | **6.0%** | T/O | T/O | 0.0% |

DPrL(DP)是唯一能扩展到N=100+的方法。训练时间方面，N=4时DPrL仅需25分钟 vs DeepStochLog 141分钟。

### 知识图谱补全

| 方法 | Family MRR | Family H@1 | WN18RR MRR | WN18RR H@1 |
|------|-----------|-----------|-----------|-----------|
| ComplEx | 0.964 | 0.938 | 0.479 | 0.438 |
| RotatE | 0.968 | 0.943 | 0.833 | 0.787 |
| R2N_SG | 0.985 | 0.981 | 0.724 | 0.699 |
| DCR_SG | 0.975 | 0.956 | 0.817 | 0.754 |
| SBR_SG | 0.981 | 0.966 | 0.840 | 0.784 |
| **DPrL (PG)** | **0.986** | **0.979** | 0.834 | 0.784 |

DPrL在Family上取得最优MRR和H@1，在WN18RR上与最优方法SBR_SG接近。

### 消融实验

- DP vs PG：DP在小目标空间上更精确（94.0% vs 93.5% @N=4），但PG在计算上更灵活
- Family测试集上：智能grounding方法需实例化18071/45466/181095个groundings（深度1/2/3），DPrL仅需12334次ground评估

### 关键发现

1. DPrL是唯一能将MNIST加法扩展到N≥100的NeSy系统
2. 在KG补全中，DPrL生成的证明树完全可解释（如uncle关系通过father→brother推导）
3. 与需要手动调参的grounding方法不同，DPrL自动学习最优探索策略
4. MDP映射使得可以根据目标空间大小灵活选择精确（DP）或近似（PG）推理

## 亮点与洞察

1. **SLP→MDP的理论桥梁**：这个映射不仅是概念上的，而且在数学上严格——策略恰好等于解析转移概率，值函数优化等价于NeSy标准目标。这打通了RL的工具箱（DP、PPO等）在符号推理中的应用
2. **目标级参数化的优势**：以完整目标（而非单个原子/项）为条件，使每步决策拥有更多上下文信息
3. **可解释推理**：每个预测对应一棵完整的证明树，比其他神经方法的黑箱输出更透明
4. **DP的意外高效**：许多看似组合爆炸的NeSy基准，编码后状态空间实际可管理，DP就能高效精确求解

## 局限与展望

1. 子句选择限于最左原子——虽保证完备性但可能效率不佳，更灵活的选择策略可更快剪枝失败推导
2. PG模式在N≥100的MNIST加法上精度为0%——巨大组合空间下策略梯度陷入次优
3. 目前用标准策略梯度，可考虑更高级的RL技术（如基于价值的方法）
4. 未探讨自动逻辑程序重构以减少搜索空间

## 相关工作与启发

- **DeepStochLog**：最接近的前作，DPrL在三个维度超越：(1)以完整目标为条件(2)支持动态输出域(3)可扩展性更好
- **MINERVA/DeepPath**：KG上的RL路径搜索，但限于KG特定结构；DPrL基于SLP可应用于任意逻辑可表达的领域
- **自动定理证明（ATP）**：ATP找任意有效证明，NeSy需按概率权衡多个证明并与子符号信号对齐——本质不同
- 对idea启发：**将结构化推理过程映射为MDP**是一种通用范式，可扩展到其他形式化推理任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （SLP→MDP映射是原创性强的理论贡献）
- 技术深度: ⭐⭐⭐⭐⭐ （形式化定义严谨，万能近似定理+等价性证明）
- 实验充分性: ⭐⭐⭐⭐ （MNIST加法可扩展性测试+KG补全，覆盖不同场景）
- 写作质量: ⭐⭐⭐⭐ （逻辑清晰，例子直观）
- 实用价值: ⭐⭐⭐⭐ （代码开源，KG推理的可解释性有实际意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Hierarchical Circuit Symbolic Discovery Framework for Efficient Logic Optimization](../../ICLR2026/reinforcement_learning/a_hierarchical_circuit_symbolic_discovery_framework_for_efficient_logic_optimiza.md)
- [\[AAAI 2026\] Good-for-MDP State Reduction for Stochastic LTL Planning](good-for-mdp_state_reduction_for_stochastic_ltl_planning.md)
- [\[AAAI 2026\] Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning](do_it_for_her_first-order_temporal_logic_reward_specification_in_reinforcement_l.md)
- [\[ICML 2026\] RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs](../../ICML2026/reinforcement_learning/rl-sph_learning_to_achieve_feasible_solutions_for_integer_linear_programs.md)
- [\[AAAI 2026\] Deep (Predictive) Discounted Counterfactual Regret Minimization](deep_predictive_discounted_counterfactual_regret_minimization.md)

</div>

<!-- RELATED:END -->
