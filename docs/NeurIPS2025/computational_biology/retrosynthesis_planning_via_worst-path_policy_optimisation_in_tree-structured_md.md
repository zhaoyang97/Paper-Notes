---
title: >-
  [论文解读] Retrosynthesis Planning via Worst-path Policy Optimisation in Tree-structured MDPs
description: >-
  [NeurIPS 2025][计算生物][逆合成规划] 将逆合成规划重构为树结构MDP中的最差路径(worst-path)优化问题——合成树的价值由最弱路径决定（任何一条死胡同路径将导致整棵树无效），提出InterRetro通过加权自模仿学习优化这一最差路径目标，在Retro*-190上达到100%成功率，路径长度缩短4.9%，仅需10%训练数据即达92%完整性能。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "逆合成规划"
  - "树结构MDP"
  - "最差路径优化"
  - "自模仿学习"
  - "无搜索推理"
---

# Retrosynthesis Planning via Worst-path Policy Optimisation in Tree-structured MDPs

**会议**: NeurIPS 2025  
**arXiv**: [2509.10504](https://arxiv.org/abs/2509.10504)  
**代码**: [GitHub](https://github.com/MianchuWang/InterRetro)  
**领域**: 强化学习 / 逆合成规划  
**关键词**: 逆合成规划, 树结构MDP, 最差路径优化, 自模仿学习, 无搜索推理

## 一句话总结

将逆合成规划重构为树结构MDP中的最差路径(worst-path)优化问题——合成树的价值由最弱路径决定（任何一条死胡同路径将导致整棵树无效），提出InterRetro通过加权自模仿学习优化这一最差路径目标，在Retro*-190上达到100%成功率，路径长度缩短4.9%，仅需10%训练数据即达92%完整性能。

## 研究背景与动机

**领域现状**：逆合成规划旨在将目标分子逐步分解为可购买的基础构建块，形成合成树。单步预测准确率已达人类水平，但多步规划仍依赖启发式搜索（如MCTS、A*），需数百次模型调用，计算代价高昂。

**现有痛点**：(1) 搜索方法（MCTS/Retro*）需大量实时计算，每个分子需数百次模型调用，限制了大规模应用。(2) 微调方法（如PDVN）通过模仿成功搜索轨迹来改善策略，但模型适应的是搜索中遇到的分子分布而非直接推理分布，导致脱离搜索后表现下降。(3) 最关键的——现有方法通常优化所有路径的**平均**表现，忽视了合成树的**最差案例**敏感性：只要一条路径无法到达可购买的构建块，整棵合成树就无效。

**核心矛盾**：合成树的成功需要**所有**叶节点都是可购买化合物，但现有优化目标关注平均表现而非最差路径。

**本文目标** 消除推理时搜索的需求，同时保证高质量的合成路线——这需要一个更合适的优化目标和无搜索的策略改进方法。

**切入角度**：从优化目标本身入手——将"最弱链条"建模为最差路径，用折扣最差路径回报替代平均累积回报。

**核心 idea**：一条合成路线只有最弱路径那么强——通过最差路径优化直接改善最容易失败的分解步骤，配合自模仿学习实现无搜索推理。

## 方法详解

### 整体框架

InterRetro作为Agent与树结构MDP交互：(1) **探索**——用当前策略从目标分子开始递归分解，构建合成树；(2) **提取**——在合成树中找到成功子树，收集所有分支决策；(3) **学习**——用加权自模仿学习更新策略和价值网络。迭代执行直到策略收敛为止。

### 关键设计

1. **最差路径目标函数**:
    - 功能：定义合成树的价值为其所有根到叶路径中最差的回报
    - 核心思路：奖励函数 $r(s) = 1$ 如果分子 $s$ 是可购买构建块，否则为0。路径回报 $\gamma^T r(s_T)$，其中折扣因子 $\gamma \in (0,1)$ 惩罚更长路径。合成树目标 $J(\pi) = \mathbb{E}_{\tau \sim \pi}[\min_{p \in P(\tau)} \sum_{t=0}^T \gamma^t r(s_t)]$。单条失败路径使整棵树价值为0
    - 设计动机：USB-50k中98.6%的反应涉及≤3个反应物，故合成树质量主要由深度（最差路径长度）而非宽度决定

2. **树MDP中的Bellman最优方程**:
    - 功能：为最差路径目标推导价值函数的递归关系和最优性条件
    - 核心思路：Q函数递推 $Q^\pi(s,a) = r(s) + \gamma(1-r(s))\min_{s' \in \mathcal{T}(s,a)} V^\pi(s')$——关键差异在于使用$\min$而非$\sum$聚合子节点。证明Bellman最优算子是压缩映射→$V^*$唯一存在且值迭代收敛
    - 设计动机：标准MDP聚合用求和/期望，但化学反应产生多个子状态(分子)，需要用$\min$捕获"木桶效应"

3. **加权自模仿学习**:
    - 功能：基于优势权重模仿过去的成功决策，确保化学可行性
    - 核心思路：策略约束在预训练单步模型$\pi^0$的支持集内：$\Pi = \{\pi | \pi(a|s)=0 \text{ whenever } \pi^0(a|s)=0\}$。更新目标 $\mathcal{L}(\theta) = -\mathbb{E}[\exp_{clip}(\beta A_\phi(s,a))\log\pi_\theta(a|s)]$，高优势反应获更高权重。理论证明 $V^{\pi^{i+1}}(s) \geq V^{\pi^i}(s)$（单调改进保证）
    - 设计动机：支持集约束保证提出的反应始终化学合理；优势加权避免盲目模仿，重点学习高质量决策

### 损失函数 / 训练策略

价值网络损失（Bellman TD）：$\mathcal{L}(\phi) = \mathbb{E}[(V_\phi(s) - (r(s) + \gamma(1-r(s))\min_{s'} V_{\phi^-}(s')))^2]$，$V_{\phi^-}$为target network。策略网络损失：$\mathcal{L}(\theta) = -\mathbb{E}[\exp_{clip}(\beta A_\phi(s,a))\log\pi_\theta(a|s)]$。使用FIFO经验回放（最大20K分支），6个并行探索进程，每次迭代收集36棵合成树后更新5次。

## 实验关键数据

### 主实验

| 基准 | 模型调用数 | InterRetro | PDVN | Retro* | MCTS |
|------|-----------|------------|------|--------|------|
| Retro*-190 | 500 | **100.0%** | 98.95% | 75.26% | 62.63% |
| Retro*-190 | Direct Gen. | **95.78%** | - | 20.00% | 20.00% |
| ChEMBL-1000 | 500 | **97.50%** | 83.50% | 74.70% | 71.90% |
| GDB17-1000 | 500 | **99.50%** | 26.90% | 7.50% | 4.50% |

### 消融实验

| 配置 | Retro*-190成功率 | 说明 |
|------|-----------------|------|
| 最差路径目标 (默认) | 100.0% | 本文方法 |
| 平均路径目标 | ~96% | 忽略最弱链条 |
| 无自模仿(仅预训练) | 16.84% | 预训练策略直接推理 |
| 10%训练数据 | ~92% | 出色的样本效率 |
| 完整训练数据 | 100.0% | 完全收敛 |

### 关键发现
- **100%成功率**：在Retro*-190上首次达到完美成功率，且路径长度比搜索方法缩短4.9%
- **无搜索推理**：Direct Generation模式下就达95.78%（对比搜索方法baseline 20%）
- **极强的样本效率**：仅10%训练数据即达92%性能
- **GDB17-1000上的巨大优势**：InterRetro 99.5% vs PDVN 26.9%——在困难分子上优势尤其显著

## 亮点与洞察
- 最差路径优化对问题的建模非常自然——"木桶效应"确实是逆合成的核心挑战
- 理论贡献扎实：唯一最优解存在性、单调改进保证、Bellman最优方程推导完整
- 实际意义重大：消除推理时搜索可将逆合成从分钟级降至毫秒级
- 10%数据就能达92%性能，暗示最差路径目标比其他目标更高效地利用经验

## 局限与展望
- 依赖预训练单步模型(Graph2Edits)的质量——如果预训练模型的支持集不包含正确反应则无法恢复
- 训练需约48小时（单A5000 GPU），虽然可接受但仍有优化空间
- 仅在USPTO-50k训练集上评估，真实世界反应可能超出该数据集覆盖范围
- Tree MDP假设反应推荐是确定性的，实际化学反应有不确定性（副反应、收率）

## 相关工作与启发
- **vs PDVN**: PDVN也通过自模仿改善搜索策略，但优化平均回报且仍依赖搜索推理；InterRetro优化最差路径且完全无搜索
- **vs MCTS/Retro***: 搜索方法需大量模型调用（500次），InterRetro直接生成达到更高成功率
- **vs DreamRetroer**: 性能相当但InterRetro不依赖任何外部模型增强

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 最差路径优化框架是逆合成领域全新视角，理论推导完整优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 三个基准、多种模型调用预算、消融充分、首次达100%成功率
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，理论与算法衔接流畅
- 价值: ⭐⭐⭐⭐⭐ 无搜索推理+100%成功率，对计算机辅助分子合成有重大实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Improved Off-policy Reinforcement Learning in Biological Sequence Design](../../ICML2025/computational_biology/improved_off-policy_reinforcement_learning_in_biological_sequence_design.md)
- [\[ICML 2026\] From Feasible to Practical: Pareto-Optimal Synthesis Planning](../../ICML2026/computational_biology/from_feasible_to_practical_pareto-optimal_synthesis_planning.md)
- [\[ICML 2026\] RETROSPECT: RETROsynthesis via Sequential Prediction, and Chemically Transformed-ranking](../../ICML2026/computational_biology/retrospect_retrosynthesis_via_sequential_prediction_and_chemically_transformed-r.md)
- [\[AAAI 2026\] Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics](../../AAAI2026/computational_biology/dual-path_knowledge-augmented_contrastive_alignment_network_for_spatially_resolv.md)
- [\[ACL 2026\] ProtoCycle: Reflective Tool-Augmented Planning for Text-Guided Protein Design](../../ACL2026/computational_biology/protocycle_reflective_tool-augmented_planning_for_text-guided_protein_design.md)

</div>

<!-- RELATED:END -->
