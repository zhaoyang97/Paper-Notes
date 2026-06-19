---
title: >-
  [论文解读] PEOAT: Personalization-Guided Evolutionary Question Assembly for One-Shot Adaptive Testing
description: >-
  [AAAI 2026][优化/理论][自适应测试] 首次提出"一次性自适应测试 (OAT)"任务，将其建模为组合优化问题，并设计 PEOAT 框架——结合个性化初始化、认知增强进化搜索和多样性保持选择策略，在无交互反馈的条件下为每位考生一次性选出最优题集，大幅超越传统 CAT 方法。 计算机自适应测试 (CAT) 的实际限制…
tags:
  - "AAAI 2026"
  - "优化/理论"
  - "自适应测试"
  - "进化算法"
  - "个性化"
  - "组合优化"
  - "智能教育"
---

# PEOAT: Personalization-Guided Evolutionary Question Assembly for One-Shot Adaptive Testing

**会议**: AAAI 2026  
**arXiv**: [2512.00439](https://arxiv.org/abs/2512.00439)  
**代码**: 无  
**领域**: 优化  
**关键词**: 自适应测试, 进化算法, 个性化, 组合优化, 智能教育

## 一句话总结

首次提出"一次性自适应测试 (OAT)"任务，将其建模为组合优化问题，并设计 PEOAT 框架——结合个性化初始化、认知增强进化搜索和多样性保持选择策略，在无交互反馈的条件下为每位考生一次性选出最优题集，大幅超越传统 CAT 方法。

## 研究背景与动机

### 计算机自适应测试 (CAT) 的实际限制

CAT 通过交互式选题和逐步能力估计来高效评估考生能力。典型流程是：选题模块 $\mathcal{M}_\pi$ 根据当前能力估计选题 → 考生作答 → 诊断模块 $\mathcal{M}_d$ 更新能力估计 → 循环。

CAT 方法可分为两类：
- **启发式方法**（MKLI、BECAT、MAAT）：基于可解释规则（如最大 Fisher 信息、KL 散度）选题
- **数据驱动方法**（BOBCAT、NCAT、GMOCAT、UATS）：用强化学习等方法学习个性化选题策略

**核心问题**：CAT 的交互式、实时逐题选择在以下场景中不可行：

**大规模考试**：交互成本高，逐题动态选择不切实际

**心理评估**：需要最小化噪声和干扰，逐题反馈可能影响评估质量

**远程/离线测试**：设备限制、响应延迟等因素制约交互

**资源受限环境**：时间敏感场景下无法支持多轮交互

### One-Shot Adaptive Testing (OAT) 任务

本文首次定义 OAT：给定考生初始能力估计 $\theta_i^0$，一次性选出固定长度 $L$ 的最优题集 $\mathcal{J}_i$，考生完成所有题目后进行单步能力更新，得到最终能力估计 $\theta_i^{final}$。

OAT 的三大挑战：

**学生适应性**：无中间反馈，需在优化过程中确保题目与个体能力匹配

**巨大搜索空间**：从大量候选题目中选出 $L$ 题的组合数是指数级的

**编码稀疏性**：候选题库远大于测试长度，编码表示面临维度灾难

### 建模思路

OAT 自然建模为双层组合优化问题：
- 外层选择题目子集 $\mathcal{J}_i$
- 内层基于模拟作答数据估计学生能力
- 目标：最终能力估计尽可能接近学生真实能力

$$\mathcal{J}_i^* = \arg\max_{\mathcal{J}_i \subseteq \mathcal{Q}_i^{untested}} \mathcal{F}(\theta_i^{final}(\mathcal{J}_i), \hat{\theta}_i)$$

## 方法详解

### 整体框架

PEOAT 由三个核心模块组成：
1. 个性化感知种群初始化 → 构建信息丰富且多样的初始种群
2. 认知增强进化搜索 → 利用认知信号进行有效探索
3. 多样性保持环境选择 → 在适应度和多样性之间取得平衡

### 关键设计

#### 1. **个性化感知种群初始化**：基于能力-难度匹配的多策略采样

**功能**：根据学生能力向量 $\boldsymbol{\theta}_i$ 和题目难度向量 $\boldsymbol{\alpha}_j$ 之间的距离，自适应构建信息丰富且多样的初始种群。

**编码方案**：每个个体是长度为 $L$ 的题目索引序列 $\mathcal{X}_i^{(j)} = [x_1, x_2, \ldots, x_L]$，各索引不重复。

**个性化距离向量**：
$$\delta_j = \|\boldsymbol{\theta}_i - \boldsymbol{\alpha}_j\|_2, \quad \forall j \in \{1, 2, \ldots, |\mathcal{Q}_i|\}$$

**三策略初始化**：
- **$\mathcal{O}_{match}$（匹配策略）**：从距离最小的 $2L$ 个题目中均匀采样 $L$ 题 → 优先选择难度匹配的题目
- **$\mathcal{O}_{diverse}$（多样策略）**：从距离最大的 $2L$ 个题目中均匀采样 → 探索能力边界
- **$\mathcal{O}_{rand}$（随机策略）**：从中间距离题目中均匀采样 → 增加随机性

每个个体随机选择一种策略，确保种群整体多样性。

**设计动机**：直接随机初始化缺乏个性化先验，导致搜索空间过大；纯匹配初始化又缺乏多样性。三策略机制在利用（exploitation）和探索（exploration）之间取得平衡。

#### 2. **认知增强进化搜索**：利用 Fisher 信息引导变异

**模式保持均匀交叉**：
- 生成二元掩码 $m_k \sim \text{Bernoulli}(0.5)$
- 两个后代通过掩码交换父代对应位置的基因
- 修复算子 $\mathcal{T}(\cdot)$ 解决重复题目，从未选题池随机替换

**认知信息引导变异**：
- 随机选择一个基因 $x_{off}$ 移除
- 基于 Fisher 信息矩阵的 Frobenius 范数作为标量信息增益：

$$\mathbf{I}_j(\boldsymbol{\theta}_i) = |\boldsymbol{\alpha}_j|^2 \cdot p_j(1-p_j)$$

其中 $p_j = \sigma(\boldsymbol{\theta}_i^\top \boldsymbol{\alpha}_j)$ 是基于 IRT 的正确率预测。

- 按归一化信息增益构建分类采样分布：
$$P(x_j \in \mathcal{Z}) = \frac{\mathbf{I}_j}{\sum_{k \in \mathcal{Z}} \mathbf{I}_k}$$

**核心思路**：information gain 高的题目（处于能力阈值附近、区分度高）被采样到的概率更大。这比随机变异更高效，确保新插入的基因既个性化又信息丰富。

#### 3. **多样性保持环境选择**：Hamming 距离过滤 + 精英保留

**适应度评估**：
- 模拟 OAT 过程：学生完成选定题目 → 诊断模型虚拟参数更新 → 在保留测试集上评估
- 混合指标：$\mathcal{F} = (\mathcal{F}_{auc} + \mathcal{F}_{acc})/2$

**选择机制**：
1. 按适应度排序，保留 top-$k$ 精英（$k = \lfloor |\mathcal{P}|/2 \rfloor$）
2. 剩余候选通过 Hamming 距离过滤：
    - 将个体编码为二元位串
    - 计算与精英池的最小 Hamming 距离
    - 仅保留 $\text{HamDist} > \tau$ 的候选

**设计动机**：纯精英选择导致种群过早收敛；Hamming 距离过滤保持基因型多样性，避免局部最优。

### 训练策略

- 诊断模型：MIRT 和 NCD 两种 backbone
- 种群大小 20，进化 15 代，交叉率 0.8，变异率 0.2
- 测试长度 $L \in \{5, 10, 15, 20\}$
- 距离阈值 $\tau$ 在 $\{0.5, 0.75, 1, 1.25, 1.5\}$ 中搜索

## 实验关键数据

### 主实验

#### JUNYI 数据集 (MIRT backbone)

| 方法 | 类型 | length=5 ACC/AUC | length=10 | length=15 | length=20 |
|------|------|-----------------|-----------|-----------|-----------|
| RAND | 启发式 | 67.98/68.24 | 74.48/73.64 | 79.60/77.73 | 82.47/80.48 |
| MKLI | 启发式 | 70.14/70.27 | 78.03/76.64 | 83.26/81.39 | 86.07/84.27 |
| BOBCAT | 数据驱动 | 69.15/71.86 | 77.05/77.60 | 81.66/81.12 | 84.29/83.43 |
| NCAT | 数据驱动 | 71.19/73.48 | 80.23/77.37 | 82.69/81.43 | 84.93/84.04 |
| UATS | 数据驱动 | 70.83/74.45 | 80.33/77.19 | 83.13/81.27 | 84.38/84.65 |
| **PEOAT** | **Ours** | **79.64/83.05** | **85.38/85.85** | **86.39/86.68** | **86.85/87.83** |

在 length=5 时，PEOAT 超越次优方法 **10.61%/10.35%**（ACC/AUC），优势非常巨大。

#### PTADisc 数据集 (NCD backbone)

| 方法 | length=5 | length=10 | length=15 | length=20 |
|------|----------|-----------|-----------|-----------|
| NCAT | 66.38/68.09 | 67.64/69.22 | 68.48/69.61 | 70.17/70.40 |
| GMOCAT | 66.47/68.24 | 67.48/68.97 | 69.14/69.49 | 69.73/70.36 |
| **PEOAT** | **69.37/70.84** | **73.65/73.58** | **75.44/75.07** | **75.91/74.93** |

在所有测试长度和两种诊断模型上，PEOAT 均全面领先。

### 消融实验

| 配置 | length=5 | length=10 | length=15 | length=20 |
|------|----------|-----------|-----------|-----------|
| w/o PI (无个性化初始化) | 下降最大 | 下降最大 | 下降最大 | 下降最大 |
| w/o CE (无认知增强进化) | 中等下降 | 中等下降 | 中等下降 | 中等下降 |
| w/o ES (无多样性选择) | 轻微下降 | 轻微下降 | 轻微下降 | 轻微下降 |
| **PEOAT 完整** | **最优** | **最优** | **最优** | **最优** |

**关键发现**：个性化初始化的贡献最大，表明将个性化先验嵌入初始种群的质量对最终性能至关重要。

#### PEOAT vs PEOAT-B (无特定设计的基础版本)

| CDM | 方法 | length=5 | length=10 | length=15 | length=20 |
|-----|------|----------|-----------|-----------|-----------|
| MIRT | PEOAT-B | 78.35/81.97 | 84.12/84.48 | 84.96/85.21 | 85.73/86.55 |
| MIRT | **PEOAT** | **79.64/83.05** | **85.38/85.85** | **86.39/86.68** | **86.85/87.83** |
| NCD | PEOAT-B | 73.27/81.81 | 80.64/85.19 | 84.73/87.59 | 86.11/88.80 |
| NCD | **PEOAT** | **74.56/83.06** | **81.90/86.47** | **85.85/88.86** | **87.34/89.78** |

即使是基础版本也已大幅超越 CAT 基线，证明将 OAT 建模为组合优化本身就是正确思路。

### 关键发现

1. PEOAT 在短测试长度下优势最为显著（length=5 提升 10%+），说明在快速评估场景中最有价值
2. 将 OAT 建模为组合优化并用进化算法求解是有效范式
3. 个性化先验对种群质量的影响大于进化算子的改进
4. Fisher 信息引导的变异比随机变异在认知诊断场景中更为有效

## 亮点与洞察

1. **OAT 任务定义**：首次提出并形式化了一个新的、有实际意义的教育评估任务，填补了 CAT 与静态测试之间的空白
2. **组合优化视角**：将教育测试问题与进化优化自然连接，跨学科思维新颖
3. **Fisher 信息驱动的变异**：巧妙结合项目反应理论 (IRT) 和进化计算，使变异算子具备教育心理学依据
4. **实用价值高**：OAT 直接服务于大规模考试、心理评估、离线测试等刚需场景

## 局限与展望

1. 每个学生都需要独立运行进化搜索（20 个体 × 15 代），计算成本随考生数量线性增长
2. 依赖预训练的认知诊断模型（MIRT/NCD）的质量，诊断模型的误差会传播到选题
3. 仅考虑固定长度测试，未探索自适应长度选题
4. 可以尝试结合深度强化学习端到端学习选题策略
5. 未考虑题目曝光率控制（同一批次不同考生可能选到相同题目）

## 相关工作与启发

- **BECAT**：使用全响应梯度近似指导 CAT 选题，理论性强
- **NCAT**：将 CAT 建模为双层强化学习问题，注意力策略选题
- **PEGA**：首个使用进化算法做个性化练习组装的工作，但面向练习推荐而非考试
- **进化优化在教育中的应用**：遗传算法在认知诊断（HGA-CDM）中已有应用，OAT 进一步拓展了应用场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — OAT 任务首次提出，组合优化建模思路新颖
- 实验充分度: ⭐⭐⭐⭐ — 两个真实教育数据集、多种基线、消融分析，但缺少大规模场景测试
- 写作质量: ⭐⭐⭐⭐ — 公式化严谨清晰，问题定义规范
- 价值: ⭐⭐⭐⭐ — 对智能教育领域有实际应用价值，但场景相对垂直

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Distribution Alignment for One-Shot Federated Learning via Optimal Transport](../../ICML2026/optimization/distribution_alignment_for_one-shot_federated_learning_via_optimal_transport.md)
- [\[AAAI 2026\] Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching](personalized_federated_learning_with_bidirectional_communication_compression_via.md)
- [\[AAAI 2026\] Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation](bridging_synthetic_and_real_routing_problems_via_llm-guided_instance_generation_.md)
- [\[AAAI 2026\] Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization](pareto-grid-guided_large_language_models_for_fast_and_high-quality_heuristics_de.md)
- [\[NeurIPS 2025\] Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing](../../NeurIPS2025/optimization/evaluating_llms_for_combinatorial_optimization_one-phase_and_two-phase_heuristic.md)

</div>

<!-- RELATED:END -->
