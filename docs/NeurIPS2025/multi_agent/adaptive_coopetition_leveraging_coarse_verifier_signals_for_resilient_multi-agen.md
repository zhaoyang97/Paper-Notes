---
title: >-
  [论文解读] Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning
description: >-
  [NeurIPS 2025 (Mathematical Reasoning and AI Workshop)][多智能体][多智能体协作] 提出 Adaptive Coopetition (AdCo) 框架，利用 UCB 多臂老虎机策略和粗粒度验证器信号，使多个 LLM 智能体在推理过程中自适应地切换协作与竞争模式，在数学推理基准上实现 20% 的相对提升。
tags:
  - "NeurIPS 2025 (Mathematical Reasoning and AI Workshop)"
  - "多智能体"
  - "多智能体协作"
  - "推理增强"
  - "UCB"
  - "竞合机制"
  - "inference-time computation"
---

# Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning

**会议**: NeurIPS 2025 (Mathematical Reasoning and AI Workshop)  
**arXiv**: [2510.18179](https://arxiv.org/abs/2510.18179)  
**代码**: [GitHub](https://github.com/AdCo-Research/adaptive-coopetition)  
**领域**: 多智能体系统 / LLM推理  
**关键词**: 多智能体协作, 推理增强, UCB, 竞合机制, inference-time computation

## 一句话总结

提出 Adaptive Coopetition (AdCo) 框架，利用 UCB 多臂老虎机策略和粗粒度验证器信号，使多个 LLM 智能体在推理过程中自适应地切换协作与竞争模式，在数学推理基准上实现 20% 的相对提升。

## 研究背景与动机

推理时计算（Inference-time Computation）是提升LLM推理能力的关键范式，但现有方法存在显著局限：

**自我修正（Self-correction）的局限**：LLM 的自我修正往往会强化模型的初始偏差，无法有效纠正根本性的推理错误

**多智能体协作（MAC）的失败**：现有多智能体方法缺乏高效的协调机制，容易导致集体错误——所有智能体可能收敛到同一个错误答案

**高性能验证器的门槛**：虽然外部验证器可以检测推理错误，但训练可靠的验证器本身需要大量资源

关键观察：**协作不总是最优的**。当多个智能体能力相近时，纯协作可能导致群体思维（groupthink），而适度的竞争可以促进解空间的探索。反之，当某个智能体明显更优时，竞争则是浪费资源。

## 方法详解

### 整体框架

AdCo 是一个多轮、多智能体的推理框架，核心流程如下：

1. **初始化**：多个 LLM 智能体各自独立生成初始答案和推理过程
2. **信号采集**：使用粗粒度验证器（如 PRM）为每个推理链打分
3. **策略选择**：基于 UCB 算法决定本轮采用协作还是竞争模式
4. **推理更新**：根据选定的模式和同伴反馈更新各智能体的推理
5. **迭代直到收敛或达到最大轮数**

### 关键设计

**UCB-based 策略选择机制**

借鉴多臂老虎机（MAB）中的Upper Confidence Bound (UCB) 算法来平衡协作和竞争：

$$UCB_i = \bar{X}_i + c\sqrt{\frac{\ln N}{n_i}}$$

其中 $\bar{X}_i$ 是策略 $i$（协作或竞争）的历史平均回报，$N$ 是总轮数，$n_i$ 是策略 $i$ 被选择的次数，$c$ 是探索参数。

**协作模式（Collaborative）**
- 智能体之间共享推理链和中间结果
- 整合其他智能体的优秀推理步骤
- 适用于某个智能体明显占优的情况

**竞争模式（Competitive）**
- 智能体各自独立改进推理，不参考他人结果
- 仅在最终答案层面进行比较和投票
- 适用于智能体能力相近、需要探索多样化解的情况

**粗粒度验证器信号**
- 不要求高精度的逐步验证器
- 仅需要粗粒度的"推理质量信号"（如整体推理链的PRM分数）
- 大幅降低了对验证器质量的依赖

### 损失函数 / 训练策略

AdCo 是一个**免训练**（training-free）的推理时框架，不需要额外的模型训练。核心是通过UCB算法在线学习最优策略分配：

- 每轮结束后，根据验证器信号更新策略的回报估计
- UCB 自然平衡了利用（exploitation）和探索（exploration）
- 随着轮数增加，策略选择趋向最优

模型列表默认使用 GPT-4o、DeepSeek-R1、Qwen-QWQ-32B 构成多样化智能体组合。

## 实验关键数据

### 主实验

在 GSM8K 和 MATH 数据集上的准确率对比：

| 方法 | GSM8K Acc (%) | MATH Acc (%) | 相对提升 (MATH) |
|------|-------------|-------------|---------------|
| Single Agent (GPT-4o) | 82.5 | 51.2 | 基线 |
| Self-correction | 83.1 | 52.4 | +2.3% |
| MAC (纯协作) | 85.3 | 54.8 | +7.0% |
| MAC (纯竞争) | 84.7 | 55.1 | +7.6% |
| majority voting | 86.2 | 56.3 | +10.0% |
| AdCo (UCB 自适应) | **88.4** | **61.5** | **+20.1%** |

### 消融实验

不同策略配置的性能对比：

| 策略 | GSM8K Acc (%) | MATH Acc (%) | 策略多样性 |
|------|-------------|-------------|----------|
| 固定协作 | 85.3 | 54.8 | 低 |
| 固定竞争 | 84.7 | 55.1 | 高 |
| 随机切换 | 85.9 | 56.7 | 中 |
| UCB (无PRM) | 86.1 | 57.3 | 中 |
| UCB + 粗粒度PRM | **88.4** | **61.5** | 自适应 |

不同智能体数量的影响：

| 智能体数量 | MATH Acc (%) | 推理开销 (相对) |
|-----------|-------------|--------------|
| 2 | 57.8 | 1.0x |
| 3 | 61.5 | 1.5x |
| 5 | 62.3 | 2.5x |
| 7 | 62.1 | 3.5x |

### 关键发现

1. **自适应策略显著优于固定策略**：UCB自适应方法在MATH上比最佳固定策略高出约6个百分点
2. **粗粒度信号足够有效**：不需要精确的逐步验证器，粗粒度PRM信号即可指导策略选择
3. **3个智能体是最佳平衡点**：性能在3个智能体时趋于饱和，继续增加带来的收益递减
4. **在高难度数据集上提升更明显**：MATH（更难）上的相对提升约20%，远高于GSM8K上的约7%
5. **鲁棒性强**：不同配置下性能波动小，说明UCB机制能有效适应不同场景

## 亮点与洞察

1. **竞合（Coopetition）概念新颖**：将博弈论中的竞合策略引入多智能体推理，为inference-time computation提供了新视角
2. **UCB 的巧妙应用**：将策略选择建模为多臂老虎机问题，利用成熟的UCB算法自然解决探索-利用权衡
3. **低门槛验证器**：不依赖高性能验证器，降低了方法的应用门槛
4. **即插即用**：作为推理时框架，不需要修改基础模型，具有良好的通用性

## 局限与展望

1. **仅验证在数学推理上**：尚未在代码生成、逻辑推理、常识推理等其他任务上验证
2. **API调用成本高**：多智能体多轮推理意味着大量API调用，实际部署成本较高
3. **策略空间有限**：仅有协作/竞争两种策略，可以考虑更细粒度的混合策略
4. **Workshop论文局限**：部分实验细节和分析深度有限，如不同难度问题上的策略分布分析
5. **PRM信号质量的影响**：虽然声称不依赖高性能验证器，但PRM信号质量对性能的具体影响未充分量化

## 相关工作与启发

- **Self-Consistency (Wang et al.)**：采样多条推理链并通过多数投票选择答案
- **Tree of Thoughts (Yao et al.)**：结构化搜索推理空间
- **Multi-Agent Debate (Du et al.)**：多智能体辩论式推理
- **Process Reward Model**：逐步验证推理正确性
- **启发**：竞合机制的思路可以推广到其他需要在exploration和exploitation之间权衡的场景，如代码调试、创意写作等

## 评分

- 新颖性：⭐⭐⭐⭐ （竞合概念新颖，UCB应用巧妙）
- 技术深度：⭐⭐⭐ （核心技术相对简单，但组合有效）
- 实验充分性：⭐⭐⭐⭐ （多维度对比，包括消融和鲁棒性分析）
- 写作质量：⭐⭐⭐⭐ （动机清晰，图表丰富）
- 综合评分：⭐⭐⭐⭐ （有价值的方向性工作，方法简洁有效）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information](../../ICML2026/multi_agent/beyond_majority_voting_llm_aggregation_by_leveraging_higher-order_information.md)
- [\[AAAI 2026\] Adaptive Theory of Mind for LLM-based Multi-Agent Coordination](../../AAAI2026/multi_agent/adaptive_theory_of_mind_for_llm-based_multi-agent_coordination.md)
- [\[CVPR 2026\] Visual Document Understanding and Reasoning: A Multi-Agent Collaboration Framework with Agent-Wise Adaptive Test-Time Scaling](../../CVPR2026/multi_agent/visual_document_understanding_and_reasoning_a_multi-agent_collaboration_framewor.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](../../ACL2026/multi_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[ICML 2025\] From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium](../../ICML2025/multi_agent/from_debate_to_equilibrium_belief-driven_multi-agent_llm_reasoning_via_bayesian_.md)

</div>

<!-- RELATED:END -->
