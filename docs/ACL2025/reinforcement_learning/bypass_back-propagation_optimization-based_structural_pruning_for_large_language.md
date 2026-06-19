---
title: >-
  [论文解读] Bypass Back-propagation: Optimization-based Structural Pruning for Large Language Models via Policy Gradient
description: >-
  [ACL 2025][强化学习][结构化剪枝] 本文提出一种基于策略梯度的LLM结构化剪枝方法，通过在概率空间中学习伯努利剪枝掩码来直接优化剪枝模型的损失函数，全程无需对LLM本身进行反向传播，仅需前向推理即可完成剪枝优化。 领域现状：大语言模型的参数量从数十亿到数千亿不等，部署成本高昂。模型剪枝（pruning）是压缩LL…
tags:
  - "ACL 2025"
  - "强化学习"
  - "结构化剪枝"
  - "策略梯度"
  - "无反向传播"
  - "伯努利分布"
  - "LLM压缩"
---

# Bypass Back-propagation: Optimization-based Structural Pruning for Large Language Models via Policy Gradient

**会议**: ACL 2025  
**链接**: [ACL Anthology](https://aclanthology.org/2025.acl-long.1421/)
**代码**: 无  
**领域**: 强化学习  
**关键词**: 结构化剪枝, 策略梯度, 无反向传播, 伯努利分布, LLM压缩

## 一句话总结

本文提出一种基于策略梯度的LLM结构化剪枝方法，通过在概率空间中学习伯努利剪枝掩码来直接优化剪枝模型的损失函数，全程无需对LLM本身进行反向传播，仅需前向推理即可完成剪枝优化。

## 研究背景与动机

**领域现状**：大语言模型的参数量从数十亿到数千亿不等，部署成本高昂。模型剪枝（pruning）是压缩LLM的主要手段之一，当前主流的LLM剪枝方法（如SparseGPT、Wanda）主要在后训练阶段（post-training）操作，无需昂贵的权重微调。

**现有痛点**：现有方法的剪枝标准通常依赖于**手工设计的启发式指标**（如权重大小、梯度信息、激活值统计量等），这些指标本质上是对"哪些参数不重要"的间接估计，可能导致次优的剪枝决策。理想的剪枝方案应该直接优化剪枝后模型的任务损失，但这需要对LLM进行反向传播，计算成本极高。

**核心矛盾**：直接优化（optimization-based）的剪枝可以获得更好的结果，但需要反向传播通过整个LLM，内存和计算开销不可接受；基于启发式指标的剪枝高效但结果次优。

**本文目标**：设计一种既是optimization-based（直接优化任务损失）又不需要对LLM进行反向传播的结构化剪枝方法。

**切入角度**：将剪枝掩码视为随机变量，通过伯努利分布参数化，利用策略梯度（policy gradient）估计器将梯度计算从LLM上解耦——只需LLM的前向传播来评估不同掩码下的损失，然后用REINFORCE算法更新伯努利分布参数。

**核心 idea**：用策略梯度优化取代通过LLM的反向传播，在概率空间中搜索最优剪枝掩码，只需前向推理就能完成基于优化的结构化剪枝。

## 方法详解

### 整体框架

输入是预训练的LLM和校准数据集，输出是剪枝后的紧凑模型。方法分为三步：(1) 初始化每个可剪枝结构单元（注意力头、FFN中间维度）对应的伯努利分布参数；(2) 迭代采样剪枝掩码、评估损失、用策略梯度更新分布参数；(3) 根据最终分布确定剪枝方案。

### 关键设计

1. **伯努利分布参数化的剪枝掩码**:

    - 功能：将离散的剪枝决策（保留/删除）转化为连续的概率优化问题
    - 核心思路：为模型中每个可剪枝单元（如第$l$层的第$h$个注意力头）关联一个伯努利分布参数 $\theta_{l,h} \in [0,1]$，表示该单元被保留的概率。剪枝掩码 $m_{l,h} \sim \text{Bernoulli}(\theta_{l,h})$ 是从该分布中采样的二值变量。通过优化 $\theta$ 参数来间接优化剪枝决策
    - 设计动机：将组合优化问题（NP-hard的最优剪枝）松弛为连续优化问题，使得梯度方法可用

2. **策略梯度估计器（Policy Gradient Estimator）**:

    - 功能：在不对LLM进行反向传播的前提下计算剪枝掩码的梯度
    - 核心思路：将剪枝过程建模为强化学习问题——伯努利参数$\theta$是策略，采样的掩码$m$是动作，剪枝后模型的负损失是奖励。使用REINFORCE算法估计梯度：$\nabla_\theta J = \mathbb{E}_{m \sim p_\theta}[\nabla_\theta \log p_\theta(m) \cdot R(m)]$，其中 $R(m)$ 只需一次前向传播即可计算。关键在于梯度 $\nabla_\theta \log p_\theta(m)$ 不涉及LLM参数，完全在伯努利分布的低维参数空间中计算
    - 设计动机：解耦"评估质量"（需要前向传播）和"优化决策"（只在$\theta$空间），绕过了对LLM反向传播的需求

3. **全局异构剪枝与指标初始化**:

    - 功能：自动为不同层分配不同的剪枝率，并可选择性地利用现有启发式指标进行初始化
    - 核心思路：由于每个结构单元有独立的伯努利参数，优化后不同层自然会得到不同的保留概率，实现异构剪枝（heterogeneous pruning）。同时，可以用现有指标（如基于梯度信息的重要性分数）来初始化伯努利参数，而非从均匀分布开始，加速优化收敛
    - 设计动机：统一剪枝率（每层删相同比例）在实践中效果差，因为不同层的冗余度不同。指标初始化结合了启发式方法的先验知识和优化方法的精确搜索

### 训练策略

使用方差缩减技术（基线减法）降低REINFORCE的高方差问题。每次迭代采样多个掩码来估计梯度。总迭代次数较少（因为参数空间小），整体计算成本主要来自LLM的多次前向传播。

## 实验关键数据

### 主实验

| 模型 | 剪枝率 | 数据集 | 本文PPL | SparseGPT PPL | Wanda PPL | LLM-Pruner PPL |
|------|--------|--------|---------|---------------|-----------|----------------|
| LLaMA-7B | 20% | WikiText2 | 优于基线 | 基线 | 基线 | 基线 |
| LLaMA-7B | 50% | WikiText2 | 显著优于 | 基线 | 基线 | 基线 |
| LLaMA-2-7B | 20% | C4 | 优于基线 | 基线 | 基线 | 基线 |
| LLaMA-3-8B | 20% | WikiText2 | 持续优于 | 基线 | 基线 | - |
| Mistral-7B | 20% | WikiText2 | 最优 | 基线 | 基线 | - |
| Vicuna-7B | 50% | WikiText2 | 优势明显 | 基线 | 基线 | 基线 |

### 消融实验

| 配置 | WikiText2 PPL | 说明 |
|------|--------------|------|
| 策略梯度优化（本文） | 最优 | 直接优化目标损失 |
| 随机初始化 + 优化 | 次优 | 收敛较慢但最终效果接近 |
| 指标初始化 + 优化 | 最优 | 结合先验+优化的双重优势 |
| 仅指标（无优化） | 较差 | 启发式方法的上限 |
| 均匀剪枝率 | 较差 | 不同层冗余度不同 |
| 异构剪枝率（本文） | 显著更优 | 自动适配各层冗余度 |

### 关键发现
- 策略梯度优化比所有启发式指标都能找到更好的剪枝方案，尤其在高剪枝率（50%）时优势更明显
- 异构剪枝（不同层不同剪枝率）比均匀剪枝提升显著，且优化后发现浅层和深层的冗余度确实不同
- 指标初始化能加速收敛，但从随机初始化也能最终收敛到接近的解
- 计算成本分析显示，本方法的总前向传播次数在可接受范围内（数百次），远低于全反向传播剪枝

## 亮点与洞察
- **"策略梯度绕过反向传播"**的思路非常巧妙——将剪枝掩码视为RL中的动作，利用REINFORCE的"免反向传播"特性来优化组合决策。这种思路可以迁移到其他需要优化离散结构决策的场景（如NAS、特征选择）
- **全局异构剪枝的自动化**解决了实践中的重要问题——不同层应该剪多少比例，以往需要手工调整或额外的搜索过程
- 方法的通用性好，理论上适用于任何Transformer结构的结构化剪枝

## 局限与展望
- 策略梯度的高方差问题虽然通过基线减法缓解，但仍可能导致优化不稳定
- 实验主要在7B-13B模型上进行，对更大规模模型（70B+）的效果和效率有待验证
- 仅评估了困惑度（PPL），未涉及下游任务（如推理、代码生成等）的评测
- 与知识蒸馏等后剪枝恢复技术的结合未被探索

## 相关工作与启发
- **vs SparseGPT**: SparseGPT通过二阶近似高效求解非结构化剪枝，但对结构化剪枝不太适用；本文专注结构化剪枝且从优化角度出发
- **vs Wanda**: Wanda使用权重×激活值作为剪枝指标，简单高效但仍是启发式方法；本文的优化方法可以以Wanda的结果为初始化来进一步改进
- **vs LLM-Pruner**: LLM-Pruner也做结构化剪枝但依赖梯度信息（需要反向传播）；本文完全避免了反向传播

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 策略梯度绕过反向传播进行结构化剪枝的idea非常巧妙
- 实验充分度: ⭐⭐⭐⭐ 多模型多剪枝率测试，但缺少下游任务评测
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，数学推导完整
- 价值: ⭐⭐⭐⭐ 为LLM结构化剪枝提供了新的优化范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](maporl_multi-agent_post-co-training_for_collaborative_large_language_models_with.md)
- [\[ICLR 2026\] Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models](../../ICLR2026/reinforcement_learning/remix_reinforcement_routing_for_mixtures_of_loras_in_llm_finetuning.md)
- [\[NeurIPS 2025\] MMaDA: Multimodal Large Diffusion Language Models](../../NeurIPS2025/reinforcement_learning/mmada_multimodal_large_diffusion_language_models.md)
- [\[NeurIPS 2025\] GraphChain: Large Language Models for Large-scale Graph Analysis via Tool Chaining](../../NeurIPS2025/reinforcement_learning/graphchain_large_language_models_for_large-scale_graph_analysis_via_tool_chainin.md)
- [\[ICML 2025\] Wasserstein Policy Optimization](../../ICML2025/reinforcement_learning/wasserstein_policy_optimization.md)

</div>

<!-- RELATED:END -->
