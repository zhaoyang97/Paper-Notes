---
title: >-
  [论文解读] RiOT: Efficient Prompt Refinement with Residual Optimization Tree
description: >-
  [ACL 2025][LLM 其他][提示学习] 提出 Residual Optimization Tree（RiOT），一种自动 prompt 优化框架，通过树结构管理优化过程、基于困惑度的节点选择增强多样性、以及文本残差连接缓解语义漂移问题。 问题背景 LLM 的性能高度依赖 prompt 设计，手动设计 prompt…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "提示学习"
  - "residual connection"
  - "perplexity"
  - "tree search"
  - "semantic drift"
---

# RiOT: Efficient Prompt Refinement with Residual Optimization Tree

**会议**: ACL 2025  
**arXiv**: [2506.16389](https://arxiv.org/abs/2506.16389)  
**代码**: [github.com/Qing1Zhong/RiOT](https://github.com/Qing1Zhong/RiOT)  
**领域**: 自动 Prompt 优化 / LLM  
**关键词**: prompt optimization, residual connection, perplexity, tree search, semantic drift  

## 一句话总结

提出 Residual Optimization Tree（RiOT），一种自动 prompt 优化框架，通过树结构管理优化过程、基于困惑度的节点选择增强多样性、以及文本残差连接缓解语义漂移问题。

## 研究背景与动机

### 问题背景
LLM 的性能高度依赖 prompt 设计，手动设计 prompt 需要大量领域知识和试错。自动 prompt 优化方法逐步兴起，但现有方法面临两个关键挑战：

### 两大挑战

**多样性不足**：现有方法（OPRO、TextGrad、ProTeGi）每轮仅生成一个候选 prompt，限制了搜索空间的广度；APE 虽然并行生成多个候选，但缺乏迭代精炼机制

**语义漂移**（Semantic Drift）：迭代优化 prompt 时，针对某个任务的优化可能损害先前任务的表现，类似于持续学习中的稳定性-可塑性困两难

### 类比启发
语义漂移问题类似于深度网络中的梯度消失和持续学习中的灾难性遗忘。RiOT 借鉴深度学习中的残差连接思想来解决这一问题。

## 方法详解

### 整体框架

RiOT 以初始 prompt $p_0$ 为根节点构建树结构。在第 $t$ 步：
1. **生成**：父节点 $p_t$ 通过 Prompt Optimization Operator $\mathcal{M}(\cdot)$ 生成 $K$ 个候选子节点
2. **选择**：基于困惑度选择最优子节点（剪枝）
3. **融合**：通过文本残差连接将父节点和最优子节点融合

### 关键设计一：基于困惑度的节点选择（Perplexity-Informed Node Selection）

传统方法仅基于候选质量选择，忽略多样性。RiOT 利用困惑度选择最优候选：

$$p_{t+1}^* = \arg\max_i PPL(p_{t+1}^{(i)})$$

**选择高困惑度的理由**：
- 从信息论角度，高困惑度意味着更低的 token 共现概率，包含更多信息
- 类似贝叶斯优化中优先探索高不确定性区域的策略
- 高困惑度的 prompt 更可能代表新颖的、未被充分探索的优化方向

### 关键设计二：文本残差连接（Text Residual Connection）

核心思想：不直接将优化后的子 prompt 作为下一轮输入，而是**融合父节点和子节点**的语义内容：

$$p_t = \mathcal{G}(p_{t-1}, p_t^*)$$

**Content Fusion Algorithm**（Algorithm 1）：
1. 将父 prompt 和子 prompt 分别分句
2. 用预训练 embedding 模型计算每句的语义表示
3. 计算父-子句子间的余弦相似度矩阵
4. **保留**父节点中与子节点高相似度（≥1-b₁）的句子（这些是被保留的有价值内容）
5. **引入**子节点中与父节点低相似度（<1-b₂）的新内容
6. 合并两者形成新 prompt

两个超参数 $b_1$ 和 $b_2$ 控制融合程度：
- $b_1 = 0.25$：控制从父节点保留的内容
- $b_2 = 0.5$：控制从子节点引入的新内容

### Prompt Optimization Operator

基于 TextGrad 作为 backbone：
1. 目标模型 $\mathcal{F}_{\text{target}}$ 用当前 prompt 在训练集上生成响应
2. 计算损失 $\mathcal{L}(p_t)$
3. 优化模型 $\mathcal{F}_{\text{opt}}$ 根据文本梯度 $\nabla_{p_t}\mathcal{L}(p_t)$ 提出候选 prompt
4. LLM 输出的固有变异性自然产生 $K$ 个不同候选

## 实验

### 实验设置
- **目标模型**：GPT-3.5-turbo（temperature=0）
- **优化模型**：GPT-4o（temperature=0）
- **Embedding 模型**：text-embedding-3-large
- **树宽度 K=3**，训练 3 epochs × batch size 4 = 15 次迭代
- **5 个 benchmark**：LogiQA 2.0、StrategyQA、Object Counting、GSM8K、Date Understanding

### 主实验结果

| 方法 | LogiQA 2.0 | StrategyQA | Object Counting | GSM8K | Date Understanding |
|------|-----------|-----------|----------------|-------|-------------------|
| Zero-Shot CoT | 59.0 | 65.8 | 71.0 | 60.2 | 76.1 |
| APE | 57.4 | 70.4 | 79.2 | 76.6 | 72.8 |
| OPRO | 58.0 | 67.0 | 79.9 | 72.8 | 76.3 |
| TextGrad | 60.0 | 68.8 | 88.3 | 74.8 | 71.9 |
| DSPy | 59.8 | 73.4 | 84.5 | 79.0 | 74.3 |
| **RiOT** | **61.4** | **74.6** | 86.9 | **81.2** | **78.2** |

- RiOT 在 **4/5** 个任务上取得最优，在 GSM8K 上比最强基线提升 **+2.2%**
- 加权平均准确率超过最佳基线 **+2.7%**

### 消融实验（GSM8K）

| 变体 | 准确率 |
|------|--------|
| RiOT（完整） | **81.2 ±1.2** |
| w/o 文本残差连接 | 68.8 ±3.9（-12.4） |
| w/o 困惑度节点选择 | 68.2 ±1.8（-13.0） |

两个核心组件各贡献约 12-13% 的绝对提升，缺一不可。

### 节点选择指标对比

| 指标 | 准确率 |
|------|--------|
| Perplexity | **81.2** |
| Entropy | 78.6（-2.6） |
| Length | 73.6（-7.6） |

困惑度显著优于熵和长度，验证了基于困惑度选择的有效性。

### 迁移性实验（在 Gemini-1.5-flash 上评估）
- **Prompt 迁移**（为 GPT-3.5 优化的 prompt 用于 Gemini）：平均略有下降
- **模型迁移**（直接为 Gemini 优化）：一致提升
- 说明优化后的 prompt 有一定跨模型迁移能力

### 关键发现
1. Few-shot CoT 的 scaling 收益递减——从 4-shot 到 20-shot 提升微乎其微
2. RiOT 是唯一在所有 5 个任务上均提升的方法，其他方法存在个别任务退化
3. 树宽度 $K=3$ 是最优，过大反而降低性能

## 亮点与洞察

1. **首个树结构 + 残差连接的 prompt 优化框架**：优雅地解决了搜索多样性和语义漂移双重问题
2. **困惑度选择的反直觉洞察**：选择"更不确定"的 prompt 反而能探索更有价值的方向
3. **文本残差连接的创新**：将深度学习中的残差思想迁移到离散文本空间，保留有价值的语义组件
4. **跨任务稳定性**：RiOT 是实验中唯一在所有任务上均有提升的方法
5. **消融实验展示两个核心组件各贡献约 12-13% 提升**，说明二者同等重要

## 局限性

1. 依赖强大的优化模型（GPT-4o）和 embedding 模型，成本较高
2. 超参数 $b_1$、$b_2$ 和 $K$ 的选择需要验证集调优
3. 在 Object Counting 上未达最优（TextGrad 88.3% vs RiOT 86.9%）
4. 文本残差连接的句子粒度融合可能丢失句内的细粒度优化

## 相关工作

- **Soft Prompt Tuning**：PEFT 类方法，需要模型参数访问
- **离散 Token 搜索**：梯度引导或 RL 方法，需要完整模型访问
- **黑盒 Prompt 优化**：APE（并行候选）、OPRO（元提示优化）、TextGrad（文本梯度）、DSPy（声明式程序合成）
- **语义漂移/灾忘**：参数隔离、正则化等持续学习方法的文本空间类比

## 评分

⭐⭐⭐⭐ — 方法设计优雅（树 + 残差 + 困惑度），实验全面且消融充分。两个核心组件的贡献明确。主要不足在于对强模型（GPT-4o）的依赖和有限的分析深度。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection](erm_prompt_optimization_memory.md)
- [\[ACL 2025\] SEE: Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization](see_strategic_exploration_exploitation_prompt_optimization.md)
- [\[ACL 2025\] Dynamic Parallel Tree Search for Efficient LLM Reasoning](dynamic_parallel_tree_search_for_efficient_llm_reasoning.md)
- [\[ACL 2025\] Efficient Universal Goal Hijacking with Semantics-guided Prompt Organization](goal_hijacking_attack.md)
- [\[ACL 2025\] OPTS: Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](bandit-based_prompt_design_strategy_selection_improves_prompt_optimizers.md)

</div>

<!-- RELATED:END -->
