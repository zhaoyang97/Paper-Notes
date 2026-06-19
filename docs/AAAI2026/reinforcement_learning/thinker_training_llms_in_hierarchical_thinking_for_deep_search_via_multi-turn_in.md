---
title: >-
  [论文解读] Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction
description: >-
  [AAAI2026][强化学习][deep search] 提出 Thinker 框架，通过分层思维（breadth decomposition + depth solving）和双重表征（自然语言 + 逻辑函数）实现结构化的深度搜索推理，配合知识边界判定减少不必要检索，以 SFT 方式训练，在多个 QA 基准上显著超越 RL-based deep search 方法。
tags:
  - "AAAI2026"
  - "强化学习"
  - "deep search"
  - "hierarchical thinking"
  - "RAG"
  - "multi-turn interaction"
  - "knowledge boundary"
---

# Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction

**会议**: AAAI2026  
**arXiv**: [2511.07943](https://arxiv.org/abs/2511.07943)  
**代码**: [OpenSPG/KAG-Thinker](https://github.com/OpenSPG/KAG-Thinker)  
**领域**: 强化学习  
**关键词**: deep search, hierarchical thinking, RAG, multi-turn interaction, knowledge boundary

## 一句话总结
提出 Thinker 框架，通过分层思维（breadth decomposition + depth solving）和双重表征（自然语言 + 逻辑函数）实现结构化的深度搜索推理，配合知识边界判定减少不必要检索，以 SFT 方式训练，在多个 QA 基准上显著超越 RL-based deep search 方法。

## 研究背景与动机
LLM 在复杂多跳推理任务中面临知识不足和幻觉问题。现有 deep search 方法主要基于端到端强化学习训练，存在四大典型问题：

- **交错求解（Interleaved Solving）**：子问题求解过程交织混乱
- **层次不清（Unclear Hierarchy）**：缺乏清晰的问题分解层次
- **粒度不一（Inconsistent Granularity）**：子问题拆分粒度不统一
- **检索低效（Inefficient Search）**：对每个问题都发起检索，即使 LLM 已知答案

核心洞察：人类专家通过结构化思维解决问题——先分解为独立子问题，再逐一解决。RL 方法难以约束推理过程的逻辑性和严谨性。

## 方法详解

### Breadth Decomposition（广度分解）
将复杂问题一次性分解为 $n$ 个原子粒度的子问题，每个子问题可独立求解：

- 定义四种 logical form 函数：**Retrieval**（检索）、**Math**（计算）、**Deduce**（推理）、**Output**（聚合输出）
- 每个子问题采用**双重表征**：
    - **Step**（自然语言）：用于通用检索器（如 E5、BGE-M3）
    - **Action**（逻辑函数）：用于结构化知识图谱检索
- 子问题间依赖通过变量传递（`#n`、`o_n`、`s_n`），保证逻辑连贯性

示例："Hit Parade of 1947 和 Khiladi 420 的导演谁先去世？"分解为 5 个子问题，依赖关系通过变量显式传递。

### Depth Solving（深度求解）
对 Retrieval 类子问题进行迭代式深度求解：

1. 执行检索 → 聚焦分析 → 推理判断
2. 若当前检索不足以回答，生成新的 logical form 继续检索
3. 迭代直到：(a) 得到答案（`<<answer>>`标签），或 (b) 达到最大搜索深度 $D$

### Knowledge Boundary Determination（知识边界判定）
**"先生成，再评估"策略**，避免不必要的外部检索：

1. LLM 先尝试用内部知识直接回答子问题
2. 双重置信度评估：
    - **Prompt-based**：自省式验证，输出 True/False
    - **Likelihood-based**：取答案 token 序列的最小生成概率 $C = \min_t p(y_t|\boldsymbol{x}, y_{<t})$，与阈值 $\tau$ 比较
3. 两者均为 True 时才跳过检索

### 训练方式
采用 multi-turn interactive SFT（而非 RL）：
- 将完整交互序列 $[S, U_1, A_1, \ldots, U_n, A_n]$ 拼接
- 仅对 Assistant 回复 token 计算 cross-entropy loss
- 优势：可控制推理风格，适用于垂直领域定制；兼容并行处理

## 实验关键数据

### 主实验（Table 1，EM 指标）

| 方法 | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | Avg |
|------|-----|----------|-------|----------|-------|---------|-----------|-----|
| Search-R1 (7B) | 0.393 | 0.610 | 0.397 | 0.370 | 0.414 | 0.146 | 0.368 | 0.385 |
| ReSearch (7B) | 0.407 | 0.611 | 0.423 | 0.419 | 0.412 | 0.205 | 0.400 | 0.411 |
| **Thinker (7B)** | **0.450** | **0.642** | **0.484** | **0.421** | **0.469** | **0.221** | **0.480** | **0.452** |

- 较 ReSearch 平均提升 **4.1%**（单跳 +4.5%，多跳 +3.9%）
- 3B 模型提升更显著：较 ReSearch 平均 +**7.9%**

### 逻辑严谨性评估（Table 2，GPT-4 评判）

| 方法 | Hierarchy | Interleaved | Granularity | Efficiency | Overall |
|------|-----------|-------------|-------------|------------|---------|
| Search-R1 | 0.813 | 0.955 | 0.852 | 0.903 | 0.638 |
| ReSearch | 0.872 | 0.967 | 0.877 | 0.922 | 0.705 |
| **Thinker** | **0.975** | **0.989** | **0.955** | **0.958** | **0.904** |

### 样本效率（Table 4）
- 仅用 **1% 数据（约数百条）** 即可达到接近 SOTA 水平（Avg 0.406 vs ReSearch 0.411）
- 完整数据 Avg 0.452

### 消融实验（Table 3）
- 移除 depth solving：Avg 下降 **3.7%**（影响最大）
- 移除 knowledge boundary：Avg 仅降 0.5%，但显著减少不必要检索次数
- 移除 logical function：整体影响小，但对知识图谱检索不可或缺

## 亮点
- **结构化推理过程**：通过分层思维使推理过程可监督、可验证，逻辑严谨性远超 RL 方法
- **双重表征**：自然语言 + 逻辑函数兼容通用检索器和结构化知识库
- **极高样本效率**：数百条训练样本即可接近 SOTA，SFT 训练成本远低于 RL
- **知识边界判定**：prompt + likelihood 双重评估有效减少噪声检索

## 局限性
- SFT 依赖高质量标注数据的构造流程，数据构造本身需要一定工程量
- 逻辑函数限定为四种预定义类型（Retrieval/Math/Deduce/Output），扩展性待验证
- 评测仅使用文本检索器（E5），knowledge graph 检索实验未在主表呈现
- 知识边界判定的阈值 $\tau$ 需要调优

## 评分
- 新颖性: ⭐⭐⭐⭐ — 分层思维 + 双重表征的设计有清晰方法论贡献，SFT 替代 RL 的思路实用
- 实验充分度: ⭐⭐⭐⭐ — 7 个数据集、多模型规模、消融和灵敏度分析齐全
- 写作质量: ⭐⭐⭐⭐ — 框架图清晰，问题定义明确
- 价值: ⭐⭐⭐⭐⭐ — 对 deep search / agentic RAG 领域有直接的实践指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Reinforcement Learning for Long-Horizon Multi-Turn Search Agents](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_for_long-horizon_multi-turn_search_agents.md)
- [\[AAAI 2026\] HCPO: Hierarchical Conductor-Based Policy Optimization in Multi-Agent Reinforcement Learning](hcpo_hierarchical_conductor-based_policy_optimization_in_multi-agent_reinforceme.md)
- [\[ICLR 2026\] AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking](../../ICLR2026/reinforcement_learning/abstral_augmenting_llms_reasoning_by_reinforcing_abstract_thinking.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](../../ICLR2026/reinforcement_learning/spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[ICLR 2026\] Masked Skill Token Training for Hierarchical Off-Dynamics Transfer](../../ICLR2026/reinforcement_learning/masked_skill_token_training_for_hierarchical_off-dynamics_transfer.md)

</div>

<!-- RELATED:END -->
