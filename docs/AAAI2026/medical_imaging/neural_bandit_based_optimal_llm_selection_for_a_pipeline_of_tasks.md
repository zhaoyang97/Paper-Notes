---
title: >-
  [论文解读] Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks
description: >-
  [AAAI 2026][医学图像][LLM选择] 提出 Sequential Bandits 算法，一种基于神经上下文多臂老虎机的在线学习方法，用于在任务流水线（如"摘要→诊断"）中为每个子任务选择最优 LLM，同时优化准确率和成本，在医学诊断和电信问答两个流水线任务上优于现有 bandit 基线。
tags:
  - "AAAI 2026"
  - "医学图像"
  - "LLM选择"
  - "多臂老虎机"
  - "神经上下文bandit"
  - "流水线任务"
  - "医学诊断预测"
  - "成本感知"
---

# Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks

**会议**: AAAI 2026  
**arXiv**: [2508.09958](https://arxiv.org/abs/2508.09958)  
**代码**: 有（含自建数据集）  
**领域**: 医学图像 / LLM 路由与选择  
**关键词**: LLM选择, 多臂老虎机, 神经上下文bandit, 流水线任务, 医学诊断预测, 成本感知

## 一句话总结
提出 Sequential Bandits 算法，一种基于神经上下文多臂老虎机的在线学习方法，用于在任务流水线（如"摘要→诊断"）中为每个子任务选择最优 LLM，同时优化准确率和成本，在医学诊断和电信问答两个流水线任务上优于现有 bandit 基线。

## 研究背景与动机

**领域现状**：随着 LLM 数量激增（包括 OpenAI/Azure 等平台的自定义 assistant），如何为特定任务选择最佳 LLM 成为关键问题。不同 LLM 在不同任务上表现差异巨大，简单选最大模型既昂贵又可能次优。

**流水线任务的挑战**：很多复杂任务需要分解为子任务（如医学报告 → 摘要 → 诊断），每个子任务可由不同 LLM 完成。关键困难在于：
   - 子任务之间存在**级联依赖**：前一个 LLM 的输出质量影响后续 LLM 的输入和性能
   - LLM 组合数指数增长
   - 成本与准确率的权衡：更强模型通常更贵

**现有方法的不足**：
   - **LLM 级联**：按预定义顺序从便宜到贵依次尝试，但多次推理浪费且序列不可自由配置
   - **LLM 路由**：单次选择一个 LLM，不适用于多子任务流水线
   - **标准 contextual bandit**：不支持子任务之间的顺序依赖关系
   - **组合 bandit**：一次性选择所有 arm，无法在观察前序结果后做决策

**切入角度**：在线学习 + 顺序 bandit，为每个（子任务, LLM）组合训练独立神经网络，利用 UCB 探索-利用权衡，同时引入成本惩罚项。

## 方法详解

### 问题建模

- 输入：查询 $q_t$ 分解为子任务 $\{T_1, T_2, \ldots, T_k\}$（DAG 结构）
- 每个子任务有 $N_i$ 个可选 LLM（arm）
- 选择 super arm $S_t = (a_{1,j}, a_{2,j}, \ldots, a_{k,j})$
- 目标：最大化净奖励 $N(t) = R(S_t, \mathbf{r}_t) - \boldsymbol{\alpha} \cdot \mathbf{C}(S_t)$，其中 $\boldsymbol{\alpha}$ 控制准确率与成本的权衡

### Sequential Bandits 算法核心

**每个（子任务, LLM）组合维护一个独立的神经网络**来预测奖励：

$$f_{i,j}(\mathbf{x}; \boldsymbol{\theta}) = \sqrt{m} \mathbf{W}^{(L)}_{i,j} \sigma(\mathbf{W}^{(L-1)}_{i,j} \sigma(\cdots \sigma(\mathbf{W}^{(0)}_{i,j} \mathbf{x})))$$

**UCB 选择策略**：对第 $i$ 个子任务的每个可选 LLM $j$，计算：

$$u_{i,j} = f_{i,j}(p_i, d_j) + \left\|\frac{\mathbf{g}_{i,j}(\mathbf{x}_t(a_{i,j}); \boldsymbol{\theta}^{t-1}_{i,j})}{\sqrt{n}}\right\|_{\mathbf{Z}^{-1}_{t-1}(a_{i,j})} - \alpha_i C_j(p_i)$$

- 第一项：神经网络预测的期望奖励（exploitation）
- 第二项：基于梯度的不确定性估计（exploration）
- 第三项：成本惩罚（cost sensitivity）

**顺序决策流程**：
1. 子任务 $T_1$：以原始查询 $q_t$ 为输入，选择 UCB 最大的 LLM
2. 子任务 $T_i$（$i>1$）：以前一子任务 LLM 的输出为输入，再选择 UCB 最大的 LLM
3. 执行所有子任务后观察各 base arm 奖励和 super arm 奖励
4. 仅更新被选中 LLM 的神经网络权重

### 成本建模

- 使用 BERT 回归模型（在 LMSYS-Chat-1M 上训练）预测输出 token 数
- 成本 = 输入 token 数 × 单价 + 预测输出 token 数 × 单价（Azure 定价）

### 与现有神经 bandit 的关键区别

1. **独立网络 vs 共享网络**：为每个 (subtask, LLM) 训练独立网络，避免共享网络导致不同 arm 的奖励估计过于相似→过早锁定次优 arm
2. 不增加训练开销：每轮只更新被选中 LLM 的网络
3. 支持子任务间的顺序依赖

## 实验

### 数据集
1. **医学诊断预测**（自建）：从 MIMIC-III 构建 100 个病人报告，去除诊断相关内容，子任务为"摘要→诊断"（2子任务）
2. **TeleQnA**：电信领域 10,000 道选择题，子任务为"摘要→答题→解释"（3子任务）

### 模型集合
- GPT-3.5-turbo、GPT-4o、Llama-3.3-70B-instruct、Mistral-3B、Phi-4
- 领域微调模型：Med（医学通识）、Tele（电信）、Med III（MIMIC-III 微调）
- GPT-3.5-turbo assistants（带检索增强）
- 成本从低到高：GPT-3.5-turbo < Llama 3.3 < Med < Tele < Med III

### 基线
- Random：随机选择
- Llama：固定选 Llama（单任务最优模型）
- Cost-Aware NeuralUCB：每个子任务一个共享网络
- Cost-Aware NeuralLinUCB：神经网络+线性层

### 主实验结果

| 设置 | Sequential Bandits vs 最强基线 |
|------|-------------------------------|
| 医学诊断（净奖励）| +7.60% vs Llama |
| 电信问答（净奖励）| +6.51% vs Random |

**关键洞察**：
- 尽管 Llama 在医学诊断单任务中准确率最高，但 Sequential Bandits 仍优于固定选 Llama，说明算法学到了更优的成本-准确率组合
- Random 在电信场景中意外优于 CA-NeuralUCB 和 NeuralLinUCB

### LLM 选择分析

- Sequential Bandits 为诊断子任务选择 Llama（49.1%）和 GPT-3.5（39.2%）最多
- 这两个恰好是最便宜的且准确率最高/次高的模型
- 基线方法更频繁选择次优且昂贵的模型（如 Med）

## 亮点与洞察

1. **首个研究 LLM 流水线选择问题**：将 LLM 选择从"单点路由"扩展到"顺序流水线"，有实际意义
2. **流水线改变最优选择**：加摘要器后，最优诊断模型从 Med III 变为 Llama，说明子任务间的交互效应不可忽视
3. **独立网络 > 共享网络**：共享网络使 arm 估计趋同，导致过早锁定在成本最低的 arm 上
4. **在线学习无需历史数据**：适用于新 LLM/自定义 assistant 上线的冷启动场景
5. **成本建模灵活**：$\alpha_i$ 可按子任务分别调整，如摘要任务输入长成本高可降低权重

## 局限性

1. 医学数据集仅 100 个报告，规模小且主要限于心脏/肾脏/肝脏/脑部疾病
2. 假设任务分解（分为哪些子任务）是预先给定的，未讨论自动分解
3. 未提供理论遗憾界证明（仅实验评估）
4. 成本模型依赖 token 数预测，对多模态 LLM 或流式输出不直接适用
5. 未考虑 LLM 推理延迟作为成本（仅在附录讨论）

## 相关工作

- **预算感知 bandit**：Castiglioni 等人的原始-对偶方案，但不支持顺序决策
- **LLM 级联**：FrugalGPT、AutoMix，按预定义序列从便宜到贵依次尝试
- **LLM 路由**：Tryage（上下文感知）、Zooter（奖励模型打分路由）、RoutingExperts（动态专家）
- **Neural Bandits**：NeuralUCB、NeuralLinUCB，本文在此基础上扩展为顺序版本

## 评分 ⭐⭐⭐⭐

- **创新性**：⭐⭐⭐⭐ — 流水线LLM选择问题定义新颖，顺序bandit对标准bandit的扩展自然且有效
- **实验**：⭐⭐⭐ — 医学数据集太小（100例），电信数据集也较标准，缺乏大规模验证
- **写作**：⭐⭐⭐⭐ — 问题定义清晰，算法描述完整
- **实用性**：⭐⭐⭐⭐ — 在LLM proliferation时代有明确的应用场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] SPA: Achieving Consensus in LLM Alignment via Self-Priority Optimization](spa_achieving_consensus_in_llm_alignment_via_self-priority_optimization.md)
- [\[ICLR 2026\] LaVCa: LLM-assisted Visual Cortex Captioning](../../ICLR2026/medical_imaging/lavca_llm-assisted_visual_cortex_captioning.md)
- [\[AAAI 2026\] Intervention Efficiency and Perturbation Validation Framework: Capacity-Aware and Robust Clinical Model Selection under the Rashomon Effect](intervention_efficiency_and_perturbation_validation_framework_capacity-aware_and.md)
- [\[ICLR 2026\] Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification](../../ICLR2026/medical_imaging/inference-time_dynamic_modality_selection_for_incomplete_multimodal_classificati.md)
- [\[CVPR 2026\] MedTVT-R1: A Multimodal LLM Empowering Medical Reasoning and Diagnosis](../../CVPR2026/medical_imaging/medtvt-r1_a_multimodal_llm_empowering_medical_reasoning_and_diagnosis.md)

</div>

<!-- RELATED:END -->
