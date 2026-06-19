---
title: >-
  [论文解读] Causal Graph based Event Reasoning using Semantic Relation Experts
description: >-
  [ACL 2025][因果推理][因果图生成] 提出基于四类语义关系专家（时间、篇章、条件、常识）多轮协作讨论的因果事件图生成框架，在零样本设置下于事件预测、事件预报等多个下游任务上取得与微调模型竞争的结果，并提供可解释的因果事件链。 - 任务定义：给定一组事件，构建全局因果事件图（节点=事件，有向边=因果关系）…
tags:
  - "ACL 2025"
  - "因果推理"
  - "因果图生成"
  - "事件推理"
  - "多智能体协作"
  - "语义关系专家"
  - "可解释预测"
---

# Causal Graph based Event Reasoning using Semantic Relation Experts

**会议**: ACL 2025  
**arXiv**: [2506.06910](https://arxiv.org/abs/2506.06910)  
**代码**: [github](https://github.com/StonyBrookNLP/causal-graphs)  
**领域**: 因果推理  
**关键词**: 因果图生成, 事件推理, 多智能体协作, 语义关系专家, 可解释预测

## 一句话总结

提出基于四类语义关系专家（时间、篇章、条件、常识）多轮协作讨论的因果事件图生成框架，在零样本设置下于事件预测、事件预报等多个下游任务上取得与微调模型竞争的结果，并提供可解释的因果事件链。

## 研究背景与动机

- **任务定义**：给定一组事件，构建全局因果事件图（节点=事件，有向边=因果关系），并利用该图辅助事件预测、预报等推理任务。
- **现有问题**：现有事件推理方法主要依赖事件共现的分布关系，缺乏对深层因果逻辑的显式建模；即使是 SOTA LLM，在标准 ICL 设置下识别因果关系的准确率仍不高。
- **核心挑战**：因果判断需要考虑事件在全局语境中的嵌入方式。例如地震本身很常见，但只有当"城市资源短缺"且"遭受损害"两个事件同时成立时，才导致"官员请求援助"——单独让 LLM 判断容易遗漏这种多事件联合的微妙因果。
- **本文方案**：设计四类语义关系专家，通过多轮辩论式协作生成全局因果图，并以因果图驱动下游推理，实现可解释的事件预测。

## 方法详解

### 整体框架

用 LLM 模拟四个关注不同语义维度的"专家"，经过独立分析→多轮讨论→裁判整合三阶段，产出全局因果事件图。随后将因果图用于下游任务：可解释事件似然预测（EEL）、事件预报（ForecastQA）、下一事件预测（Narrative Cloze）。

### 关键设计

**1. 四类语义关系专家**

每个专家被赋予不同的因果性判断视角：

| 专家 | 关注维度 | 核心思路 |
|------|---------|---------|
| 时间专家 (Temporal) | 事件时序关系 | 时间先后是因果的必要条件，通过筛选时序合理的事件对缩小搜索空间 |
| 篇章专家 (Discourse) | 共享实体关系 | 共享实体的事件对更可能存在因果链——对实体的操作可能触发后续事件 |
| 条件专家 (Conditional) | 反事实前置条件 | 通过反事实推理判断：移除事件 A 后事件 B 是否仍会发生，识别必要前提 |
| 常识专家 (Commonsense) | 隐含背景知识 | 捕捉未在文本中显式提及的中间知识，桥接表面上无直接关系的事件对 |

**2. 多轮协作讨论机制**

采用"关注点分离"策略，避免让 LLM 一次性处理所有维度：

- **初始化**：四个专家各自独立生成因果关系判断及推理依据
- **多轮讨论**（最多 3 轮）：每轮中每个专家获取其他所有专家的响应，分析后修订自己的因果链接列表并给出修改理由；专家可以接受、反驳或补充其他专家的观点
- **裁判整合**：一个 Causality Judge LLM 汇总所有讨论结果，解决剩余分歧，输出最终因果图

**3. 基于因果图的下游推理（CGEL）**

将因果图用于可解释事件似然预测：给定已观测事件集合和查询事件，判断查询事件能否插入因果图中——若可插入则认为 likely，同时输出一条因果事件链作为解释。该方法零样本、不需在下游任务上微调。

### 训练策略

本方法为纯推理时框架，无需训练或微调。使用 GPT-4o 和 Llama-70B-instruct 作为基础 LLM，通过精心设计的 prompt 实现各专家角色分配与讨论协议。

## 实验关键数据

### 主实验：因果图生成质量（CRAB 数据集，图级别指标）

| 方法 | LLM | BAcc | F1:Causal | F1:Non-Causal | Macro F1 |
|------|-----|------|-----------|---------------|----------|
| Direct (零样本直接生成) | GPT-4o | 70.86 | 66.17 | 76.80 | 71.48 |
| Pairwise (逐对判断) | GPT-4o | 73.93 | 62.99 | 82.37 | 72.68 |
| Experts wo collab | GPT-4o | 74.92 | 70.21 | 78.23 | 74.22 |
| **Collab with experts** | **GPT-4o** | **79.27** | **75.62** | **82.80** | **79.21** |
| Direct | Llama-70B | 63.08 | 53.42 | 69.35 | 61.39 |
| **Collab with experts** | **Llama-70B** | **73.69** | **73.31** | **71.67** | **72.49** |

### 下游任务结果

| 任务 | 系统 | 准确率 |
|------|------|--------|
| 事件预报 (ForecastQA) | GPT-4 baseline | 51.3% |
| | One-shot baseline | 50.0% |
| | **CGEL（本文）** | **62.7%** |
| | BERT-large + MDS（微调） | 67.4% |
| 下一事件预测 (NC) | ELM | 46.0% |
| | EGELM | 50.0% |
| | **CGEL with context** | **61.0%** |

EEL 任务中 CGEL vs One-shot baseline：因果性维度赢 41.6%，信息量维度赢 48.4%，连贯性维度赢 37.0%。

### 消融实验

| 设置 | BAcc | Macro F1 | 相对完整方法下降 |
|------|------|----------|----------------|
| Collab with experts（完整） | 79.27 | 79.21 | — |
| Collab wo experts（无专家角色） | 75.39 | 75.51 | -3.70 |
| 去掉时间专家 | 77.51 | 77.72 | -1.49 |
| 去掉前置条件专家 | 77.48 | 77.26 | -1.95 |
| 去掉篇章专家 | 78.32 | 78.29 | -0.92 |
| 去掉常识专家 | 78.88 | 78.85 | -0.36 |

去掉任何一个专家均导致性能下降，前置条件专家和时间专家影响最大。

### 辩论轨迹分析

| 专家 | 初始与 gold 重叠 | 讨论后与 gold 重叠 | 贡献度 | 错误翻转率 |
|------|------------------|-------------------|--------|-----------|
| 时间专家 | 13% | 33% | 64% | 0% |
| 篇章专家 | 17% | 24% | 64% | 0% |
| 前置条件专家 | 17% | 22% | 46% | 67% |
| 常识专家 | 22% | 26% | 57% | 0% |

时间专家初始最弱但经讨论后提升最大，前置条件专家错误翻转率最高。

## 亮点与洞察

1. **异构专家协作优于同构辩论**：不同于 ChatEval 等让多个相同角色 LLM 辩论的方式，本文为每个 agent 赋予不同的语义关系专长，实现了真正的"关注点分离"。实验证明去掉专家角色（Collab wo experts）BAcc 下降近 4 个点。

2. **零样本即可竞争微调模型**：CGEL 在 ForecastQA 上达到 62.7%，接近 BERT-large 微调的 67.4%，且无需任何任务特定训练数据，同时还能输出因果事件链作为解释——这是微调模型做不到的。

3. **辩论过程可分析可调试**：详细追踪了每个专家在讨论中的翻转、添加和冲突模式，形成了透明的决策路径，有助于后续改进。

## 局限性与改进方向

1. 依赖基础 LLM 的因果理解能力，可能与人类因果感知存在偏差。
2. 多轮多专家讨论的计算成本较高（每个场景需要多次 LLM 调用）。
3. 未建模因果强度的等级化判断，仅做二元因果/非因果分类。
4. GPT-4 作为评估器可能偏向自身生成内容。
5. 可扩展更多类型的语义关系专家以及更多领域/语言的验证。

## 评分

- **创新性**: ★★★★☆ — 异构语义专家协作生成全局因果图的框架新颖，与已有多智能体辩论方法有本质区别
- **实用性**: ★★★★☆ — 零样本可解释，适用于事件预测/预报/解释等多种场景
- **实验充分度**: ★★★★★ — 内在评估+三个外在任务+消融+辩论轨迹分析，评估维度全面
- **写作质量**: ★★★★☆ — 动机清晰，框架描述系统，但部分符号说明稍显冗余

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CoA-Reasoning: Explorations on Counterfactual Analysis in Physical Reasoning of LVLMs](coa-reasoning_explorations_on_counterfactual_analysis_in_physical_reasoning_of_l.md)
- [\[CVPR 2026\] CGU-Bayes: Causal Graph Uncertainty-Guided Bayesian Inference for Domain Generalization](../../CVPR2026/causal_inference/cgu-bayes_causal_graph_uncertainty-guided_bayesian_inference_for_domain_generali.md)
- [\[ACL 2026\] iTAG: Inverse Design for Natural Text Generation with Accurate Causal Graph Annotations](../../ACL2026/causal_inference/itag_inverse_design_for_natural_text_generation_with_accurate_causal_graph_annot.md)
- [\[ICML 2026\] From Observation to Intervention: A Causal Audit of Expert Importance in Mixture-of-Experts Models](../../ICML2026/causal_inference/from_observation_to_intervention_a_causal_audit_of_expert_importance_in_mixture-.md)
- [\[ACL 2025\] Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation](reasoning_is_all_you_need_for_video_generalization_a_counterfactual_benchmark_wi.md)

</div>

<!-- RELATED:END -->
