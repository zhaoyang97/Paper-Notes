---
title: >-
  [论文解读] Rerank Before You Reason: Analyzing Reranking Tradeoffs through Effective Token Cost in Deep Search Agents
description: >-
  [ACL 2026 Findings][信息检索/RAG][深度搜索智能体] 本文系统研究了深度搜索智能体中 listwise 重排序的效率-效果权衡，提出 Effective Token Cost (ETC) 指标，发现中等深度重排序通常比增加搜索时推理预算更具成本效益，在更低 token 开销下达到相当甚至更高的端到端准确率。
tags:
  - "ACL 2026 Findings"
  - "信息检索/RAG"
  - "深度搜索智能体"
  - "重排序"
  - "推理预算分配"
  - "有效token成本"
  - "检索增强推理"
---

# Rerank Before You Reason: Analyzing Reranking Tradeoffs through Effective Token Cost in Deep Search Agents

**会议**: ACL 2026 Findings  
**arXiv**: [2601.14224](https://arxiv.org/abs/2601.14224)  
**代码**: [https://github.com/sahel-sh/DeepHone](https://github.com/sahel-sh/DeepHone)  
**领域**: 信息检索  
**关键词**: 深度搜索智能体, 重排序, 推理预算分配, 有效token成本, 检索增强推理

## 一句话总结
本文系统研究了深度搜索智能体中 listwise 重排序的效率-效果权衡，提出 Effective Token Cost (ETC) 指标，发现中等深度重排序通常比增加搜索时推理预算更具成本效益，在更低 token 开销下达到相当甚至更高的端到端准确率。

## 研究背景与动机
**领域现状**: 深度研究智能体通过迭代检索和推理来回答复杂多跳查询，在推理密集型 benchmark 上表现优异，但大量 test-time compute 带来显著的效率问题。

**现有痛点**: 现有评估多依赖不透明的 Web 搜索 API，混淆了检索质量与外部服务行为；重排序在深度搜索流水线中的角色和成本效益尚未被系统研究。

**核心矛盾**: 增加推理预算（更多思考 token）可以提高准确率但成本急剧上升；检索阶段的质量提升（通过重排序）可能是更高效的替代方案，但缺乏统一的度量来比较这两种投资。

**本文目标**: 量化深度搜索智能体中重排序与推理预算的效率-效果权衡，找到最优的计算分配策略。

**切入角度**: 在 BrowseComp-Plus 基准（固定人工验证语料）上控制变量实验，引入 ETC 指标统一衡量不同配置的成本。

**核心 idea**: 将计算预算从搜索推理转向检索重排序，可以在更低成本下获得更好的端到端效果。

## 方法详解

### 整体框架
本文不训练新模型，而是搭一套受控实验流水线去回答一个工程问题：同样的算力，花在"让 agent 多思考"还是"先把检索做好"更划算。流水线为：搜索智能体（gpt-oss-20b/120b）迭代生成查询，qwen3-embedding-8b 检索 top-$d$ 候选，再经 listwise 重排序（oss-20b/120b 或轻量 qwen3-reranker-0.6b）后把 top-5 喂回智能体，循环直到找到答案。所有配置都在固定语料的 BrowseComp-Plus 上跑，并统一用 ETC 指标折算真实成本后再比较，从而把"重排序投资"和"推理预算投资"放到同一把尺子上。

### 关键设计
**1. 有效 token 成本（ETC）：用一把统一的尺子折算不同 token 的真实代价**

简单数 token 数无法反映真实开销——非缓存输入、缓存输入、含推理链的输出在硬件吞吐和 API 定价上差异巨大，输出 token 的解码成本尤其高。ETC 把三者加权求和：$\text{ETC} = \text{Input}_{nc} + \alpha \cdot \text{Input}_c + \beta \cdot \text{Output}_t$，其中 $\alpha$（0.1–0.5）建模缓存复用带来的折扣，$\beta$（3–7）建模输出解码的高昂代价。有了 ETC，重排序消耗的 token 和推理消耗的 token 才能被公平地放在一起比较，这是全文所有结论的计价基础。

**2. 多维度重排序分析框架：控制变量，分离每个旋钮的边际收益**

深度搜索同时有多个可调旋钮，混在一起看不出谁在起作用。本文固定检索器（qwen3-embedding-8b），系统地变化重排序深度 $d \in \{10, 20, 50\}$、重排序模型规模（oss-20b/120b）和推理级别（low/med/high），在 830 个查询上做全面消融。这种网格式控制变量让"加深重排序"与"加大推理预算"各自的边际增益被清晰量化，从而支撑"先排后推更省"的核心论断。

**3. 异构设置与广义 ETC：把轻量交叉编码器纳入同一成本框架**

listwise 重排序本身也烧 token，于是论文进一步问：能否用更便宜的交叉编码器替代？它用 qwen3-reranker-0.6b 作 cross-encoder（只输出 yes/no logits），并把 ETC 扩展为 $\text{ETC} = \text{ETC}_{agent} + \gamma \cdot \text{ETC}_{reranker}$，其中 $\gamma = 0.32$ 反映两者的 FLOPs 差异。轻量重排序器峰值性能略低，但 $\gamma \ll 1$ 让它在效率优先场景下极具吸引力，也证明了 ETC 框架能跨模型规模、跨重排序范式做成本比较。

实验侧的固定设置：重排序基于 RankLLM 框架（window size 20），端到端准确率用 LLM-as-a-judge（oss-120b）平均 5 次判分。

## 实验关键数据

### One-shot 重排序效果（NDCG@5）

| 配置 | d=0 | d=10 | d=20 | d=50 |
|------|-----|------|------|------|
| 无重排序 | 19.72 | — | — | — |
| oss-20b-low | — | 27.30 | 32.28 | 35.89 |
| oss-20b-med | — | 28.34 | 34.37 | 39.86 |
| oss-120b-low | — | 29.64 | 35.69 | 44.10 |
| oss-120b-med | — | 29.78 | 36.63 | **46.05** |

### 端到端深度搜索准确率（Accuracy %）

| 搜索智能体 | d=0 | d=10 | d=20 | d=50 |
|------------|-----|------|------|------|
| oss-20b-low | 17.33 | 19.64 | 22.96 | 25.73 |
| oss-20b-med | 35.59 | 39.04 | 43.39 | 49.11 |
| oss-20b-high | 41.38 | 48.31 | 52.70 | 55.97 |
| oss-120b-low | 28.17 | 31.76 | 36.92 | 42.65 |
| oss-120b-med | 40.43 | 48.65 | 53.96 | 57.97 |
| oss-120b-high | 51.20 | 54.65 | 58.99 | **63.35** |

### 关键发现
- 重排序深度的边际收益最大：d=10→20 提升 5-7 NDCG@5 点，d=20→50 再提升 3.5-10 点
- oss-120b-med + d=20（准确率 53.96%）即可匹配 oss-120b-high + d=0（51.20%），且 ETC 显著更低
- 搜索调用次数在启用重排序后仅降低 ≤12%，说明增益主要来自检索质量提升而非行为变化
- 异构设置（qwen3-reranker-0.6b）保留了大部分端到端增益（52.91% vs 55.97% at oss-20b-high, d=50），但 ETC 大幅降低（$\gamma=0.32$）
- 高推理设置下重排序反而降低延迟（150.7s vs 184.7s at d=50 vs d=0），因搜索调用减少

## 亮点与洞察
- ETC 指标设计巧妙，可适配 API 定价和自部署两种场景，为 RAG 系统的成本优化提供了统一框架
- "先排后推"的核心洞察简单有效：与其让 agent 多思考，不如先给它更好的文档
- 异构 ETC 的扩展使得跨模型规模的成本比较成为可能
- 延迟分析揭示了非直觉结果：重排序在高推理场景下反而加速

## 局限与展望
- 仅使用 gpt-oss 系列模型，未验证商业模型或开源推理模型的泛化性
- BrowseComp-Plus 使用固定语料而非真实 Web 搜索，结论向生产环境迁移需谨慎
- 重排序采用固定 top-d 选择，自适应文档子集选择可能进一步提升效率
- 保留完整交互历史（自动截断至 128K token），未探索显式压缩或摘要策略

## 相关工作与启发
- **BrowseComp-Plus**: 受控的深度搜索评估基准，固定检索阶段以隔离变量
- **RankLLM**: listwise 重排序框架，本文的实验基础设施
- **AgentDiet**: 减少智能体轨迹中的冗余上下文，与本文的检索端优化互补
- 启发：在 RAG/Agent 系统中，计算预算分配问题值得更多关注——检索质量的提升往往比推理量的增加更具性价比

## 评分
- 新颖性: ⭐⭐⭐⭐ ETC 指标新颖实用，"先排后推"的实证洞察有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 830 查询、多配置消融、异构设置、延迟分析，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，实验设计描述详尽
- 价值: ⭐⭐⭐⭐⭐ 对深度搜索系统的工程实践有直接指导意义

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] RRRA: Resampling and Reranking through a Retriever Adapter](../../AAAI2026/information_retrieval/rrra_resampling_and_reranking_through_a_retriever_adapter.md)
- [\[ACL 2026\] Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](enhancing_llm-based_search_agents_via_contribution_weighted_group_relative_polic.md)
- [\[ICLR 2026\] Hybrid Deep Searcher: Scalable Parallel and Sequential Search Reasoning](../../ICLR2026/information_retrieval/hybrid_deep_searcher_scalable_parallel_and_sequential_search_reasoning.md)
- [\[ICML 2026\] ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards](../../ICML2026/information_retrieval/reseek_a_self-correcting_framework_for_search_agents_with_instructive_rewards.md)
- [\[ACL 2026\] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](can_compact_language_models_search_like_agents_distillation-guided_policy_optimi.md)

</div>

<!-- RELATED:END -->
