---
title: >-
  [论文解读] PaSa: An LLM Agent for Comprehensive Academic Paper Search
description: >-
  [ACL2025][LLM Agent][学术论文搜索] PaSa 是一个基于 LLM 的学术论文搜索智能体，通过自主调用搜索工具、阅读论文和导航引用网络来实现全面准确的学术文献检索，经 RL 训练后在真实场景中大幅超越 Google Scholar 和 GPT-4o。 学术论文搜索是科研的核心任务，但也是特别具有挑战性的信…
tags:
  - "ACL2025"
  - "LLM Agent"
  - "学术论文搜索"
  - "强化学习"
  - "引用网络"
  - "论文检索"
---

# PaSa: An LLM Agent for Comprehensive Academic Paper Search

**会议**: ACL2025  
**arXiv**: [2501.10120](https://arxiv.org/abs/2501.10120)  
**代码**: [bytedance/pasa](https://github.com/bytedance/pasa)  
**领域**: LLM Agent  
**关键词**: 学术论文搜索, LLM Agent, 强化学习, 引用网络, 论文检索  

## 一句话总结

PaSa 是一个基于 LLM 的学术论文搜索智能体，通过自主调用搜索工具、阅读论文和导航引用网络来实现全面准确的学术文献检索，经 RL 训练后在真实场景中大幅超越 Google Scholar 和 GPT-4o。

## 研究背景与动机

学术论文搜索是科研的核心任务，但也是特别具有挑战性的信息检索问题。它要求长尾专业知识、综述级别的全面覆盖以及处理细粒度查询的能力。例如查询 "哪些研究使用基于 UCB 的算法聚焦于非平稳强化学习？"——Google Scholar 等通用搜索工具对此类复杂查询往往力不从心。

研究者通常不仅使用搜索工具，还会阅读相关论文、检查引用关系来执行全面的文献调研。这一过程耗时巨大。尽管 LLM 在信息检索增强方面已有探索（如查询改写），但将 LLM 作为自主 Agent 来模拟人类文献调研的完整行为——搜索、阅读、引用追踪——仍是空白。

## 方法详解

### 整体框架

PaSa 系统由两个 LLM Agent 组成（基于 Qwen2.5-7B）：

- **Crawler（爬取器）**：负责自主收集论文，最大化召回率
- **Selector（选择器）**：负责精确判断论文是否满足查询需求，强调精确率

### Crawler 的设计

Crawler 执行 token 级别的马尔可夫决策过程（MDP），动作空间对应 LLM 词表。注册了三个函数：

| 函数 | 功能 |
|------|------|
| [Search] | 生成搜索查询，调用搜索工具，将结果加入论文队列 |
| [Expand] | 生成子章节名称，将该章节引用的所有论文加入队列 |
| [Stop] | 重置上下文为用户查询+队列中下一篇论文 |

工作流程：Crawler 接收用户查询后，可以反复执行 Search（用不同查询多次搜索）、读取论文后执行 Expand（追踪引用网络发现更多相关论文）、或 Stop（切换到下一篇论文）。探索深度限制为 3 层。

### Crawler 的训练

**两阶段训练**：

**阶段 1：模仿学习**
- 为 5,000 个查询生成示范轨迹，进行监督微调
- 学习率 1e-5，batch size 4，1个 epoch

**阶段 2：强化学习（Session-Level PPO）**

面临的挑战：
- **稀疏奖励**：AutoScholarQuery 中的论文集仅是实际合格论文的子集
- **长轨迹**：完整轨迹可能涉及数百篇论文，超出 LLM 上下文长度

解决方案——**Session-Level PPO**：

将轨迹划分为一系列 session，每个 session 以 [Stop] 结束。定义两类初始状态：$S_q$（仅含查询）和 $S_{q+p}$（含查询和论文）。

奖励设计：
$$r(s_t, a_t) = \alpha \times \sum_{i=1}^{n_t} \mathbb{I}(q, p_i, t) - c(a_t)$$

其中 $\mathbb{I}$ 判断论文是否匹配查询。为缓解稀疏奖励，使用 Selector 作为辅助奖励模型——当 Selector 判定论文满足查询或论文在标注集中时，给予正奖励。

回报估计结合 session 内折扣因子 $\gamma_0$ 和跨 session 折扣因子 $\gamma_1$，并加入 per-token KL 惩罚防止过度优化。

### Selector 的设计

接收查询和论文（标题+摘要），输出：
1. 决策 token（True/False）
2. 推理理由

关键设计：决策 token 前置于理由，使 Selector 在 Crawler 训练时可作为单 token 奖励模型。优化方式为模仿学习。

### 数据集构建

#### AutoScholarQuery
- 从 ICLR 2023、ICML 2023、NeurIPS 2023、ACL 2024、CVPR 2024 论文的 Related Work 章节构建
- 用 GPT-4o 从引用关系中生成细粒度学术查询
- 含 33,511 / 1,000 / 1,000 训练/验证/测试实例
- 人工评估：94% 查询合格，93.7% 论文匹配

#### RealScholarQuery
- 50 个真实世界学术查询
- AI 研究者在 PaSa demo 上提交的真实查询
- 专业标注者（顶尖大学教授）审核所有候选论文
- 每个查询平均审核 76 篇候选论文，标注成本 $304/查询
- 平均每个查询关联 15.82 篇答案论文

## 实验

### 基线方法
- Google / Google Scholar（直接搜索）
- Google with GPT-4o（GPT-4o 改写查询后搜索）
- ChatGPT（搜索增强 GPT-4o）
- GPT-o1（无外部搜索）
- PaSa-GPT-4o（用 GPT-4o 实现 PaSa Agent）

### AutoScholarQuery 主实验

| 方法 | Precision | Recall | Recall@20 | Recall@50 | Recall@100 |
|------|-----------|--------|-----------|-----------|------------|
| Google | - | - | 0.1568 | 0.1891 | 0.2015 |
| Google + GPT-4o | - | - | 0.1921 | 0.2450 | 0.2683 |
| ChatGPT | 0.0507 | 0.3046 | - | - | - |
| PaSa-GPT-4o | 0.1457 | 0.3873 | - | - | - |
| **PaSa-7B** | **0.1448** | **0.4834** | **0.5301** | **0.6334** | **0.6947** |

PaSa-7B 相比 Google + GPT-4o 在 Recall@20 上提升 **33.80%**，Recall@50 提升 **38.83%**。

### RealScholarQuery 实验

| 方法 | Precision | Recall | Recall@20 | Recall@50 |
|------|-----------|--------|-----------|-----------|
| Google + GPT-4o | - | - | 0.2020 | 0.2573 |
| PaSa-GPT-4o | 0.4721 | 0.3075 | - | - |
| **PaSa-7B** | **0.5146** | **0.6111** | **0.5798** | **0.6563** |

PaSa-7B 在真实场景中优势更大：比 PaSa-GPT-4o 在 recall 上提升 **30.36%**，precision 提升 **4.25%**。

### Selector 评估

| 方法 | Precision | Recall | F1 |
|------|-----------|--------|----|
| GPT-4o | 0.96 | 0.69 | 0.80 |
| Qwen2.5-7B | 1.00 | 0.38 | 0.55 |
| **PaSa Selector** | **0.95** | **0.78** | **0.85** |

Selector 的 F1 达到 85%，超越 GPT-4o 的 80%。

### 消融实验

| 设置 | Crawler Recall (Auto) | Recall (Auto) | Crawler Recall (Real) | Recall (Real) |
|------|----------------------|---------------|----------------------|---------------|
| w/o [Expand] | 0.3355 | 0.2536 | 0.3359 | 0.2890 |
| w/o RL 训练 | 0.6556 | 0.4210 | 0.4847 | 0.4115 |
| w/o Selector as RM | 0.7041 | 0.4458 | 0.5994 | 0.5148 |
| **PaSa-7B** | **0.7931** | **0.4834** | **0.7071** | **0.6111** |

- 移除 [Expand]（引用网络导航）后 Recall 降幅最大（约 50%+），证明引用网络追踪是核心能力
- RL 训练带来约 6-20% 的提升
- Selector 作为辅助奖励模型也贡献显著

### 关键发现

1. **7B 模型击败 GPT-4o Agent**：PaSa-7B 经过 RL 训练后，性能显著超越用 GPT-4o prompt 实现的 PaSa-GPT-4o
2. **引用网络导航至关重要**：Crawler 在深入引用网络时发现大量相关论文，即使中间论文与查询不直接相关
3. **合成数据上训练，真实场景泛化**：仅在 AutoScholarQuery 上训练，在 RealScholarQuery 上泛化效果更强
4. **Ensemble 进一步提升**：采样解码运行两次 Crawler 可额外提升 3-4% Crawler Recall

## 亮点与洞察

1. **模仿人类文献调研的完整流程**：不仅搜索，还阅读论文和追踪引用，这种设计远超简单的查询改写范式
2. **Session-Level PPO 的创新**：优雅解决了长轨迹和稀疏奖励问题，使 RL 在 Agent 长轨迹任务中可行
3. **Selector 双重角色**：既是最终过滤器又是 RL 的辅助奖励模型，单一组件发挥双重作用
4. **高质量数据集**：AutoScholarQuery 从顶会论文的 Related Work 构建，数据质量极高；RealScholarQuery 标注成本极高（$304/查询），代表性强
5. **实用价值突出**：在线 demo（pasa-agent.ai）已上线，论文搜索是研究者的刚需

## 局限性

1. 搜索工具限于 Google + arXiv，未覆盖其他学术数据库
2. Crawler 探索深度限制为 3 层，可能遗漏更深层引用网络中的论文
3. AutoScholarQuery 仅限 AI 领域顶会，对其他学科的泛化性未验证
4. RealScholarQuery 仅 50 个查询，规模较小
5. 论文获取依赖 ar5iv，部分论文可能无法获取完整内容

## 相关工作

- **LLM 在科学发现中的应用**：思路生成、实验设计、论文写作等，但文献调研自动化仍不足
- **LLM Agent**：AGILE、ReAct 等框架用于工具使用和规划，本文采用 AGILE 框架
- **学术搜索增强**：查询改写和检索增强方法在一般 IR 中有效，但缺乏面向学术搜索的 Agent 系统

## 评分

⭐⭐⭐⭐⭐ — 系统设计优雅、实验全面、实际应用价值极高。Session-Level PPO 的创新解决了 Agent RL 训练的实际难题，7B 模型击败 GPT-4o Agent 的结果令人印象深刻。数据集构建质量高，论文搜索作为科研刚需的场景选择极具影响力。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Paper2Figure: A Multi-Agent Collaborative System for Figure Generation Towards Academic Research Paper](../../CVPR2026/llm_agent/paper2figure_a_multi-agent_collaborative_system_for_figure_generation_towards_ac.md)
- [\[ICML 2025\] Improving LLM Agent Planning with In-Context Learning via Atomic Fact Augmentation and Lookahead Search](../../ICML2025/llm_agent/improving_llm_agent_planning_with_in-context_learning_via_atomic_fact_augmentati.md)
- [\[ACL 2025\] LegalAgentBench: Evaluating LLM Agents in Legal Domain](legalagentbench_evaluating_llm_agents_in_legal_domain.md)
- [\[ACL 2025\] LLM Agents Making Agent Tools](llm_agents_making_agent_tools.md)
- [\[ACL 2025\] Enhancing LLM Agent Safety via Causal Influence Prompting](enhancing_llm_agent_safety_via_causal_influence_prompting.md)

</div>

<!-- RELATED:END -->
