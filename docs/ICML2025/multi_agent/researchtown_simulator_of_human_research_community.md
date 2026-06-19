---
title: >-
  [论文解读] ResearchTown: Simulator of Human Research Community
description: >-
  [ICML 2025][多智能体][多智能体模拟] 提出 ResearchTown，一个基于 agent-data 图和 TextGNN（文本空间消息传递）的多智能体框架，将人类科研社区建模为异构图，统一模拟论文阅读、论文写作和审稿三大核心研究活动，并通过节点掩码预测任务 (ResearchBench) 进行可扩展、客观的仿真质量评估。
tags:
  - "ICML 2025"
  - "多智能体"
  - "多智能体模拟"
  - "图神经网络"
  - "科研社区仿真"
  - "文本消息传递"
  - "自动化科研"
---

# ResearchTown: Simulator of Human Research Community

**会议**: ICML 2025  
**arXiv**: [2412.17767](https://arxiv.org/abs/2412.17767)  
**代码**: [ulab-uiuc/research-town](https://github.com/ulab-uiuc/research-town)  
**领域**: LLM评测  
**关键词**: 多智能体模拟, 图神经网络, 科研社区仿真, 文本消息传递, 自动化科研

## 一句话总结

提出 ResearchTown，一个基于 agent-data 图和 TextGNN（文本空间消息传递）的多智能体框架，将人类科研社区建模为异构图，统一模拟论文阅读、论文写作和审稿三大核心研究活动，并通过节点掩码预测任务 (ResearchBench) 进行可扩展、客观的仿真质量评估。

## 研究背景与动机

**核心问题**：能否用 LLM 模拟人类科研社区？回答这一问题有两大意义：(1) 理解既有研究 idea 背后的发现过程；(2) 民主化与加速新研究 idea 的发现。

**现有方法的不足**：

- 已有多智能体框架（社会模拟、游戏模拟等）无法处理科研社区中复杂的协作活动（如多人合作写论文、同行评审）
- 现有科研自动化工作局限于单一任务（如 idea 生成、代码实验），或仅关注单智能体工作流
- 无法模拟多背景研究者之间的协作——而这正是现代科研的基本形态

**关键观察**：深度互联的科研社区可以自然地表示为图结构（引用网络、学术社交网络等），而引入 LLM 可以将传统的预测/分析扩展为动态模拟和实时预测。

## 方法详解

### 整体框架

ResearchTown 的核心架构包含三个层次：

1. **Agent-Data 图**：一种新型异构图，包含两类节点——agent 节点（研究者）和 data 节点（论文），以及三类边——agent-agent (ℰ_aa)、agent-data (ℰ_ad)、data-data (ℰ_dd)
2. **TextGNN**：在 agent-data 图上进行文本空间消息传递的推理框架
3. **ResearchBench**：基于节点掩码预测任务的评估基准

整个仿真流程是一个 2 层 GNN：第一层为论文阅读（信息聚合），第二层为论文写作 + 审稿（生成最终输出）。

### 关键设计

#### 1. Agent-Data 图的定义

Agent-data 图 𝒢 = (𝒱, ℰ) 的独特之处在于：

- **Data 节点**：携带文本属性 $\mathbf{x}_v$（如论文全文）
- **Agent 节点**：携带 agent 函数 $f_u(\cdot)$（即以特定 prompt 配置的 LLM），而非嵌入向量
- Agent 节点本质上是 data 节点上的函数：$\mathbf{x}_{uv} = f_u([\mathbf{x}_u, \mathbf{x}_v])$

在科研社区图 (Community Graph) 中，具体化为：研究者为 agent 节点，论文为 data 节点；边包括引用关系 (ℰ_dd)、作者关系和审稿关系 (ℰ_ad)；省略 ℰ_aa（可通过 2-hop 路径推断）。

#### 2. TextGNN 消息传递机制

TextGNN 与标准 GNN 的关键区别：所有隐状态定义在**文本空间** $\Sigma^*$ 而非**嵌入空间** $\mathbb{R}^d$。

对 agent 节点 $u$ 的第 $k$ 层更新：

$$\mathbf{h}_u^{(k)} = f_u\Big([\mathbf{h}_u^{(k-1)}, \{f_a([\mathbf{h}_a^{(k-1)}, \mathbf{h}_u^{(k-1)}, \mathbf{h}_d^{(k-1)}]) \mid (u,a) \in \mathcal{E}_{aa}, (u,d) \in \mathcal{E}_{ad}\}]\Big)$$

对 data 节点 $v$ 的第 $k$ 层更新使用全局 agent 函数 $f_g(\cdot)$（无特定 profile）聚合来自邻居 agent 和 data 节点的消息。

#### 3. 三阶段仿真流程

**Stage 1 — 论文阅读**：新增 agent 节点，聚合邻域论文信息生成研究者 profile

$$\mathbf{h}_u = f_u\Big([\{\mathbf{h}_d \mid (u,d) \in \mathcal{E}_{ad}\}]\Big)$$

这是一种特殊的消息传递：agent 节点初始为空，读完论文后生成 profile。

**Stage 2 — 论文写作**：新增 data 节点，由多个 agent 协作生成论文内容

$$\mathbf{h}_v = f_g\Big([\{f_a([\mathbf{h}_a, \mathbf{h}_d]) \mid (v,a) \in \mathcal{E}_{ad}, (v,d) \in \mathcal{E}_{dd}\}]\Big)$$

每个作者 agent 基于自身 profile 和引用论文生成消息，再由全局函数聚合。

**Stage 3 — 审稿**：reviewer agent 基于论文内容、自身专业背景和相关文献生成评审意见

$$\mathbf{r}_v = f_g\Big([\mathbf{h}_v, \{f_a([\mathbf{h}_a, \mathbf{h}_v, \mathbf{h}_d]) \mid (v,a) \in \mathcal{E}_{ad}, (v,d) \in \mathcal{E}_{dd}\}]\Big)$$

与前两阶段不同：审稿者并非作者，且论文节点此时已有内容。

#### 4. ResearchBench 评估框架

评估核心思想——**节点掩码预测**：

- 在社区图中遮蔽某个论文节点的内容 $\mathbf{h}_v^*$
- 用 ResearchTown 从邻域信息重建该节点
- 用 text-embedding-3-large 计算重建结果与真实结果的余弦相似度

基准包含 1,000 个论文写作任务和 200 个审稿任务，来源于 NeurIPS 2024 和 ICLR 2024。

### 损失函数 / 训练策略

本文并非训练模型，而是利用现有 LLM（GPT-4o-mini）作为 agent 函数的骨干，温度设为 0 保证可复现性。评估通过嵌入相似度度量而非梯度优化。

四种聚合策略用于消融分析：

- **AGG-self**：仅目标节点自身
- **AGG-agent**：目标节点 + 邻居 agent 节点
- **AGG-data**：目标节点 + 邻居 data 节点
- **AGG-global**（即 ResearchTown）：目标节点 + 所有邻居

## 实验关键数据

### 主实验

**论文写作仿真**（text-embedding-3-large 相似度 ×100）：

| 聚合策略 | Easy | Medium | Hard | Overall |
|---------|------|--------|------|---------|
| AGG-self | 46.42 | 45.92 | 45.90 | 46.08 |
| AGG-agent | 56.90 | 55.55 | 53.26 | 55.24 |
| AGG-data | 74.36 | 66.42 | 56.02 | 65.30 |
| AGG-global (ResearchTown) | **73.79** | **67.85** | **60.89** | **67.51** |

**审稿仿真**（Strength/Weakness 为嵌入相似度，ΔS 为评分差异）：

| 聚合策略 | Strength ↑ | Weakness ↑ | ΔS ↓ | 平均分 |
|---------|-----------|-----------|------|--------|
| AGG-self | 51.23 | 47.16 | 1.27 | 5.33 |
| AGG-agent | 51.66 | 46.75 | 1.19 | 5.40 |
| AGG-data | 51.45 | 47.62 | 1.26 | 5.30 |
| AGG-global | 51.51 | 47.17 | 1.55 | 5.00 |

### 消融实验

**不同 LLM 骨干对比**：

| 聚合策略 | 论文写作 (Qwen/GPT/DS) | 审稿 ΔS (Qwen/GPT/DS) |
|---------|----------------------|---------------------|
| AGG-self | 46.45 / 46.08 / 48.62 | 1.36 / 1.27 / 1.11 |
| AGG-agent | 53.91 / 55.24 / 56.19 | 1.41 / 1.19 / 1.05 |
| AGG-data | 65.03 / 65.30 / 65.05 | 1.28 / 1.26 / 1.07 |
| AGG-global | **65.30** / **67.51** / **65.33** | **0.79** / 1.51 / **0.81** |

**新颖性与可行性评估**（0-10 分）：

| 评估方式 | 仿真-新颖性 | 仿真-可行性 | 真实-新颖性 | 真实-可行性 |
|---------|-----------|-----------|-----------|-----------|
| LLM 评估 | 7.39 | 6.82 | 7.85 | 7.13 |
| 人工评估 | 5.50 | 7.98 | 5.90 | 7.85 |

### 关键发现

1. **引用论文比作者 profile 更重要**：AGG-data (65.30) 显著优于 AGG-agent (55.24)，说明参考文献是论文写作的核心信息源
2. **多研究者协作提升困难任务表现**：Hard 集上 AGG-global (60.89) 大幅超过 AGG-data (56.02)，提升 4.87 分——研究者提供了多跳论文信息
3. **审稿模拟难度高于论文写作**：审稿相似度（~47-51）远低于论文写作（~67），因真实审稿数据更嘈杂多样
4. **Agent 数增加持续提升质量**：1→2 个 agent 提升最显著（49.0→52.7），边际递减
5. **DeepSeek-v3 表现最优**，GPT-4o-mini 次之，Qwen-2.5-7B 最弱，与通用能力排名一致

## 亮点与洞察

- **图+LLM 的优雅结合**：将 GNN 的消息传递从嵌入空间迁移到文本空间，为多智能体系统提供了统一的理论框架（TextGNN），思路清晰且可扩展
- **评估方法创新**：节点掩码预测避免了传统人工评估的主观性和高成本，提供了可扩展的客观评估方式
- **跨学科 idea 生成**：ResearchTown 能组合 NLP+天文学、NLP+犯罪学等罕见领域组合，生成现实中不存在的研究方向
- **隐状态设计巧妙**：将论文压缩为 bullet-point 紧凑形式作为隐状态，既保留关键信息又控制上下文长度

## 局限与展望

1. **审稿模拟质量有限**：Weakness 识别相似度仅 ~47，实际审稿涉及更深层推理和领域专精
2. **跨学科组合过多时失败**：当组合 4+ 个不相关领域时，输出变得不连贯、流于表面（简单堆砌术语）
3. **仅使用摘要级信息**：论文阅读和写作阶段仅使用摘要，而审稿阶段才使用全文，丢失较多细节
4. **缺乏迭代反馈机制**：真实科研中作者会根据审稿意见修改论文，当前框架是单向流水线
5. **单模态限制**：仅处理文本，未涉及代码、图表、数据等多模态研究材料
6. **评估方法本身的局限**：嵌入相似度不等于研究质量；人工与 LLM 评估在内在质量维度上相关性低

## 相关工作与启发

- **与 AI Scientist (Lu et al., 2024) 的区别**：AI Scientist 关注单智能体端到端科研流程，ResearchTown 关注多智能体社区协作模拟
- **与 ResearchAgent (Baek et al., 2024) 的区别**：ResearchAgent 聚焦 idea 迭代生成，ResearchTown 建模完整的社区研究活动
- **TextGNN 的启发**：将 GNN 消息传递推广到文本空间的思路可应用于其他需要结构化协作的场景（如企业知识管理、教育协作等）
- **对科研工具的启示**：未来论文推荐/审稿匹配系统可借鉴 agent-data 图的建模方式

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|----------|------|
| 创新性 | ⭐⭐⭐⭐ | TextGNN + agent-data 图的统一框架设计新颖 |
| 技术深度 | ⭐⭐⭐⭐ | 形式化定义完整，从 GNN 自然推导出研究活动建模 |
| 实验充分度 | ⭐⭐⭐⭐⭐ | 消融全面，多模型/多嵌入/多维度评估 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，图表丰富 |
| 实用价值 | ⭐⭐⭐ | 仿真质量尚有差距，但方向有前景 |
| 总评 | ⭐⭐⭐⭐ | 高水平系统工作，开辟了科研社区仿真新范式 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] PaperMentor: A Human-Centered Multi-Agent Writing Tutor for AI Research Papers on Overleaf](../../ACL2026/multi_agent/papermentor_a_human-centered_multi-agent_writing_tutor_for_ai_research_papers_on.md)
- [\[NeurIPS 2025\] MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems](../../NeurIPS2025/multi_agent/metamind_modeling_human_social_thoughts_with_metacognitive_multi-agent_systems.md)
- [\[ACL 2026\] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey](../../ACL2026/multi_agent/llm-based_human-agent_collaboration_and_interaction_systems_a_survey.md)
- [\[ACL 2026\] RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems](../../ACL2026/multi_agent/roadmapper_a_multi-agent_system_for_roadmap_generation_of_solving_complex_resear.md)
- [\[ICLR 2026\] UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking](../../ICLR2026/multi_agent/uis-digger_towards_comprehensive_research_agent_systems_for_real-world_unindexed.md)

</div>

<!-- RELATED:END -->
