---
title: >-
  [论文解读] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey
description: >-
  [ACL 2026 Findings][多智能体][human-in-the-loop] 本文首次系统性梳理"LLM 基础的人-agent 协作系统（LLM-HAS）"——把人重新拉回 agent loop，从环境/画像、人类反馈、交互类型、编排范式、通信结构 5 个维度建立统一分类，并补充了一个 A1–A5 的 Human Agency Scale 量化"任务里到底该让人参与多深"。
tags:
  - "ACL 2026 Findings"
  - "多智能体"
  - "human-in-the-loop"
  - "agent orchestration"
  - "human feedback"
  - "human agency scale"
  - "LLM-HAS"
---

# LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey

**会议**: ACL 2026 Findings  
**arXiv**: [2505.00753](https://arxiv.org/abs/2505.00753)  
**代码**: https://github.com/HenryPengZou/Awesome-Human-Agent-Collaboration-Interaction-Systems  
**领域**: 人机协作 / LLM Agent / 综述  
**关键词**: human-in-the-loop、agent orchestration、human feedback、human agency scale、LLM-HAS

## 一句话总结
本文首次系统性梳理"LLM 基础的人-agent 协作系统（LLM-HAS）"——把人重新拉回 agent loop，从环境/画像、人类反馈、交互类型、编排范式、通信结构 5 个维度建立统一分类，并补充了一个 A1–A5 的 Human Agency Scale 量化"任务里到底该让人参与多深"。

## 研究背景与动机
**领域现状**：LLM agent 调研近年多在卷"完全自治"：单 agent (AutoGPT)、多 agent (MetaGPT)、长程任务执行（SWE-Agent）等都把"减少人介入"当目标。

**现有痛点**：完全自治路线撞了三堵墙——(1) 可靠性：幻觉在多步链式 action 里被放大；(2) 复杂性：科学、医疗、长上下文连贯性等任务超出 LLM 单独可及范围；(3) 安全/伦理：金融、医疗、安全场景下不可逆 action 风险陡增。已有的 LLM agent / multi-agent / specific-app 综述都不专门讨论"人怎么有效介入"。

**核心矛盾**：当前社区把"自治程度"当成单一进度条往满拉，但很多真实任务的最优点在 augmentation 而非 automation；缺乏一个统一框架来描述"人在什么时间、以什么方式、什么粒度、和 agent 怎么交互"。

**本文目标**：(a) 定义 LLM-HAS 并区分于 single agent / multi-agent；(b) 把现有工作沿 5 个维度归类；(c) 系统化人类反馈的类型/粒度/时机；(d) 给出一个量化"自治 vs 增强"程度的 Human Agency Scale；(e) 总结 prompting / SFT / RL 三类实现路线及代表 benchmark；(f) 提出 5 大开放挑战。

**切入角度**：把"人"显式建模为 LLM-HAS 的 first-class 组件（Lazy User vs Informative User），并借鉴 multi-agent system 的通信/编排概念扩展到人-agent 场景。

**核心 idea**：一个 LLM-HAS = Environment & Profiling + Human Feedback + Interaction Type + Orchestration + Communication，配合 Human Agency Scale 标定参与深度。

## 方法详解

### 整体框架
作者把 LLM-HAS 拆成 5 个正交核心维度 + 1 个跨维度量表：

- **Environment & Profiling**：物理世界 vs 虚拟仿真；single/multi-human × single/multi-agent 共 4 种拓扑；人画像分 Lazy / Informative，agent 画像按角色（通用助手、数学专家、机器人等）。
- **Human Feedback**：类型（Evaluative / Corrective / Guidance / Implicit）× 粒度（Coarse / Fine）× 时机（Initial / During / Post）。
- **Interaction Type**：Collaboration（最常见，分 Delegation / Supervision / Cooperation / Coordination 4 子类）、Competition、Coopetition。
- **Orchestration**：Task Strategy（One-by-One vs Simultaneous）× Temporal Synchronization（Synchronous vs Asynchronous）。
- **Communication**：Structure（Centralized / Decentralized / Hierarchical）× Mode（Conversation / Observation / Shared Message Pool）。
- **Human Agency Scale (A1–A5)**：A1 Full Automation → A2 Minimal Human Input → A3 Equal Partnership → A4 Agent-Assisted → A5 Human-Driven，A1–A2 属 Automation，A3–A5 属 Augmentation。

### 关键设计

**1. Human Feedback 三维分类（Type × Granularity × Phase）：把"人怎么给反馈"从单一打分拆成可定位的坐标系**

过去谈人类反馈往往就一个"打分"，可实际系统里反馈的形态千差万别，没法横向比较。本文把它拆成三个正交轴：类型分 Evaluative（像 RLHF 的偏好打分）、Corrective（像 PRELUDE 学用户编辑）、Guidance（像 InteractGen 用 demo 引导）、Implicit（像 VeriPlan 观察用户滑块行为）四类；粒度分 holistic 与 segment-level 两档；时机分 Initial / During / Post 三段。三轴交叉出 24 格分析坐标，任意一篇工作的反馈机制都能被精确编码成一个三元组（如 (Corrective, Fine, During)）。

这套拆法把"反馈复杂度"变成了可比较的设计选择：粗粒度评估好收集但 credit assignment 弱，细粒度反馈精确却加重用户负担，时机则决定了系统能实时纠错还是只能 offline 学习。设计者据此能在"信号质量 vs 用户成本"之间做明确权衡，而不是笼统说一句"我们用了人类反馈"。

**2. Human Agency Scale（A1–A5）：用 5 档量化"任务里人该参与多深"**

社区习惯把"自治程度"当成一根往满拉的进度条，可很多真实任务的最优点其实在 augmentation 而非 automation——这件事一直停留在口水仗，缺一把尺。本文给出 A1–A5 五档量表：A1 全自动、A2 关键点 spot-check、A3 平等协作（双方都比单干强）、A4 agent 需大量人输入、A5 人主导 agent 仅辅助；其中 A1–A2 归为 Automation，A3–A5 归为 Augmentation。

这把尺最直接的用处是给 benchmark 设计者一个参照。现有 benchmark 几乎只评 A1 场景（agent 能多接近全自动），却忽略了医疗诊断、法律咨询这类天然落在 A3–A5 的任务；有了量表，"该不该让 agent 全干"就从立场之争变成了可分类、可评测的研究问题。

**3. Interaction Type 四子类 Collaboration（Delegation / Supervision / Cooperation / Coordination）：拒绝把"协作"当一个原子词**

把"协作"当成一个笼统的词，会让综述退化成一堆协作工作的大杂烩——因为不同协作形态需要的反馈机制、通信模式、自治度完全不同。本文按"谁主导 + 是否动态"把 Collaboration 进一步切成四个子类：Delegation 是一上来给完整指令、agent 自治执行（如 FineArena 的投资偏好委托）；Supervision 是实时监督加随时介入（如 teleoperator 监控机器人）；Cooperation 是双方自愿联合达成同一目标（如 CoELA 体感 agent）；Coordination 是分工并同步、重在避免冲突（如共享工作空间任务）。

四分之后，每篇工作的交互形态都能对号入座，"这种协作该配什么反馈、什么通信结构"的讨论也有了落点。配合并列的 Competition、Coopetition，整个 Interaction Type 维度就覆盖了人-agent 之间从合作到竞争的连续谱。

### 损失函数 / 训练策略
本文是综述无训练。但系统对比了三大实现路线：
- **Prompting-based**（MToM、Collaborative Gym、Magentic-UI）：灵活、零训练成本，但 brittle、跨 session 不积累；
- **SFT-based**（PRELUDE、XtraGPT、Ask-before-Plan）：把交互轨迹转成持续行为改进，更稳但贵；
- **RL-based**（UserRL、SWEET-RL、ReHAC、MUA-RL）：长程多轮优化，但 reward 设计/样本效率/稳定性挑战大，近期多采用 prompting/SFT 引导 + RL 微调的混合管线。

## 实验关键数据

### 主实验
作者整理了不同领域代表性 datasets / benchmarks（节选自 Table 4）：

| 领域 | 代表 Benchmark | 代表工作 |
|------|----------------|----------|
| Embodied AI | PARTNR / MINT / IGLU Multi-Turn / TaPA | PARTNR (Chang 2024)、TaPA (Wu 2023) |
| Conversational | WEBLINX / Ask-before-Plan / HOTPOTQA / WildSeek | Co-STORM、ReHAC、WebLINX |
| Software Dev | ConvCodeWorld / ColBench / RECODE-H / MINT | SWEET-RL、ConvCodeWorld、RECODE-H |
| Gaming | CuisineWorld / MineWorld | MindAgent、MineWorld |
| Healthcare | EmoEval / GenoTEX | EmoAgent、GenoMAS |
| Retail / Travel | τ-Bench / τ2-Bench / UserBench | τ-Bench (Yao 2025)、UserBench (Qian 2025) |
| Finance | FinArena-Low-Cost | FineArena |
| Web / Computer Use | InterruptBench | InterruptBench (Zou 2026) |

3 个代表 LLM-HAS 框架的特征对比：

| 框架 | 交互类型 | 关键特性 |
|------|---------|----------|
| Collaborative Gym (Shao 2024) | Async + Collab | 同时评 outcome + 交互质量 |
| COWPILOT (Huq 2025) | Sync + Suggest-then-Execute | Chrome 插件，web 导航人监督 |
| DPT-Agent (Zhang 2025) | Real-time Sync | Dual Process Theory，快/慢双系统 |

### 消融（按 Human Feedback 维度的能力对比，作者总结自 Table 1）

| 反馈类型 | 收集难度 | 信号精度 | 代表工作 |
|----------|---------|----------|----------|
| Evaluative | 低（打分/preference） | 弱、缺 credit assignment | MINT、EmoAgent、SOTOPIA |
| Corrective | 中（编辑/修改） | 强、可直接学策略 | SymbioticRAG、SWEET-RL、AI Chains |
| Guidance | 中-高（demo/instruction） | 强、可 bootstrap | Hierarchical Agent、Ask-before-Plan |
| Implicit | 低（观察行为） | 弱+ambiguous | MTOM、Attentive Support、MineWorld |

### 关键发现
- 当前 LLM-HAS 研究**严重 agent-centered**——绝大多数把人当被动评估者，agent 主动观察人/教人的方向（ConvCodeWorld 之外）几乎空白。
- 用 LLM 模拟人（CollabLLM、user simulator）和真人之间的 gap 完全未量化；模拟人极少出现真人的 grammar error 和模糊表达，可能让 benchmark 系统性偏离真实部署。
- 评测严重偏重 task accuracy，没有任何 benchmark 标准化测量"人工作负荷 / cognitive load / coordination cost"，这导致一个 task 显示"协作好"可能只是把成本转嫁给了人。
- 安全性几乎被所有 LLM-HAS 工作回避（MetaGPT、MINT 都没考虑 prompt injection / data exfiltration / interrupt safety），与高风险落地领域严重不匹配。

## 亮点与洞察
- "5 维度分类 + Human Agency Scale" 是把"人-agent 协作"领域 from 散点工作 to 二维坐标系的范式贡献，未来工作都可以快速 self-locate。
- "Human Feedback Type × Granularity × Phase 3D 分类"非常实用——一篇论文的反馈机制可以被精确编码为 (Corrective, Fine, During)，方便横向对比与设计空间探索。
- 强调"很多任务最优点在 Augmentation 而非 Automation"是对当前 LLM agent 社区"卷自治"风潮的及时降温，与 Mitchell et al. 2025 的"完全自治 agent 不该被开发"形成共鸣。
- 提出 4 个开放挑战 (Human Flexibility, Agent-Centered Bias, Inadequate Evaluation, Safety) 对接下来的 benchmark 建设几乎是 to-do list。

## 局限与展望
- 作者自陈：可能漏了 cognitive science 等交叉学科 preprint；本质上是 NLP/agent conference 中心视角。
- 5 维分类相互之间存在轻度冗余（Communication Mode 的 Observation 与 Implicit Feedback 重叠），未来可压缩成更紧凑的本体。
- 没有给出系统的"框架推荐表"——比如"医疗诊断协作该选 A3 + Corrective Fine During + Hierarchical 通信"这种处方型建议会让综述更有可操作性。
- 自己想到的：Human Agency Scale 给了 5 档但没给"如何选择"的算法；未来可以做一个 "task → agency level" 的回归模型，输入任务属性自动推荐协作深度。

## 相关工作与启发
- **vs LLM Multi-Agent 综述（Tran 2025 / Wu 2025）**：那些只覆盖 agent-agent 通信和编排；本文把"人"当 first-class agent 重写了同样的概念体系。
- **vs LLM Agent 综述（Wang 2024a / Li 2024）**：那些以 single-agent 模块（memory/planning/tool use）为脊柱；本文以"协作维度"为脊柱，正交且互补。
- **vs Human-in-the-Loop ML 综述（Wu 2022b）**：传统 HITL 主要在监督学习数据标注层面；本文在 agent decision loop 层面，时间尺度和动态性更复杂。
- **vs Human-AI Teaming 综述（Vats 2024 / Lou 2025）**：更偏 HCI 视角，本文偏 NLP/agent 系统视角，相互补足。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个专门覆盖 LLM-HAS 的综述，5 维分类法 + Agency Scale 是新的分析框架。
- 实验充分度: ⭐⭐⭐⭐ 涉及框架/数据集/benchmark 表很全（Table 4-7），主表覆盖 50+ 工作。
- 写作质量: ⭐⭐⭐⭐ 结构层次清楚、术语统一；个别小节略冗余。
- 价值: ⭐⭐⭐⭐⭐ 给"人该怎么留在 LLM agent loop 里"这一关键但被忽视的问题打了第一根桩。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Searching for Synergy in Shared Workspace Human-AI Collaboration](../../ICML2026/multi_agent/searching_for_synergy_in_shared_workspace_human-ai_collaboration.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[NeurIPS 2025\] MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems](../../NeurIPS2025/multi_agent/metamind_modeling_human_social_thoughts_with_metacognitive_multi-agent_systems.md)
- [\[ICML 2026\] Representational Similarity and Model Behavior in Multi-Agent Interaction](../../ICML2026/multi_agent/representational_similarity_and_model_behavior_in_multi-agent_interaction.md)
- [\[ACL 2026\] Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)

</div>

<!-- RELATED:END -->
