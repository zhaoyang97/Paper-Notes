---
title: >-
  [论文解读] BIRD-INTERACT: Re-imagining Text-to-SQL Evaluation via Lens of Dynamic Interactions
description: >-
  [ICLR2026][LLM评测][text-to-SQL] BIRD-INTERACT 把单轮 text-to-SQL 评测改造成一个带"会模拟用户提问、会管知识库、会跑测试用例"的动态交互环境，覆盖完整 CRUD 操作并故意往任务里注入歧义，用 c-Interact（协议引导对话）和 a-Interact（自主智能体）两种设置考察 LLM 的交互能力——结果连最强的 GPT-5 在 600 题全集上也只解出 8.67%（c）/17.00%（a），暴露出当前模型"会写 SQL 但不会通过交互把任务问清楚"的短板。
tags:
  - "ICLR2026"
  - "LLM评测"
  - "text-to-SQL"
  - "交互式评测"
  - "benchmark"
  - "用户模拟器"
  - "测试时扩展"
---

# BIRD-INTERACT: Re-imagining Text-to-SQL Evaluation via Lens of Dynamic Interactions

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=nHrYBGujps](https://openreview.net/forum?id=nHrYBGujps)  
**论文**: [项目页](https://bird-interact.github.io)（BIRD Team）  
**代码**: https://bird-interact.github.io （有）  
**领域**: LLM评测 / Benchmark / Text-to-SQL  
**关键词**: text-to-SQL, 交互式评测, benchmark, 用户模拟器, 测试时扩展

## 一句话总结
BIRD-INTERACT 把单轮 text-to-SQL 评测改造成一个带"会模拟用户提问、会管知识库、会跑测试用例"的动态交互环境，覆盖完整 CRUD 操作并故意往任务里注入歧义，用 c-Interact（协议引导对话）和 a-Interact（自主智能体）两种设置考察 LLM 的交互能力——结果连最强的 GPT-5 在 600 题全集上也只解出 8.67%（c）/17.00%（a），暴露出当前模型"会写 SQL 但不会通过交互把任务问清楚"的短板。

## 研究背景与动机

**领域现状**：LLM 在 Spider、BIRD 这类单轮 text-to-SQL benchmark 上已经表现得相当亮眼，给一句话需求就能直接吐出正确 SQL。学界由此涌现大量基于 LLM 的 NLIDB（自然语言数据库接口）方法，似乎"把自然语言翻成 SQL"这件事快被做完了。

**现有痛点**：但真实的数据库使用从来不是"一句完美需求 → 一条正确查询"。它是一个有状态、会演化的迭代对话：用户的需求天然带歧义（什么叫"需要紧急处理"？），第一版 SQL 可能跑出语法错误要根据反馈调试，确认正确后用户还会追问依赖前一步结果的后续问题。现有的多轮 text-to-SQL benchmark 在两点上严重失真——其一，它们大多把对话历史当成**静态脚本**：每个 LLM 都被喂同一条预先录好的、干净的对话轨迹（没有失败尝试、没有岔开、没有澄清），于是无法奖励聪明的交互策略，也无法惩罚把对话搞砸的模型；其二，它们的**任务范围太窄**，几乎只覆盖只读的 SELECT 查询（BI 报表场景），完全忽略了 DBA 日常里同样高频的增删改（INSERT/UPDATE/DELETE）、改表结构（ALTER TABLE）、事务控制等数据管理操作。

**核心矛盾**：评测要逼近"真实生产级数据库助手"，就必须同时复原两件被现有 benchmark 阉割掉的东西——**动态交互**（模型得自己去问清歧义、自己从错误中恢复）和**完整操作谱系**（不只读还要写）。而要让交互可大规模、无人工地评，又会引入一个新难题：用 LLM 当用户模拟器很容易"作弊"（泄露 ground-truth SQL）或"跑偏"（偏离原始任务），让评测失去公平性。

**本文目标**：构建一个 benchmark，把"缺失的真实感"补回来——具体拆成：① 一个高保真、可端到端自动运行的交互环境；② 两种贴合真实场景的评测设置；③ 一套覆盖完整 CRUD、带可执行测试用例、每题都需要动态交互才能解的挑战性任务集。

**切入角度**：作者站在 LIVESQLBENCH（一个支持 DML/DDL、自带可执行数据库沙盒和分层知识库的单轮 benchmark）之上，把它"动态化"。关键洞察是：交互的难点可以被工程化地注入和度量——歧义可以人工注入并配一个唯一澄清，用户模拟器可以用"函数驱动"的方式约束住，让它既灵活又不作弊。

**核心 idea**：用"动态交互的视角"重新构想 text-to-SQL 评测——把数据库 + 分层知识库 + 元数据 + 一个函数驱动的用户模拟器耦合成沙盒，故意制造歧义和后续追问，逼模型在有限交互预算下学会"问对问题、用对资源、从错误里恢复"。

## 方法详解

### 整体框架

BIRD-INTERACT 是一个 benchmark（不是模型），所以"方法"其实是**怎么把一个静态单轮数据集，构造成一个能自动跑动态交互评测的环境**。整条管线可以这样看：拿 LIVESQLBENCH 的单轮任务作原料 → 由 12 位专家标注员往里**注入歧义**并**追加后续子任务** → 配上一个**两阶段函数驱动的用户模拟器**充当"会答澄清、会给反馈、但不泄答案"的人类 → 最后用 **c-Interact / a-Interact 两种设置**在有限预算下跑评测，每个子任务的对错由可执行测试用例判定。

数据规模上，最终得到 900 道交互式任务：全集 BIRD-INTERACT-FULL 600 题（展开最多 11,796 次动态交互）用于全面评测，精简集 BIRD-INTERACT-LITE 300 题（数据库更干净）用于细粒度行为分析和快速迭代。每题固定 2 个子任务（$n=2$）：一个带歧义的优先子任务 $q_1$，和一个依赖前一步状态的后续子任务 $q_2$。

形式化地，交互被建模成 text-to-SQL 系统 $S_\theta$ 与用户模拟器 $U_\gamma$ 在数据库环境 $E=\{D, M, K\}$（可执行数据库 $D$、schema 元数据 $M$、外部知识 $K$）上的多轮协作。对子任务 $q_i$，第 $t$ 轮交互为：

$$u_i^t = U_\gamma(h_i^{t-1}, q_i, E), \quad s_i^t = S_\theta(h_i^{t-1}, u_i^t, E), \quad h_i^t = h_i^{t-1} \oplus \langle u_i^t, s_i^t \rangle$$

其中 $h_i^t$ 是到第 $t$ 轮的交互历史，$\oplus$ 表示在 prompt 里做文本拼接。关键约束：只有第一个子任务成功通过测试用例，后续子任务才会被释放——优先子任务一旦失败，整个 session 立即终止。

### 关键设计

**1. 歧义注入：把"一问就清楚"的任务变成"不问清楚就做不出"**

现有 benchmark 的任务太"干净"，模型不需要交互就能做对，自然测不出交互能力。BIRD-INTERACT 的做法是系统性地往单轮查询和环境里注入三类歧义，并保证"每个歧义都配一个唯一的澄清来源"。第一类是**表层用户查询歧义**：包括意图级（用户措辞含糊，如"elderly people"到底指多少岁）和实现级（意图清楚但细节欠定，如小数精度）。第二类是**知识歧义**，往外部分层知识库（HKB）里制造不完整——HKB 把领域知识组织成有向无环图（DAG）的节点，比如"urgent care"→"AVS"→"IF/CPI"这样一条多跳推理链；作者要么删掉孤立知识条目（one-shot 歧义），要么**遮蔽多跳链的中间节点**（knowledge chain breaking，如把"AVS"这个事实从 HKB 里 mask 掉），让推理链断裂、必须靠向用户澄清才能续上。第三类是**环境歧义**：数据库里天然存在的 NULL 等噪声，让"这种情况该怎么处理"本身就不确定。质量控制保证：注入歧义后的查询**不澄清就无解、一旦澄清就能完全重建**——这正是逼出交互的关键。

**2. 后续子任务与状态依赖：让"追问"真正依赖前一步做了什么**

真实用户的意图会随会话演化（改条件、加过滤、探索相关方面），所以每个优先子任务都被追加一个后续子任务，按一套 5 类分类法精心设计。本设计的真正贡献在于引入了**子任务间的状态依赖**——这是和 Spider 系列、SParC、CoSQL 等数据集的本质区别：后续子任务可能依赖前一步**修改过的数据库状态**或**新创建的对象**（比如前一步 CREATE 出来的表/函数）。模型必须对"前一步把数据库改成了什么样"做推理，才能写出后续子任务的 SQL。这把评测从"独立的几条查询"升级成了"有状态、长程的问题求解过程"。

**3. 两阶段函数驱动用户模拟器：既要像人一样灵活，又不能泄题或跑偏**

用 LLM（哪怕 GPT-4o）直接当用户模拟器有两个致命问题：会泄露 ground-truth SQL 里的信息，会偏离原始任务要求。作者的解法是把模拟器拆成两阶段、用"函数/符号动作"把它约束住。**第一阶段**让一个 LLM 充当语义解析器，把系统提出的澄清请求映射到三个预定义动作之一：`AMB()` 对应那些已被预标注、配好关键 SQL 片段的歧义；`LOC()` 处理落在预标注歧义之外但合理的澄清（如 SQL 格式、某个子组件的问题），此时用基于 AST 的检索去定位相关 SQL 片段；`UNA()` 直接拒绝不当请求（如试图套出 ground-truth 答案）。**第二阶段**才让模拟器根据选中的动作 + 标注好的 GT SQL 与澄清来源生成最终回复。这样一来，模拟器的行为既**可预测、可控**（不会泄答案、不会跑偏），又能产生多样、贴合上下文的交互。这是把"用 LLM 模拟人"从不可控变可控的工程关键。

**4. 两种评测设置 + 预算约束：分别考"会不会按流程沟通"和"会不会自主规划"**

同一套任务用两种设置评，对应 LLM 的两种真实角色。**c-Interact（协议引导）**把模型当对话助手：session 按 $q_1 \to q_2$ 两阶段顺序展开，模型可以先澄清再生成 SQL $\sigma_1$，通过测试后用户才抛出连贯的后续 $q_2$；每个子任务给**一次调试机会**（收到执行反馈后可提交一版修订查询），但调试要扣奖励以反映额外计算成本。其预算是对澄清轮数的约束 $\tau_{\text{clar}} = m_{\text{amb}} + \lambda_{\text{pat}}$，$m_{\text{amb}}$ 是解清所有歧义所需的最小轮数（等于标注歧义个数），$\lambda_{\text{pat}}$ 是模拟用户耐心、给模型额外澄清机会的可调参数。**a-Interact（自主智能体）**只给一个高层目标，遵循 ReAct 范式让模型自主规划：把数据库、元数据、HKB、用户模拟器都封装成可调用工具，模型自己决定何时查库、查文档还是问用户，动作空间含 9 个离散动作。其预算按动作计费 $B = B_{\text{base}} + 2 m_{\text{amb}} + 2\lambda_{\text{pat}}$（基础预算 $B_{\text{base}}=6$），不同动作消耗不同预算，逼模型在"彻底"和"高效"之间权衡。两种设置都引入**预算感知**机制（告知模型剩余预算），并支持低预算压力测试，专门考"会不会问对问题、会不会做规划"。

评测指标有两个：**成功率（SR）**，每个子任务通过全部测试用例记 1 否则记 0，分别报告 $q_1$、$q_2$ 的 SR（在线评测）；**归一化奖励（Normalized Reward）**，按优先级加权（优先子任务占 70%、后续占 30%）归一到 $[0,1]$，用于交互结束后的离线行为分析。

## 实验关键数据

作者用一个全新的 PostgreSQL 14 Docker 实例评测了 7 个前沿 LLM（2 开源 + 5 闭源），默认用户耐心 $\lambda_{\text{pat}}=3$、a-Interact 基础预算 6，温度 0、top_p 1，单次运行。

### 主实验

下表为 BIRD-INTERACT-FULL（600 题）上的整体成功率（c-Interact 数值为调试后，括号内 +n 是调试带来的增益），BI=商业智能查询，DM=数据管理查询：

| 设置 | 模型 | 优先子任务 SR(%) | 后续子任务 SR(%) | Reward*(%) | 均价(USD) |
|------|------|------|------|------|------|
| c-Interact | GPT-5 | 14.50 | 8.67 | 12.58 | $0.08 |
| c-Interact | Claude-Sonnet-4 | 22.33 | 14.17 | 18.35 | $0.29 |
| c-Interact | Gemini-2.5-Pro | **25.00** | **16.33** | **20.92** | $0.04 |
| a-Interact | Qwen-3-Coder-480B | 13.33 | 4.17 | 10.58 | $0.07 |
| a-Interact | Claude-Sonnet-4 | 27.83 | 12.67 | 23.28 | $0.51 |
| a-Interact | GPT-5 | **29.17** | **17.00** | **25.52** | $0.24 |

核心信号是难度极高：最强的 Gemini-2.5-Pro / GPT-5 在两种模式下也只拿到 20.92% / 25.52% 的可得奖励，端到端成功率在 c-Interact 不超过 16.33%、a-Interact 不超过 17.00%，给未来留出大量提升空间。

### 消融 / 分析实验

| 分析 | 关键发现 | 说明 |
|------|---------|------|
| 交互模式对比 | GPT-5：c-Interact 14.50%（最差）↔ a-Interact 29.17%（最好） | 同一模型在两种模式下表现可天差地别，交互模式是成败决定性因素 |
| Memory Grafting | 把 Qwen-3-Coder / O3-mini 的澄清历史"嫁接"给 GPT-5 后，GPT-5 显著涨点 | 证明 GPT-5 短板在沟通/交互而非 SQL 生成能力 |
| ITS（交互测试时扩展） | Claude-3.7-Sonnet 随交互轮数增加，性能单调上升 | 提出 ITS Law：给足交互轮数，性能可追平甚至超过理想化单轮任务 |
| BI vs DM | BI 查询显著比 DM 难 | DM 操作模式标准化、可预测；BI 需要复杂领域业务逻辑和分析推理 |

### 关键发现
- **GPT-5 的"反差"最值得玩味**：它在协议引导的 c-Interact 里垫底（14.50%），却在自主探索的 a-Interact 里登顶（29.17%）。作者通过 Memory Grafting 实验把别的模型的澄清对话历史喂给它，性能立刻上来——说明它"会写 SQL"但"不会按既定流程把需求问清楚"，瓶颈在交互沟通而非生成能力。
- **后续子任务普遍比优先子任务难**：越往后上下文越长、拼接越多，长上下文成了交互式 text-to-SQL 的瓶颈；加上后续子任务依赖前一步修改的数据库状态，对推理要求更高。
- **交互是可以"测试时扩展"的资源**：ITS 实验表明，把额外的交互机会高效转化为信息增益的模型（如 Claude-3.7-Sonnet），性能能随轮数稳定爬升，逼近甚至反超"信息全给齐"的理想单轮场景。

## 亮点与洞察
- **把"交互能力"从玄学变成可注入、可度量的工程量**：歧义不是天上掉下来的，而是按表层/知识/环境三类系统注入、每个配唯一澄清，并保证"不问就无解、问了就能重建"——这让 benchmark 能精确地把"需要交互"逼出来，而不是靠运气。
- **两阶段函数驱动用户模拟器**是最可复用的 trick：先用 LLM 把澄清请求解析成 `AMB/LOC/UNA` 三个符号动作、再据此受控生成回复，干净利落地解决了"LLM 模拟用户会泄题/跑偏"的老大难。这套"语义解析→符号动作→受控生成"的两段式思路，可直接迁移到任何需要可控 LLM 模拟器的交互式评测（客服、工具调用、多轮 agent）。
- **同一套任务、两种设置（对话 vs 智能体）**的对照设计，揭示了"交互模式 × 模型"的强交互效应——这提醒大家：评一个模型的交互能力，单一协议远远不够。
- **Memory Grafting 这个诊断实验很漂亮**：通过"移植别人的对话历史"把"生成能力"和"沟通能力"解耦开来定位瓶颈，是一种很值得借鉴的归因方法。

## 局限与展望
- **覆盖完整 CRUD 但仍限于关系型数据库 + PostgreSQL**：对 NoSQL、图数据库、跨库联邦查询等场景未覆盖，外推性待验证。
- **每题固定 2 个子任务**（$n=2$）：虽然引入了状态依赖，但相比真实 DBA 动辄十几轮、多目标演化的会话仍偏短，长程依赖的难度可能被低估。
- **用户模拟器再可控也仍是 LLM 近似真人**：函数驱动约束住了泄题/跑偏，但模拟用户的耐心、表达风格、追问逻辑是否真能代表真实用户分布，仍是开放问题；且评测因成本只做单次运行（temperature=0），方差未被刻画。
- **歧义注入依赖 12 位专家标注**，规模化扩展到新领域/新库的人力成本不低，自动化注入流程是值得探索的方向。

## 相关工作与启发
- **vs Spider / BIRD（单轮 text-to-SQL）**：它们只评"一句话→一条 SQL"的生成能力，BIRD-INTERACT 把评测对象换成了"在歧义、错误、演化需求中通过多轮交互解决问题"的完整能力，难度量级提升（最强模型从单轮高分掉到 8.67%）。
- **vs SParC / CoSQL / 现有多轮 text-to-SQL**：它们把对话当**静态脚本**，每个模型被喂同一条预录轨迹，无法奖励聪明交互、无法惩罚搞砸对话；BIRD-INTERACT 用活的用户模拟器让交互轨迹由模型自己生成，且引入子任务间状态依赖和完整 CRUD（不只 SELECT）。
- **vs LIVESQLBENCH（构建基座）**：BIRD-INTERACT 直接站在它的可执行沙盒、DML/DDL 支持和分层知识库（HKB）之上，核心贡献是把这个**单轮** benchmark 动态化、交互化。
- **vs MINT 等 LLM-as-user 交互 benchmark**：它们也用 LLM 模拟用户但存在泄题/跑偏问题；BIRD-INTERACT 的两阶段函数驱动模拟器是针对性的修复。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用动态交互视角重构 text-to-SQL 评测，歧义注入 + 两阶段函数驱动模拟器 + 双设置都是扎实的原创设计。
- 实验充分度: ⭐⭐⭐⭐ 7 个前沿模型 × 双设置 × 600 题，外加 Memory Grafting、ITS、BI/DM 等分析；唯单次运行、未刻画方差略可惜。
- 写作质量: ⭐⭐⭐⭐⭐ 动机推导清晰，图 1 的任务全景和形式化定义把"难在哪"讲得很透。
- 价值: ⭐⭐⭐⭐⭐ 暴露了"会写 SQL ≠ 会交互"的真实短板，为交互式数据库助手提供了高难度、可自动评测的标尺。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] MCTS-SQL: Light-Weight LLMs can Master the Text-to-SQL through Monte Carlo Tree Search](../../AAAI2026/llm_evaluation/mcts-sql_light-weight_llms_can_master_the_text-to-sql_through_monte_carlo_tree_s.md)
- [\[ACL 2025\] Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks](../../ACL2025/llm_evaluation/somethings_fishy_in_the_data_lake_a_critical_re-evaluation_of_table_union_search.md)
- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](../../ACL2026/llm_evaluation/comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)
- [\[ICLR 2026\] Can Vision–Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective](can_vision_language_models_assess_graphic_design_aesthetics_a_benchmark_evaluati.md)
- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](../../ACL2026/llm_evaluation/minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)

</div>

<!-- RELATED:END -->
