---
title: >-
  [论文解读] LLM Agents Are the Antidote to Walled Gardens
description: >-
  [ICML 2026][LLM Agent][universal interoperability] 这是一篇 ICML 2026 立场论文，主张 LLM 智能体能够通过自动格式转换 + 拟人化 UI 交互"绕过"主流平台的 API 封闭策略，实现"通用互操作性"（universal interoperability），从而瓦解传统网络效应造成的"围墙花园"，但同时需要 ML 社区主动建立 agent-friendly 接口、安全机制和生态基础设施来管控随之而来的安全、法律与新一层 lock-in 风险。
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "universal interoperability"
  - "walled gardens"
  - "data portability"
  - "agent security"
---

# LLM Agents Are the Antidote to Walled Gardens

**会议**: ICML 2026  
**arXiv**: [2506.23978](https://arxiv.org/abs/2506.23978)  
**代码**: 无（立场论文）  
**领域**: LLM Agent / 互操作性 / AI 治理  
**关键词**: LLM agent, universal interoperability, walled gardens, data portability, agent security

## 一句话总结
这是一篇 ICML 2026 立场论文，主张 LLM 智能体能够通过自动格式转换 + 拟人化 UI 交互"绕过"主流平台的 API 封闭策略，实现"通用互操作性"（universal interoperability），从而瓦解传统网络效应造成的"围墙花园"，但同时需要 ML 社区主动建立 agent-friendly 接口、安全机制和生态基础设施来管控随之而来的安全、法律与新一层 lock-in 风险。

## 研究背景与动机
**领域现状**：Internet 底层协议（TCP/IP、HTTP、DNS）天生开放，但应用层却被一系列"围墙花园"瓜分——社交网络互不连通，企业软件用私有 API，移动平台限制开发者只能在封闭生态里写代码。GDPR 的数据可携权、欧盟 DMA 的强制互操作条款只是反应式、慢节奏的局部修补。

**现有痛点**：互操作性长期没做成，技术上是因为构建和维护跨服务集成又贵又繁琐（schema 对齐、版本兼容、错误处理、业务规则编码），战略上是因为占主导的平台没动力让用户轻易迁走数据（强网络效应 → 高切换成本 → 锁定用户）。法律上 ToS 普遍禁止自动化访问，监管又落后于平台演化速度。

**核心矛盾**：用户福利和市场竞争需要可携性与互操作，但平台的商业利益恰恰建立在封闭之上；传统标准化（SOAP、REST、Semantic Web）与监管干预都无法以足够快的速度撕开这道墙。

**本文目标**：(1) 论证当下 LLM 智能体的两项能力为何已经从根本上改变了互操作性的成本结构；(2) 给出"通用互操作性"这一新范式的正反面分析；(3) 提出 ML 社区应当主动建设的三类基础设施来引导这一趋势走向良性。

**切入角度**：作者把 LLM agent 视作"通用适配器"（universal adapter）——任何对人类可读的 GUI、对机器可读的 API，它都能在运行时动态适配，让"开 API"还是"不开 API"对平台而言不再具有战略意义。

**核心 idea**：用 LLM 智能体在运行时动态生成 schema 映射、glue 代码和 UI 操作脚本，把过去要几周工程的集成压缩为几次 prompt，从而让"是否开放 API"这一战略选择失去意义；与其抵抗，不如趁智能体生态尚未成熟，主动建好接口、安全与治理的脚手架。

## 方法详解

### 整体框架
这是一篇立场论文，不提算法，而是构建一条"为什么 LLM agent 能瓦解围墙花园、ML 社区又该如何引导它"的论证链，全文按 Background → Universal Adapter → Universal Interoperability → Call to Action → Alternative Views 推进。它先用过去标准化（XMPP/ActivityPub/FHIR/ISO 20022）和监管（DMA/GDPR/ACCESS Act）的失败做对照组，再把 LLM agent 定义为"既懂自然语言/代码/结构化格式、又能调 API 或模拟用户操作"的通用适配器（universal adapter）——它的格式自动翻译 + 鲁棒 UI 交互两项能力，共同把平台"开/不开 API"的战略选择"折现"为零，于是 universal interoperability（用 LLM 适配器在运行时动态发现操作、推断 schema、生成 glue code/UI 动作）成为新范式。作者不靠跑模型，而靠两类硬证据撑起紧迫感：WebArena 成功率 18 个月从 8.87% 涨到 71.6%（接近人类 78.24%）、头部实验室（ChatGPT Atlas、Claude in Chrome、Gemini Computer Use、Perplexity Comet、Edge Copilot、Nova Act）全部上线 production web agent，说明"agent 已经在干这件事"；Perplexity 涉嫌违反 robots.txt 与版权、Akirabot 用 OpenAI 模型 + CAPTCHA bypass 在 80,000+ 站点投垃圾，说明"无序的绕墙已在野外发生"，于是问题从"会不会发生"被改写成"以什么方式发生"。§6 还把监管优先 / 本体论优先 / 安全保守派 / 经济可持续性四种反对意见反向吸收进自己的方案。

### 关键设计
论文最具"可被实施"成分的是 §5 给 ML 社区的三类基础设施，这里把它们当作三个核心论点精读——它们共同回答"既然趁智能体生态尚未成熟该主动搭脚手架，那到底搭什么"。

**1. Agent-Friendly Interfaces：用最小元数据让 agent 跳过试错循环**

针对的痛点是 agent 现在只能靠"试错—失败—调 prompt"的循环去猜接口背后的隐含业务规则。作者的解法是在现有 API/网页上叠加最小侵入的注解、而不是另起一套新标准：对 REST 这类机器接口，让服务方在 OpenAPI schema 外补一段自然语言 rationale（可由非技术人员甚至 LLM 撰写），最简形式就是给一个博客/官方文档链接，进阶形式是开一个专答 schema 澄清问题的 LLM 端点；对网页，则在 DOM 里嵌一个 manifest，把按钮/表单字段映射到具体 endpoint（如把 "Submit Order" 按钮标注为 `POST /api/order`），让 agent 直接跳过 UI 调 API。`llms.txt`（Howard, 2024）是这一方向当下最早的雏形。这样研究问题就被收窄成"多少元数据才够、静态注解与动态 explanation 服务如何搭配最优"，比等一个全新标准被广泛采纳现实得多。

**2. Security by Design：围绕 agent 的三层运行时安全架构**

当 agent 自主操作关键数据流时，既要保护用户、也要保护被访问的网站，作者因此把"agent 自主性"和"站点可控性"拆到不同层，提出一套三层运行时强制架构。第一层是 signed permission documents（参考 South et al., 2025），给每个 agent 签发可验证的权限文档，声明允许的 endpoints、数据使用策略、速率上限与委托权；第二层是 runtime policy checker，在每个 action 执行前比对权限文档，违规即阻断或上报；第三层是 automatic rollback / kill-switch，监控到越界行为就回退或终止。难点集中在第二层——既要低延迟不拖慢 agent workflow、又要低假阳性不频繁打断用户，作者建议用学习型 policy 分类器 + 符号检查器混合实现，并配合 ToolEmu（模拟外部 API）、AgentSims（合成任务环境）、SandboxEval（隔离容器测越权）等沙箱把 agent 安全测试纳入 CI。它的类比很直白：universal interoperability 需要一个等价于 OAuth 的"agent 权限/速率/委托标准"，否则站点只能一刀切封禁 AI 流量，反而把互操作性堵死。

**3. Ecosystem Infrastructure：开放协议、技术债治理与防垄断三管齐下**

这条针对的是 universal interoperability 最大的反噬——"墙"从 API 层滑落到 agent/模型层，换一层马甲继续 lock-in。在协议层，作者支持 Google A2A 与 Anthropic MCP 这类开放协议，但提醒它们都源自单一公司、自带 lock-in 风险，解法是积极参与 W3C AI Agent Protocol、Lightweight Agent Standards、NANDA、Eclipse LMOS 等多方治理工作组，并让 agent 框架自带 A2A↔MCP↔其他协议的 adapter。在技术债层，社区需要维护开源的"集成参考实现"模板（auth/pagination/error/rate-limit），API 提供方应发布机器可读 changelog（如 OpenAPI diff 格式），让 agent 能扫描下游 adapter 的破坏性变更；每段生成的 connector 都带版本和生成元数据，长期目标是 agent 自身能感知更新、自动 deprecate 过时 adapter。在防垄断层，开源 agent 框架与模型是对抗"框架偏袒自家服务"（agent-layer favoritism）的最佳防线，agent 须可审计地 log 它评估与选择服务的依据——否则底层 API 再开放也毫无意义。

## 实验关键数据

### 主"对比表"：通用互操作性 vs 既有范式
作者并未跑实验，但 §4 显式做了一张范式对比，整理如下：

| 范式 | 接口契约 | 适配方式 | 主要弱点 |
|------|----------|----------|----------|
| 静态中间件 / 自研 adapter | 预编程 | 工程师手写 | 维护成本高，跨服务难复用 |
| Semantic Web（RDF/OWL） | 全局本体 | schema 注册中心 | 本体工程门槛高，semantic drift |
| 标准化 API（OpenAPI/GraphQL） | 预定义合约 | 自动生成 client | 仍需统一规范，跨平台覆盖有限 |
| RPA / 规则爬虫 | 无 | UI 脚本 | 对页面变更脆弱，无语义理解 |
| **Universal Interoperability**（本文） | **运行时推断** | **LLM 动态生成 glue/UI 动作** | 安全、技术债、agent-layer lock-in |

### 关键定量证据：Web agent 能力的指数级增长

| 时间 | 评测 / 事件 | 数值或现象 |
|------|-------------|------------|
| 2023-03 | WebArena 成功率（Zhou et al., 2024） | 8.87% |
| 2026-01 | WebArena 成功率（Guo et al., 2026） | 71.6%（接近人类 78.24%） |
| 2024–2026 | 头部实验室发布 production web agent / agent browser | 6+（OpenAI、Anthropic、Google、Perplexity、Microsoft、Amazon） |
| 2024–2025 | 野外越界案例 | Perplexity 违反 robots.txt/版权、Akirabot 在 80k+ 网站投垃圾消息 |

### 关键发现
- 三年内 WebArena 涨了近一个数量级，说明"绕过 UI"已从研究 toy 变成 production 现实；论证的时间窗口确实关闭得很快。
- "野外案例"已先于学术框架出现，这把作者的立场从"未来畅想"拔高到了"必须 now-or-never 行动"。
- 即便作者列出诸多风险，但所有风险都被归类为"engineering and governance challenges rather than insurmountable obstacles"——这是这篇论文最关键的修辞立场。

## 亮点与洞察
- **"经济学 + 工程学"双视角**：很多 LLM agent 论文只谈技术，本文从 Katz–Shapiro / Farrell–Saloner 的网络效应理论切入，把 LLM agent 的实际作用解释为"降低 multi-homing 与 switching cost"，给纯技术读者补了一节产业组织课。
- **"墙不靠拆，而是被绕没"**：传统对抗围墙花园的思路是 GDPR/DMA 那种"强制开 API"，作者直接给出技术性绕过路径——agent 可以爬 UI、可以拟人，平台关 API 反而没意义。这是一个让监管者也会重新思考策略的 reframe。
- **三层运行时安全架构很可迁移**：signed permission document → runtime policy checker → automatic rollback 这套架构，对任何想做 agent 框架/agent 网关的人都是直接可参考的设计蓝图，可对应到 MCP / A2A 的真实工程。
- **吸纳反方观点的写法值得学**：§6 的四个 Alternative Views 不是"打脸对手"而是"反向吸收进自己方案"，例如把 regulation-first 转译为"agent 接口要内嵌 consent/规制"，这种讨论方式让立场论文更立体。

## 局限与展望
- **没有给出 universal interoperability 的可量化定义或 benchmark**：什么算"足够通用"？哪些维度上对比？目前停留在概念层面，未来 ML 社区可设计"端到端 universal-adapter benchmark"。
- **runtime policy checker 的可行性是关键瓶颈但只有大方向**：低延迟 × 低假阳性的策略校验在真实工业 agent 流量下究竟能不能做出来，作者只给了 hybrid（学习 + 符号）的范畴，没给原型。
- **agent-layer lock-in 的反垄断手段偏弱**：仅依赖"开源模型 + 透明日志"很难制约头部 agent 框架的网络效应；可能需要类似"agent 等价于 OAuth"的强制标准。
- **对内容生产者经济模型的冲击讨论较浅**：当用户不再访问网页而是让 agent 抓内容时，广告收入与内容生态的可持续性问题作者只点了 §6 经济可持续性反方，没给出严肃方案。

## 相关工作与启发
- **vs ReAct / WebArena / Gorilla / RestGPT 等 agent 实证工作**：那些工作回答"能不能让 agent 干这事"，本文则把"能干"翻译成"产业层会发生什么"，是 agent 研究的 macro-level 配套立场。
- **vs A2A (Google, 2025) / MCP (Anthropic, 2024)**：A2A、MCP 是协议层的具体提案，本文把它们放到 ecosystem infrastructure 框架里，并提醒"单公司发起的开放协议依旧有 lock-in"，提出多方治理 + 协议互转 adapter 作为防御。
- **vs Bainbridge "Ironies of Automation" (1983)**：作者把这个经典 HCI/可靠性观点搬进 agent 时代——人从操作者变监视者，干预能力下降——为 §4.2 的安全风险论证打底，启发我们写 agent 安全研究时也可引用这条系统工程经典。
- **vs Kapoor et al. (2025) on agent-layer favoritism / Agranat & Gal (2025) on AI agents and network effects**：这些是关于"agent 是否会成为新一层垄断"的同期前沿论文，本文把它们整合到 §4.2 风险与 §5.3 防垄断里。

## 评分
- 新颖性: ⭐⭐⭐⭐ 提出 "universal interoperability" 这一术语并把 LLM agent 与产业组织理论显式接通，框架性贡献明显，但单个论点（agent 绕墙、agent-layer lock-in、permission documents）在 22~26 年间已陆续被分散讨论过。
- 实验充分度: ⭐⭐⭐ 立场论文不要求 benchmark，但作者用 WebArena 趋势 + 真实越界案例做证据，比一般 position paper 扎实；缺少对 policy checker / agent-friendly schema 的原型验证。
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰、文献覆盖跨经济学/法律/系统/ML，Alternative Views 写法是范本，适合做 position paper 的写作模板。
- 价值: ⭐⭐⭐⭐ 对 agent 框架开发者、政策制定者、平台战略团队都有直接行动意义；对纯技术研究者价值偏低，但给出了若干可立项的研究方向（policy checker、agent-friendly metadata、OpenAPI diff、open-source agent framework 反垄断评估）。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Constitutional Black-Box Monitoring for Scheming in LLM Agents](constitutional_black-box_monitoring_for_scheming_in_llm_agents.md)
- [\[ICML 2026\] Agent-Omit: Adaptive Context Omission for Efficient LLM Agents](agent-omit_adaptive_context_omission_for_efficient_llm_agents.md)
- [\[ICML 2026\] ExCyTIn-Bench: Evaluating LLM Agents on Cyber Threat Investigation](excytin-bench_evaluating_llm_agents_on_cyber_threat_investigation.md)
- [\[ICML 2026\] Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use](reward_hacking_benchmark_measuring_exploits_in_llm_agents_with_tool_use.md)
- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)

</div>

<!-- RELATED:END -->
