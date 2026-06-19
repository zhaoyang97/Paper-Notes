---
title: >-
  [论文解读] A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?
description: >-
  [ACL 2026][LLM安全][Computer-Using Agent] 本文给"计算机使用智能体 (CUA)"的安全研究做了第一次系统化梳理，把 124 篇相关论文整理成"内在威胁 × 外在威胁 × 防御 × 评测"四维分类框架，并指出现有 CUA 的最大缺口是 UI grounding 鲁棒性与跨平台对抗评测。
tags:
  - "ACL 2026"
  - "LLM安全"
  - "Computer-Using Agent"
  - "GUI 智能体"
  - "安全威胁"
  - "提示学习"
  - "defense taxonomy"
---

# A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?

**会议**: ACL 2026  
**arXiv**: [2505.10924](https://arxiv.org/abs/2505.10924)  
**代码**: 无  
**领域**: 多模态 VLM (综述)  
**关键词**: Computer-Using Agent, GUI 智能体, 安全威胁, prompt injection, defense taxonomy

## 一句话总结
本文给"计算机使用智能体 (CUA)"的安全研究做了第一次系统化梳理，把 124 篇相关论文整理成"内在威胁 × 外在威胁 × 防御 × 评测"四维分类框架，并指出现有 CUA 的最大缺口是 UI grounding 鲁棒性与跨平台对抗评测。

## 研究背景与动机
**领域现状**：以 OpenAI 的 o3/o4-mini、PC-Agent、AppAgent、SeeAct 为代表的 CUA 把多模态感知、推理与 GUI 自动化结合，已能自主完成"网购、订机票、填表"等复杂任务，正快速被工业界部署。

**现有痛点**：CUA 同时把"LLM 的固有漏洞 + GUI 操控权限 + 多模态输入"叠加在一起，攻击面急剧扩大：视觉 grounding 误差、响应延迟、HTML 解析陷阱都可能被滥用为数据泄露或目标劫持的入口。但目前的安全研究散落在 prompt injection、jailbreak、backdoor 等各自孤立的方向，缺少统一视角。

**核心矛盾**：单一 LLM 的安全研究无法直接迁移——CUA 多了"环境"（GUI 状态、外部 API）这一不可信通道，许多原本不存在的攻击面（环境注入、reasoning gap、web hacking）由此涌现。同时，"自主操作权"使得任何成功的攻击都直接产生物理世界后果（资金转账、文件删除）。

**本文目标**：(i) 给出适合安全分析的 CUA 统一定义；(ii) 系统枚举所有内在/外在威胁；(iii) 把现有防御方案与威胁逐一对齐；(iv) 总结评测 benchmark、指标与数据集。

**切入角度**：从"智能体 = 感知/大脑/行动 三件套 + 环境"这一架构视图入手，把每一类组件作为攻击面来枚举威胁，再反向梳理对应的防御。这种"组件中心"视角比"按攻击名分类"更不易遗漏新型攻击。

**核心 idea**：用一个"威胁来源 × 受影响组件 × 威胁模型"三轴的可扩展分类框架，把碎片化的 CUA 安全研究统一收编，并暴露真正的安全空白。

## 方法详解

### 整体框架
这篇综述要解决的是 CUA 安全研究"散落各处、缺统一视角"的问题，做法是把 124 篇文献收编进一个三层威胁-防御对照体系。它先给 CUA 一个适合安全分析的统一定义——把智能体拆成"感知 / 大脑 / 行动"三件套外加一个不可信的"环境"通道，并细分出 OS Agent、GUI Agent、Web Agent、设备控制 Agent 四类子型；再以组件为锚枚举 8 类内在威胁与 8 类外在威胁，反向梳理 14 类防御方法逐一对齐；最后铺开按平台（Web / Mobile / 通用）划分的 benchmark 与 metric 全景。整条主线是"先定义攻击面 → 枚举威胁 → 对齐防御 → 给评测地图"。

### 关键设计

**1. 三轴威胁分类法：用"来源 × 组件 × 威胁模型"取代"按攻击名分类"，让分类法长期不过时**

传统安全综述按攻击名（jailbreak、backdoor……）分类，每出现一种新攻击就得新增一栏，框架越长越乱。本文转而给每个威胁打三轴标签——威胁来源（Env / Prompt / Model / User）× 受影响组件（Perception / Brain / Action）× 威胁模型，外加一个"何时发生"的阶段标签（development / deployment / architecture / training）。在这个矩阵里，内在威胁来自 agent 自身缺陷：感知层是"UI 理解与 grounding 困难"，大脑层集中了"调度错误、目标错配、幻觉、上下文超长、社会文化偏见、响应延迟"六种，行动层是"API 调用错误"；外在威胁来自攻击者：对抗攻击、直接/间接 prompt injection、jailbreak、记忆提取与注入、backdoor、reasoning gap attack（多模态信号冲突）、system sabotage、web hacking。因为标签锚在组件而非攻击名上，新攻击诞生后只需归入对应组件即可，分类法不必推倒重来。

**2. 防御-威胁对齐矩阵：把 14 类防御和 16 类威胁做显式映射，让工程师照着选组合**

枚举完威胁后，光有威胁清单工程上没法直接用——读者真正想问的是"我要防 X 攻击，业内有哪几招""某种防御能覆盖多少威胁"。本文把 14 类防御（环境约束、输入校验、防御性 prompt、数据净化、对抗训练、输出监控、模型审查、跨验证、持续学习、透明化、拓扑引导、感知协同、规划架构强化、合规规则）逐条标上同样的三轴标签——目标组件、强化的框架元素、对应威胁编号（In./Ex.X），再与内在/外在威胁对齐。比如"环境约束"主防外在威胁（限制 agent 与 GUI 交互的权限），"规划架构强化"则同时缓解内在的调度错误和外在的 reasoning-gap 攻击。这样一来 web 应用要防护就能直接读出"环境约束 + 输入校验 + 跨验证"这种组合，避免重复造轮子。

**3. 跨平台 benchmark/metric 全景：按平台铺开评测地图，并把混杂的指标归成三组**

CUA 跨平台差异极大——Mobile 屏幕受限、Web 文本动态、桌面 API 复杂，连评测时用哪个 benchmark 都不好选。本文按 Web / Mobile / General-purpose 三类平台整理安全 benchmark，把散乱的指标归成三组：任务完成（TSR、Helpfulness）、中间步骤（SSR、Total Correct Prefix）、安全鲁棒性（ASR、CuP、F1、RR、LR、AR、TS）；测量方式也分成 Rule-based、LLM-as-a-judge、Manual 三类。这张地图让研究者能按目标平台和关注维度快速挑到合适的评测环境。

### 损失函数 / 训练策略
本文为综述，无独立训练。作者梳理了防御侧的核心训练目标作为参考：对抗训练（注入对抗样本提升鲁棒性）、数据净化（剔除被污染样本）、持续学习与自演化（基于环境反馈在线更新策略），以及合规规则学习（把 SOP / 伦理规范编码进训练目标）。

## 实验关键数据
本文为综述，下面整理其汇总的 CUA 安全 benchmark 与威胁分布数据。

### 主实验：CUA 安全 benchmark 全景（节选）

| Benchmark | 平台 | 主要威胁焦点 | 评测方式 |
|-----------|------|-------------|----------|
| ST-WebAgentBench | Web | safety + trustworthiness | Rule-based |
| BrowserART | Web | jailbreak | LLM-as-judge |
| PrivacyLens | Web/通用 | 隐私泄露 | LLM-as-judge |
| MobileSafetyBench | Mobile | 多种安全风险 | Rule-based |
| Hijacking JARVIS | Mobile | 第三方注入 | Rule-based |
| AgentDojo | 通用 | prompt injection | Rule-based |
| AgentHarm | 通用 | 有害指令 | LLM-as-judge |
| OS-Harm | 通用 | OS 层攻击 | LLM-as-judge |
| TrustAgent | 通用 | 综合可信 | LLM-as-judge |
| RedTeamCUA | 混合 Web/OS | 红队对抗 | LLM-as-judge |
| AgentHazard | 通用 | 有害行为 | LLM-as-judge |

### 消融实验：威胁-组件影响矩阵（简化）

| 威胁类别 | 主要来源 | 主受影响组件 | 触发阶段 |
|----------|----------|-------------|---------|
| UI Grounding 困难 | Env | Perception | Development |
| 调度错误 | Prompt | Brain | Development |
| 目标错配 | Prompt | Brain | Deployment |
| 幻觉 | Prompt/Model | Brain | Deployment |
| 对抗攻击 | Env | Perception | Runtime |
| Prompt Injection | Env/Prompt | Brain/Action | Runtime |
| Memory Attack | Model | Brain/Action | Runtime |
| Backdoor | Model | Brain/Action | Training |
| Reasoning Gap | Model | Perception | Runtime |
| Web Hacking | Env | Action | Runtime |

### 关键发现
- **Brain 组件是最高危位置**：8 类内在威胁中 6 类、8 类外在威胁中 5 类都直接攻击或经过 Brain，说明 LLM 的推理/规划层是 CUA 的薄弱核心。
- **环境注入是 CUA 特有攻击面**：间接 prompt injection（污染网页、文件、UI 元素）几乎仅在 CUA 设定下成立，传统 LLM benchmark 完全捕捉不到。
- **评测过度集中在 Web 平台**：本综述列出的 benchmark 中 Web 类最多，Mobile/混合环境严重不足，缺乏跨平台一致的安全评估标准。

## 亮点与洞察
- **组件中心的分类法**：把"威胁锚到 Perception/Brain/Action"而非"按攻击名分类"是一个长青设计，新攻击诞生后只需归到已有组件，分类法不需推倒重来。
- **防御-威胁矩阵直接可用**：作者把 14 类防御和 16 类威胁做了显式映射，工程师可以照着矩阵选择组合（如 web 应用：环境约束 + 输入校验 + 跨验证）。
- **"reasoning gap attack"被首次正式纳入**：这种通过多模态信号冲突诱导 agent 犯错的攻击（如截图里的弹窗文本与 HTML 不一致）是 CUA 独有的，未来需要多模态对齐式防御。
- **明确指出 transparency 是治理瓶颈**：作者点名 OpenAI 等厂商未公开 safety policy 与评测结果，呼吁建立独立审计与披露框架。

## 局限与展望
- **作者承认**：仅覆盖到截稿前的公开英文文献，可能漏掉新兴攻击和工业界内部研究；只做架构层分析，未对各防御做相对效力的实证评测。
- **额外局限**：威胁矩阵中"affected component"的标签依赖人工判断，对模糊攻击（如同时影响 Perception 和 Brain 的 reasoning gap）有定义重叠；防御-威胁对齐矩阵不区分"主要防"与"次要防"的覆盖度差异。
- **改进方向**：未来工作应做端到端实测——在统一 benchmark 上跑所有防御组合，给出 Pareto 前沿；同时建立多语言、跨文化的安全评测扩展，目前的 benchmark 几乎都基于英文 GUI。

## 相关工作与启发
- **vs LLM Safety 综述**（Shi 2024、Ma 2025）：他们关注模型层威胁（jailbreak、hallucination），本文把视角扩展到"模型 + 环境 + 多模态" 三方耦合，揭示 CUA 独有的环境注入攻击面。
- **vs OS Agent 综述**（Hu 2024）：他们偏能力综述（架构、记忆、规划），本文专攻安全维度，构成互补关系。
- **vs Sager et al. 2025**（AI Agents for Computer Use 综述）：他们梳理 GUI agent 的能力与落地，本文则提供这些 agent 部署前必须解决的"安全清单"。

## 评分
- 新颖性: ⭐⭐⭐⭐ 第一份针对 CUA 的安全综述，"组件中心"分类法与 reasoning gap attack 的引入有原创性。
- 实验充分度: ⭐⭐⭐ 综述类工作，覆盖 124 篇文献但无原创实验；防御对比缺少端到端实证。
- 写作质量: ⭐⭐⭐⭐ 三轴矩阵与防御-威胁对齐表非常工程友好；JARVIS/Ultron 隐喻让安全研究更有传播力。
- 价值: ⭐⭐⭐⭐⭐ 对要部署 CUA 的工程团队是必读手册；对学术研究者明确指出 mobile/cross-platform 评测和透明性治理两个空白方向。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Why Agents Compromise Safety Under Pressure](why_agents_compromise_safety_under_pressure.md)
- [\[ACL 2026\] On Safety Risks in Experience-Driven Self-Evolving Agents](on_safety_risks_in_experience-driven_self-evolving_agents.md)
- [\[ACL 2026\] LeakDojo: Decoding the Leakage Threats of RAG Systems](leakdojo_decoding_the_leakage_threats_of_rag_systems.md)
- [\[ICLR 2026\] Breaking Agent Backbones: Evaluating the Security of Backbone LLMs in AI Agents](../../ICLR2026/llm_safety/breaking_agent_backbones_evaluating_the_security_of_backbone_llms_in_ai_agents.md)
- [\[ACL 2026\] AgentMark: Utility-Preserving Behavioral Watermarking for Agents](agentmark_utility-preserving_behavioral_watermarking_for_agents.md)

</div>

<!-- RELATED:END -->
