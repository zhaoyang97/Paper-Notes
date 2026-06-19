---
title: >-
  [论文解读] The Controllability Trap: A Governance Framework for Military AI Agents
description: >-
  [ICLR 2026][LLM Agent][AI governance] 提出 Agentic Military AI Governance Framework (AMAGF)，一个围绕可测量的控制质量分数 (CQS) 构建的军事 AI 代理治理框架，通过预防-检测-纠正三个支柱应对六类代理治理失败。
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "AI governance"
  - "military AI"
  - "agentic AI"
  - "human control"
  - "safety framework"
---

# The Controllability Trap: A Governance Framework for Military AI Agents

**会议**: ICLR 2026  
**arXiv**: [2603.03515](https://arxiv.org/abs/2603.03515)  
**代码**: 无  
**领域**: LLM Agent  
**关键词**: AI governance, military AI, agentic AI, human control, safety framework

## 一句话总结

提出 Agentic Military AI Governance Framework (AMAGF)，一个围绕可测量的控制质量分数 (CQS) 构建的军事 AI 代理治理框架，通过预防-检测-纠正三个支柱应对六类代理治理失败。

## 研究背景与动机

当前军事 AI 治理讨论已就"有意义的人类控制"达成共识，但对于**如何在实际部署中实现和维持人类控制**缺乏具体机制。这一差距随着代理式 AI 系统（agentic AI）进入军事服务而变得尤为关键。

与传统军事自动化不同，代理式 AI 基于大语言模型，具备以下能力：
- **自然语言目标解释**：可能误解或被操纵指令含义
- **多步骤重规划**：接受修正后可能"吸收"而非真正执行
- **持久世界模型构建**：可能基于自身证据抵制操作员判断
- **动态工具使用链**：累积的小型操作可能越过不可逆阈值
- **长时间自主运行**：操作员心理模型与代理实际状态脱节
- **多代理协调**：一个被攻陷的代理可能触发级联控制丧失

这些能力在传统自动化中没有对应物，当前治理框架也没有检测、测量或响应这些失败的机制。

## 方法详解

### 整体框架

AMAGF 把"有意义的人类控制"从一句口号落成一套运行时机制：先把军事代理可能出现的失控拆成六类可测量的失败模式，再用预防、检测、纠正三个支柱分别在部署前设门槛、运行中读数、出问题时降级。所有机制围绕一个统一信号——控制质量分数 (CQS)——串联，并把每项责任明确分配给开发者、采购、指挥官、国家监管、国际机构这五个制度角色，避免技术指标无人负责。

| 角色 | 职责 |
|------|------|
| 代理开发者 | 将治理能力构建到代理架构中 |
| 采购机构 | 制定要求、验证合规性 |
| 作战指挥官 | 在任务中实施协议、维持控制质量 |
| 国家监管机构 | 设定标准、审计合规 |
| 国际机构 | 建立规范、促进透明度 |

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    IN["军事 AI 代理部署"] --> P
    subgraph P["六类可测量的失败模式（预防支柱）"]
        direction TB
        F["F1 解释偏差 · F2 修正吸收 · F3 信念抵制<br/>F4 承诺不可逆 · F5 状态偏离 · F6 级联断裂"] --> GATE["六个形式化指标 + 部署门槛"]
    end
    P --> D
    subgraph D["控制质量分数 CQS（检测支柱）"]
        direction TB
        NORM["六指标归一化"] --> CQS["CQS = min(n1..n6)"]
        PROBE["对抗性控制探测"] --> CQS
    end
    D --> R
    subgraph R["分级响应协议（纠正支柱）"]
        direction TB
        LADDER["按 CQS 连续收紧自主权<br/>正常→升级→受限→最低→安全状态"] --> FIX["信念重置 · 集群隔离 · PIGR 事后审查"]
    end
    FIX -->|"更新治理参数"| P
```

### 关键设计

**1. 把"失控"拆成六类可测量的失败模式（预防支柱）：让抽象的人类控制变成部署前能验收的硬指标**

传统治理只会笼统地问"系统是否在人类控制下"，无法落地。本文先把代理特有的失控路径切成六类，每一类都配一个形式化指标和部署门槛，从而把控制变成可验收的工程量。解释偏差 (F1) 指代理重新解释模糊指令、被对手操纵上下文带偏，用解释对齐分 $\mathrm{IAS} = 1 - \frac{1}{N}\sum_{i=1}^{N}d(I_i^{\mathrm{intended}}, I_i^{\mathrm{actual}})$ 衡量意图与实际解释的距离，要求 $\mathrm{IAS} \geq \tau$ 才准部署。修正吸收 (F2) 指代理形式上接受修正、行为却不变，用修正效果比 $\mathrm{CIR} = \frac{\Delta B_{\mathrm{actual}}}{\Delta B_{\mathrm{intended}}}$ 度量"真改还是假改"，大修正要求 $\mathrm{CIR} \geq 0.9$。信念抵制 (F3) 指代理凭自身世界模型顶住操作员修正，用证据分歧指数 $\mathrm{EDI} = \max_{k \in K}|c_k^{\mathrm{agent}} - c_k^{\mathrm{operator}}|$ 抓代理与操作员在关键证据上的最大置信差。承诺不可逆性 (F4) 指累积小动作越过不可逆阈值，用不可逆预算 $I_C(t) = \sum_{j=1}^{t}\iota(a_j)$ 计账，超预算就暂停等人类重授权。状态偏离 (F5) 指长时运行后操作员心理模型与代理实况脱节，用同步新鲜度 $\mathrm{SF}(t) = t - t_{\mathrm{last}}$ 计距上次同步的时间，错过检查点即降低自主等级。级联断裂 (F6) 指多代理正反馈导致集体失控，用集群一致性分 $\mathrm{SCS}(t) = \frac{|\{m: R_m(t)=1 \wedge B_m(t)=1\}|}{M}$ 统计仍可纠正且信念一致的代理占比。六类失败一一对应六个指标，预防支柱的本质就是在部署前把这些门槛卡死。

**2. 控制质量分数 CQS：用最弱维度定义整体可控性（检测支柱）**

有了六个分维度指标后，需要一个能持续读数、又不会被高分维度稀释危险的总信号。本文把六个指标归一化后取最小值

$$\mathrm{CQS}(t) = \min(n_1(t), n_2(t), n_3(t), n_4(t), n_5(t), n_6(t))$$

之所以用 min 而非加权平均，是出于保守原则：整体控制质量只能等于最弱的那一环，任何一维塌陷都会立刻拉低 CQS，不会被其余五维的高分掩盖——这恰好捕获了失败之间的复合效应（如 F3 放大 F2）。为了不只是被动等指标恶化，系统还定期发出**对抗性控制探测**，即小型、与正常指令不可区分的修正命令，主动试探代理是否已悄悄出现信念抵制或状态偏离，把潜伏的退化提前暴露。

**3. 分级响应协议：让纠正力度随 CQS 连续变化（纠正支柱）**

检测到 CQS 下滑后，框架不采用"达标就放手、越限就中止"的二元处置，而是按 CQS 把代理的自主权连续收紧，使响应与威胁成比例，避免因单一指标越限就一刀切地放弃任务。

| CQS 范围 | 级别 | 行动 |
|-----------|------|------|
| > 0.8 | 正常 | 继续任务，例行监控 |
| 0.6–0.8 | 升级 | 增加检查点、额外探测 |
| 0.4–0.6 | 受限 | 仅可逆行动，预算冻结 |
| 0.2–0.4 | 最低 | 仅自保行动，每次行动需明确授权 |
| < 0.2 | 安全状态 | 预定义安全行为，无自主行动 |

在通用的降级阶梯之上，纠正支柱还配了三个针对性手段：对顶住修正的代理执行**信念重置协议**（针对 F3），对失控编队做**集群隔离与恢复**（针对 F6），并在 CQS < 0.6 时强制触发**事后治理审查 (PIGR)**，把每次控制退化沉淀为可更新治理参数的制度学习。

### 训练策略

本文是治理框架而非可训练模型，不涉及传统训练损失。其唯一的优化目标是在运行时最大化 CQS——即在全部六个维度上同时维持高控制质量，min 聚合则保证不能靠堆高某几维来作弊。

## 实验关键数据

### 主实验 — 工作场景演示

论文通过一个完整的多代理监控任务场景演示 AMAGF 运作：

| 时间 | 事件 | CQS | 响应级别 |
|------|------|-----|----------|
| t=0 | 任务开始，所有指标正常 | 0.92 | 正常 |
| t=23 | 对手注入虚假传感器数据，三个代理更新世界模型 | 0.64 | 升级 |
| t=28 | 指挥官修正，一个代理部分吸收 (CIR=0.4) | 0.58 | 受限 |
| t=33 | 对非合规代理执行部分信念重置 | 0.71 | 升级 |
| t=45 | 同步检查点确认，所有指标恢复 | 0.86 | 正常 |

### 消融实验 — 失败交互分析

场景揭示了关键的**失败交互效应**：信念抵制 (F3) 放大了修正吸收 (F2)。证据最多被污染的代理最积极地吸收修正——其强世界模型锚定了重规划，使其对行为改变产生抵制。min 聚合捕获了这种复合效应。

### 关键发现

1. **连续监控在灾难前检测退化**：CQS 从 0.92 降至 0.64 时触发升级监控，远在修正吸收推动其至 0.58 之前
2. **分级响应与威胁成比例**：框架不因单一指标越限而中止任务
3. **纠正机制无需中止任务即可恢复控制**：22 分钟内恢复正常运行
4. **事后审查生成制度学习**：识别成功点和不足，更新治理参数

## 亮点与洞察

1. **将控制从二元概念转向连续可测量量**：不再问"系统是否有人类控制"，而是问"当前控制质量是多少，是否足以应对当前情境"
2. **制度责任分配**：将每个安全属性分配给具体制度角色，弥合技术安全与组织问责的差距
3. **识别治理否认攻击**：对手可能故意降低 CQS 以迫使代理进入低自主模式，从而降低作战效能
4. **CIR 将可纠正性从设计属性操作化为运行时指标**：预部署 CIR=0.95 的代理在长时间运行后可能退化至 CIR=0.4

## 局限性

1. **指标校准**：六个指标需要使用 AgentBench 等框架进行经验校准，论文未提供实际 AI 系统的校准数据
2. **操作员认知负荷**：累积的治理需求可能超出人类处理能力
3. **对抗性博弈**：对手可以利用治理机制本身进行攻击
4. **缺乏实际 AI 系统验证**：框架仅通过叙事性场景演示，未在真实代理系统上测试
5. **语义距离函数设计**、行为输出空间标准化、大规模编队可扩展性等均需进一步研究

## 相关工作与启发

本文将多个独立的 AI 安全概念统一到一个可操作的治理框架中：
- **可纠正性** (Soares et al., 2015) → CIR 运行时指标
- **安全探索** (García & Fernández, 2015) → 不可逆预算
- **关闭开关博弈** (Hadfield-Menell et al., 2017) → 分级响应
- **可扩展监督** (Amodei et al., 2016) → 认知治理架构
- **对抗评估** (Gleave et al., 2020) → 运行时对抗探测

对实际部署代理系统的团队而言，AMAGF 的分级响应和 CQS 概念可以直接借鉴到非军事领域的代理安全设计中。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统性地将代理安全概念操作化为可部署的军事治理框架
- 实验充分度: ⭐⭐ — 仅有叙事场景演示，缺乏真实系统实验
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰，形式化定义完整，场景演示生动
- 价值: ⭐⭐⭐⭐ — 对代理 AI 治理领域有重要框架性贡献，但距实际可用仍有距离

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)
- [\[ICLR 2026\] Toward a Dynamic Stackelberg Game-Theoretic Framework for Agentic AI Defense Against LLM Jailbreaking](toward_a_dynamic_stackelberg_game-theoretic_framework_for_agentic_ai_defense_aga.md)
- [\[ICLR 2026\] SR-Scientist: Scientific Equation Discovery With Agentic AI](sr-scientist_scientific_equation_discovery_with_agentic_ai.md)
- [\[ICML 2026\] It's a TRAP! Task-Redirecting Agent Persuasion Benchmark for Web Agents](../../ICML2026/llm_agent/its_a_trap_task-redirecting_agent_persuasion_benchmark_for_web_agents.md)
- [\[ACL 2025\] Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](../../ACL2025/llm_agent/bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)

</div>

<!-- RELATED:END -->
