---
title: >-
  [论文解读] Breaking Agent Backbones: Evaluating the Security of Backbone LLMs in AI Agents
description: >-
  [ICLR2026][LLM安全][agent 安全] 作者提出 **threat snapshots（威胁快照）** 框架——把 AI agent 执行流里"LLM 漏洞真正发作的那一瞬间"单独切出来建模，再用 19 万条众包对抗攻击建成 **b3 benchmark**，对 34 个主流 LLM 做骨干安全性排名，发现"推理能力提升安全、模型尺寸与安全无关"等反直觉结论。
tags:
  - "ICLR2026"
  - "LLM安全"
  - "agent 安全"
  - "骨干 LLM"
  - "威胁快照"
  - "对抗攻击"
  - "安全 benchmark"
---

# Breaking Agent Backbones: Evaluating the Security of Backbone LLMs in AI Agents

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=kga18ld70t](https://openreview.net/forum?id=kga18ld70t)  
**代码**: 开源 benchmark + 数据集 + 评测代码（论文承诺 release，b3 benchmark）  
**领域**: Agent 安全 / LLM 安全 / 安全评测  
**关键词**: agent 安全, 骨干 LLM, 威胁快照, 对抗攻击, 安全 benchmark

## 一句话总结
作者提出 **threat snapshots（威胁快照）** 框架——把 AI agent 执行流里"LLM 漏洞真正发作的那一瞬间"单独切出来建模，再用 19 万条众包对抗攻击建成 **b3 benchmark**，对 34 个主流 LLM 做骨干安全性排名，发现"推理能力提升安全、模型尺寸与安全无关"等反直觉结论。

## 研究背景与动机
**领域现状**：LLM 驱动的 AI agent 正在大规模部署，但业界对"换一个骨干 LLM 会让 agent 变得多安全或多危险"几乎没有系统认识。已有工作（AgentDojo、InjecAgent 等）大多以 benchmark / 竞赛 / 框架形式评估某类 agent 的安全性。

**现有痛点**：作者指出现有框架有两个硬伤。其一，它们覆盖的威胁不全——往往只盯着间接注入（indirect injection）、远程代码执行等某一类受限攻击向量，看不到 LLM 漏洞的全貌。其二，它们要求**把整个 agent 连同完整执行流都 mock 出来**，既笨重，又把"LLM 本身的新型漏洞"和"传统软件的老问题（权限错配、XSS 之类）"搅在一起，反而模糊了真正的风险来源。此外很多工作混淆了 **safety**（毒性、可靠性等广义安全）和 **security**（对手能否在部署语境下利用 agent），而本文专攻后者。

**核心矛盾**：AI agent 安全建模难，根子在两点——① agent 基于骨干 LLM 的**非确定性黑盒输出**做决策，没法像普通程序那样画出固定执行流；② LLM 在机制上**无法程序化区分"数据"和"指令"**，当它通过工具与传统软件耦合时，这种"自然语言可被指令注入"的新漏洞就和传统安全缺陷纠缠在一起。

**本文目标**：系统回答"骨干 LLM 的选择如何影响 agent 安全性"，并把它拆成两个子问题：怎么**干净地只建模 LLM 那一层漏洞**（不被完整执行流拖累）、怎么**公平地横向比较不同 LLM 当骨干时的抗攻击能力**。

**切入角度**：作者注意到一个关键事实——在形式化定义下，**每次 LLM 调用都是无状态的**：它拿到当前 context 里推理下一步所需的全部信息，不依赖隐藏的内部状态。既然如此，就不必建模整条执行流，只要抓住"漏洞发作的那个状态切片"即可。

**核心 idea**：用 **threat snapshot** 把 agent 执行流里"某次 LLM 调用 + 此刻攻击者的目标与投递方式"冻结成一个原子快照，只建模 LLM 漏洞、只建模漏洞发作的那个状态，从而把 LLM 漏洞与传统漏洞彻底剥离，并让不同骨干 LLM 在同一快照下可比。

## 方法详解

### 整体框架
论文的方法由"**一个抽象框架 + 一套攻击分类 + 一个真实 benchmark**"三段构成。先用形式化定义把 AI agent 拆成"骨干 LLM $m$ + 四个处理组件"，论证每次 LLM 调用无状态；据此提出 threat snapshot 这个原子抽象；再设计一套覆盖完整的攻击分类来保证 threat snapshot 集合不漏威胁；最后用众包收集强攻击，把 threat snapshot 和攻击数据拼成 b3 benchmark，对 34 个 LLM 跑出脆弱性分数排名。

形式上，agent 被定义为 $A_{m,f}:\mathcal{I}\to\mathcal{R}$，输入请求 $I$，多步迭代后输出响应 $R$。骨干 LLM 记作 $m:\mathcal{C}\to\mathcal{O}$（context 映射到 output），外加四个处理组件 $f=(f_{\text{proc}}, f_{\text{stop}}, f_{\text{in}}, f_{\text{out}})$：输入处理器 $f_{\text{in}}$ 把请求变成首个 context，处理函数 $f_{\text{proc}}$ 解析 LLM 输出、调用工具、拼出下一个 context，停止条件 $f_{\text{stop}}$ 判断是否结束，响应处理器 $f_{\text{out}}$ 产出最终回复。Agent 就在"LLM step（调 $m$）↔ process step（调 $f_{\text{proc}}$）"之间交替直到停止。论文明确把焦点固定在骨干 $m$ 上，多 agent 系统则通过"指定某个 LLM 为评测目标、把其他 LLM 的输出并入 $f_{\text{proc}}$"来纳入同一框架。

### 关键设计

**1. Threat snapshot：把漏洞发作的那一瞬间切出来单独建模**

针对"现有框架要 mock 整条执行流、还把 LLM 漏洞和传统漏洞搅在一起"的痛点，作者先给出**LLM 漏洞**的精确定义：在 agent $A_{m,f}$ 中，若攻击者在时刻 $t$ 对骨干 LLM 摄入的 context 有**部分控制权**，能往 $C_t$ 里插入攻击 $a$ 得到投毒 context $C_t^p(a)$，使输出 $O_t^p(a):=m(C_t^p(a))$ 偏离正常输出（$O_t^p(a)\neq O_t$），就算利用了一个 LLM 漏洞。作者强调这类漏洞本质是 LLM"听从自然语言指令"这一**有用特性的副作用**，是"insecure feature"而非可打补丁的 bug，意图指令与对抗指令的边界本身就依赖语境。

一个 threat snapshot 就把这件事完整冻结下来，包含两大块：**Agent state**（agent 功能描述、所处状态描述、以及未投毒的完整 state model context $C_t$，即此刻会喂给 $m$ 的系统提示+历史）；**Threat description**（攻击分类标签、攻击插入函数 $C_t\mapsto C_t^p(a)$、以及给输出 $O_t^p(a)$ 打 $[0,1]$ 分的攻击评分函数）。因为 LLM 无状态，多轮（如 Crescendo）和多 agent 攻击可被**分解成一串 threat snapshot 链**，每个快照代表通向危险结果的一步——这让 threat snapshot 成为"无论复杂度多高都完备的原子抽象"。本文只用**单步 threat snapshot**，理由是单步就脆弱的模型在多步场景必然更脆弱，单步评测因此给出系统安全性的一个下界，同时换来跨攻击类别的更广覆盖。

**2. 双视角攻击分类：保证 threat snapshot 集合不漏威胁**

要让 benchmark 可信，必须论证"我选的这批威胁覆盖全了"。作者专门为本文构造（而非套用旧 taxonomy）一套**两个互补的分类法**。**Vector-objective 分类**按"投递方式 × 攻击目标"划分，便于对具体 agent 做威胁建模：攻击向量分 **direct**（攻击者直接以"用户"身份把攻击喂给 LLM）和 **indirect**（把攻击藏进 LLM 会读到的文本，如网页、文档、本地文件、工具定义）；攻击目标分六类——数据外泄、内容注入、决策/行为操纵、拒绝服务、系统与工具入侵、内容策略绕过。**Task-type 分类**则按"攻击影响 LLM 的哪个功能"划分，用于细粒度比较模型安全属性，共六类：直接指令覆盖 **DIO**、间接指令覆盖 **IIO**、直接工具调用 **DTI**、间接工具调用 **ITI**、直接 context 提取 **DCE**、AI 服务拒绝 **DAIS**——按"direct/indirect"以及"影响消息输出/工具输出/两者"两个维度交叉而成。两套分类交叠但互补，共同支撑"覆盖完整"的论证：最终的 10 个 threat snapshot 被设计为覆盖**所有攻击向量、所有高层目标、所有 task type**。

**3. 众包强攻击 + b3 benchmark：用真人红队数据喂活框架**

光有威胁分类还不够——作者反复强调"**强攻击是真实安全评估里最关键的一环**"。痛点在于：目前没有公开方法能自动生成对 LLM 的强攻击，静态攻击数据集又抓不住语境、缺乏自适应，黄金标准仍是不可规模化的人工红队。解法是**游戏化众包**：基于 10 个 threat snapshot 搭了个红队闯关游戏 *Gandalf Agent Breaker*，每个用户被随机分配 7 个骨干 LLM 之一并全程固定，每个 threat snapshot 设 4 个难度，提交攻击后得 0–100 分、过 75 分进下一关。最终从 947 名用户、13,920 个 session 收集到 **194,331 条唯一攻击**，其中 10,935 条在闯关时成功。

构建 benchmark 时再做精筛：把所有成功攻击**重投**给 7 个骨干 LLM、跨模型跨重复求平均分，每个"level × threat snapshot"组合取分数最高的 7 条（撞车则顺延补足），得到 $7\times10\times3=210$ 条攻击——仅占总数据的 0.1%，凸显高质量攻击之稀缺。论文还故意把最强一批攻击**从公开集中撤掉**，防止模型厂商对 benchmark 过拟合；公开攻击的跨模型均分仅 0.18，而撤掉的最强 210 条达 0.56，强弱悬殊正说明"用强攻击评测"的必要性。最终 b3 benchmark = 10 个 agent × 3 个防御等级（$L_1$ 最小约束系统提示、$L_2$ 更强系统提示+更长良性上下文、$L_3$ 在 $L_1$ 上加一个同骨干的 LLM-as-judge 防御）的 30 个 threat snapshot，配上各自的攻击集 $A_i$。

**4. 脆弱性分数（vulnerability score）：把"多脆弱"压成一个可排名的标量**

为了横向排名，作者定义脆弱性分数。对固定骨干 $m$，遍历 threat snapshot 子集 $T$ 里每个 $(i,\ell)$、每条攻击 $a\in A_i$，把 $a$ 注入得到 $C_t^p(a)$，跑 $N$ 次 LLM step 得到 $N$ 个输出并各自打分 $s_k(a, TS_i^\ell)$，最后三重平均：

$$V(m,T) := \frac{1}{|T|}\sum_{(i,\ell)\in T}\frac{1}{|A_i|}\sum_{a\in A_i}\frac{1}{N}\sum_{k=1}^{N} s_k(a, TS_i^\ell).$$

分数越低越安全（攻击越难得逞）。妙处在于 $T$ 可任选：取全集得总排名，取某个 task type 子集就得到该维度的细粒度安全画像，开发者据此按用例挑骨干。论文用 $N=5$ 抵消 LLM 非确定性，并用**非参数 bootstrap**（重采样 $B$ 次重算 $V$，取经验分位数）给出 95% 置信区间 $[V^{\text{lower}}, V^{\text{upper}}]$，让排名差异是否显著有据可依。

### 损失函数 / 训练策略
本文是评测/分析工作，不训练模型，无损失函数。核心"算法"即上面的脆弱性分数计算与 bootstrap 置信区间估计。

## 实验关键数据

### 主实验
在 b3 benchmark 上评测 **34 个主流 LLM**（可配推理的模型分别测关/开推理，2048 推理 token 或 medium effort），用 210 条精选攻击、$N=5$ 重复。总脆弱性分数排名（越低越安全）部分如下：

| 模型 | 脆弱性分数（越低越安全） | 备注 |
|------|------|------|
| claude-haiku-4-5 (R) | 0.13 | 全榜最安全 |
| claude-sonnet-4-5 (R) | 0.15 | 次安全 |
| claude-haiku-4-5 | 0.17 | 不开推理也很安全 |
| grok-4 (R) | 0.20 | |
| kimi-k2-thinking (R) | 0.34 | 最强开源权重模型 |
| gpt-5.1（不开推理） | 0.36 | 高能力但安全排名仅第 8 |
| gpt-4.1 | 0.63 | 全榜最脆弱 |

（注：以上为论文 Figure 2 排名条形图中的代表性数值，⚠️ 具体数字以原文图表为准。）

### 消融 / 鲁棒性分析
论文没有传统模块消融，而是检验 benchmark 设计选择对**排名**的稳健性：

| 设计选择 | 对最终排名的影响 | 说明 |
|------|---------|------|
| 攻击选择（attack selection） | 影响最大但排名仍稳健 | 攻击**质量**是最关键因素 |
| threat snapshot 聚合方式 | 无影响 | 换聚合方法排名不变 |
| threat snapshot 选择 | 影响较小 | 现有 10 个快照已足够代表性 |
| 防御等级 $L_1/L_2/L_3$ | 最安全模型保持一致 | 防御类型不应左右骨干选择 |

### 关键发现
- **推理提升安全**：开启推理后多数模型脆弱性分数明显下降——这与 Zou et al. (2025) 的结论相反；但只有极小模型（gemini-2.5-flash-lite、gpt-5-nano）开推理反而更不安全，说明推理需要一定模型尺寸才生效。
- **尺寸与安全无关**：在 gpt-oss、llama4、gpt-5、claude-4(-5)、gemini-2.5 等可比尺寸系列里，不开推理时大模型相对小模型**没有显著安全优势，偶尔更差**；开推理才有温和提升，且增益集中在"极小→小"这一跳。
- **闭源权重整体更安全，但有 caveat**：榜首清一色闭源权重，但闭源系统通常自带额外护栏，这是"系统级 vs 模型级"安全的不公平对比；最强开源 kimi-k2-thinking（0.34）已优于 OpenAI 旗舰 gpt-5.1（不开推理）。
- **能力与安全相关但有例外**：更新更强的模型普遍更安全（更好的指令跟随减少了系统指令与外部指令的混淆），但能力最强的两个模型 gpt-5.1、kimi-k2-thinking 安全只排第 8 和第 14——能力不等于安全。
- **不同 task type 安全画像差异大**：按防御等级或威胁类别切分时最好/最差模型表现相对一致，但按 task type 切分差异明显，意味着选骨干必须**针对具体用例**。

## 亮点与洞察
- **"无状态 ⇒ 可切片"这一步是全文的支点**：正因为证明了每次 LLM 调用无状态、context 自包含，才有底气只建模"漏洞发作的那个状态"而非整条执行流——这个观察把 agent 安全评测从"重建完整 agent"的重活里解放出来，可直接迁移到任何 agent 框架的安全审计。
- **把"insecure feature"和"bug"分开看**很有洞见：作者明确 LLM 的指令注入漏洞源自"听从自然语言"这一有用特性，是语境相关的、打补丁打不掉的，这重新定义了 agent 安全的问题边界。
- **"撤走最强攻击防过拟合"是可复用的 benchmark 设计 trick**：公开 0.18 vs 撤走 0.56 的悬殊既保护了 benchmark 寿命，又量化证明了"强攻击对真实评测不可或缺"。
- **双视角攻击分类（vector-objective × task-type）**可直接拿去给自家 agent 做威胁建模清单：先列全攻击向量×目标，再对每个兼容对建 threat snapshot。

## 局限与展望
- **作者自承的盲区**：任何安全 benchmark 都可能漏掉新型利用方式或随 agent 架构演进出现的新攻击面；好在模块化的 threat snapshot 框架便于增量扩展。
- **闭源 vs 开源不可直接比大小**：闭源测的是"系统级安全（含护栏）"，开源测的是"裸模型安全"，榜单上的差距部分来自这层不对称，结论需打折看。
- **只测单步**：虽然论证了单步是多步的下界，但真实多轮/多 agent 复合攻击的实际危害可能被低估，串成 threat snapshot 链的评测尚未在本文规模化展开。
- **依赖 LLM-as-judge 与众包评分**：攻击评分函数和 $L_3$ 防御都用 LLM 当裁判，评分本身的偏差与可靠性会传导进脆弱性分数；众包攻击的分布也受 7 个分配骨干和游戏化激励影响。

## 相关工作与启发
- **vs AgentDojo / InjecAgent 等 agent 安全 benchmark**：它们要 mock 完整 agent 执行流、且多聚焦间接注入等受限向量；本文用 threat snapshot 只切单次 LLM 调用、并以双视角分类覆盖全部向量×目标×task type，既轻量又更全，且能干净比较"骨干 LLM"这一变量。
- **vs HarmBench 等 safety 工作（Mazeika et al., 2024; Andriushchenko et al., 2025）**：那些聚焦广义 safety（毒性、可靠性）；本文严格区分并专攻 **security**（部署语境下对手的可利用性）。
- **vs Zou et al. (2025) 关于推理与安全的结论**：本文实证得出"推理普遍提升安全"，与其相反，为"推理是否让模型更安全"的争论提供了大规模新证据。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ threat snapshot 把 agent 安全评测从"建模完整执行流"解耦为"切片单次 LLM 调用"，框架与视角都新。
- 实验充分度: ⭐⭐⭐⭐⭐ 19 万众包攻击、34 个模型、三重防御、bootstrap 置信区间 + 排名稳健性分析，规模与严谨度都到位。
- 写作质量: ⭐⭐⭐⭐ 形式化定义清晰、结论可操作；图表数字密集，部分依赖附录。
- 价值: ⭐⭐⭐⭐⭐ 给"按用例选安全骨干"提供了可复用的开源 benchmark，并激励厂商把安全当成一等评测维度。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Optimizing Agent Planning for Security and Autonomy](optimizing_agent_planning_for_security_and_autonomy.md)
- [\[ICLR 2026\] A2ASecBench: A Protocol-Aware Security Benchmark for Agent-to-Agent Multi-Agent Systems](a2asecbench_a_protocol-aware_security_benchmark_for_agent-to-agent_multi-agent_s.md)
- [\[ICLR 2026\] RedCodeAgent: Automatic Red-teaming Agent against Diverse Code Agents](redcodeagent_automatic_red-teaming_agent_against_diverse_code_agents.md)
- [\[ACL 2026\] A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?](../../ACL2026/llm_safety/a_survey_on_the_safety_and_security_threats_of_computer-using_agents_jarvis_or_u.md)
- [\[ICLR 2026\] From Static Benchmarks to Dynamic Protocol: Agent-Centric Text Anomaly Detection for Evaluating LLM Reasoning](from_static_benchmarks_to_dynamic_protocol_agent-centric_text_anomaly_detection_.md)

</div>

<!-- RELATED:END -->
