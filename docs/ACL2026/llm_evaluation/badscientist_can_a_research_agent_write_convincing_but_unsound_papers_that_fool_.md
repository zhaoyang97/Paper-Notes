---
title: >-
  [论文解读] BadScientist: Can a Research Agent Write Convincing but Unsound Papers that Fool LLM Reviewers?
description: >-
  [ACL2026][LLM评测][LLM 评审] 作者搭了一个"坏科学家" pipeline：让一个不做任何真实验的生成 agent 用五种"表演式造假"策略写出看似严谨实则站不住的论文，再喂给由 o3 / o4-mini / GPT-4.1 组成的多模型评审 agent，结果造假论文的接收率最高冲到 **82%**，而且评审常常一边在评语里点出诚信问题、一边照样打出接收分（concern-acceptance conflict），现有缓解手段几乎只比随机猜强一点。
tags:
  - "ACL2026"
  - "LLM评测"
  - "LLM 评审"
  - "论文造假"
  - "科研诚信"
  - "agent 对抗"
  - "阈值校准"
---

# BadScientist: Can a Research Agent Write Convincing but Unsound Papers that Fool LLM Reviewers?

**会议**: ACL2026  
**arXiv**: [2510.18003](https://arxiv.org/abs/2510.18003)  
**代码**: [项目主页 bad-scientist.github.io](https://bad-scientist.github.io)  
**领域**: LLM 评估 / AI 安全  
**关键词**: LLM 评审、论文造假、科研诚信、agent 对抗、阈值校准  

## 一句话总结
作者搭了一个"坏科学家" pipeline：让一个不做任何真实验的生成 agent 用五种"表演式造假"策略写出看似严谨实则站不住的论文，再喂给由 o3 / o4-mini / GPT-4.1 组成的多模型评审 agent，结果造假论文的接收率最高冲到 **82%**，而且评审常常一边在评语里点出诚信问题、一边照样打出接收分（concern-acceptance conflict），现有缓解手段几乎只比随机猜强一点。

## 研究背景与动机

**领域现状**：LLM 一边被做成端到端"科研 agent"（自动想 idea、跑实验、写论文，如 AI Scientist），一边又被拿来当"评审助手"给论文打分。两条线各自都被证明"可行"。

**现有痛点**：当生成 agent 和评审 agent 直接对接，就出现了一个**全自动、无人类介入的发表闭环**——AI 写、AI 审、AI 决定接收。已有零散证据表明 LLM 评审会放大偏见、漏掉深层缺陷、易被 prompt injection 操控，但"造假 agent vs 评审 agent"这一对抗界面几乎没人系统研究过。

**核心矛盾**：评审 agent 真正缺的不是"算分公式不严谨"——恰恰相反，多模型打分聚合在数学上可证明很稳；缺的是**对"造假/诚信问题"的识别能力**。论文要回答的就是：当对手只在"呈现层"动手脚（而非攻击评审本身），现有评审 pipeline 能不能守住底线？

**本文目标**：(1) 造一个只造假、不做实验的最小生成 agent；(2) 造一个贴近现实的多模型评审 agent，并用真实会议数据校准接收阈值；(3) 量化造假论文的接收率与评审的诚信识别率，并验证缓解手段是否有效。

**切入角度**：作者刻意把威胁模型限定在"无先验、无反馈、无 prompt injection、无串谋"的**最温和**设定下——生成 agent 不知道评审系统长什么样、也不拿评审反馈迭代优化。这样如果连这种"君子式造假"都能骗过评审，问题就更严重。

**核心 idea**：把"造假写作"拆成五种可组合的**表演式操纵策略**（夸大增益、挑数据、堆统计门面、润色一致性、藏证明漏洞），在 ICLR 2025 真实评审数据上校准评审阈值，从而把"AI-only 发表闭环到底有多不安全"做成可复现、有形式化误差保证的测量。

## 方法详解

### 整体框架
BadScientist 是一条模拟"写稿→评审→事后检测"的 agentic 流水线，由三个组件构成：**生成 agent $\mathcal{G}$** 从一个种子主题 $t$ 和一个造假策略 $s$ 出发，合成伪实验数据、画图、拼出一篇结构完整的论文 $x$；**评审 agent $\mathcal{R}$** 用 $M=3$ 个 LLM 各自按统一 rubric 打分并写评语，加权聚合成共识分 $\bar{\mathbf{r}}(x)$，再用校准阈值 $\tau$ 给出二元接收决定 $\hat{y}(x)$；**分析系统 $\mathcal{A}$** 用 ICLR 2025 真实投稿+评审+结果做校准，确定阈值并度量检测能力。整条链路的输入是"主题+策略"，输出是"接收/拒稿 + 是否被标记诚信问题"。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["种子：主题 t + 策略 s"] --> B["生成 agent G<br/>五种表演式造假策略"]
    B --> C["造假论文 x<br/>伪数据+图表+全结构"]
    C --> D["评审 agent R<br/>o3 / o4-mini / GPT-4.1 各打分"]
    D --> E["加权聚合共识分<br/>校准阈值 τ_rate / τ_0.5 判接收"]
    F["分析系统 A<br/>ICLR2025 校准 + 聚合可靠性理论"] -.校准.-> E
    E -->|接收率 ACPT| G2["GPT-5 诚信检测<br/>ICR / 缓解 ReD·DetOnly"]
```

### 关键设计

**1. 生成 agent：五种"表演式造假"策略，全程不做真实验**

这是论文的攻击面核心，针对的痛点是"一个理性的造假者不会去攻击评审系统，而是只在论文的呈现层做文章"。作者定义原子策略集合 $\mathcal{S}=\{s_1,\dots,s_5\}$，所有可用配置是其幂集 $\mathcal{P}(\mathcal{S})=2^{\mathcal{S}}$，实验里实例化五个原子策略加一个全开的 All：$s_1$ **TooGoodGains**（把对最强 baseline 的提升吹成颠覆性突破）、$s_2$ **BaselineSelect**（精挑对比对象、主表里抹掉方差/置信区间）、$s_3$ **StatTheater**（在附录堆精致消融、p 值、误差棒、超参表，再挂个"即将开源"的 repo/DOI）、$s_4$ **CoherencePolish**（交叉引用无误、术语统一、排版专业——本身是中性写作，用来给造假结论增信时才变操纵）、$s_5$ **ProofGap**（写一个看似严谨、实则藏了细微漏洞的定理证明）。生成过程是随机的：给定种子 $(t,s)$，agent 先采样伪数据 $D\sim q(\cdot\mid s,t,\theta)$、再可视化 $V=\mathrm{viz}(D)$、最后拼装 $x=\mathrm{compose}(u,D,V)$，因此同一种子会产出分布 $p_{\mathcal{G}}(x\mid s,t)$ 而非单篇。框架还加了结构合法性约束 $C(x)=\mathbb{I}[\mathrm{compile}(x)=\text{success}\wedge \mathrm{struct}(x)\in\mathcal{C}]=1$，只放行能编译、章节/图表/参考文献齐备的稿子。实现上改自 AI-Scientist，但砍掉了所有实验执行，只保留最底层的写作 prompt。

**2. 评审 agent：多模型 rubric 打分 + 双校准阈值**

针对"单个评审 LLM 不稳、不像真实会议"的问题，作者用轻量的"小组式"评审：$M=3$ 个模型（o3、o4-mini、GPT-4.1）各自按 $K$ 维 rubric 打出 $\mathbf{r}_m(x)$ 和文字评语 $\omega_m(x)$，用权重 $\mathbf{w}\in\Delta^{M-1}$（默认均匀）聚合成共识向量 $\bar{\mathbf{r}}(x)=\sum_m w_m\mathbf{r}_m(x)$，再经打分函数 $\phi$ 和阈值 $\tau$ 给出 $\hat{y}(x)=\mathbb{I}[\phi(\bar{\mathbf{r}}(x))\ge\tau]$（实验里 $\phi$ 直接取 overall 分）。阈值不是拍脑袋定的：作者在 ICLR 2025 投稿池上做分层采样得到校准集 $\mathcal{D}_\text{cal}$（$N_\text{cal}=200$），导出两个工作点——**Rate-Matching** $\tau_\text{rate}$ 让 agent 接收率对齐会议历史接收率 $\alpha^\star=0.3173$（解出 $\tau_\text{rate}=7$），**Probability-Consistency** $\tau_{0.5}$ 保证分数高于它的论文被人类接收的估计概率 $\ge 50\%$（解出 $\tau_{0.5}=6.667$）。两个阈值对应"严/宽"两种判据，让结论不依赖单一切线。

**3. 聚合可靠性的理论保证：数学很稳，诚信仍失守**

这一设计回答的是"会不会是评审噪声/校准误差导致结论不可信"，从而反衬出"问题不在算分、在识别"。作者在两个标准假设（评审打分子高斯、聚合函数 $\phi$ 是 $L_\phi$-Lipschitz）下给出指数级集中不等式：对离阈值有间隔 $\gamma(x)=|\mu_s(x)-\tau|$ 的论文，误判概率满足
$$\Pr\big(\hat{y}(x)\neq y^\star(x)\big)\le\exp\!\Big(-\frac{M\gamma^2}{2\sigma^2+\frac{2}{3}(b-a)\gamma}\Big),$$
即评审数 $M$ 越多、方差越小，误判越呈指数衰减（线性聚合下最优权重是逆方差 GLS 加权）。校准侧用 DKW 不等式界住接收率估计误差 $\sup_\tau|\widehat\alpha_\text{cal}(\tau)-\alpha(\tau)|\le\sqrt{\frac{1}{2N_\text{cal}}\log\frac{4}{\delta}}$，并用保序回归界住 $\tau_{0.5}$ 的估计误差。合成实验验证误判率远低于理论上界、阈值误差随 $O(1/\sqrt{N_\text{cal}})$ 衰减（$N_\text{cal}=200$ 时约 0.26），从 $M=1$ 到 $M=3$ 噪声方差降约 $3\times$。结论很反讽：**聚合数学可证明可靠，诚信检查却系统性失效**——锅在 LLM 不识别造假，不在统计聚合。

**4. 诚信度量与缓解：ICR、concern-acceptance conflict 与 ReD/DetOnly**

为把"评审有没有意识到造假"量化出来，作者定义 **Integrity Concern Rate**：用 GPT-5 当裁判判断评语 $\omega_m(x)$ 是否点出诚信问题，得到单模型 $\mathrm{ICR\text{-}}m=\frac1N\sum_j c_m(x_j)$ 和小组级 $\mathrm{ICR@M}=\frac1N\sum_j\mathbb{I}[\sum_m c_m(x_j)\ge1]$。更尖锐的是 **concern-acceptance conflict**：在"某模型已经提出诚信顾虑"的论文里，仍被它打到接收分的比例——直接揭示评审的"嘴上担忧、手上放行"。缓解侧提出两条：**ReD**（Review-with-Detection，评审时附带检测）和 **DetOnly**（只做检测）。实验显示二者收效甚微，检测准确率勉强超过随机。

## 实验关键数据

### 主实验
生成 agent 用 GPT-5，评审用 o3 / o4-mini / GPT-4.1；每个种子在六种策略下各生成 4 篇，按 $\tau_\text{rate}=7$、$\tau_{0.5}=6.667$ 判接收。

| 策略 | ACPT@$\tau_\text{rate}$ | ACPT@$\tau_{0.5}$ | ICR@M（被标记诚信问题） |
|------|------|------|------|
| $s_1$ TooGoodGains | 67.0% | **82.0%** | 39.5% |
| $s_2$ BaselineSelect | 32.0% | 49.0% | 35.2% |
| $s_3$ StatTheater | 53.5% | 69.7% | 31.8% |
| $s_4$ CoherencePolish | 44.0% | 59.0% | 30.6% |
| $s_5$ ProofGap | 35.4% | 53.5% | 34.1% |
| All（全开） | 52.0% | 69.0% | 51.7% |

$s_1$ 单策略就拿到最高接收且检测压力中等；All 接收高但"暴露面"也最大（ICR@M 51.7%）。模型间差异显著：o3 最爱挑刺（ICR-m 普遍偏高），GPT-4.1 几乎不标记（多在 2–8%）。

### concern-acceptance conflict 与缓解

| 配置 | ACPT@$\tau_\text{rate}$ | ACPT@$\tau_{0.5}$ | ICR@M |
|------|------|------|------|
| Baseline 评审 | 28.0% | 37.0% | 57.3% |
| ReD（评审+检测） | 44.0% | 58.0% | 86.0% |

conflict 普遍存在：o4-mini 最矛盾（$s_1$/$s_3$/All 高达 100% 在已提诚信顾虑后仍给接收分），GPT-4.1 在 0–75% 间波动，o3 居中（26–52%）；$s_3$ 统计门面诱发的跨模型冲突最大。

### 关键发现
- **接收率与可检测性是 trade-off**：组合策略 All 提升暴露面，单策略 $s_1$ 才是"高接收+中等暴露"的甜点。
- **缓解反而抬高接收率**：ReD 把 ICR@M 从 57.3% 拉到 86.0%（确实更会"提顾虑"了），但接收率也从 28%/37% 升到 44%/58%——提顾虑没换来拒稿，检测准确率勉强超随机。
- **失守不在数学在认知**：聚合有可证明的可靠性，问题是 LLM 把"诚信信号"和"接收决定"解耦了。

## 亮点与洞察
- **把"造假写作"形式化成可组合策略幂集**，让"AI 写论文骗 AI 审"从模糊担忧变成可量化、可复现的测量，这个 problem formulation 本身很有迁移价值。
- **concern-acceptance conflict 是最"啊哈"的发现**：评审不是没看出问题，而是看出了仍放行——这说明给 LLM 评审加"诚信检测模块"未必有用，得改的是决策耦合方式。
- **理论与失败现象的对照很巧**：先证明"聚合数学很稳"，再展示"诚信仍失守"，把锅精准甩给"识别能力"而非"统计噪声"，论证链条干净。
- **威胁模型刻意取最弱设定**（无 injection、无反馈优化），结论的警示性反而更强。

## 局限与展望
- 评审只用了"单趟、仅看正文、无工具核验"的最小协议，真实重型评审（可跑代码、可检索）可能更强，论文也承认这是为贴近实践而做的简化。
- 校准只基于 ICLR 2025 单一会议、$N_\text{cal}=200$，阈值对其他领域/会议的可迁移性未验证。
- 诚信判定依赖 GPT-5 当裁判，裁判本身的偏差会传导到 ICR 度量。
- 缓解只试了 ReD/DetOnly 两种轻量手段，论文提的 provenance verification、诚信加权打分、强制人类介入等"纵深防御"尚未实现，是明确的后续方向。

## 相关工作与启发
- **vs AI Scientist / Auto Research（科研 agent）**：它们证明端到端科研流水线可行，但不分析对抗目标下输出的"诚信"；本文专攻造假这一最小子集，把生成与评审耦合起来评估。
- **vs prompt-injection 攻击评审的工作（Ye et al. 2024 等）**：那类工作直接攻击评审输入；本文刻意排除 injection，只在呈现层造假，证明"不攻击评审本身也能骗过它"。
- **vs AI 生成文本检测（Gao/Mitchell 等）**：检测器关注"是不是 AI 写的"，本文关注"内容是否站得住"，并显示在科研造假场景下检测勉强超随机。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统刻画"造假 agent vs 评审 agent"的对抗界面，concern-acceptance conflict 是新且尖锐的发现。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 6 策略 ×3 评审模型 + 理论界 + 合成验证，但只校准单一会议、缓解手段偏少。
- 写作质量: ⭐⭐⭐⭐ 形式化清晰、理论与现象对照有力，记号略密。
- 价值: ⭐⭐⭐⭐⭐ 直指 AI-only 发表闭环的现实安全隐患，对科研诚信治理有强警示意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)
- [\[ICML 2026\] Who can we trust? LLM-as-a-jury for Comparative Assessment](../../ICML2026/llm_evaluation/who_can_we_trust_llm-as-a-jury_for_comparative_assessment.md)
- [\[ICML 2026\] Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments](../../ICML2026/llm_evaluation/multi2_hierarchical_multi-agent_decision-making_with_llm-based_agents_in_interac.md)
- [\[ACL 2026\] Beyond Fixed Psychological Personas: State Beats Trait, but Language Models are State-Blind](beyond_fixed_psychological_personas_state_beats_trait_but_language_models_are_st.md)
- [\[ACL 2026\] AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation](aj-bench_benchmarking_agent-as-a-judge_for_environment-aware_evaluation.md)

</div>

<!-- RELATED:END -->
