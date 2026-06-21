---
title: >-
  [论文解读] C-Evolve: Consensus-based Evolution for Prompt Groups
description: >-
  [ICLR 2026][LLM Agent][提示词进化] C-Evolve 把"进化出一个最优提示词"改成"进化出一组互补的提示词"，用一个衡量提示词在群体投票中贡献度的 voting score：作为进化适应度，让多个提示词聚合后达成共识，从而突破单提示词的能力天花板。 领域现状：GPT-4.1、Claude 这类闭源大…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "提示词进化"
  - "共识聚合"
  - "多数投票"
  - "岛屿模型"
  - "复合 AI 系统"
  - "黑盒优化"
---

# C-Evolve: Consensus-based Evolution for Prompt Groups

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=iyxRqYiCbF](https://openreview.net/forum?id=iyxRqYiCbF)  
**代码**: 待确认  
**领域**: LLM Agent / 提示词优化 / 进化算法  
**关键词**: 提示词进化, 共识聚合, 多数投票, 岛屿模型, 复合 AI 系统, 黑盒优化  

## 一句话总结
C-Evolve 把"进化出一个最优提示词"改成"进化出一组互补的提示词"，用一个衡量提示词在群体投票中贡献度的 **voting score** 作为进化适应度，让多个提示词聚合后达成共识，从而突破单提示词的能力天花板。

## 研究背景与动机
**领域现状**：GPT-4.1、Claude 这类闭源大模型只能通过 API 访问，无法微调权重，于是基于提示词的黑盒优化成为主流。其中进化类方法（GEPA 用自然语言反思 + Pareto 选择、AlphaEvolve 用岛屿模型进化整段程序）最为强大，它们都在一个种群里反复变异、选择，最终挑出**单个**适应度最高的提示词作为全局最优解。

**现有痛点**：单个最优提示词存在天然的表达瓶颈。复杂任务往往有多个侧面的要求，一条提示词很难面面俱到——它在某些 case 上必然失败。传统机器学习早就证明集成（ensemble）比单分类器更强，self-consistency、多智能体也都验证了"多个专家投票胜过单个专家"。但**在提示词优化领域，如何系统性地聚合多条提示词的输出来达成共识，几乎没人深入研究**。

**核心矛盾**：如果只是事后把几条各自最优的提示词凑成一组投票，效果并不好——因为它们是按"个体得分"进化出来的，彼此高度同质，犯的错也相似，聚合后无法互补。论文实验证实：直接把 AlphaEvolve 三个岛各自最优的提示词组队投票，准确率仍停在 41.15%，毫无提升。问题在于**进化目标错了**：要想群体投票强，进化时就该奖励"善于配合"的提示词，而不是"单打独斗强"的提示词。

**本文目标**：找到最优**提示词组** $G^* = \arg\max_G \mathbb{E}_{(x,m)\sim\tau}[\mu(C(y_G), m)]$，其中 $C$ 是共识聚合器，让这组提示词的聚合输出（而非任一个体）在任务上最优。

**核心 idea**（投票分驱动进化）：用一个 **voting score** 度量每条提示词在它所参与的所有群体中的平均贡献，把它作为进化适应度，使得"更可能组成高分群体"的提示词被保留繁殖、"拖后腿"的被淘汰，进化方向天然指向高质量共识。

## 方法详解

### 整体框架
C-Evolve 在 AlphaEvolve 的**岛屿进化模型**基础上改造：初始化 $|G|$ 个岛屿（默认 3 个），各岛并行独立进化以保持种群多样性，最后从每个岛各取一条提示词组成投票群体。整个流程分两阶段——先 **warm-up 阶段**用个体得分把每条提示词本身练强，再 **voting 阶段**改用群体共识的投票分作适应度，让提示词学会互补配合。进化由一个 evolver LLM 完成：它读取被选提示词及其执行反馈，refine 出新个体。推理时，从每个岛取 EMA 投票分最高的个体组成最终群体，聚合得出答案。

```mermaid
flowchart TD
    A[初始化 |G|=3 个岛<br/>每岛容量 N_max=10] --> B[Warm-up 阶段<br/>适应度=个体分 s_ind]
    B --> C[Voting 阶段开始<br/>初始化 s_EMA = warm-up 末个体分]
    C --> D[每岛采样1个体→evolver LLM<br/>用群体反馈进化出新个体]
    D --> E[跨岛采样 n_c=10 个投票群体<br/>每组各岛取1人, 共识聚合器评分]
    E --> F[算 voting score 式3<br/>→ EMA 更新 s_EMA 式4]
    F --> G[淘汰各岛 s_EMA 最低个体]
    G --> H{迭代结束?}
    H -- 否 --> D
    H -- 是 --> I[每岛取 s_EMA 最高个体<br/>组成最终群体, 共识推理]
```

### 关键设计

**1. Voting score：把"个体强"换成"团队贡献强"的适应度。** 这是 C-Evolve 区别于一切单提示词进化的核心。一个朴素想法是直接淘汰"最差群体"里的所有成员，但群体之间成员高度重叠，最差群体里也可能混着别处的好成员，一刀切毫无道理。论文转而度量**每个个体在所有含它的群体中的平均共识表现**：
$$s_{\Pi,\text{voting}} = \frac{\sum_{k=1}^{n_c} \mathbb{I}(\Pi \in G_k)\cdot \mathbb{E}_{(x,m)\sim D_{met}}\big[\mu(C(y_{G_k}), m)\big]}{\sum_{k=1}^{n_c}\mathbb{I}(\Pi \in G_k)}$$
其中 $\mathbb{I}(\Pi\in G_k)$ 表示个体 $\Pi$ 是否属于群体 $G_k$。投票分越高，说明含它的群体越容易给出正确共识，它就越该被保留来组队。把它当适应度，进化自然就奖励"善于配合"的提示词——这正是 Table 2 里 C-Evolve（43.88%）反超 AlphaEvolve 直接组队（41.15%）的根本原因，哪怕 C-Evolve 单个体得分（40.47/39.79/38.77）还不如 AlphaEvolve 的 41.15。

**2. EMA 平滑：让适应度反映"当前种群"而非陈年旧账。** voting score 在不同迭代算出来的值差异很大，因为种群一直在变。如果简单对一个个体历史上所有群体表现求平均，会把早已被淘汰的旧成员组成的群体也算进来、还给同样权重，这无法反映该个体在**当前种群**里的真实贡献。论文用指数移动平均更新：
$$s_{\Pi,\text{EMA}} \leftarrow \alpha\cdot s_{\Pi,\text{EMA}} + (1-\alpha)\cdot s_{\Pi,\text{voting}}$$
$\alpha\in[0,1]$ 平衡历史与当下，给近期表现更高权重、压制久远历史的影响。voting 阶段开始时用 warm-up 末尾的个体分初始化 $s_{\Pi,\text{EMA}}$，之后每轮更新完淘汰各岛 $s_{\Pi,\text{EMA}}$ 最低者。消融显示简单平均只有 41.16%，EMA 在 $\alpha=0.8$ 时最佳达 42.85%。

**3. 共识聚合器：闭/开放任务分而治之。** 群体内每条提示词先各自产出 $y = \Phi(x;\Pi)$，再由聚合器 $C$ 整合成 $y_{final}=C(y_G)$。闭式任务（多选、数学）直接**多数投票**取票最多的答案；开放式任务（自由文本）则用 **LLM-based 聚合器**挑出"覆盖最多其他回答内容的最具代表性回答"。消融对比两种 LLM 聚合策略：LLM-summary（综合所有回答生成新答案）只有 38.66%，而 **LLM-selection（从已有回答里选最具代表性的）** 达 42.85%，因为选择保留了原始输出的忠实性，而综合容易"编"出没有任何个体支撑的内容。

**4. 双数据集 + 群体反馈进化。** 进化用 $D_{met}$（配评测指标 $\mu$）算分、用独立的 $D_{feed}$ 生成详细执行反馈喂给 evolver。warm-up 阶段反馈含个体内各提示词、系统各模块的输入输出及个体指标分；voting 阶段则升级为**群体反馈**——额外加入分组信息、组内每个个体的输出、群体达成的共识、共识在 $D_{met}$ 上的得分。evolver LLM 据此诊断"错在哪个模块、何时出错"，refine 出更会配合的新个体。岛屿间还设 10% 迁移率避免陷入局部最优。

## 实验关键数据

### 主实验表格
跨 2 个开放式任务（HotpotQA、IFBench）+ 3 个闭式任务（HoVer、MATH、GPQA），在开源 Qwen3-8B 和闭源 GPT-4.1-mini 上对比：

| 模型 | 方法 | HotpotQA | IFBench | Hover | MATH | GPQA | 平均提升 |
|------|------|----------|---------|-------|------|------|----------|
| Qwen3-8B | Baseline | 50.03 | 31.29 | 37.66 | 67.66 | 41.43 | - |
| | GEPA | 65.72 | 34.01 | 43.33 | - | - | +8.02 |
| | AlphaEvolve | 65.31 | 41.15 | 44.66 | 82.66 | 43.08 | +9.75 |
| | **C-Evolve** | **70.67** | **43.88** | **50.33** | **85.33** | **47.15** | **+13.85** |
| GPT-4.1-mini | Baseline | 44.24 | 40.13 | 42 | 78.66 | 46.34 | - |
| | GEPA | 68.39 | 46.59 | 39 | - | - | +9.2 |
| | AlphaEvolve | 67.31 | 45.24 | 50.33 | 92.66 | 63.01 | +13.42 |
| | **C-Evolve** | **70.64** | **47.96** | **51.66** | **95.33** | **66.26** | **+16.09** |

C-Evolve 全部任务 SOTA：Qwen3-8B 上平均超 baseline +13.85%、超 GEPA +5.83%、超 AlphaEvolve +4.1%；更强的 GPT-4.1-mini 上仍超 GEPA +6.89%、超 AlphaEvolve +2.67%。

### 消融实验表格

| 消融项 | 设置 | 准确率(%) |
|--------|------|-----------|
| LLM 聚合器 | LLM-summary | 38.66 |
| | **LLM-selection** | **42.85** |
| 投票分平滑 | Group Average | 41.16 |
| | EMA, α=0.5 | 42.51 |
| | **EMA, α=0.8** | **42.85** |
| | EMA, α=0.95 | 41.50 |

（均在 IFBench / Qwen3-8B 上，warm-up 50 轮 + voting 50 轮）

### 关键发现
- **共识对难题增益最大**：MATH 按难度分层（Table 3），单个体在 Level 5 全部跌破 67%，而 C-Evolve 投票后在 Level 3–5 分别达 92.00 / 82.66 / 68.33%，专啃单提示词解不了的硬骨头。在"三个体中恰好两个达成共识"的 25% 问题里，C-Evolve 解对 49.33%，而单最优提示词只有 40%。
- **多岛进化出差异化分工**：IFBench 上三个岛各自演化出不同侧重——岛1 主"核心约束优先 + 精修修正"，岛2 主"约束预声明 + 分层校验"，岛3 主"任务优先级分层 + 关键失败点验证"；t-SNE 和 Levenshtein 距离显示 voting 阶段三岛逐渐形成各自簇，互补性是性能来源。
- **进化曲线对比**：AlphaEvolve 早期快升但很快饱和，C-Evolve 进入 voting 阶段后继续爬升，说明共识驱动的适应度提供了单提示词进化耗尽后的新优化空间。

## 亮点与洞察
- **重定义了进化目标**：从"找单个最优解"转向"找最优互补组合"，把集成学习的智慧第一次系统性地注入提示词进化。关键洞见是——要让群体投票强，进化时的适应度就必须是"团队贡献"而非"个人成绩"，否则进化只会产出同质、错误相关的提示词，聚合无增益。
- **voting score + EMA 的组合很巧**：voting score 解决"如何归因群体表现到个体"，EMA 解决"如何只用当前种群的有效信号"，两者配合让淘汰决策既公平又紧跟种群演化。
- **岛屿模型在这里不只是防局部最优，而是多样性的生产引擎**：并行隔离进化天然孕育出分工不同的提示词，正好满足"互补才能共识增益"的前提，框架自洽。

## 局限与展望
- **推理成本翻倍**：群体有 $|G|$ 条提示词，每个 query 要跑 $|G|$ 次再聚合，开放式任务还要额外一次 LLM 聚合，token 开销随群体规模线性增长，论文未充分讨论成本-收益权衡。
- **岛屿数固定为 3**：$|G|=3$ 是经验设定，更多岛是否持续增益、何时边际递减、最优群体大小如何随任务自适应，缺乏系统分析。
- **依赖 evolver 与聚合器的能力**：进化质量和共识质量都受底层 LLM 制约，弱模型上 voting score 的反馈信号可能噪声更大；LLM-based 聚合器本身也可能引入偏差。
- **代码与 AlphaEvolve 复现**：对比的 AlphaEvolve 系作者自行复现（原方法未开源），公平性需谨慎看待。

## 相关工作与启发
- **进化式提示词优化**：GEPA（自然语言反思 + Pareto 选择）、AlphaEvolve（岛屿模型进化程序）、PromptQuine（ICL 压缩剪枝）都只追求单一最优解，C-Evolve 指出这限制了泛化——任务复杂或动态变化时，单解无法探索提示词间的协同。
- **LLM 共识**：self-consistency（多输出多数投票）、多智能体系统（多角色协作）都体现"委员会胜过单专家"，但它们的共识是事后机制；C-Evolve 把共识做成**进化的内在驱动力**，让进化全程都在为"建立高质量共识"服务。
- **启发**：任何"先各自优化再事后集成"的 pipeline 都值得反思——如果集成是最终目标，优化目标就该直接对齐集成表现。这个"用群体贡献分替代个体分作适应度"的思路，可迁移到模型集成、检索器组合、多智能体角色进化等场景。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 把集成/共识思想首次系统性嵌入提示词进化，voting score 作适应度的设计点题精准，从"单最优"到"最优组合"的目标重定义有真正的概念贡献。
- **实验充分度**: ⭐⭐⭐⭐ — 2 模型 × 5 任务覆盖开/闭式，对比 GEPA/AlphaEvolve 两个强基线，难度分层、多岛分工可视化、EMA/聚合器消融齐全；扣分在推理成本与岛屿数缺系统分析、AlphaEvolve 系自行复现。
- **写作质量**: ⭐⭐⭐⭐ — 动机递进清晰（单解瓶颈→集成启发→进化目标错配→voting score），公式与图示到位，Table 2 的"直接组队无效"对照极具说服力。
- **价值**: ⭐⭐⭐⭐ — 黑盒模型时代提示词优化是刚需，"共识驱动进化"为该方向打开了单提示词之外的优化空间，且思路可迁移性强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] ChatInject: Abusing Chat Templates for Prompt Injection in LLM Agents](chatinject_abusing_chat_templates_for_prompt_injection_in_llm_agents.md)
- [\[ICLR 2026\] AlphaAgentEvo: Evolution-Oriented Alpha Mining via Self-Evolving Agentic Reinforcement Learning](alphaagentevo_evolution-oriented_alpha_mining_via_self-evolving_agentic_reinforc.md)
- [\[ACL 2026\] Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](../../ACL2026/llm_agent/mem2evolve_towards_self-evolving_agents_via_co-evolutionary_capability_expansion.md)
- [\[ICML 2026\] EvoClaw: Evaluating AI Agents on Continuous Software Evolution](../../ICML2026/llm_agent/evoclaw_evaluating_ai_agents_on_continuous_software_evolution.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](../../ACL2026/llm_agent/agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)

</div>

<!-- RELATED:END -->
