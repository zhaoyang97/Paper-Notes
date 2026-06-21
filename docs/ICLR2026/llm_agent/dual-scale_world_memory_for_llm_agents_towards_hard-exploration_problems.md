---
title: >-
  [论文解读] Dual-Scale World Memory for LLM Agents towards Hard-Exploration Problems
description: >-
  [ICLR 2026][LLM Agent][硬探索] 提出 GLoW：用"全局轨迹前沿 + 局部多路径优势反思"的双尺度文本世界记忆武装 LLM 智能体，在 Jericho 文字游戏的稀疏奖励硬探索任务上刷新 LLM 方法的 SOTA，并以少 100–800× 的环境交互逼近最强 RL 方法。 领域现状：LLM 智能体（R…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "硬探索"
  - "LLM 智能体"
  - "世界记忆"
  - "Go-Explore"
  - "优势学习"
  - "Jericho"
---

# Dual-Scale World Memory for LLM Agents towards Hard-Exploration Problems

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=bH5uHIVtTe](https://openreview.net/forum?id=bH5uHIVtTe)  
**代码**: [https://github.com/mnskim/glow](https://github.com/mnskim/glow)  
**领域**: LLM Agent / 探索与决策  
**关键词**: 硬探索, LLM 智能体, 世界记忆, Go-Explore, 优势学习, Jericho  

## 一句话总结
提出 GLoW：用"全局轨迹前沿 + 局部多路径优势反思"的双尺度文本世界记忆武装 LLM 智能体，在 Jericho 文字游戏的稀疏奖励硬探索任务上刷新 LLM 方法的 SOTA，并以少 100–800× 的环境交互逼近最强 RL 方法。

## 研究背景与动机
**领域现状**：LLM 智能体（ReAct、Reflexion 等）凭借预训练知识在机器人规划、软件工程、网页自动化上表现亮眼，但在"硬探索"问题上仍远逊于人类。硬探索问题的典型特征是**巨大的状态-动作空间、欺骗性局部最优、稀疏奖励**——例如 Jericho 中的 Zork1，词表 697 词、最多五词指令，每步理论动作高达 $O(697^5)\approx1.64\times10^{14}$，而只有极少数语法且语境合理。

**现有痛点**：硬探索对 LLM 智能体提出两个核心挑战。一是**全局学习**——需要长期累积"哪些发现有价值"的知识，而 ReAct/Reflexion 只做局部试错、没有长期知识沉淀机制，探索容易困死在局部最优。二是**局部试错**——需要从稀疏反馈中快速精炼探索策略，但现有自反思本质是从单条轨迹估计 Q 值，在稀疏奖励下方差极高、因果归因常出错。另一方面，RL/MCTS 类方法（XTX、MC-DML）虽强，却要几十万到上百万次环境交互，样本效率极差。

**核心矛盾**：选择"回到哪个状态继续探索"（select）与"从该状态怎么探索"（explore）都需要从历史经验中做结构化学习，**但二者所需的尺度不同**——前者要全局视野判断价值高地与瓶颈，后者要局部细粒度的动作进展信号。把两者混为一谈，要么探索盲目，要么知识断层。

**本文目标**：让 LLM 智能体在不依赖海量交互的前提下，具备持续突破稀疏奖励瓶颈的探索能力。

**核心 idea**：**双尺度文本世界记忆**。基于 Go-Explore 的"选择—探索"交替框架，作者把世界记忆拆成两层：全局尺度维护一个**价值排序的轨迹前沿**（保留高价值发现"怎么到达、为何卡住"的完整时序上下文），由 LLM 做价值分解以做有原则的状态选择；局部尺度用 **Multi-path Advantage Reflection（MAR）**，对同一起点的多条轨迹做优势推断，把稀疏奖励"稠密化"为进展信号来指导探索。世界记忆不建模转移动力学，而是把探索经验编码成结构化**文本**表示。

## 方法详解

### 整体框架
GLoW 沿用 Go-Explore 家族的 select↔explore 迭代循环，但把"状态档案 + 启发式"升级为双尺度世界记忆。每轮迭代：① 全局世界记忆 $g_{\text{LLM}}$ 分析轨迹前沿 $F$，产出带价值标注的关键状态集 $W_{\text{global}}$，再用 $\text{align}_{\text{LLM}}$ 从档案 $A$ 中选出最值得返回的状态 $s_{\text{next}}$；② 回放动作序列返回该状态后，进入局部探索阶段，连续采样 $n$ 条轨迹，每条之后用 MAR 提炼局部世界记忆 $W_{\text{local}}$ 指导下一条；③ 新轨迹回填前沿 $F$ 与档案 $A$。

```mermaid
flowchart TD
    F[轨迹前沿 F<br/>价值排序的 top-k 高价值轨迹] -->|gLLM 价值分解| WG[全局世界记忆 Wglobal<br/>关键状态 (s, v, v')]
    A[状态档案 A] -->|alignLLM 对齐选择| SN[选定状态 s_next]
    WG --> SN
    SN --> EXP[局部探索阶段<br/>采样 n 条轨迹]
    EXP -->|MAR 多路径优势反思| WL[局部世界记忆 Wlocal<br/>关键状态优势 As*]
    WL -->|指导 πexplore| EXP
    EXP -->|新轨迹回填| F
    EXP -->|新状态回填| A
```

### 关键设计

**1. 价值排序的轨迹前沿：用完整轨迹替代孤立状态档案**。传统 Go-Explore 只存离散状态，丢掉了"动作-观测序列"上下文，在稀疏奖励下无法做准确的信用分配。GLoW 改为维护一个轨迹前沿 $F=\{\tau_1,...,\tau_k\}$，保留 $k$ 条按价值函数 $v:T\to\mathbb{R}$ 排序的最高价值完整轨迹。价值取整条 episode 中的最大累计回报 $v(\tau_i)=\max_{t\in[1,T]}\sum_{j=1}^{t}r_j^i$——这对 Jericho 中途可能遇负奖励或致死的稀疏结构尤为合适。每条新轨迹通过滑窗机制更新前沿：$F_{t+1}=\text{top-k}(F_t\cup\{\tau_{\text{new}}\},v)$，让更优轨迹替换陈旧者。由此任意状态都能反推其可达价值 $v(s_i)=\max_{\tau\in F,s_i\in\tau}v(\tau)$。正因为保留完整序列，前沿才能跨轨迹推断出像 Zork1"先拿灯和剑、再下地窖才得奖励"这类被稀疏反馈掩盖的序列依赖。

**2. 全局价值分解：把 UCB 的探索-利用拆解搬进 LLM 推理**。受 UCB 的价值分解 $\bar V(s)+c\sqrt{\log(N)/n_s}$ 启发，GLoW 让 $g_{\text{LLM}}$ 跨前沿轨迹分析，给每个关键状态标注一对值 $W_{\text{global}}=g_{\text{LLM}}(F)=\{(s_i,v_i,v_i')\}$：$v_i$ 是已达成价值（对应利用项），$v_i'$ 是 LLM 推断的未来潜力价值（对应探索奖励项）。关键在于 $v_i'$ **无法从轨迹分数直接得出**——它要求 LLM 推理"轨迹为何失败、解决当前瓶颈能带来什么进展"。比如多条高价值轨迹汇聚却都卡在某状态，说明其后方藏着未探索高价值区，该瓶颈状态便被赋予高 $v_i'$。这等于用 LLM 对瓶颈的语义分析实现了"不确定性下的乐观主义"，把 UCB 的统计奖励换成语义推断。随后 $\text{align}_{\text{LLM}}(s,W_{\text{global}})$ 为档案中每个状态打分，$s_{\text{next}}=\arg\max_s \text{score}[s]$，天然在"靠近已验证高奖励区"（利用）和"靠近高潜力瓶颈"（探索）间取得平衡。

**3. Multi-path Advantage Reflection（MAR）：从 Q 值反思走向优势反思**。作者指出现有自反思相当于从单轨迹估 Q 值，而稀疏奖励下 Q 值估计方差极大、易错误归因。借鉴优势函数 $A(s,a)=Q(s,a)-V(s)$ 用基线降方差、以及 PRM 研究中"优势比 Q 值更能捕捉进展信号"的结论，MAR 仿照 TRPO 在同一起点做多条 rollout 的思路，对同起点 $n$ 条轨迹做对比推断。它是一个 LLM 算子，输入局部轨迹集 $T_s$ 与前沿轨迹 $F$，输出结构化文本 $W_{\text{local}}=\{(s_1^*,A_{s_1^*}),...,(s_k^*,A_{s_k^*})\}$（典型 2–4 个关键状态）。两条设计原则保证语义优势推断的准确性：**多轨迹对比**让 LLM 聚合分歧结果、识别好/坏动作并聚焦在信号最丰富的关键状态；**前沿轨迹作稳定参照**充当"价值基线"，用基于上下文的推理代替数值相减来判断新轨迹是否构成真实进展。由此把稀疏奖励"伪稠密化"为像"即便当下无奖励也该先拿剑"这类进展信号。

**4. 优势驱动的探索策略**。局部世界记忆通过 LLM 智能体定义的策略增强探索：$\pi_{\text{explore}}(a|s_t,h_t)=\text{Agent}_{\text{LLM}}(s_t,h_t,W_{\text{local}},T_s,F)$，其中 $h_t$ 是当前轨迹历史，$T_s$ 是本阶段已采样轨迹，策略同时利用 $W_{\text{local}}$ 学到的优势与前沿 $F$ 的成功策略。为应对 Jericho 指数级动作空间，采用"自由生成 + Jericho 合法动作软约束"的混合动作生成方案——这也是作者复现的 LLM baseline 远超此前文献中近零分的关键。

## 实验关键数据

### 主实验表格
在 Jericho 10 个游戏上评测，每法 3 次取均值±标准差。GLoW 在 7/10 游戏上刷新 LLM 方法 SOTA，在 3/10 上为全局最优、5/10 为次优。

| 游戏 | XTX (RL, 800k步) | MC-DML (MCTS, ~400k步) | ICRL (LLM, 1k步) | IGE (LLM, 1k步) | **GLoW (1k步)** |
|------|------|------|------|------|------|
| Zork1 | 103.4 | 48.66 | 51.7 | 44.3 | **73.0** |
| Deephome | 77.7 | 67.0 | 24.0 | 71.3 | **75.0** |
| Ludicorp | 78.8 | 19.67 | 32.0 | 28.3 | **73.7** |
| Enchanter | 52.0 | 20.0 | 43.3 | 50.0 | **61.7** |
| Ztuu | – | 23.67 | 16.7 | 15.0 | **29.3** |
| Temple | – | 8.0 | 8.0 | 8.0 | **13.0** |
| Balances | 24 | 10.0 | 11.7 | 10.0 | **16.7** |

GLoW 仅用 **1,000 次交互**，即可在 Deephome/Ludicorp 上逼近用 800× 交互的 XTX，并在 Enchanter 上反超 XTX；相比同为 LLM-Go-Explore 的 IGE，在 8/10 游戏上更优。

### 消融实验表格
逐一移除组件（Zork1/Deephome/Ludicorp 为例）：

| 变体 | Zork1 | Deephome | Ludicorp | Balances |
|------|------|------|------|------|
| GLoW (Full) | **73.0** | **75.0** | **73.7** | **16.7** |
| ✗ MAR（局部世界记忆） | 70.0 | 56.7 | 54.7 | 11.7 |
| ✗ Wglobal（全局价值分解） | 62.0 | 61.3 | 63.3 | 13.3 |
| ✗ 轨迹前沿 F | 61.7 | 57.7 | 63.3 | 11.7 |
| ✗ 全部（≈IGE+多路径反思） | 51.3 | 56.0 | 22.0 | 10.0 |
| Standard IGE | 44.3 | 71.3 | 28.3 | 10.0 |

### 关键发现
- **三组件协同而非简单叠加**：单把多路径反思加到 IGE 上（✗全部变体）相比 IGE 并无清晰提升，说明 GLoW 的性能来自全局价值分解、轨迹前沿与 MAR 的互补协同。
- **优势学习胜过自反思**：把 MAR 换成 Reflexion（同样多路径但只做单轨迹反思）多数游戏明显掉分，验证优势式公式更善用多轨迹信息。
- **$n$ 的探索-利用权衡**：每状态探索次数 $n$ 越大局部学习越深、越小则状态选择越频繁更易逃离局部最优；实验表明 $n=3$（前沿大小 $k=5$）取得最佳平衡。
- **缩放到更强 LLM**：换用更强 LLM 后，GLoW 在 6 个困难/极端游戏中的 4 个上超越所有先前方法。

## 亮点与洞察
- **"文本世界记忆"而非动力学世界模型**：把 LLM 世界模型理解为对任务相关知识的隐式表示，用结构化文本沉淀探索经验，绕开了显式建模转移函数的难题，天然适配 LLM 的推理接口。
- **把经典 RL 理论"语义化"**：UCB 的探索奖励 → LLM 对瓶颈的潜力价值 $v'$；优势函数的数值基线 → 前沿轨迹作上下文参照基线。这种"用 LLM 推理实现 RL 原理"的映射很有启发，且附录给出了多轨迹对比降方差的理论动机。
- **样本效率的量级突破**：1,000 次交互逼近甚至局部反超百万级交互的 RL，凸显 LLM 智能体一旦配上对的记忆结构，在硬探索上的潜力被严重低估。
- **诚实的 baseline 复现**：作者用混合动作生成让 ReAct/Reflexion/ICRL 从"近零分"提升到与 RL 基线相当，澄清了此前文献对 LLM 智能体 Jericho 能力的低估。

## 局限与展望
- **确定性环境假设**：状态返回靠回放动作序列，依赖环境确定性；随机环境下的扩展仅在附录讨论，未实证。
- **LLM 调用成本**：select 与 explore 各阶段都重度依赖 LLM 推理（$g_{\text{LLM}}$、$\text{align}_{\text{LLM}}$、MAR），虽然环境交互少，但 API 调用与 token 开销不可忽视。
- **依赖合法动作软约束**：所有方法都假设可访问 Jericho 提供的合法动作集，迁移到无此先验的真实开放环境时探索难度会上升。
- **领域特异的新颖性判据**：档案去冗余用的是 domain-specific novelty，跨任务通用性待验证。
- **评测范围**：仍局限于文字游戏 Jericho，向机器人/网页等连续或多模态硬探索域的泛化尚未展开。

## 相关工作与启发
- **Go-Explore 家族**：原始 Go-Explore（启发式选择+随机探索）→ XTX（模仿学习选择+DQN 好奇心探索）→ IGE（两阶段都用 LLM）。GLoW 的两点创新是轨迹前沿+LLM 价值分解的有原则选择，以及 MAR 的局部优势学习。
- **LLM 智能体与自反思**：ReAct、Reflexion、ICRL 提供局部试错与多 episode 记忆，但缺长期价值结构；GLoW 把"自反思"升级为"多路径优势反思"。
- **RL 探索理论**：UCB/乐观主义、优势函数、TRPO 多 rollout、PRM 的进展信号——这些经典思想被系统地"语义化"迁移进 LLM 推理，是本文方法论上的核心借鉴。
- **启发**：对做 LLM 智能体长程任务的研究者，"把记忆按全局/局部双尺度拆分，并分别对接 RL 中的状态选择与优势估计"是一个可复用的设计范式，尤其在稀疏奖励、需要持续探索的场景。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 双尺度文本世界记忆 + 把 UCB 价值分解与优势函数系统映射进 LLM 推理（MAR）的组合很新颖，虽建立在 Go-Explore 框架上但创新点清晰且自洽。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 RL/MCTS/LLM 三类共 11 个 baseline、10 个游戏、3 次重复，消融逐组件拆解并验证协同效应，还分析了 $n$ 的权衡与更强 LLM 缩放；略欠随机环境与更广域的验证。
- **写作质量**: ⭐⭐⭐⭐ 动机—理论映射—方法—实验逻辑顺畅，图 1/图 2 与 Zork1 troll 例子把抽象机制讲得很直观。
- **价值**: ⭐⭐⭐⭐ 以 100–800× 更少交互逼近最强 RL，刷新 LLM 在 Jericho 硬探索的 SOTA，为"记忆结构化驱动 LLM 智能体探索"提供了有说服力的范式与可复现代码。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Meta-RL Induces Exploration in Language Agents](meta-rl_induces_exploration_in_language_agents.md)
- [\[ICLR 2026\] Go-Browse: Training Web Agents with Structured Exploration](go-browse_training_web_agents_with_structured_exploration.md)
- [\[AAAI 2026\] DEPO: Dual-Efficiency Preference Optimization for LLM Agents](../../AAAI2026/llm_agent/depo_dual-efficiency_preference_optimization_for_llm_agents.md)
- [\[ICLR 2026\] VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications](vitabench_benchmarking_llm_agents_with_versatile_interactive_tasks_in_real-world.md)
- [\[ICML 2026\] Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents](../../ICML2026/llm_agent/memory_is_reconstructed_not_retrieved_graph_memory_for_llm_agents.md)

</div>

<!-- RELATED:END -->
