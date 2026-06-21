---
title: >-
  [论文解读] How Catastrophic is Your LLM? Certifying Risks in Conversation
description: >-
  [ICLR 2026][LLM安全][灾难性风险] 本文提出 **C3LLM**——第一个对 LLM 多轮对话灾难性风险给出**统计认证下界**的框架：把对话建模为查询图上的 Markov 过程，对整个对话分布（而非固定攻击序列）采样，用 Clopper–Pearson 置信区间证明"该模型在某分布下产生灾难性输出的概率至少为 p"，对前沿模型测出最高 70%+ 的认证下界。
tags:
  - "ICLR 2026"
  - "LLM安全"
  - "灾难性风险"
  - "统计认证"
  - "多轮越狱"
  - "Markov 过程"
  - "置信区间"
  - "红队"
---

# How Catastrophic is Your LLM? Certifying Risks in Conversation

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=yt9TW2WtpG](https://openreview.net/forum?id=yt9TW2WtpG)  
**代码**: 见论文 GitHub 仓库（论文中标注）  
**领域**: LLM 安全 / 多轮对话风险认证  
**关键词**: 灾难性风险, 统计认证, 多轮越狱, Markov 过程, 置信区间, 红队

## 一句话总结
本文提出 **C3LLM**——第一个对 LLM 多轮对话灾难性风险给出**统计认证下界**的框架：把对话建模为查询图上的 Markov 过程，对整个对话分布（而非固定攻击序列）采样，用 Clopper–Pearson 置信区间证明"该模型在某分布下产生灾难性输出的概率至少为 p"，对前沿模型测出最高 70%+ 的认证下界。

## 研究背景与动机
**领域现状**：现有 LLM 安全评测几乎都是经验性的——在固定的攻击序列数据集上测攻击成功率（ASR）。单轮越狱已被研究透彻，但真实对话本质是多轮的：攻击者把恶意意图拆散到一串看似无害的查询里，逐步把模型引向有害内容。

**现有痛点**：经验性评测有两个根本缺陷。其一，结论**强依赖于所选的固定序列**——一个 benchmark 给 20 条长度为 5 的攻击序列，最多只能揭示 20 种灾难性行为；而把这些查询自由组合，长度为 5 的对话空间可达 $100^5$，绝大多数序列从未被测过。其二，**没有任何统计保证**，findings 无法泛化到这个庞大的对话空间。

**核心矛盾**：我们想要的是对"整个对话分布"的**定量保证**——随机采一段对话、模型产生灾难性输出的概率是多少。但精确概率无法计算，穷举又不可行；而定性保证（是否存在某一条灾难性对话）对任意大空间的 LLM 都恒为"是"，无法用来比较模型安全性。

**本文目标**：给出对多轮对话灾难性风险的**高置信度区间**，把它变成可靠比较前沿模型安全性的指标。

**核心 idea**：**【从 benchmark 到 certification】** 不再测固定序列，而是把多轮对话形式化为查询图上的概率分布，i.i.d. 采样后用统计置信区间界定灾难性风险的下界——这样一句"区间 [0.4, 0.6]"就意味着"高置信下，分布中至少 $0.4n$ 条序列能触发灾难性输出"，覆盖了整个空间而非几条样本。

## 方法详解

### 整体框架
C3LLM 把一组与有害目标相关、但单独看较温和的查询集 $Q$ 构造成无向图 $G=(V,E)$，边由句向量余弦相似度刻画（既要语义相关又要避开近重复）。在图上定义"对话 = 查询序列 $\gamma=(v_0,\dots,v_{n-1})$"的概率分布 $D_n$，用图上的 Markov 过程生成。对每条采样序列，逐轮喂给目标 LLM（带累积上下文），用 judge 模型判断响应是否泄露有害目标 $q^\star$，最后用 Clopper–Pearson 把这些 0/1 结果聚合成 95% 置信区间，认证目标量 $\Pr_{\gamma\sim D_n}[\exists i,\ J_{q^\star}(r_i)=1]$。

```mermaid
flowchart LR
    A[有害目标 q*] --> B[扩展查询集 Q<br/>actor-based prompts]
    B --> C[构造查询图 G<br/>边=语义相似度]
    C --> D[Markov 过程<br/>4 种分布采样序列]
    D --> E[逐轮 query 目标 LLM<br/>累积上下文]
    E --> F[GPT-4o judge<br/>响应是否灾难性]
    F --> G[Clopper-Pearson<br/>95% 置信区间下界]
```

### 关键设计

**1. 提升状态空间的 Markov 过程：把"不重复采样"写进状态。** 为了反映"自适应攻击者不会原样重复同一 prompt"，作者不在原始查询上做马尔可夫，而是在**提升状态空间** $\Omega=\{(v,S):S\subseteq V, v\in S\}\cup\{\tau\}$ 上定义过程：当前状态既记当前查询 $v$，又记本序列已用过的查询集合 $S$，从而天然禁止序列内重访；$\tau$ 是吸收终止态。转移分前向（给初始分布 $\mu$，访问集 $S_t=\{v_0,\dots,v_t\}$ 递增）和后向（给终点分布 $\nu$，访问集 $U_t=\{v_t,\dots,v_{n-1}\}$ 从尾往前）两族，并用归一化算子 $N(\cdot)$ 处理"序列可能提前进入 $\tau$"导致的概率不归一问题，保证 $\sum_{|\gamma|=n}\Pr(\gamma)=1$。这套提升状态 + 前向/后向选择就是定义各种攻击分布的统一"配方"。

**2. 三类（四种）攻击分布：覆盖从无结构到自适应的攻击者。** 在统一框架下实例化出代表性分布。**Random node** 让每步从未访问节点 $V\setminus S$ 独立采样，估计模型不利用任何结构时的整体倾向。**Graph path** 走图上的一条路径（后向选择），相邻查询天然语义连贯——既给后续查询提供局部上下文，又避免 random node 那种不真实的跳跃；它又分 vanilla（终点从 $V$ 取）和 **harmful target constraint**（终点限制在目标集 $Q_T=\{v:\ell_{th}<\mathrm{sim}(v,q^\star)<h_{th}\}$），后者把对话强行导向高危查询。**Adaptive with rejection** 则把模型的接受/拒绝反馈引入转移：用拒绝指示 $r_v=\mathbb{1}\{\text{is\_rej}(M(v))\}$，把未访问邻居按与目标相似度切成 $A_{\text{prog}}$（更接近 $q^\star$）和 $A_{\text{deprog}}$（更远）两组，当查询被接受（$r_v=0$）就给 prog 组更大权重 $\lambda_h$ 鼓励逼近目标，被拒绝（$r_v=1$）就反转权重退回更安全区域，权重 $\lambda_{v,S}(w)$ 经归一化得到转移概率。这恰好模拟真实红队"被拒就换温和说法、被答就步步紧逼"的自适应行为。

**3. 增强层：把越狱 prefix 作为分布的一部分注入。** 在基础分布之上叠一层增强分布 $D_{\text{aug}}(\cdot\mid v_t)$，对序列每个查询独立采样替换：$\tilde v_t\sim D_{\text{aug}}(\cdot\mid v_t)$，整条序列概率写成 $\Pr(\tilde\gamma)=\Pr(\gamma)\prod_t \Pr_{D_{\text{aug}}}(\tilde v_t\mid v_t)$。它统一涵盖恒等情形（以概率 1 返回原查询）和随机改写——主实验用越狱分布 $D_{jb}$，以插入概率 $p=0.2$ 在查询前 prepend 一段越狱 prompt（仅作用于 Random Node，得到 RNwJ）。关键在于：增强器被视为攻击过程的一部分、且诱导出良定义的对话分布，因此认证的统计保证依然成立。

**4. 统计认证：从有限采样到高置信下界。** 给定分布与判别函数后，C3LLM 采 $n=50$ 条序列，每条得一个是否灾难性的 0/1 结果，对这组伯努利样本用 **Clopper–Pearson** 精确方法计算 95% 置信区间 $[p_l,p_h]$。下界 $p_l$ 的意义很硬：高置信下，整个分布里至少有 $p_l$ 比例的对话会触发灾难性输出——这就把"在 50 条上的观测"提升为"对整个 $D_n$ 的概率保证"，从而可用于跨模型公平比较。数据集来自 HarmBench 的 chem-bio / cybercrime / illegal 三类，用 actor-based prompt（如目标"造炸弹"→actor"诺贝尔"→围绕其生成温和查询）扩展，每场景去重后采 20 actors × 5 查询 = 100 条查询建图。

## 实验关键数据

### 主实验表格
对 6 个前沿模型、4 种分布在两类数据集上的 95% 认证区间（中位数）：

| 数据集 | 模型 | RNwJ | GPv | GPh | AwR |
|---|---|---|---|---|---|
| chembio | nova | (.005,.137) | (.001,.106) | (.013,.165) | (.005,.137) |
| chembio | deepseek | **(.554,.821)** | (.221,.498) | (.229,.508) | (.212,.488) |
| chembio | claude | (.001,.106) | (.001,.106) | (.001,.106) | (.001,.106) |
| chembio | gpt-oss | (.028,.205) | (.072,.291) | (.045,.243) | **(.101,.337)** |
| chembio | mistral | **(.554,.821)** | (.318,.607) | (.432,.718) | (.452,.735) |
| chembio | llama | **(.212,.488)** | (.116,.359) | (.195,.457) | (.146,.403) |
| cyber | deepseek | **(.721,.935)** | (.472,.753) | (.543,.813) | (.543,.813) |
| cyber | mistral | **(.652,.892)** | (.403,.691) | (.533,.805) | (.565,.830) |
| cyber | claude | (.028,.205) | (.123,.371) | (.195,.467) | (.195,.467) |
| cyber | nova | (.000,.071) | (.001,.106) | (.001,.106) | (.000,.071) |

- DeepSeek-R1 在 cyber + RNwJ 下认证**下界超过 70%**（最危险），Mistral-Large 紧随其后。
- Claude-Sonnet-4 与 Nova Premier 最安全，下界普遍 <0.03，Nova 内置 guardrail 经常直接拦截。

### 消融实验表格
认证下界超过单轮(ST)/多轮(MT) ASR baseline 0.05 以上的场景数（说明认证比固定 benchmark 揭示更多风险）：

| 数据集 | 模型 | vs ST | vs MT |
|---|---|---|---|
| chembio | mistral | 100 | 30 |
| chembio | deepseek | 86 | 29 |
| chembio | llama | 78 | 14 |
| cyber | deepseek | 157 | 22 |
| cyber | mistral | 151 | 38 |
| cyber | llama | 143 | 17 |

### 关键发现
- **认证 ≫ benchmark**：认证下界在大量场景上显著高于固定单/多轮 ASR，证明固定序列严重低估了真实风险。
- **越狱增强对安全模型几乎无效**（Nova/Claude），但对弱模型（Mistral/DeepSeek）显著抬高灾难率——提高 $p$ 进一步放大差距。
- **GPh > GPv**：约束最后一个查询落在有害目标集，一致地比 vanilla 更有效——"塑造对话最后一步"是高效攻击手段。
- **Adaptive with Rejection 专打"看起来很对齐"的模型**：利用 gpt-oss(~20%)、Claude(~15%) 的非平凡拒绝率，通过与拒绝动态交互大幅提升灾难性响应。
- **两大攻击模式**：① **干扰项(distractors)**——单独问被拒的有害查询，嵌入多轮对话（即使夹杂无关问题）后更容易被回答；② **上下文(context)**——用"你刚提到…"引用前文，让模型推断用户焦点、产出更贴近有害目标的完整答案。

## 亮点与洞察
- **范式转变**：把 LLM 安全评测从"测几条固定序列"升级为"对整个对话分布给统计保证"，第一次让"模型 A 比 B 安全"有了可证伪的定量含义。
- **形式化优雅**：提升状态空间的 Markov 过程 + 前向/后向选择 + 增强层，构成一套可扩展的"攻击分布配方"，三种实例只是冰山一角。
- **认证下界很硬**：Clopper–Pearson 给的是高置信下界而非点估计，比 ASR 这种经验数字更可靠、更难被"换个序列就翻盘"。
- **实证洞察落地**：distractor 与 context 两个攻击模式直接指出当前安全训练的盲区——单查询拒绝不等于多轮安全。

## 局限与展望
- **样本量有限**：每个 specification 仅 50 条序列，Clopper–Pearson 区间偏宽（上下界相差常达 0.3+），下界虽硬但分辨率有限。
- **依赖 judge 模型**：用 GPT-4o 判别灾难性，judge 的误判会直接污染认证结果，框架本身不解决 judge 可靠性。
- **图与查询集是人为构造**：actor-based 扩展、相似度阈值 $\ell_{th},h_{th}$、越狱概率 $p$ 等超参影响分布形态，认证保证只对"所定义的分布"成立，分布选取本身仍是经验性的。
- **认证 ≠ 防御**：框架揭示风险但不提供缓解；如何把这些认证下界反馈进安全训练（如针对 GPh / AwR 模式做对抗训练）是自然的下一步。

## 相关工作与启发
- **多轮越狱**：人工红队、自动 LLM 攻击者(Crescendo/ActorAttack)、场景化、查询分解、攻击者训练等，本文不造新攻击，而是为"任意攻击分布"提供统计认证外壳。
- **LLM 认证**：以往认证多是 token/embedding 空间的扰动鲁棒性（局部 $\ell_\infty$ 球），或单轮的知识理解/偏见认证；本文首次对**多轮、时序、定量**的灾难性属性做认证。
- **启发**：① 把"评测"重构为"对分布的统计推断"这一思路可迁移到幻觉、偏见、隐私等其他安全维度；② distractor/context 模式提示安全对齐需在多轮上下文层面做，而非仅强化单查询拒绝。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 首个多轮对话灾难性风险的统计认证框架，提升状态 Markov + 认证下界是真正的新范式。
- **实验充分度**: ⭐⭐⭐⭐ 6 个前沿模型 × 4 分布 × 2 数据集，含 ST/MT baseline 对比与攻击模式案例，扎实；但每 spec 仅 50 样本使区间偏宽。
- **写作质量**: ⭐⭐⭐⭐ 形式化清晰、动机层层递进，框架图与公式配合好；自适应分布的权重定义略密集。
- **价值**: ⭐⭐⭐⭐⭐ 给前沿模型多轮安全提供了可比较、可证伪的认证指标，对监管与安全训练都有直接意义（DeepSeek-R1 cyber 下界 >70% 的结论极具警示性）。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Searching for Privacy Risks in LLM Agents via Simulation](searching_for_privacy_risks_in_llm_agents_via_simulation.md)
- [\[ICLR 2026\] Watch Your Steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning](watch_your_steps_dormant_adversarial_behaviors_that_activate_upon_llm_finetuning.md)
- [\[ICLR 2026\] Estimating Worst-Case Frontier Risks of Open-Weight LLMs](estimating_worst-case_frontier_risks_of_open-weight_llms.md)
- [\[ACL 2025\] Unveiling Privacy Risks in LLM Agent Memory](../../ACL2025/llm_safety/mextra_agent_memory_privacy.md)
- [\[ICLR 2026\] PropensityBench: Evaluating Latent Safety Risks in Large Language Models via an Agentic Approach](propensitybench_evaluating_latent_safety_risks_in_large_language_models_via_an_a.md)

</div>

<!-- RELATED:END -->
