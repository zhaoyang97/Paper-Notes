---
title: >-
  [论文解读] Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives
description: >-
  [ICML 2026 Oral][对话系统][按 token 计费] 本文把 LLM-as-a-Service 建模成"委托-代理"问题，证明现在主流的"按 token 收费"机制天然激励服务商把同一字符串重新切成更长的 token 序列来超额收费，并且即使强制服务商公开 next-token 分布，多收费而不被发现也只是 NP-Hard 而非不可行——作者给出一个简单启发式算法在保持合理性的前提下实测最多多收 11.2% 的 token，最后证明唯一能消除该激励的可加性定价机制是"按字符长度线性计费"。
tags:
  - "ICML 2026 Oral"
  - "对话系统"
  - "按 token 计费"
  - "激励兼容"
  - "分词多重性"
  - "按字符计费"
  - "委托代理"
---

# Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives

**会议**: ICML 2026 Oral  
**arXiv**: [2505.21627](https://arxiv.org/abs/2505.21627)  
**代码**: https://github.com/Human-Centric-Machine-Learning/token-pricing (有)  
**领域**: AI 安全 / 机制设计 / LLM-as-a-Service 定价  
**关键词**: 按 token 计费、激励兼容、分词多重性、按字符计费、委托代理

## 一句话总结
本文把 LLM-as-a-Service 建模成"委托-代理"问题，证明现在主流的"按 token 收费"机制天然激励服务商把同一字符串重新切成更长的 token 序列来超额收费，并且即使强制服务商公开 next-token 分布，多收费而不被发现也只是 NP-Hard 而非不可行——作者给出一个简单启发式算法在保持合理性的前提下实测最多多收 11.2% 的 token，最后证明唯一能消除该激励的可加性定价机制是"按字符长度线性计费"。

## 研究背景与动机
**领域现状**：云端 LLM 服务（OpenAI / Gemini / Anthropic 等）几乎清一色采用 pay-per-token 计费：用户提交 prompt，服务商在自家硬件上跑模型生成输出，按返回的 token 数乘以单价收费。用户能看到的只有返回字符串和声称的 token 数，模型内部用什么 vocabulary、实际怎么切分、next-token 分布是什么样的，都在服务商一侧。

**现有痛点**：分词不唯一。同一个字符串 "Damascus" 既可以被切成 `|Dam|ascus|`（2 个 token），也可以被切成 `|Da|ma|s|cus|`（4 个 token），用户对此完全不知情。服务商完全可以把真实生成的 2-token 序列"重新报"成 4-token 来双倍收费，而字符串完全没变，用户没有任何技术手段察觉。

**核心矛盾**：信息不对称（asymmetry of information）造成的道德风险（moral hazard）——服务商完整观察生成过程，用户只观察并支付最终上报的 token 序列；只要"按 token 计费"成立且 vocabulary 中存在多字符 token，把短分词换成长分词在数学上严格能涨收入。

**本文目标**：分解为三个子问题。(1) 在 pay-per-token 下，服务商有没有结构性的撒谎激励？(2) 如果强制服务商公开 next-token 分布，让用户用"该分词在模型下可信吗"来反查，撒谎是否就被堵死了？(3) 有没有一个原则上消除这种激励的定价机制？

**切入角度**：作者用契约理论里的委托-代理（principal-agent）框架建模——把用户当 principal、服务商当 agent、计费规则当 contract，然后系统性地刻画"激励兼容"（incentive-compatibility）这一性质：在该性质下，服务商如实上报永远不比撒谎差。这是一个在拍卖、机制设计、保险等场景反复使用过的范式，第一次被搬到 LLM 定价上。

**核心 idea**：用 token 长度计费一定不激励兼容，唯一可加且激励兼容的方式是按字符数线性计费；过渡时只需令 $r_c = r_o \cdot \mathrm{tpc}$（tpc 为平均每字符 token 数），服务商平均利润率可以保持不变。

## 方法详解

### 整体框架
全文不是提出一个新模型，而是把"按 token 收费的 LLM 服务到底安不安全"做成一条完整的论证链：先用契约理论里的委托-代理框架把服务过程形式化，再证明 pay-per-token 天然激励服务商把同一字符串重切成更长的 token 序列多收钱，最后给出唯一能消除这种激励的计费方式并附上无痛迁移公式。

形式化的基本设定是：用户提交 prompt，服务商在自家硬件上跑出真实 token 序列 $\mathbf{t}$（对应字符串 $s = \mathrm{str}(\mathbf{t})$），然后按某个上报策略 $\pi$ 给出 $\tilde{\mathbf{t}} \sim \pi(\mathbf{t})$，唯一硬约束是 $\mathrm{str}(\tilde{\mathbf{t}}) = s$——用户看到的文本一字不差，看不到的只是它被切成了几个 token。服务商效用写成 $U_\pi(\tilde{\mathbf{t}}, \mathbf{t}) = r(\tilde{\mathbf{t}}) - c_\text{gen}(\mathbf{t}) - c_\pi(\mathbf{t})$：收入 $r$ 由计费规则决定，生成成本 $c_\text{gen}(\mathbf{t}) \approx c_o \cdot \mathrm{len}(\mathbf{t})$ 与真实 token 数成正比，$c_\pi$ 是执行上报策略本身的代价（不验证的策略 $c_\pi = 0$，需要跑 forward pass 验证合理性的策略 $c_\pi = c_v$ 是常数）。后文所有结论都建立在"激励兼容"这一性质上——按 Definition 4，如实上报策略 $\pi_0$ 满足 $U_{\pi_0}(\mathbf{t}, \mathbf{t}) \geq U_\pi(\tilde{\mathbf{t}}, \mathbf{t})$ 对一切策略弱占优，即诚实永远不吃亏。

### 关键设计

**1. 撒谎激励的形式化 + 零成本启发式（Algorithm 1）：戳破"按 token 收费没问题"的假象**

最朴素的"按 token"收费是一种可加机制 $r(\tilde{\mathbf{t}}) = \sum_i r(\tilde{t}_i)$，简化成 $r(\tilde{\mathbf{t}}) = r_o \cdot \mathrm{len}(\tilde{\mathbf{t}})$。这一形式立刻暴露问题：对任何成本相同的两个上报策略 $\pi, \pi'$，只要 $\mathrm{len}(\tilde{\mathbf{t}}) > \mathrm{len}(\tilde{\mathbf{t}}')$ 就有 $U_\pi > U_{\pi'}$——上报序列越长利润越高，撒谎不是个例外而是结构性最优解。Algorithm 1 用一个零计算实现把它具象化：维护当前序列，每轮枚举所有"还能再切成两个非空子词"的 token，从合法切分里随机挑一个执行，切 $m$ 次直到所有 token 都成单字符或到达上限，全程不碰 GPU 因为根本不验证合理性。作者用它先说明问题在竞争市场里的杀伤力——作弊方只要把序列拉长 $1/\alpha$ 倍，就能在每 token 单价比对手低 $\alpha$ 倍的情况下拿到同样收入却抢走更多用户，撒谎从"道德瑕疵"变成"抢市场的武器"。代价是这种乱切出来的序列在真实模型下几乎都不合理，一查就露馅，于是引出第二个更狡猾的算法。

**2. 可信赖的启发式撒谎（Algorithm 2）+ NP-Hard 屏障：透明也堵不住作弊**

第二个问题是：如果强制服务商公开 next-token 分布（top-$p$ 采样下，每步合理候选集 $\mathcal{V}_p$ 是累积概率 $\geq p$ 的最小集合），让用户能反查"这个分词在模型下可信吗"，撒谎是不是就被堵死了？作者先证明"找最长的合理分词"是 NP-Hard（Theorem 3，从 Hamiltonian Path 归约），意味着最优作弊不可多项式求解——但这恰恰是个陷阱：复杂度高不等于经济上安全。Algorithm 2 利用 BPE 里"id 越高 token 越长"的经验规律，每轮挑当前序列中 id 最大的 token，把它切成两个子 token $(t_1', t_2')$ 使 $\min(\mathrm{id}(t_1'), \mathrm{id}(t_2'))$ 尽可能大（max-min 启发式，让切出来的两半都仍是模型熟悉的常见词），切 $m$ 次后跑一次 forward pass 验证每一步都满足 $\hat{t}_i \in \mathcal{V}_p(\hat{\mathbf{t}}_{\leq i-1})$；合理就上报，不合理就退回真实序列稳赚不赔。它划算的判据是 $\mathbb{E}[\mathrm{plausible}(\hat{\mathbf{t}})] \cdot m \cdot r_o > c_v$，即"合理时多收的 token 钱 > 一次验证的固定能耗"——因为验证成本 $c_v$ 与序列无关而每多切一个 token 就多赚一个 $r_o$，在主流利润率下这个不等式轻松成立，实测对 $p=0.99$ 能把超额收入做到 10% 以上且净效用始终为正。透明因此只是把作弊从"随便切"收紧到"贴着合理边界切"，并没有真正保护用户。

**3. 激励兼容定价的充要刻画 + 平滑迁移公式：按字符收费是唯一解**

第三步回答"那到底该怎么收费"。作者先证 Proposition 5：激励兼容要求收入 $r(\tilde{\mathbf{t}})$ 只能依赖字符串 $\mathrm{str}(\tilde{\mathbf{t}})$ 而不能依赖具体怎么切——否则服务商总能挑最贵的那种分词撒谎。再证 Theorem 6：在可加前提下，激励兼容当且仅当 $r(\mathbf{t}) = \sum_{\sigma \in \Sigma} \mathrm{count}_\sigma(\mathbf{t}) \cdot r(\sigma)$，也就是按字符线性计费；若每个字符同价 $r_c$，则 $r(\mathbf{t}) = |\mathrm{str}(\mathbf{t})| \cdot r_c$ 是唯一选择（推论 7 由此直接断言：只要 vocabulary 含多字符 token，pay-per-token 一定不激励兼容）。这把"用什么计费"从工程偏好压成了一条充要定理——不是建议而是数学必然。为了让现有 API 能无痛切换，作者给出迁移公式 $r_c = r_o \cdot \mathrm{tpc}$，其中 tpc 是数据集上"每输出 token 对应的字符数"的样本均值，按此换算平均利润率不变。代价是单条样本利润率会波动（生成成本随 token 数走、收入随字符数走，二者比值会变），但这反而是良性激励：服务商想多赚就得去造更好的 tokenizer / 模型把字符压得更紧，而不是靠重切字符串占用户便宜。

### 损失函数 / 训练策略
本文是机制设计 + 理论 + 实证的工作，不涉及训练 loss。真正起作用的"超参数"是实验旋钮：top-$p$ 采样的 $p \in \{0.90, 0.95, 0.99\}$、温度 $T = 1.3$、Algorithm 2 的迭代数 $m$（实验显示对每个 $p$ 都存在单峰最优值），以及基准利润率 $\rho_o \in \{0.2, 0.4, 0.6\}$。撒谎是否划算被压成一条可直接对照真实能耗账单计算的判据 $\rho(\mathbf{t}) > 1 - \mathbb{E}[\mathrm{plausible}(\hat{\mathbf{t}})] \cdot m \cdot c_o / c_v$——即利润率与验证成本之比超过阈值就有利可图。

## 实验关键数据

### 主实验
评测在 LMSYS Chatbot Arena 的 600 条 prompt 上，模型用 Llama / Gemma / Ministral 系列，温度 1.3，top-$p$ 采样。

| 模型 / 配置 | $p=0.99$ 超额收费 | $p=0.95$ 超额收费 | $p=0.90$ 超额收费 | 说明 |
|--------|------|------|------|------|
| Llama-3.2-1B + Alg. 2 | 11.2% | 1.8% | 0.28% | $p$ 越大、采样空间越宽，合理性约束越松，作弊空间越大 |
| Llama-3.2-1B 效用增益 (任何 $\rho_o$) | >10.5% | 1.7% | 0.4% | 即扣掉验证 forward pass 成本后仍净赚 |
| pay-per-char 后利润率 > 0 比例（$\rho_o = 0.2 / 0.4 / 0.6$） | — | 92.4% / 93.3% / 94.1% | — | 迁移后绝大多数样本仍盈利 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Algorithm 1（无验证） | $m$ 越大超额收费越多（线性），但合理性快速崩到 ~0 | 证明"不验证"撒谎容易被反查 |
| Algorithm 2 关于 $m$ | 超额收费 vs $m$ 呈单峰，最优 $m$ 随 $p$ 减小而减小 | $m$ 太大合理性掉得太快，反而拉低期望收益 |
| 利润率 $\rho_o$ 变化 | $\rho_o$ 越小相对效用增益越大 | 价格战中作弊收益相对放大，激励反而更强 |

### 关键发现
- 关键设计 2 贡献最大：Algorithm 2 + Theorem 3 共同回答了"透明能否堵住作弊"，结论是"理论上半堵（NP-Hard），实际上不堵（启发式照样赚 10%）"。
- $p$ 越大可作弊空间越大，但 $p$ 也是真实生成的多样性旋钮——这把"高温/高 $p$"创意写作场景指认为最脆弱场景。
- 利润率越低撒谎的相对回报越高，意味着低毛利的小服务商作弊动机最强；这与"小服务商口碑成本低"的市场现实叠加形成系统性风险。

## 亮点与洞察
- 用 Hamiltonian Path 归约把"最长合理分词"打成 NP-Hard，然后立刻用 max-min id 启发式打脸"那 NP-Hard 总安全了吧"——理论与实证组合拳非常清晰，证明算法复杂度并不等于经济安全。
- Theorem 6 把"什么计费方式才能消除撒谎激励"压成一个充要刻画，pay-per-character 不是众多候选之一而是（同价字符前提下）唯一解，这种"必然性"叙述对推动政策极有力。
- $r_c = r_o \cdot \mathrm{tpc}$ 这条迁移公式可以直接套用到任何现有 API，不改模型不改 tokenizer 不改架构，迁移成本仅是一次数据集统计，非常工程友好。

## 局限与展望
- 作者承认：pay-per-character 不能阻止服务商让模型啰嗦（artificially verbose）多输出字符；这种攻击需要质量度量类机制（如 Saig et al. 2025 的 pay-for-performance）配合。
- 假设服务商不能伪造 next-token 分布本身或换 tokenizer——对闭源模型这层假设其实很强；作者建议用 trusted execution environments / zero-knowledge proofs 解决。
- 实验只在开源模型（Llama / Gemma / Ministral）上做，prompt 来自 LMSYS Chatbot Arena 这个被批评过代表性不足的平台；闭源模型 + 真实生产流量上的效果仍开放。
- 分析只到单用户-单服务商微观层，多服务商竞争 + 用户选择反馈这类宏观市场动力学留给后续。

## 相关工作与启发
- **vs Saig et al. (2025)**：他们也用 principal-agent，但针对的是"服务商用便宜模型却按贵模型计费"这种模型替换攻击，提出 pay-for-performance；本文针对的是"同一模型内的分词重报"，两者正交互补。
- **vs Sun et al. (2025) / Cai et al. (2025)**：这两篇是审计/检测向（验证 reasoning step 是否被注水 / 验证模型是否被替换），本文是机制设计向（直接换计费规则消除激励），层次不同。
- **vs Ahia et al. (2023)**：他们指出不同语言被 BPE 切出来的 token 数差异巨大，导致非英语用户在 pay-per-token 下被天然多收费；本文的 pay-per-character 顺带也修了这个公平性问题——多语言用户每字符同价。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一篇把 LLM 定价当作机制设计问题严肃刻画，并给出充要定理的工作。
- 实验充分度: ⭐⭐⭐⭐ 三族开源模型 + 600 条多语言 prompt 覆盖了主要情形，但缺闭源模型与生产级流量验证。
- 写作质量: ⭐⭐⭐⭐⭐ 论证链清晰：建模 → 揭露激励 → 启发式实证 → 不可避免性定理 → 平滑迁移配方，一气呵成。
- 价值: ⭐⭐⭐⭐⭐ 直接面向 LLM 商业化的核心定价规则，结论可直接进入监管与合同条款讨论，影响面很大。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Your Students Don't Use LLMs Like You Wish They Did](../../ACL2026/dialogue/your_students_dont_use_llms_like_you_wish_they_did.md)
- [\[ACL 2025\] Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles](../../ACL2025/dialogue/know_you_first_and_be_you_better_modeling_human-like_user_simulators_via_implici.md)
- [\[ICML 2026\] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)
- [\[ACL 2026\] Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation](../../ACL2026/dialogue/cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti.md)
- [\[ICML 2025\] Investigating Non-Transitivity in LLM-as-a-Judge](../../ICML2025/dialogue/investigating_non-transitivity_in_llm-as-a-judge.md)

</div>

<!-- RELATED:END -->
