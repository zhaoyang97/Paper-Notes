---
title: >-
  [论文解读] Position: Stop Anthropomorphizing Intermediate Tokens as Reasoning/Thinking Traces!
description: >-
  [ICML2026][可解释性][中间 token] 这是 Kambhampati 团队的立场文（position paper），核心主张：把推理模型（如 DeepSeek-R1）在给出答案前吐出的「中间 token」称作 "reasoning trace / thinking trace"（推理痕迹/思考痕迹）是一种**危险的拟人化**——它（1）是一厢情愿、（2）几乎没有实证支持、（3）制造对模型的虚假信任、（4）把社区推向无意义的研究方向；作者用一系列实验证据（A* 迷宫 trace 互换、trace 长度与问题复杂度脱钩、人类信任实验）论证 trace 的语义与最终解答正确性**根本脱钩**…
tags:
  - "ICML2026"
  - "可解释性"
  - "中间 token"
  - "思维链"
  - "拟人化"
  - "RLVR"
---

# Position: Stop Anthropomorphizing Intermediate Tokens as Reasoning/Thinking Traces!

**会议**: ICML2026  
**arXiv**: [2504.09762](https://arxiv.org/abs/2504.09762)  
**代码**: 无（立场文，无代码）  
**领域**: 可解释性（LLM 推理 · 思维链 · AI 安全）  
**关键词**: 中间 token, 思维链, 拟人化, 可解释性, RLVR

## 一句话总结
这是 Kambhampati 团队的立场文（position paper），核心主张：把推理模型（如 DeepSeek-R1）在给出答案前吐出的「中间 token」称作 "reasoning trace / thinking trace"（推理痕迹/思考痕迹）是一种**危险的拟人化**——它（1）是一厢情愿、（2）几乎没有实证支持、（3）制造对模型的虚假信任、（4）把社区推向无意义的研究方向；作者用一系列实验证据（A* 迷宫 trace 互换、trace 长度与问题复杂度脱钩、人类信任实验）论证 trace 的语义与最终解答正确性**根本脱钩**，呼吁社区停止赋予中间 token "面向用户的可解释性"，信任应来自对解答本身的验证。

## 研究背景与动机
**领域现状**：中间 token 生成（ITG, intermediate token generation）——模型在输出解答前先生成一长串 token——已成为提升 LLM 推理性能的标准手段。R1 类模型在后训练阶段用外部验证器（rule-based reward model）给"中间 token + 解答"轨迹打分，只奖励最终解答正确、**不对中间 token 直接施压**（即 RLVR，RL with Verified Rewards）。

**现有痛点**：尽管中间 token **确实**提性能这一事实无争议，但社区普遍把它们叫作"思维链/推理痕迹/思考痕迹"，并默认：①它们对用户**可解释**、能透视模型"在想什么"；②它们的正确性/可解释性与最终解答**强相关甚至因果相关**；③它们的**长度**反映"思考努力/问题难度"。R1 论文甚至把 trace 里出现 "aha" 当成模型"顿悟"的标志。

**核心矛盾**：这些假设几乎全无严格证据。中间 token 看起来像人类草稿（夹杂 "hmm…"、"wait a minute"、"aha!"），但"长得像"不等于"用途相同"，更不等于能当成透视模型思考的窗口。一旦把拟人化叙事当真，就会催生大量看似合理实则站不住脚的研究（追求 trace 可读性、按 trace 长度衡量难度、声称 transformer 学会了比 A* 更优的算法等）。

**本文目标**：（1）厘清"中间 token"到底是什么、与事后辩解/工具调用如何区分；（2）罗列拟人化的具体危害；（3）用可形式验证的实验证明 trace 语义与解答正确性脱钩、trace 长度与问题复杂度脱钩；（4）给出行动号召。

**核心 idea**：用中性术语 **derivational trace（推导痕迹）** 取代"思维链/推理痕迹"，并以"prompt + 中间 token 是否以任何**逻辑可解释**的方式导向解答"为检验标尺——大量证据表明答案是否定的，因此中间 token 不该被当作面向用户的可信解释。

## 方法详解
立场文没有 pipeline，论证骨架是"定义中性术语 → 列危害 → 摆实证 → 发号召"。下面按论证支柱拆解。

### 整体框架
论证分四步：① 先把语言上的拟人化拆掉——用 derivational trace 指代"解答前的未过滤 token"，并明确**排除**两类东西：事后辩解/总结（如 o1 隐藏真实 token 只给 sanitized summary）、以及 agentic 场景里的工具调用与工具返回（它们必须对外部有语义）；② 列出拟人化叙事衍生的一串"不健康"研究后果；③ 用**可形式验证**的实验（A* 迷宫 trace 互换、QA 子问题分解、噪声 trace 蒸馏）证明 trace 的语义正确性与解答正确性脱钩，再用 MDP 分析证明 trace 长度被 RL 机制性地拉长、与问题复杂度无关；④ 用人类受试实验证明"展示 trace 反而放大虚假信任"，最后给出号召并提出"中间 token = 学到的 prompt 增强"这一替代解释。全文的检验标尺始终是：trace 是否以**逻辑可解释**的方式导向解答，而非仅仅改变了下一个 token 的条件分布。

### 关键设计

**1. 去拟人化定义 + 危害清单**

作者首先做术语手术：把"chain of thought / reasoning trace"换成中立的 **derivational trace**，并划清三类 token 的边界——本文针对的只是"模型在解答前吐出的、对 LLM 之外无需任何语义的未过滤中间 token"，**不**包括事后辩解（o1 给的 sanitized summary）、也**不**包括 agentic 场景的工具调用（对外部环境的承诺，必须有外部语义，且在非遍历环境下可能不可逆）。在此基础上罗列拟人化的危害：把 trace 当"可解释"催生了一堆只是"让 trace 长得像伪英语"的工作（R1 甚至专门加训练去掉中英混杂）；默认 trace 正确性与解答正确性强相关，以至厂商研究发现"答案常与其 trace 不一致"竟引发惊讶；把 trace 长度当难度/努力的度量；甚至声称 Searchformer 的 transformer 因 trace 比 A* 短就"比 A* 更优"（而 A* 本是图搜索可证最优算法）。

**2. trace 语义与解答正确性脱钩（可形式验证的核心证据）**

这是全文最硬的实证支柱。作者在 A* 迷宫寻路上训练 transformer：用 A*（开/闭表操作）trace + 解路径训练，因为 trace 由 A* 生成，推理时可用 A* **形式验证** trace 是否真导向解。发现：trace 有效性与解答正确性**只有松散相关**，尤其在分布外问题上。更尖锐的是**因果干预**——把不同实例之间的 trace **互相调换**（语义上完全错误），模型性能竟**维持甚至提升**；用全对 trace 和全错（swapped）trace 训练，测试准确率都很高，**只在两者混合时才掉**。再加 GRPO 后训练实验：后训练提升解答准确率，却**不**提升 trace 语义有效性（哪怕训在无关 trace 上）。QA 域同样：用错误中间 trace 配正确解答做 SFT，甚至**超过**用正确 trace 的设置，且大量"解答对但 trace 错"的假阳性。Li et al. 噪声蒸馏、Dualformer 截断 A* trace 也都印证：**对模型有用的不是 trace 的语义，而是训练数据中一致的模式**让模型把自己拟合上去。

**3. trace 长度 ≠ 问题复杂度 / 思考努力**

针对"长度=思考努力"的拟人化，作者两路反驳。实证一：在不同难度迷宫上看 trace 长度——分布内看着像"按问题自适应计算"，但分布外这种相关**崩掉**；训在复杂迷宫上的模型遇到平凡的"no-maze"实例（起点终点都在空地、A* 几乎零计算）反而吐出极长 trace，甚至撑爆上下文窗口。理论二：Samineni et al. 分析 R1 的 MDP 形式化——在"状态=token 序列、把终端奖励**均匀分摊**到中间 token"的结构假设下，RL 被**机制性地激励**去生成更长的中间 token 序列，这被误读成"学会了更强的推理"。其根因是：把最终奖励平均分到每个中间 token 短路了 RL 本该做的信用分配，使 RL 退化成"过滤式的 on-policy 监督微调"。所以 trace 变长是奖励结构的产物，与问题从头算的计算复杂度关系很弱。

**4. 虚假信任的实证 + 行动号召**

作者把危害落到用户层面：在"用户无法独立验证解答"的真实设定下做人类受试实验，发现**展示 trace 或其摘要会无差别地抬高用户对模型预测的信任，无论答案对错**——用户因此更容易接受错误答案，即制造**虚假信任**。这正是危险所在：强大 AI 可能用"风格上像样的伪推理痕迹"利用用户认知弱点说服其相信错误答案。号召因此自然导出：停止把人类对中间 token 的解读当作解答可信度的代理；信任应来自对**解答本身**的验证（用户/第三方/问题类专用验证器，如 LLM-Modulo）；既然中间 token 主要是帮 LLM 而非用户，就别再为了"让用户看着舒服"把它约束成伪英语格式。作者进而提出替代解释——**把中间 token 看成"学到的 prompt 增强"**：对任务 $T$ 存在某个增强 $\mathrm{PA}$ 使 $\mathbb{P}(\mathrm{Sol}(\mathrm{LLM}(T+\mathrm{PA}),T))>\mathbb{P}(\mathrm{Sol}(\mathrm{LLM}(T),T))$，挑战在于学一个 $\mathrm{PA}=f_\theta(T,\mathrm{LLM})$；这种增强**无需对人类可解释**（正如对抗 jailbreak 用的乱码串也能生效），中间 token 不过是模型把当前实例拉近训练分布的"脚手架"。

### 一个例子：DeepSeek-R1 的 "aha moment"
R1 论文把 trace 里出现 "aha" token 当成模型"突然顿悟"的标志性行为。作者用这个例子戳破拟人化：人说 "aha" 确实标记了一次内部状态的突变，但模型**没有任何这种内部状态**——下一次前向传播相比 "aha" 之前，唯一差别只是上下文里多了 "aha" 这一个 token 而已。把 "aha" 解读为有意义的顿悟，正是"推导痕迹有语义、且类比人类推理"这一长期未审视假设的典型体现。这个例子把"trace 长得像人类思考"与"trace 其实无对应内部计算"的鸿沟讲得很直白。

## 实验关键数据
立场文的"实验"是作者及合作者一系列证伪研究，下面归纳其结论。

### A* 迷宫 trace 互换的核心发现

| 训练 trace 类型 | 测试解答准确率 | trace 语义有效性 |
|------|------|------|
| 全部正确 trace | 高 | 较高 |
| 全部互换（swapped）trace | **仍高** | 很低 |
| 正确/互换混合 | **下降** | 低 |
| GRPO 后训练（含训在无关 trace） | 进一步提升 | **不提升** |

结论：解答准确率与 trace 语义正确性**脱钩**，对模型有用的是训练数据的一致模式而非 trace 语义。

### 拟人化叙事 vs 实证反驳

| 拟人化假设 | 实证反驳 |
|------|------|
| trace 语义正确 ⇒ 解答正确 | swapped trace 同样高准确率；QA 出现大量"解答对、trace 错"假阳性 |
| trace 长度 = 问题难度/思考努力 | "no-maze"平凡实例反而吐超长 trace；分布外相关性崩塌 |
| RL 拉长 trace = 学会更强推理 | MDP 分析：均匀分摊终端奖励机制性地激励更长序列（≈过滤式 SFT） |
| 展示 trace 提升透明度/可信度 | 人类实验：trace 无差别抬高信任，放大对**错误**答案的接受 |

### 关键发现
- **最反直觉的一点**：用**语义完全错误**（互换）的 trace 训练，模型解答准确率不降反维持/提升，只在"对错混合"时才掉——说明模型从 trace 里榨取的是"一致的可拟合模式"，不是推理语义。
- **可解释性与有用性反向**：另一项人类受试研究显示，最帮模型涨点的 R1 式冗长 trace，在用户的可解释性、忠实性、可预测性维度上得分**最低**；而用户觉得最可懂的可验证 trace，解答准确率反而上不去——存在强烈的"对模型有用 ↔ 对用户可懂"解耦。
- **现状反讽**：除 R1 外，OpenAI/Google/Anthropic 等前沿厂商**已不展示真实中间 token**（以专有为由，却仍向用户计费），实际上已在遵循本文立场；反倒是研究社区还在指望中间 token 能给用户一个可解释的窗口。

## 亮点与洞察
- **trace 互换实验是杀手锏**：用 A* 这种可形式验证的域，把"语义对/错"做成可控变量，直接证明"语义错也照样涨点"，比任何定性分析都有力——这种"可验证域 + 因果干预"的范式可迁移到任何想检验 CoT 是否真有语义的研究。
- **MDP 视角解释 trace 变长**：把"RL 后训练让 trace 变长"从"学会推理"重新解释为"均匀奖励分摊的机制性副产物"，戳破了一个被广泛误读的现象，对理解 RLVR 本质很有启发。
- **prompt 增强替代解释**：把中间 token 看成无需人类可解释的"prompt 脚手架"，并用对抗 jailbreak 的乱码串类比，给"为什么无语义 trace 也有效"提供了一个干净的框架，也顺势支持"可以用非语言 token / 连续潜空间 token 做中间 token"的方向。

## 局限与展望
- **作者承认的局限**：立场是"中间 token **不必然**有可解释语义"，而非"永远不可能有"——偶然的可解释性可能存在，只是不可依赖；且这一立场需更多证据支撑，目前主要靠作者团队自己的系列实验。
- **自己发现的局限**：核心证据集中在 A* 迷宫、QA 等**可形式验证**或受限格式的域，泛化到通用 LRM（R1 的自由 trace 无法形式验证）时是外推；人类信任实验的设定（用户无法验证）虽现实但结论的边界条件需注意；"prompt 增强"只是 speculative 的替代解释，尚未给出可学的 $f_\theta$。
- **改进思路**：作者建议把信任交给解答验证（LLM-Modulo / 问题类验证器）；并指出 agentic 场景需区分"内部中间 token"与"工具调用"（后者是对外部的承诺、必须有语义），其团队的 LLM-Process-Modulo 是一个控制运行时行为的方向。

## 相关工作与启发
- **vs DeepSeek-R1（"aha moment" / 长度=推理）**：R1 论文主推"trace 体现思考、变长=学会推理"，本文正面反驳——"aha" 无对应内部状态、长度由奖励结构机制性拉长，是全文主要靶子。
- **vs Searchformer / Dualformer（trace 越短越优 / 截断 trace）**：Searchformer 称 trace 比 A* 短即"更优"，本文指出 A* 本是可证最优、该说法站不住；Dualformer 截断 A* trace（破坏语义）仍涨点，反被本文当作"语义无关"的证据。
- **vs CoT 监控/可解释性倡议（Korbak et al.）**：有人呼吁"保护 CoT 脆弱的可监控性"以便安全监控，本文认为这种联系可能只是幻觉，在安全攸关域不应依赖中间 token 做监控，第三方验证解答才是正道。

## 评分
- 新颖性: ⭐⭐⭐⭐ 用可验证域 + trace 互换的因果干预，把"CoT 是否有语义"做成可证伪命题，立论锋利
- 实验充分度: ⭐⭐⭐⭐ 汇集 A* 迷宫、QA、噪声蒸馏、MDP 分析、人类信任多条证据，但多在受限域、对通用 LRM 是外推
- 写作质量: ⭐⭐⭐⭐ 术语手术清晰、危害清单与证据层层递进、"aha"例子点睛
- 价值: ⭐⭐⭐⭐⭐ 直接挑战"思维链可解释"主流叙事，对 RLVR 理解、AI 安全监控、可解释性研究都有纠偏意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] When Thinking Backfires: Mechanistic Insights Into Reasoning-Induced Misalignment](../../ICLR2026/interpretability/when_thinking_backfires_mechanistic_insights_into_reasoning-induced_misalignment.md)
- [\[ICML 2026\] Learn from A Rationalist: Distilling Intermediate Interpretable Rationales](learn_from_a_rationalist_distilling_intermediate_interpretable_rationales.md)
- [\[ICML 2026\] AI Engram: In Search of Memory Traces in Artificial Intelligence](ai_engram_in_search_of_memory_traces_in_artificial_intelligence.md)
- [\[ICML 2026\] Position: Ideas Should be the Center of Machine Learning Research](position_ideas_should_be_the_center_of_machine_learning_research.md)
- [\[ICML 2026\] LatentLens: Revealing Highly Interpretable Visual Tokens in LLMs](latentlens_revealing_highly_interpretable_visual_tokens_in_llms.md)

</div>

<!-- RELATED:END -->
