---
title: >-
  [论文解读] Do LLMs Capture Embodied Cognition and Cultural Variation? Cross-Linguistic Evidence from Demonstratives
description: >-
  [ACL 2026][可解释性][指示词] 作者用「this/that」与「这/那」这类指示词（demonstrative）作为探针，构建中英双语对照数据集（80 题/语 × 4 cue × 4 perspective × 5 场景），用 320 名母语者的 6,400 条响应建立人类基线，发现英语者擅长 proximal–distal 区分但弱于他者视角，中文者反之；而 5 个 SOTA LLM 既无法稳定区分近–远，也无跨文化差异，普遍退回到 English-centric 推理或"All of the above"安全 fallback。
tags:
  - "ACL 2026"
  - "可解释性"
  - "指示词"
  - "具身认知"
  - "跨文化"
  - "对称指数"
  - "proximal/distal"
  - "自我/他者视角"
---

# Do LLMs Capture Embodied Cognition and Cultural Variation? Cross-Linguistic Evidence from Demonstratives

**会议**: ACL 2026  
**arXiv**: [2604.25423](https://arxiv.org/abs/2604.25423)  
**代码**: 待确认 (论文未直接给出)  
**领域**: 跨文化 / LLM 评测 / 具身认知  
**关键词**: 指示词、具身认知、跨文化、对称指数、proximal/distal、自我/他者视角

## 一句话总结
作者用「this/that」与「这/那」这类指示词（demonstrative）作为探针，构建中英双语对照数据集（80 题/语 × 4 cue × 4 perspective × 5 场景），用 320 名母语者的 6,400 条响应建立人类基线，发现英语者擅长 proximal–distal 区分但弱于他者视角，中文者反之；而 5 个 SOTA LLM 既无法稳定区分近–远，也无跨文化差异，普遍退回到 English-centric 推理或"All of the above"安全 fallback。

## 研究背景与动机

**领域现状**：LLM 在文本任务上突飞猛进，但学界对其是否真正掌握 grounded cognition（具身知识）仍有激烈争论。多数评测仍是知识/推理 benchmark，缺少专门探测「物理空间感」「视角切换」「文化语用」的探针。

**现有痛点**：grounded knowledge 在文本里几乎不被显式表述（说话人不会在小说里写「我现在面对桌子，桌上有杯子靠近我」），LLM 难以从纯文本学到。但要测它有没有学到，又缺少干净的语言学探针——已有 benchmark 多关注 reasoning 或 multimodal 输入，没有把「空间指示」这种 universal 且仅由几个词承载的现象单独拎出来。

**核心矛盾**：(i) demonstrative 全人类通用（2-3 岁就习得），是最 embodied 的语言现象之一；(ii) 但 demonstrative 解读高度依赖说话人物理位置 + 对话伙伴视角 + 文化习俗，纯文本里几乎没有这种 grounding 信号；(iii) 不同语言文化对「proximal-distal」和「self-other」的偏好不同——这正是测试 LLM 是否真学到「文化变异」的绝佳场景。

**本文目标**：(1) 用 demonstrative 设计可控实验，探测 LLM 是否掌握 embodied spatial grounding；(2) 比较 LLM 跨语言行为是否反映中英语用文化差异；(3) 建立人类基线，验证两种语言下 proximal-distal vs perspective-taking 的不对称。

**切入角度**：选 demonstrative 作探针有 3 个好处——它是 (a) universal 现象但跨语言用法有微妙差异，(b) 通常隐含在语境里不被显式书写（对 LLM 是 hard probe），(c) 任务设计可量化（多选 + Symmetry Index）。

**核心 idea**：用「pair-to-pair 多选 + 4 cue 条件 + 5 场景 + 反向逻辑选项（All of the above）」的精巧实验设计，强迫 LLM 暴露它是否真理解 proximal-distal 的 mutual exclusivity，并跨语言比较其行为是否随语言文化变化。

## 方法详解

### 整体框架

本文不训练模型，而是用一套精心控制的指示词实验，把"模型有没有真正 ground 空间含义"转化成可观测的选择行为。数据集共 160 题（80 题/语 × 中英两语），每题是一道四选一多选题：开头描述两个角色相对而坐的场景、指定红色标注的 speaker、给一条蓝色指令（target 对象用花括号标注，如 `{fruit}`），再反向追问"Done! Are there any items left on the place?"——即先剔除被指代的对象再问剩下什么。四个选项分别是近 speaker 的 proximal、近 interlocutor 的 distal、中间干扰项 middle，以及逻辑陷阱 "All of the above"。题目沿 4 个 cue 条件（纯指示词 / 纯代词 / demo+加强代词 / demo+冲突代词，其中纯指示词为主实验）和 5 个场景（搭配 eat、hide、take 等不同动词）展开。人类基线由 4 个独立调查构成（每 cue 一个，各 40 人 × 20 题 × 2 语），共 320 名母语者、6,400 条响应，中文走 Credamo、英文走 Prolific。LLM 侧评测 5 个 SOTA 模型（闭源 GPT-5.1、Claude-Sonnet-4.5、Gemini-2.5-Pro 与开源 DeepSeek-V3.1、Qwen3-Max），用最简洁的 zero-shot prompt refinement（"Please reply [The answer is: 1, 2, 3, 4], and give a brief reason."），每模型 10 runs 取平均，共 4,800 实例，run-to-run 标准差仅 0.02–0.08。

### 关键设计

**1.「Are there any items left」反向追问 + 含逻辑陷阱的四选项：让 grounding 能力在选择行为里现形。**

直接问"该拿哪一个"太容易被模型猜中，于是改成先剔除被指代对象、再反问剩下什么，并在候选里同时埋入近/远/中三个互斥具体项与一个反逻辑的 "All of the above"。关键在于，proximal 和 distal 在物理上天然互斥，真正理解这两个词的被试绝不会选 "All of the above"——人类选它的比例只有约 0.5%。这就把"语言理解"直接转成"行为可观察"：模型选 "All of the above" 的比例本身就量化了它对 mutual exclusivity 的掌握程度。实测中 Gemini-2.5-Pro 在 self-distal 条件下选 4 的比例高达 60%、Qwen3-Max 更达 84%，这种"安全 fallback"行为正是模型没有 ground 住空间含义的直接证据。

**2. Pair-to-pair × 4 cue × 4 perspective 的交叉控制：把混淆维度拆开，逼出 disentangled 能力**

只看单一维度（如 proximal 准确率）会被 perspective 等因素混淆，因此实验把 proximity（proximal/distal）、perspective（self/other）、pronoun reinforcement（无/有/冲突）三个维度交叉起来。每个场景生成 4 题（2 proximity × 2 perspective）一组配对，避免单题随机性，并以 Symmetry Index 量化对称性；cue 条件从纯指示词到纯代词再到加强/冲突代词，可分离"指示词理解"与"代词依赖"两种来源；中英同设计对照则用来揭示文化差异。正是这套设计让作者能 isolate 出"英语者在 distal 上准确、却在 perspective 切换时崩溃"这类细粒度跨文化特征，而非笼统的一个准确率。

**3. Symmetry Index (SI) 量化配对响应分布的对称性：替代不适用的 accuracy**

在没有唯一正确答案的开放式指代实验里，传统 accuracy 失效，作者借鉴 Robinson 1987 的步态对称性分析，定义对称指数

$$\mathrm{SI} = \frac{|A_1 - B_2| + |B_1 - A_2|}{A_1 + A_2 + B_1 + B_2},$$

其中 $A_1, A_2$ 与 $B_1, B_2$ 是两个被比较条件下两类响应的计数。以 0.1 为阈值：低 SI（<0.1）表示两条件下行为高度对称（如 Self-Proximal 与 Self-Distal 呈镜像，说明被试稳定区分近远），高 SI 则表示某条件下行为崩溃。相比 chi-square，SI 能更直观地刻画多类响应的均衡性，正是靠它才量化出"英语者同视角内 proximal-distal 对称、中文者跨视角对称"这组互补的跨语言模式。

### 损失函数 / 训练策略

本文不训练模型，只做评测。LLM 与人类的响应分布是否同源用 Rao-Scott adjusted chi-square test 检验，分布距离用 Jensen-Shannon divergence（JSD）量化；每模型 10 runs 取平均，run-to-run 标准差 0.02–0.08。

## 实验关键数据

### 主实验：人类 vs 5 LLM 在 only-demonstrative 条件下的响应分布（节选 self-distal 与 other-proximal 条件）

| 条件 | 类别 | Human-en | Human-zh | GPT-5.1-en | GPT-5.1-zh | Gemini-2.5-Pro-en | Gemini-2.5-Pro-zh | Qwen3-Max-en | Qwen3-Max-zh |
|------|------|----------|----------|------------|------------|--------------------|--------------------|---------------|---------------|
| Self-Proximal | (2,3) | **76.5%** | **84.5%** | 72.0% | 80.0% | 62.0% | 48.0% | 20.0% | 12.0% |
| Self-Proximal | (4) "All" | 1.0% | 0% | 20.0% | 18.0% | 36.0% | 18.0% | **80.0%** | **54.0%** |
| Self-Distal | (1,3) | **81.5%** | 52.0% | 30.0% | 24.0% | 10.0% | 10.0% | 0% | 0% |
| Self-Distal | (2,3) | 18.5% | **44.5%** | 0% | 0% | 0% | 0% | 12.0% | 2.0% |
| Self-Distal | (4) | 0% | 0% | 40.0% | 46.0% | 42.0% | 28.0% | **84.0%** | 74.0% |
| Other-Proximal | (1,3) | **62.5%** | **86.0%** | 60.0% | 70.0% | 36.0% | 36.0% | 0% | 0% |
| Other-Distal | (2,3) | **64.0%** | 53.0% | 30.0% | 48.0% | 16.0% | 10.0% | 2.0% | 2.0% |
| Other-Distal | (4) | 0% | 0% | 46.0% | 32.0% | 56.0% | 46.0% | **80.0%** | 64.0% |

### 消融：人类 Symmetry Index（人类基线）

| 比较 | English SI | Chinese SI |
|------|-----------|-----------|
| Self-Proximal vs Self-Distal | **0.0309 ✓** | 0.3472 ✗ |
| Other-Proximal vs Other-Distal | **0.0254 ✓** | 0.3541 ✗ |
| Self-Proximal vs Other-Proximal | 0.1731 ✗ | **0.0131 ✓** |
| Self-Distal vs Other-Distal | 0.1646 ✗ | **0.0077 ✓** |

✓ = SI<0.1（高对称），✗ = SI>0.1（不对称）。揭示完美对照：**英语者**在「同一视角下的 proximal-distal 对比」上对称（强空间区分），但「跨视角时」崩溃；**中文者**反之，跨视角对称（流畅切换 perspective），但 distal 解读模糊。

### 关键发现
- **LLM 无法理解 proximal-distal 的互斥**：人类几乎不选 "All of the above"（~0.5%），但 Qwen3-Max 选 4 的比例在 self-distal 上高达 84%（en）/ 74%（zh），Gemini 在 self-proximal-en 上选 4 的比例 36%，Deepseek-v3.1 在多条件下 40–68%。这种「安全 fallback」直接证明模型没把 demonstrative 当作空间互斥范畴处理，而是当成模糊的 puzzle 选项。
- **LLM 无跨语言文化差异**：人类在 English 中 Self-Distal 选 (1,3) 比例 81.5% / Chinese 52.0%（差 29.5 个百分点），LLM 完全无此差异——Gemini-2.5-Pro 在 4 个条件上 (2,3 / 2 / 1,3 / 1) 中英差异只有 0–6 个百分点。这意味着 LLM 在中文输入下并未启用中文语用习惯，而是用英语化推理框架处理任何语言。
- **Self-perspective 上 LLM 偶尔接近人类**：唯一在 chi-square 上 p > 0.05 的条件是 self-perspective proximal，可能因为这是最简单、最常见的语境，训练数据里充分覆盖。
- **代词显著帮助 LLM**：在 reinforcing pronoun 条件下，Claude-Sonnet-4.5 选 (1,3) 在 English self-distal 达 80%，逼近人类 89.5%（vs only-demo 条件下仅 34%）。说明 LLM 强烈依赖显式 lexical cue（代词）而非真正的空间 grounding，一旦代词去掉，model behavior 退化。
- **冲突代词揭示「代词覆盖指示词」**：在 demo + inconsistent pronoun 条件下，人类和模型都倾向跟随代词，证明代词的 definiteness 信号比 demonstrative 的空间信号更强——但 LLM 在此条件下仍频繁选 (1,2) 或 (4)，违反互斥性，说明它的"代词理解"也是浅层模式匹配。
- **跨 prompt 策略不变**：附录测试了 zero-shot / CoT / few-shot / role-play / prompt refinement 5 种策略，效果差异在 run-to-run 噪声范围内，说明这不是 prompt 工程问题而是 grounding 能力的真实缺失。

## 亮点与洞察
- **「逻辑陷阱选项」是诊断 grounding 的优雅手段**：把 "All of the above" 放进近/远互斥选项里，人类只 0.5% 选，LLM 高达 80%——一个单一指标就把 grounding 能力暴露出来。这种「让被试在不知不觉中暴露概念结构」的实验设计可以迁移到任何 LLM 能力探测（如时间因果、数量级、唯一性约束）。
- **指示词作为 grounding 探针的天才之处**：它是 (i) 全人类语言通用（universal benchmark）、(ii) 仅靠 2-3 个词承载、(iii) 高度依赖物理 grounding 但训练语料几乎不含、(iv) 跨语言用法有明确文化差异。一个语言学小现象同时勾出 embodied cognition + cultural variation 两条主线。
- **Symmetry Index 比 Accuracy 更适合无 GT 实验**：当任务本身没有唯一正确答案（如 ambiguous referential context），SI 量化「同被试在配对条件下行为的内部一致性」比看 accuracy 信息量大得多。可推广到任何"无 GT 但需测一致性"的语用实验。
- **「LLM 是 English-centric reasoner」是有定量证据的**：4 个条件下中英行为差异 0–6 个百分点，但人类对应差异 30%——这个对比直接证伪了「多语 LLM 真的理解多语文化」的乐观叙事。该结果对 multilingual / multicultural alignment 研究有直接的反驳价值。
- **「pronoun > demonstrative」for LLMs 与人类一致**：LLM 和人类都对 explicit lexical cue 更敏感，这说明问题不在「模型完全不懂代词」而在「模型只会浅层 cue matching，碰到 cue 弱（如纯 demonstrative + 需要 perspective inference）就崩溃」。
- **个体差异（individual variation）是新挑战**：人类响应在某些条件下分裂为多模式（50/50 split on Chinese distal），但 LLM 总输出 single "expert" answer——作者点出这是当前训练范式的根本限制，呼吁 individualized 模型。

## 局限与展望
- **纯文本输入限制**：作者明确承认，demonstrative 本质需要 multimodal grounding（视觉 + 空间 + 物理交互），纯文本无法完整考察具身能力，建议后续做多模态 + 3D simulation 实验。
- **数据集小（160 题）**：为保 controlled 设计而牺牲规模，统计 power 受限，且不同 discourse 场景覆盖少。
- **只有中英两语**：demonstrative 在西班牙语等是 3 元（aquí / ahí / allá），日语有 ko/so/a 三元，本研究无法覆盖更丰富的 typological 变异。
- **缺乏「为什么」分析**：发现 LLM English-centric 但没诊断到底是 (a) 训练数据 English 主导、(b) 后训练 RLHF 用英文 reward、还是 (c) tokenization/embedding 层面的英文偏置。
- **未探究 fine-tuning 是否能修复**：所有结论建立在 SOTA 现成模型上，没尝试用中文 grounding 数据 fine-tune 看能否提升。
- **改进方向**：(i) 用图像/3D 场景作多模态 demonstrative benchmark；(ii) 扩 typologically diverse 语言（西班牙语 3 元、日语 ko/so/a、阿拉伯语等）；(iii) 测试 RLHF 后的模型 vs base model 的差异，分离训练数据 vs 对齐范式的贡献；(iv) 探索 individualized 训练（让模型保留多种合理解读而非塌缩到 single answer）。

## 相关工作与启发
- **vs 传统 grounding benchmark (Embodied AI / Robot QA)**：那些 benchmark 需要 visual / 3D 输入，本文用纯文本探针就让 LLM 露馅，且任务设计极简，复现成本低。
- **vs Kauf et al. (2023) "Event knowledge in LLMs"**：Kauf 团队用 plausibility judgments 探测事件常识，本文用 referential disambiguation 探测空间 grounding，是同思路（minimal-pair 语言学探针）但聚焦更 embodied 的现象。
- **vs Xu et al. (2025) "LLMs recover non-sensorimotor but not sensorimotor features"**：该研究用 concept feature 任务发现 LLM 对 sensorimotor 维度学得差，本文从语用层面（demonstrative）独立验证同结论。两者互补，构成对 LLM grounding 能力的 converging evidence。
- **vs Demonstratives 语言学经典工作（Bühler 1934, Diessel 1999, Peeters 2016）**：那些工作用 EEG / 对话实验测人类是 egocentric 还是 sociocentric，本文把同样的实验范式迁移到 LLM 评测，并给出「两种倾向跨语言共存」的新证据。
- **vs 通用文化 benchmark（如 CulturalBench, CultureAtlas）**：那些主要测事实层（节日、饮食、礼仪），本文测语用层（指示词解读），更深入到语言行为本身的文化变异，是文化评测向「深层结构」的推进。
- **vs Hall (1976) high-context vs low-context theory**：作者用实验数据为 Hall 的中英 high/low-context 理论提供了 demonstrative 层面的支持——中文模糊但灵活，英文精确但僵硬，正与 high/low-context 一致。
- **启示**：任何 multilingual LLM benchmark 都应包含一组「人类有明显跨文化差异」的对照任务，专门检验模型是否真的在「按文化推理」而非「按英文推理然后翻译」。这个思路可推广到 politeness、honorifics、metaphor 等更多语用现象。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用 demonstrative 这一语言学小现象同时探测 embodied cognition + cultural variation，视角独特；"All of the above" 逻辑陷阱设计极妙
- 实验充分度: ⭐⭐⭐⭐ 320 母语者 + 5 LLM × 10 runs + 4 cue × 4 perspective 控制；遗憾是数据集只 160 题且只覆盖中英两语
- 写作质量: ⭐⭐⭐⭐⭐ 实验设计阐述清晰，SI 公式 + 选项设计 + 跨语言对照逻辑环环相扣；图 1/2/3 直观
- 价值: ⭐⭐⭐⭐⭐ 对 multilingual LLM 评测领域提出关键反驳——"多语 ≠ 多文化"，且诊断方法可复用到其它 grounding 探测任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations](do_llms_know_tool_irrelevance_demystifying_structural_alignment_bias_in_tool_inv.md)
- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)
- [\[AAAI 2026\] PragWorld: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics](../../AAAI2026/interpretability/pragworld_a_benchmark_evaluating_llms_local_world_model_under_minimal_linguistic.md)
- [\[ACL 2026\] Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining](crosscoding_through_time_tracking_emergence_consolidation_of_linguistic_represen.md)
- [\[CVPR 2026\] Where Culture Fades: Revealing the Cultural Gap in Text-to-Image Generation](../../CVPR2026/interpretability/where_culture_fades_revealing_the_cultural_gap_in_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
