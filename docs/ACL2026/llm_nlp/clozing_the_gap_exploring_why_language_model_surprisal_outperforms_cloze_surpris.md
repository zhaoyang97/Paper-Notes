---
title: >-
  [论文解读] Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal
description: >-
  [ACL2026][LLM 其他][cloze surprisal] 论文系统比较 cloze 反应和 GPT2 surprisal 对人类逐词阅读时间的解释力，并通过三类概率干预证明 LM surprisal 的优势主要来自更高概率分辨率、能区分语义相近词和能给低频词分配细粒度概率。 领域现状：语言理解研究常用 surpr…
tags:
  - "ACL2026"
  - "LLM 其他"
  - "cloze surprisal"
  - "LM surprisal"
  - "阅读时间"
  - "语言预测"
  - "认知建模"
---

# Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal

**会议**: ACL2026  
**arXiv**: [2601.09886](https://arxiv.org/abs/2601.09886)  
**代码**: https://github.com/sathvikn/cloze-surprisal  
**领域**: NLP理解 / 心理语言学 / LM surprisal  
**关键词**: cloze surprisal、LM surprisal、阅读时间、语言预测、认知建模

## 一句话总结
论文系统比较 cloze 反应和 GPT2 surprisal 对人类逐词阅读时间的解释力，并通过三类概率干预证明 LM surprisal 的优势主要来自更高概率分辨率、能区分语义相近词和能给低频词分配细粒度概率。

## 研究背景与动机
**领域现状**：语言理解研究常用 surprisal 描述词在上下文中的可预测性，并用它预测阅读时间、眼动等加工努力。传统做法依赖 cloze task，即让人补全下一个词；近年越来越多研究改用语言模型条件概率。

**现有痛点**：LM surprisal 在拟合阅读时间上常优于 cloze surprisal，于是有人主张直接用 LM 取代 cloze。但两者来源不同：cloze 来自人类离线单次生产，样本数通常只有几十到一百；LM 来自大规模语料训练，可以给任意 continuation 分配细粒度概率。

**核心矛盾**：LM 概率拟合人类数据更好，不代表它“因为更像人类预测”而更好。它可能只是因为概率分辨率更高、词表覆盖更全、训练语料更大，或者捕捉了与人类预测无关的统计结构。

**本文目标**：先复现 GPT2 surprisal 相比 cloze surprisal 对阅读时间的优势，再通过有针对性的概率 manipulations 解释优势来自哪里，避免把更高拟合度误读为更强认知合理性。

**切入角度**：作者不只比较两个 predictor，而是人为降低或改写 GPT2 概率的某些能力：把概率分辨率降到 cloze 样本数、把语义相似词聚成簇、只允许高频词概率，从而观察拟合度下降多少。

**核心 idea**：通过干预 LM 概率的分辨率、语义区分和低频词覆盖，反推出 LM surprisal 为什么比 cloze surprisal 更能预测阅读时间。

## 方法详解
论文包含三个实验。实验 1 比较 cloze 与 GPT2 surprisal 对阅读时间的增量解释力；实验 2 对 GPT2 概率做三类干预来测试优势来源；实验 3 尝试用 similarity-adjusted surprisal 将 cloze response 与 LM probability 结合。

### 整体框架
输入是多个英文阅读时间数据集中的逐词上下文和目标词。作者为每个词计算 cloze-based predictability 和 GPT2-based predictability，再把它们作为 predictor 加入 linear mixed-effects regression，用 held-out log likelihood 衡量对阅读时间的解释力。核心比较是“只加 cloze”、“只加 GPT2”和“两者都加”谁能解释更多方差。

### 关键设计

**1. 强 cloze baseline 构造：先把 cloze 这一侧调到最优，再谈 LM 的优势从哪来**

如果 cloze 处理本身偏弱，那么"LM 拟合更好"就可能只是输在起跑线上。cloze 概率天生有两个麻烦：未被任何人补全的词得到零计数，而 raw probability 到底该不该取负对数也没有定论。为此作者把这两个自由度系统地搜了一遍——平滑参数取 add-one smoothing 的 $V\in\{50,100,200,500,1000,2000\}$，函数形式则覆盖 raw probability、raw surprisal 以及多种 surprisal power transform 共 6 种。在六个阅读时间指标上拟合最好的组合是 $S(w_t)^2$ 配 $V=200$，于是这一档被定为后续比较的 cloze 基线。只有把 cloze 调到它自己能达到的最好状态，后面 GPT2 仍然胜出才说明优势是真实的、而不是 cloze 处理草率带来的假象。

**2. 三类 GPT2 概率干预：人为"砍掉"LM 的某种能力，看阅读时间拟合掉多少**

复现出 GPT2 优于 cloze 之后，真正的问题是这个优势具体来自哪几种能力。作者不去比新 predictor，而是对 GPT2 概率做三种定向削弱，分别对应一个假设：H1（resolution）把 GPT2 分布按 cloze response 的样本量重新采样，再用 count-and-divide 估计概率，等于把 LM 拉到和 cloze 一样低的分辨率；H2（semantics）用 GPT2 token embedding 做 k-means 聚类，把目标词概率替换成它所属语义簇的概率，抹掉 couch/sofa 这类近义词之间的细分；H3（frequency）把低频 token 概率直接置零、只在高频词表上重归一化，去掉 LM 给罕见 continuation 分配细粒度概率的能力。哪一个干预让阅读时间拟合显著下降，就说明被砍掉的那种能力正是 LM 优势的来源之一——三个干预后拟合全都下降，于是三种能力都被坐实。

**3. Similarity-adjusted surprisal 组合尝试：让相似候选也"算半个命中"，看 cloze 与 LM 能否互补**

count-and-divide 的硬伤是只认完全一致：人类补的是 sofa、目标词却是 couch 时，目标词概率被记为零，哪怕两者语义几乎等价。SA surprisal 想修这一点，按候选 response 与目标词的 embedding 距离加权概率质量

$$P_S(w_t\mid context)=\sum_{w'\in R} z(w_t,w')\,P(w'\mid context),$$

其中 $z(w_t,w')$ 是相似度权重，$R$ 是候选集合，作者分别拿 cloze responses 和 GPT2 samples 当 $R$ 试。直觉上这能把"预测了近义词"也纳入加工便利性，但实验里两个版本对阅读时间的拟合反而更差——简单的相似度加权并没能让 cloze 与 LM 互补，这条负结果反过来提示二者的融合需要更精细的认知假设，而非一个 embedding 距离就能搞定。

### 损失函数或训练策略
本文不训练神经模型，统计建模采用 linear mixed-effects regression。每个 reading-time measure 先建 baseline，包含词长、词位、unigram surprisal、眼动中前词是否 fixated 等控制变量；再加入 cloze 或 GPT2 predictor。拟合优劣用 10-fold cross-validation 的 held-out log likelihood，显著性用 paired permutation test，并对多重比较做 Bonferroni correction。

## 实验关键数据

### 主实验

| 实验 / 设置 | 关键结果 | 解释 |
|-------------|----------|------|
| Cloze transform 搜索 | $S(w_t)^2$ + $V=200$ 的 log-likelihood 增益最高，约 153.8 | cloze probability 转为 surprisal 后明显更好，线性 probability 不够 |
| Cloze vs GPT2 | GPT2 surprisal 在 6 个 measure 中 4 个显著优于 cloze，反向不成立 | GPT2 surprisal 通常 subsume cloze surprisal，尤其在 eye-tracking 上 |
| H1 resolution | 将 GPT2 降采样成 cloze 样本数后拟合显著下降 | LM 优势很大部分来自高分辨率概率 |
| H2 semantics | 按 embedding cluster 合并语义相近词后拟合下降 | LM 对 couch/sofa 等相近词做细粒度区分有助于拟合 RT |
| H3 frequency | 限制到高频词表后拟合下降 | LM 给低频 continuation 分配细粒度概率也有贡献 |

| 数据集 | 阅读时间指标 | Cloze response 规模 | 说明 |
|--------|--------------|--------------------|------|
| BK21 SPR | self-paced reading | 每句约 90 responses | 操纵 high/moderate/low cloze，同一 target word |
| Provo ET | first-pass / go-past | 每词约 40 responses | 55 篇短段落，478 人提供 cloze |
| UCL SPR/ET | SPR / first-pass / go-past | 每词约 80 responses | 205/361 句，短句且高频词较多 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| Raw cloze probability | log-likelihood 增益约 92 左右 | 不如 surprisal 变换 |
| Raw cloze surprisal | log-likelihood 增益约 150 | 负对数概率是更合理的加工努力 predictor |
| Cloze $S^2$ + V=200 | log-likelihood 增益约 153.8 | 作者后续采用的 cloze baseline |
| Manipulated GPT2 variants | 三种均显著低于原始 GPT2 | 三个假设都得到支持 |
| SA cloze / SA GPT2 | 对 RT 拟合较差 | 简单相似度加权并未成功改进 cloze-LM 结合 |

### 关键发现
- LM surprisal 的优势不是单一原因，而是高分辨率概率、语义细分能力和低频词概率覆盖共同作用。
- 当 GPT2 概率被降采样为 cloze-like estimates 后，它不再稳定优于 cloze surprisal，这说明 cloze task 的样本分辨率限制非常关键。
- SA surprisal 的负结果同样有价值：把相似候选简单加权不一定比 count-and-divide 更好，cloze 与 LM 的融合需要更精细的认知假设。

## 亮点与洞察
- 论文没有停留在“LM 拟合更好”这个经验事实，而是做了非常干净的机制拆解，这对心理语言学里使用 LLM 指标很重要。
- H1 的设计尤其直观：让 GPT2 像 cloze task 一样只给有限样本，一下子暴露了 cloze 低分辨率的影响。
- 这篇论文提醒我们，预测人类阅读时间的好 predictor 不一定是人类大脑真实使用的 predictor；拟合度和认知解释力之间需要额外论证。

## 局限与展望
- 所有实验都基于英语模型和英语 native speaker 阅读时间数据，跨语言、跨书写系统和多语 LM 的结论尚不清楚。
- GPT2-small 是经典心理语言学 baseline，但不代表现代 LLM；更大模型、instruction-tuned 模型或不同 tokenizer 可能呈现不同概率特性。
- H2/H3 的概率 manipulations 基于 token-level 处理和 embedding cluster，作者也承认可能较粗糙，未来可用更认知合理的语义空间和词频建模。
- cloze task 本身是离线生产任务，收集更多 response 虽能提高分辨率，但未必解决实时预测与显性生产之间的机制差异。

## 相关工作与启发
- **vs 传统 cloze surprisal**: Cloze 直接来自人类，但样本少、零概率多、离线生产偏差大；LM surprisal 分辨率高但认知来源不透明。
- **vs Shain / Michaelov 等 LM surprisal 研究**: 这些工作强调 LM surprisal 更能预测阅读时间，本文进一步问“为什么更好”，并指出不能简单等同于更人类化。
- **vs similarity-adjusted surprisal**: SA surprisal 试图把语义相近候选纳入预测便利性，但本文实验显示简单版本效果不佳，启发后续做更结构化的人类预测模型。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 不是新模型，但用干预实验解释 LM surprisal 优势很有洞察力。
- 实验充分度: ⭐⭐⭐⭐☆ 多数据集、多指标、多 manipulation，统计比较严谨；模型范围可继续扩展。
- 写作质量: ⭐⭐⭐⭐☆ 心理语言学问题讲得清楚，实验链条紧密，公式和统计细节较多但必要。
- 价值: ⭐⭐⭐⭐☆ 对使用 LLM 概率做认知建模的人很有警示和方法参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] The Impact of Token Granularity on the Predictive Power of Language Model Surprisal](../../ACL2025/llm_nlp/token_granularity_impact.md)
- [\[ACL 2026\] Expect the Unexpected? Testing the Surprisal of Salient Entities](expect_the_unexpected_testing_the_surprisal_of_salient_entities.md)
- [\[ACL 2026\] An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal](an_existence_proof_for_neural_language_models_that_can_explain_garden-path_effec.md)
- [\[ACL 2026\] Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models](mind_the_gap_how_elicitation_protocols_shape_the_stated-revealed_preference_gap_.md)
- [\[ACL 2026\] Why Did Apple Fall: Evaluating Curiosity in Large Language Models](why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)

</div>

<!-- RELATED:END -->
