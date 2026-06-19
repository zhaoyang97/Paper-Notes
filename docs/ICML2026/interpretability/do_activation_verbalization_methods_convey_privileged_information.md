---
title: >-
  [论文解读] Do Activation Verbalization Methods Convey Privileged Information?
description: >-
  [ICML 2026][可解释性][激活语言化] 本文系统证明：当前流行的激活语言化方法（Patchscopes / LIT / SelfIE）在被用作 LLM 可解释性工具时，其性能完全可以由 "verbalizer 模型自己的知识" 解释，不需要任何 target 模型的内部激活——意味着这些工具在现有 benchmark 上看起来 work 是因为基准本身设计有缺陷，且当 verbalizer 知识超过 target 时会编造出 target 根本不具备的 "解释"。
tags:
  - "ICML 2026"
  - "可解释性"
  - "激活语言化"
  - "Patchscopes"
  - "LIT"
  - "忠实性"
  - "特权知识"
---

# Do Activation Verbalization Methods Convey Privileged Information?

**会议**: ICML 2026  
**arXiv**: [2509.13316](https://arxiv.org/abs/2509.13316)  
**代码**: https://github.com/millicentli/verb_faithfulness  
**领域**: 可解释性 / LLM 探针 / 评测基准批判  
**关键词**: 激活语言化, Patchscopes, LIT, 忠实性, 特权知识

## 一句话总结
本文系统证明：当前流行的激活语言化方法（Patchscopes / LIT / SelfIE）在被用作 LLM 可解释性工具时，其性能完全可以由 "verbalizer 模型自己的知识" 解释，不需要任何 target 模型的内部激活——意味着这些工具在现有 benchmark 上看起来 work 是因为基准本身设计有缺陷，且当 verbalizer 知识超过 target 时会编造出 target 根本不具备的 "解释"。

## 研究背景与动机
**领域现状**：理解 LLM 内部表征是可解释性领域的核心难题。近年涌现一类 "verbalization" 方法——用第二个 LLM（verbalizer $\mathcal{M}_2$）把目标模型（target $\mathcal{M}_1$）的隐状态翻译成自然语言描述，代表工作包括 Patchscopes（将 token 激活 patch 到 prompt 中相应位置）、SelfIE（同源思路）、LIT（fine-tune verbalizer 学习一层全部 token 的激活矩阵）。这些方法被宣称为 "理解 LLM 的计算" 的工具。

**现有痛点**："verbalizer 的输出反映了 target 的内部表征" 这一关键假设从未被严格检验。verbalizer 本身就是 LLM，自带世界知识，回答时到底是用 target 给的激活、还是用自己脑子里的常识，无法区分。如果它纯靠常识也能答对，那这种 "解释" 对可解释性没任何价值——你解释的不是模型而是世界。

**核心矛盾**：可解释性要求 verbalizer 传达 "privileged information"（必须通过内部激活才能获得的信息）；但 LLM 强大的参数知识使它在多数任务上仅凭输入文本就能答对，那 "借激活回答" 和 "不借激活回答" 不可区分。

**本文目标**：(1) 检验现有 benchmark 是否要求 verbalizer 真正使用 target 的激活；(2) 如果不要求，构造能区分 "知识来自 target 还是 verbalizer" 的对照实验；(3) 看 verbalizer 在知识冲突时优先信谁。

**切入角度**：把 verbalization 当成一个 NLP "shortcut learning" 问题来批判——如果模型能在不看真正应该看的输入下答对，就说明评测本身有 shortcut；类比 VQA 里的 prior bias。

**核心 idea**：设计三组对照——(a) zero-shot baseline 直接给 $\mathcal{M}_2$ 看输入不喂激活，看它能答多少；(b) activation inversion 把激活反转回输入文本，看其信息量；(c) 知识错配实验故意让 $\mathcal{M}_2$ 知道而 $\mathcal{M}_1$ 不知道某事实，看 verbalizer 报告谁的答案。

## 方法详解

### 整体框架
本文是一篇评测批判性研究，没有新模型，要解决的是 "verbalizer 的输出到底反映了 target 的内部激活、还是 verbalizer 自己的知识" 这个从未被检验的假设。它把问题转成三组层层递进的反事实对照：先用 zero-shot 基线测 "不喂激活能答多少"，再用激活反演测 "激活里到底有没有超出输入文本的信息"，最后用知识错配实验测 "知识冲突时 verbalizer 信谁"。三组实验都共享同一个判定标准——output 包含 ground-truth substring（忽略大小写）即算对，与 prior verbalization 工作对齐——并用 McNemar test + Bonferroni 校正做显著性检验。

### 关键设计

**1. Zero-shot 基线：用最朴素的反事实测激活的必要性**

verbalization 方法被推荐作可解释性工具，前提是 "激活带来了只靠输入拿不到的信息"，但这点从没被反过来证伪。本文针对 Patchscopes / LIT 沿用的 6 类 feature extraction 数据集（country_curr / food_country / ath_pos / ath_sport / prod_comp / star_const），令 $\mathcal{M}_1 = \mathcal{M}_2 = $ Llama3.1-8B-Instruct 或 Ministral-8B-Instruct，把 $x_{\text{input}}$ + question 直接拼起来问 $\mathcal{M}_2$，全程不做任何激活 patch，再与 LIT 和 Patchscopes（layer 1-15 平均）对比。这是最严苛的必要性测试：如果不喂激活就能与 verbalization 持平甚至更高，那激活的边际贡献就是零乃至负的，"这些方法揭示了 target 内部状态" 的合法性当场崩塌。

**2. 激活反演：构造 "激活 = 输入有损副本" 的替代解释**

即便有人坚持 "Patchscopes 的成功来自激活里的额外信息"，也得先排除一种平凡解释——那点信息只是输入文本的复述。为此本文训一个 T5-Base 或 Llama3 反演器，把 $\mathcal{M}_1$ 的 layer-$\ell$ 激活映射回近似输入 $\hat{x}$，再把 $\hat{x}$ 当普通输入交给 $\mathcal{M}_2$ 做 prompt + answer，全程仍不直接 patch 激活。如果这条 "反演 → zero-shot 回答" 的 pipeline 就能逼近 Patchscopes / LIT 的性能，就说明真正起作用的是激活里残留的输入信息加上 verbalizer 自身常识，而不是 target 对输入做的任何 "特权处理"。论文还拆出单层（$\ell=15$）与多层平均两档对比，确认不同 patch 强度下结论一致。

**3. 知识错配实验：直接测忠实性**

前两组实验只能说明 benchmark 有 shortcut，还不能证明 verbalizer 不忠实；这一组才是核心的忠实性测试。本文构造 (subject, relation, object) 三元组并分成两类——(a) $\mathcal{M}_1$ 知道但 $\mathcal{M}_2$ 不知道（例如 fine-tune $\mathcal{M}_1$ 学一个新事实）；(b) $\mathcal{M}_2$ 知道但 $\mathcal{M}_1$ 不知道——然后把 verbalization 输出与两个模型各自独立的 zero-shot 输出对照。若 verbalization 倾向 (a)，说明它确实在描述 target 的知识；若倾向 (b)，说明它在拿自己的常识编造 "target 的解释"。结果接近 (b)：当知识冲突时，verbalizer 频繁 fabricate 出 target 根本不具备的答案，这是全文最具杀伤力的 unfaithfulness 证据。

整套实验所用模型与配置为：$\mathcal{M}_1$ / $\mathcal{M}_2$ 取 Llama3.1-8B-Instruct 与 Ministral-8B-Instruct；LIT 沿用 LatentQA 数据集 fine-tune verbalizer；跨家族 verbalization 时额外学一个 affine map 把激活从 Llama3 空间映到 Ministral 空间。

## 实验关键数据

### 主实验
Llama3 / Ministral 上 6 类 feature extraction，$\mathcal{M}_1 = \mathcal{M}_2$，layer 1-15 平均（Table 1）：

| 方法 | country_curr | food_country | ath_pos | ath_sport | prod_comp | star_const | 平均 |
|------|--------------|--------------|---------|-----------|-----------|------------|------|
| Llama3 LIT | 0.79 | 0.45 | 0.66 | 0.84 | 0.67 | 0.41 | 0.64 |
| Llama3 Patchscopes | 0.31 | 0.21 | 0.41 | 0.73 | 0.32 | 0.28 | 0.38 |
| **Llama3 zero-shot** | **0.82** | **0.58** | 0.59 | 0.76 | 0.67 | 0.43 | **0.64** |
| Ministral LIT | 0.77 | 0.48 | 0.59 | 0.78 | 0.67 | 0.39 | 0.61 |
| Ministral Patchscopes | 0.14 | 0.01 | 0.22 | 0.61 | 0.47 | 0.15 | 0.27 |
| **Ministral zero-shot** | **0.85** | 0.45 | 0.57 | **0.83** | **0.68** | **0.67** | **0.68** |

zero-shot 与 LIT 平均分相同（0.64 vs 0.64），Ministral 上 zero-shot 反而更高（0.68 vs 0.61）——意味着 verbalization 没带来任何 "特权" 信息。

### 消融实验
跨模型 verbalization（$\mathcal{M}_1 = $ Llama3, $\mathcal{M}_2 = $ Ministral，Table 2）：

| 方法 | $\mathcal{M}_2$ | 平均准确率 | 说明 |
|------|----------------|------------|------|
| Patchscopes | Llama3（同源）| ~0.12 | 单层 $\ell=15$ |
| Patchscopes | Ministral（跨）| ~0.08 | 跨家族崩塌 |
| LIT | Llama3 | ~0.64 | 同源工作 |
| LIT | Ministral | ~0.08 | 跨家族 fine-tune 也救不了 |

激活反演（Table 3-4）：用 T5-Base 反演激活回近似输入再让 Llama3 回答，平均 0.27（country_curr 0.42），与 Patchscopes 单层 ($\ell=15$) 0.10 相比反而更高——说明反演 + zero-shot 的组合就足以解释 Patchscopes 的全部 "性能"。LIT 的高分（0.79）大部分也来自反演 + verbalizer 自身知识。

### 关键发现
- **zero-shot 持平 LIT**：彻底打脸 "LIT 揭示 target 内部状态" 的宣传——同样模型同样输入，问还是不问激活结果一样。
- **Patchscopes 不及 zero-shot**：在多数任务上 Patchscopes 性能反而低于直接问 verbalizer，说明 patch 激活把信息搞乱了，添了噪声而不是揭示了什么。
- **跨家族 verbalization 几乎失效**：affine map 之后 Patchscopes 平均 ~0.08，LIT 即使重新 fine-tune 也只有 ~0.08，说明这些方法严重依赖 "$\mathcal{M}_1 = \mathcal{M}_2$" 这种作弊条件。
- **知识错配下 fabricate**：当 $\mathcal{M}_2$ 知道某事实而 $\mathcal{M}_1$ 不知道时，verbalization 输出 $\mathcal{M}_2$ 的答案——意味着它编造了 target 根本不具备的知识，这是最直接的 unfaithfulness 证据。
- **反演几乎解释了一切**：把激活反演回近似输入再问 zero-shot 模型，能复现 Patchscopes 的多数性能，意味着这些方法的 "成功" 是激活泄漏输入 + verbalizer 自身常识的组合，没有任何 "特权" 成分。

## 亮点与洞察
- **极简却致命的对照设计**：zero-shot 基线这种朴素到不像研究的操作，却直接戳破了整个子领域的方法论假设。验证了 "做基线" 在 AI 研究里的不可替代价值。
- **"特权信息" 的概念框架**：从知识论（Alston 1971）借来 "privileged knowledge" 概念，给可解释性研究提供了清晰的 evaluation criterion——这是 verbalization 是否成立的判定标准。
- **激活反演作为 null hypothesis**：用 inversion 模型构造 "激活含有的输入信息" 这一替代解释，是个非常聪明的反事实——一旦反演能达到与 verbalization 相同的性能，verbalization 就再难自圆其说。
- **批判性 ICML 论文**：不是发明新方法而是证伪老方法，对学界的方法论健康度有重要意义；这种工作在大模型时代尤其稀缺也尤其必要。
- **同时质疑 benchmark 和方法**：作者点明很多 verbalization benchmark 本身设计有缺陷（不要求特权信息），未来研究应先修评测。

## 局限与展望
- **只测了 feature extraction 和 factual recall**：未覆盖更复杂的 verbalization 用例如行为解释、reasoning trace、危险知识检测，结论可能不完全推广。
- **未提出修复方案**：批判清楚了，但 "该如何设计真正测试 privileged information 的 benchmark" 没给完整方案，只在结尾呼吁需要 controlled tasks。
- **跨家族 affine mapping 可能没充分调优**：跨家族失败也可能是映射没学好，而非根本不可行；需要更彻底的对照实验。
- **依赖 "知道与否" 的二元标签**：知识错配实验里 "$\mathcal{M}_1$ 知道但 $\mathcal{M}_2$ 不知道" 的判定本身就模糊，模型对事实的掌握有概率分布。
- **激活反演用 T5-Base / Llama3 训了多大算力**：成本未充分披露，可能影响该 "替代解释" 的强度。
- **未来方向**：作者建议设计 "target 模型才有的知识" 的合成任务做 ground-truth 测试；这是非常合理的下一步研究方向。

## 相关工作与启发
- **vs Ghandeharioun 2024 (Patchscopes)**: Patchscopes 原文宣称揭示 LLM 计算；本文用 zero-shot 反例打掉这种宣称。是直接证伪关系。
- **vs Pan 2026 (LIT)**: LIT 通过 fine-tune verbalizer 学习激活，本文证明 LIT 的高分也可被 "反演 + verbalizer 知识" 解释，且跨家族失效。
- **vs Belrose 2023 (TunedLens) / nostalgebraist 2020 (logitlens)**: 这些 lens 方法都是 Patchscopes 的特例，同样面临本文的批判——产生的描述是否真在传递 target 特有信息。
- **vs VQA prior bias 工作 (Goyal 2017)**: 同样是 "不看应该看的输入也能答对" 的 shortcut 问题，本文把这种批判范式移植到 LLM 可解释性。
- **启发**：这种 "反事实评测" 思路应该推广到几乎所有 LLM evaluation 上——任何宣称需要某种特殊能力 / 输入的 benchmark 都该测一遍 zero-shot 替代基线。

## 评分
- 新颖性: ⭐⭐⭐⭐ 不是发明新方法，但 "用 zero-shot + 反演 + 知识错配" 三件套系统证伪整个 verbalization 子领域的方法论假设，框架级新颖。
- 实验充分度: ⭐⭐⭐⭐ 双模型家族 × 6 类 feature extraction × Patchscopes/LIT × 反演两种 inverter × 单层/多层平均，覆盖度足；只是任务限于 QA-style 抽取，复杂行为解释未涉及。
- 写作质量: ⭐⭐⭐⭐⭐ 论点清晰、对照实验层层递进（先 zero-shot → 再反演 → 再知识错配），statistical significance 标注规范，每张表都直接服务论点。
- 价值: ⭐⭐⭐⭐⭐ 对可解释性社区有 "刹车" 价值，迫使后续 verbalization 工作必须先证明 benchmark 不能被 shortcut，是真正改变研究范式的批判性论文。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Do Different Prompting Methods Yield a Common Task Representation?](../../NeurIPS2025/interpretability/do_different_prompting_methods_yield_a_common_task_representation_in_language_mo.md)
- [\[ICML 2026\] Steer Like the LLM: Activation Steering that Mimics Prompting](steer_like_the_llm_activation_steering_that_mimics_prompting.md)
- [\[ICML 2026\] On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders](on_the_relationship_between_activation_outliers_and_feature_death_in_sparse_auto.md)
- [\[ICLR 2026\] Concepts' Information Bottleneck Models](../../ICLR2026/interpretability/concepts_information_bottleneck_models.md)
- [\[ICML 2026\] In Defense of Information Leakage in Concept-based Models](in_defense_of_information_leakage_in_concept-based_models.md)

</div>

<!-- RELATED:END -->
