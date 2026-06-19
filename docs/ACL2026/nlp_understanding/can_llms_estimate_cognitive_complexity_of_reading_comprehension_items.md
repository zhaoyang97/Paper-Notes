---
title: >-
  [论文解读] Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?
description: >-
  [ACL2026][NLP理解][阅读理解难度] 这篇论文构建 ReCo 阅读理解认知复杂度数据集，并系统评估 8 个 LLM 是否能自动判断题目所需的证据范围和表述转换层级，结果显示强模型能接近但仍明显低于专家，尤其不擅长识别完整证据集合和细粒度词序转换。 领域现状：阅读理解题目的难度通常依赖学生作答后的 CTT/IRT…
tags:
  - "ACL2026"
  - "NLP理解"
  - "阅读理解难度"
  - "认知复杂度"
  - "证据范围"
  - "转换层级"
  - "元认知分析"
---

# Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?

**会议**: ACL2026  
**arXiv**: [2510.25064](https://arxiv.org/abs/2510.25064)  
**代码**: https://github.com/SeonjeongHwang/ReCo  
**领域**: NLP理解 / 教育测评 / LLM评测  
**关键词**: 阅读理解难度、认知复杂度、证据范围、转换层级、元认知分析

## 一句话总结
这篇论文构建 ReCo 阅读理解认知复杂度数据集，并系统评估 8 个 LLM 是否能自动判断题目所需的证据范围和表述转换层级，结果显示强模型能接近但仍明显低于专家，尤其不擅长识别完整证据集合和细粒度词序转换。

## 研究背景与动机
**领域现状**：阅读理解题目的难度通常依赖学生作答后的 CTT/IRT 统计，或者由专家在出题阶段人工估计；NLP 侧也会抽取句长、词汇熟悉度、选项相似度等文本特征来解释难度。

**现有痛点**：这些方法要么发生在考试之后，要么只看表层语言特征。真正影响学习者负担的因素往往出现在答题推理过程中，例如需要跨几句话找证据、选项和原文之间是否只是词面匹配还是需要推断，这类认知特征过去主要依赖人工标注。

**核心矛盾**：LLM 已经具备很强的阅读理解能力，但“能答对”不等于“能解释题目为什么难”。如果模型能自动估计认知复杂度，它可以帮助出题前难度分析；如果不能，则说明 LLM 的推理能力和元认知意识之间仍有缺口。

**本文目标**：作者围绕两个认知维度展开：Evidence Scope 衡量判断答案需要引用多少文本证据，Transformation Level 衡量题干陈述和原文证据之间的语言转换程度。核心问题是：LLM 能否像专家一样给阅读理解题打上这些认知标签？

**切入角度**：论文不直接让 LLM 预测总体难度，而是把难度拆成可解释的认知因素，并用专家标注的数据集做分类评测。这比直接预测“难/中/易”更能看出模型到底理解了哪些推理负担。

**核心 idea**：用 LLM 自动估计阅读理解题的认知复杂度，同时把主任务拆成细粒度子任务，检验模型的答题能力和对自身证据/转换过程的识别能力是否一致。

## 方法详解
论文的“方法”不是提出新模型，而是提出一个数据集与评测协议。作者先从真实考试阅读理解题构造 TFNG 形式的 ReCo，再定义两个认知复杂度标签，最后用多种 prompting 策略评估不同 LLM 的分类能力和错误模式。

### 整体框架
输入是一篇阅读文章、一个陈述句以及该陈述的事实标签。模型需要输出两个维度之一的认知复杂度标签：证据范围任务要求判断证据是单句、多句还是不足；转换层级任务要求判断陈述与证据之间是词面匹配、改写、词序变化还是推断。输出结果与专家标注比较，主指标为 Macro F1。

### 关键设计
**1. ReCo 数据集构造：把多选 True/False 题拆成可标注的 TFNG 样本**

要研究"题目为什么难"，先得有带认知标签的题。作者从 RACE++ 的多选 True/False（MTF）题入手：每题含一篇 passage 和 4 个选项，作者把它拆成 `(passage, statement, factuality)` 三元组；对 False 样本，专家再写出最小修改后的 True 陈述，以便后续标注转换层级。之所以选 TFNG 形式，是因为它天然覆盖从直接匹配到多句整合、从原文可证到证据不足的多种认知负担，比普通抽取式 QA 更适合观察阅读理解题的难点来源。

**2. 双维度认知标签：用证据范围和转换层级刻画答题负担**

表层特征（句长、词频）解释不了真正的答题负担，作者因此定义两个可操作标签。**Evidence Scope（证据范围）** 衡量判断答案要引用多少文本证据，分为 single-sentence evidence、multi-sentence evidence、insufficient evidence 三类；**Transformation Level（转换层级）** 衡量陈述与证据之间的语言转换程度，在单句证据上用 5 级——word matching、transformed word matching、paraphrasing、transformed paraphrasing、inference，在多句场景下简化为 word matching、paraphrasing、inference 三级。前者对应"要读多少文本"，后者对应"从证据到答案要做多少语言/语义变换"，两者都比表层语言特征更贴近实际的推理过程。

**3. LLM 评测与细粒度诊断：不止看主任务，还要定位错误来源**

作者在 Gemma2、Mistral、Qwen2.5、GPT-4o 等多个模型上，用 standard prompting、CoT、CoT self-consistency 做 zero/one/few-shot 分类，主指标为 Macro F1。但只看主任务 F1 分不清模型是"读不懂文章"还是"答对了却说不清自己用了哪些证据"，所以作者进一步把任务拆成 falsifiability、evidence sentence counting、inference detection、paraphrasing detection、phrase reordering detection 等子任务。这些子任务直接探针模型的元认知短板——它能不能准确复盘自己引用了哪些句子、做了哪种转换。

### 损失函数或训练策略
本文没有训练新模型，主要使用推理时提示策略。标准提示直接要求输出标签；CoT 提示要求先逐步分析再预测；self-consistency 在 CoT 条件下采样 10 次，使用 top-$k=20$、top-$p=0.8$、temperature $0.7$，再按优先规则聚合答案。为避免简单样本抬高分数，作者还排除 GPT-4o 用 zero-shot CoT 可直接正确分类的过易样本。

## 实验关键数据

### 主实验

| 任务 | 最佳模型 / 设置 | 最佳 Macro F1 | 人类专家 | 关键结论 |
|------|------------------|---------------|----------|----------|
| 阅读理解事实判断 | GPT-4o CoT 1-shot | 84.4 | 未作为主对比 | 多数强模型能答题，说明认知标签错误不主要来自基础阅读失败 |
| Evidence Scope | GPT-4o CoT 1-shot | 74.8 | 87.0 | 模型能近似证据范围，但离专家仍有约 12 F1 差距 |
| Transformation Level 3级 | Mistral-24B CoT-SC zero-shot | 82.0 | 84.9 | 开源模型可接近专家，3级标签相对可学 |
| Transformation Level 5级 | GPT-4o CoT zero-shot | 61.3 | 83.0 | 细分词序重排后性能大幅下降 |

| ReCo 统计 | 数量 |
|-----------|------|
| Test passages | 151 |
| Test statements | 498，含 revised true 后为 671 |
| Demonstration passages | 83 |
| Demonstration statements | 278，含 revised true 后为 371 |
| Evidence Scope 分布 | single 388 / multi 243 / insufficient 145 |
| 3级 Transformation 分布 | word matching 123 / paraphrasing 189 / inference 319 |

### 消融实验

| 分析项 | 结果 | 说明 |
|--------|------|------|
| 5级 vs 3级 TL | 3级最高 82.0，5级最高 61.3 | 词序重排和改写组合是最难被稳定识别的细粒度维度 |
| Evidence sentence selection | GPT-4o precision 88.8 / recall 79.2 / F1 80.0 | 模型偏向少选证据句，precision 高但 recall 低 |
| Deep reasoning mode | Qwen3-32B thinking mode 低于非 thinking | 更长推理并不等于更好的认知复杂度分类 |
| Prompting | one-shot/few-shot 不总是更好 | 大模型在少样本演示下有时反而退化，说明标签边界不只是示例覆盖问题 |

### 关键发现
- LLM 的阅读理解能力和认知复杂度估计能力并不同步：模型可以答对题，却说不清自己到底引用了哪些证据或进行了何种表述转换。
- Evidence Scope 的主要瓶颈是证据句数量识别，模型倾向于只选一两句显眼证据，忽略人类标注中必要但细微的句子。
- Transformation Level 的主要瓶颈是 phrase reordering，模型常把 transformed word matching 当成普通 word matching，也会混淆 paraphrasing 和 transformed paraphrasing。

## 亮点与洞察
- 论文把“题目难度”拆成更可解释的认知标签，而不是让 LLM 直接给一个粗粒度难度分数；这让评测结果可以落到出题、改题和教学诊断的具体环节。
- ReCo 的设计很巧：TFNG 题能自然产生证据不足、多句整合和表述转换三类难点，比普通抽取式 QA 更适合分析阅读理解的认知负担。
- 最有启发的结果是“推理强”不等于“元认知强”。Qwen3 thinking mode 的退化说明，分类人类认知过程可能更依赖细粒度模式识别，而不是更长的抽象推理链。

## 局限与展望
- 数据来自 RACE++ 英语考试题，任务格式集中在 TFNG，结论能否迁移到开放问答、主旨题、作者意图题或其他语言仍需要验证。
- 标注只保留至少两位专家一致的样本，提升了可靠性，但也可能过滤掉真实考试中最有争议、最能体现难度边界的题目。
- 评测主要依赖提示工程，没有训练专门的认知复杂度模型；未来可以尝试用 ReCo 微调小模型，或把证据句检索和转换分类拆成显式多阶段系统。
- Transformation Level 的 5 级标签对模型很难，后续可考虑引入对齐式证据标注、句法重排检测器或可视化解释，让模型先定位原文片段再分类。

## 相关工作与启发
- **vs 传统 IRT/CTT 难度估计**: IRT/CTT 依赖学生作答后的统计，本文在出题前基于题目文本和认知标签估计复杂度，优势是可解释和可预分析，劣势是不能直接替代真实学生表现。
- **vs 表层文本特征难度预测**: 句长、词频、语义相似度易自动抽取，但解释不了跨句证据和推断负担；本文的证据范围与转换层级更贴近答题过程。
- **vs LLM 直接难度评分**: 直接问 LLM “这题难不难”容易得到黑箱判断；本文要求模型给出可核验的认知标签，更适合发现模型在元认知上的具体失误。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 把 LLM 难度估计落到教育心理学认知维度上，问题设定清晰且有数据集贡献。
- 实验充分度: ⭐⭐⭐⭐☆ 模型、prompt、子任务和错误分析都比较完整，但任务来源和语言范围仍偏窄。
- 写作质量: ⭐⭐⭐⭐☆ 结构清楚，标签定义和分析逻辑扎实，部分表格信息密集但结论明确。
- 价值: ⭐⭐⭐⭐☆ 对自动出题、阅读理解测评和 LLM 元认知研究都有直接启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Can LLMs Reliably Simulate Real Students' Abilities in Mathematics and Reading Comprehension?](../../ACL2025/nlp_understanding/can_llms_reliably_simulate_real_students_abilities_in_mathematics_and_reading_co.md)
- [\[ACL 2025\] Automatic Generation of Inference Making Questions for Reading Comprehension Assessments](../../ACL2025/nlp_understanding/automatic_generation_of_inference_making_questions_for_reading_comprehension_ass.md)
- [\[ACL 2026\] Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs](reasoning-based_refinement_of_unsupervised_text_clusters_with_llms.md)
- [\[ACL 2026\] Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs](creating_conlangs_to_probe_the_metalinguistic_grammatical_knowledge_of_llms.md)
- [\[ACL 2026\] Test-Time Reasoners Are Strategic Multiple-Choice Test-Takers](test-time_reasoners_are_strategic_multiple-choice_test-takers.md)

</div>

<!-- RELATED:END -->
