---
title: >-
  [论文解读] Can Large Language Models Match the Conclusions of Systematic Reviews?
description: >-
  [ICLR 2026][医疗 LLM][系统综述] 作者构建了 **MedEvidence** 基准——把 100 篇 Cochrane 系统综述（SR）的结论改写成 284 道封闭式问答，并配上综述所依据的原始研究，让 LLM 在"看到和专家相同材料"的受控条件下复现专家结论；评测 25 个 LLM 后发现：推理不一定更好、模型越大边际收益越小、医学微调反而掉点，模型普遍缺乏对低质量证据的"科学怀疑"，至少 37% 的题答不对专家结论。
tags:
  - "ICLR 2026"
  - "医疗 LLM"
  - "系统综述"
  - "医学循证"
  - "LLM 评测"
  - "证据合成"
  - "科学怀疑精神"
  - "MedEvidence"
---

# Can Large Language Models Match the Conclusions of Systematic Reviews?

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=uIJyYkOgAy](https://openreview.net/forum?id=uIJyYkOgAy)  
**代码/数据**: MedEvidence 基准（论文承诺公开）  
**领域**: 医学 NLP / LLM 评测基准  
**关键词**: 系统综述, 医学循证, LLM 评测, 证据合成, 科学怀疑精神, MedEvidence  

## 一句话总结
作者构建了 **MedEvidence** 基准——把 100 篇 Cochrane 系统综述（SR）的结论改写成 284 道封闭式问答，并配上综述所依据的原始研究，让 LLM 在"看到和专家相同材料"的受控条件下复现专家结论；评测 25 个 LLM 后发现：推理不一定更好、模型越大边际收益越小、医学微调反而掉点，模型普遍缺乏对低质量证据的"科学怀疑"，至少 37% 的题答不对专家结论。

## 研究背景与动机
**领域现状**：科学文献指数增长，做一篇系统综述平均要耗费约 67 周的密集人力，于是 Deep Research、Elicit、OpenEvidence 等 LLM 辅助工具被迅速部署，连美国 FDA 都在 2025 年 5 月启动了 LLM 辅助的科学评审试点。LLM 似乎要接管循证医学这块基石。

**现有痛点**：但人们对 LLM 在系统综述这件事上的真实能力理解得很浅。一方面，过去评测 LLM 医学能力多是考"静态内部知识"（如 USMLE 选择题），考的是知识召回；另一方面，少数评测"摘要生成"质量的工作（Wallace、O'Doherty 等）缺乏可验证的标准答案、必须请医学专家逐句核对，既慢又难规模化，样本量普遍 N<10 或仅几个 case。已有的事实核查类基准（MedREQAL、HealthFC）要么不提供综述用到的原始来源（退化成知识召回），要么只给"已经合成好的分析"作为证据（退化成信息检索），都没有真正考察"跨多篇未合成的原始研究做证据推理"这件核心难事。

**核心矛盾**：系统综述的难点恰恰在于——要在多篇研究类型、样本规模、偏倚风险各不相同、甚至结论互相冲突的原始文献之间，权衡证据强度、对低质量结果保持怀疑，最后给出可靠推荐。这套"跨源合成 + 批判性怀疑"的能力，现有基准统统没考。

**本文目标**：把问题剥离到最干净的形式——**给 LLM 提供和专家完全相同的源研究，它能否复现专家系统综述里的逐条结论？** 去掉文献检索、筛选、长摘要写作这些干扰变量，只考核心的证据合成推理。

**核心 idea**：**【受控封闭问答】** 把综述结论转成"比较干预 A 与对照 B 时某结局是 higher/lower/same/uncertain/insufficient 的五选一封闭题"，标准答案直接来自 Cochrane 专家结论，从而把原本需要专家逐句核对的开放评测，变成可大规模自动判分的精确匹配任务。

## 方法详解

### 整体框架
MedEvidence 的本质是一条**"专家结论 → 封闭问答 → 配齐源文献 → 可答性校验"**的四阶段人工策展流水线，再叠加一层 LLM 辅助的元数据标注（来源一致性、医学专科）。最终交付 284 道带丰富元数据的题目，配 329 篇被引研究（114 篇有全文），随后在其上系统评测 25 个 LLM。

```mermaid
flowchart LR
    A[Cochrane 系统综述<br/>2014-2024 开源] --> B[阶段1 综述筛选<br/>所有来源须在 PubMed 可得]
    B --> C[阶段2 结论转问答<br/>读 Main Results<br/>抽统计性结论→五选一]
    C --> D[阶段3 相关研究选择<br/>按 meta 分析权重<br/>挑支撑结论的源文献]
    D --> E[阶段4 可答性校验<br/>有效研究权重≥75%]
    E --> F[(MedEvidence<br/>284 题 / 100 SR)]
    F --> G[LLM 辅助元数据<br/>来源一致性 + 专科]
    G --> H[评测 25 个 LLM<br/>精确匹配判分]
```

### 关键设计

**1. 数据来源选择：以 Cochrane 为黄金标准锚定标签可信度。** 数据全部取自 Cochrane 经 PubMed 公开的系统综述——这是由三万多名志愿临床作者维护、格式高度标准化、循证医学界长期公认的金标准来源。标准化格式让"结论→问答"的转换可以系统化进行，而 Cochrane 用 GRADE 框架显式标注的**证据确定性**（high/moderate/low/very low）正好成为分析模型行为的天然分层维度。源文献全文优先从 BIOMEDICA 数据集（CC-BY 4.0，可再分发）获取，拿不到全文的退而用 PubMed Entrez API 取摘要。

**2. 结论到封闭问答的人工转换：把开放评测压成可自动判分。** 三名有 1–5 年研究生教育背景的标注员阅读综述摘要的 "Main Results" 子节，找出"统计性地比较某干预与对照"的结论句，改写成统一句式"Is [结局] higher, lower, or the same when comparing [干预] to [对照]?"，答案落入五个固定类别。这里两个边界类别定义得很讲究：**insufficient data** 指综述作者明说没有研究或数据不足以分析该组合；**uncertain effect** 指做了分析但因证据问题无法下定论。正是这两类"承认不确定"的题，后面成了暴露 LLM 弱点的探针。

**3. 相关研究选择 + 可答性校验：保证"材料足够复现结论"。** 这是建库最难的一步——必须确保模型拿到的源研究真的"够"推出专家结论。标注员依据综述附录里 meta 分析对各来源的"权重"挑选支撑研究；随后做可答性校验：一道题被判为可答，当且仅当 meta 分析中**至少 75% 的总权重来自"有效研究"**。"有效"定义为该研究同时给出干预组和对照组的数值、且含组间差异的统计细节（原始计数、p 值、置信区间或风险比）。最常见的丢弃原因是：综述把多研究结局做了汇总，但相关研究的摘要里却没清晰报告该结局的统计量。这一步把"标签噪声"控制住了——答错不能赖材料不全。

**4. LLM 辅助的来源一致性（source concordance）标注：量化"证据冲突度"。** 用基准里最强的 DeepSeek-V3，**每次只喂一篇相关源研究**去回答该题，若单篇得出的分类与最终标准答案一致就算这篇"同意"。把"同意"的源文献占比定义为 source concordance（full / mixed / no agreement），公式上即 $\text{concordance} = \frac{\#\{\text{单源答案} = \text{SR标准答案}\}}{\#\text{相关源研究}}$。这个指标后来成了最有解释力的自变量：它直接刻画了"证据之间有多冲突"，而 LLM 表现随它单调变化，揭示了模型在冲突证据下的崩溃。

## 实验关键数据

### 主实验（25 个 LLM，zero-shot 精确匹配）

| 模型 | 平均准确率（95% CI） | 备注 |
|------|------------------|------|
| DeepSeek-V3 | **62.40%** (56.35, 68.45) | 最强 |
| GPT-4.1 | 60.40% (54.30, 66.50) | 前沿模型 |
| 人类临床专家（受时间限制） | **< 75%（最佳，仍高于所有模型）** | 粉色虚线参照 |
| 其余 23 个模型 | 均显著低于上述 | 含推理/医学/不同规模 |

要点：**没有任何模型达到或超过最佳专家**，而专家还是在限时、无法做综述作者那种深度分析的条件下完成的；前沿模型在约 **37%** 的题上答不对专家结论。

### 关键消融与分析

| 分析维度 | 发现 |
|----------|------|
| 推理 vs 非推理 | DeepSeek-V3 > 其推理版 R1；推理**不一定**更好 |
| Token 长度 | 准确率随输入 token 增长**显著下降**（即便 80% 数据能塞进上下文窗口） |
| 结局类别召回 | higher/lower 最好 → no difference/insufficient 次之 → **uncertain effect 最差**（模型不愿表达不确定，倾向硬下结论） |
| 证据确定性 | 准确率随 GRADE 证据等级**单调上升** |
| 来源一致性 | 单调上升：DeepSeek-V3 在 100% 一致时 92.45%，0% 一致时仅 41.21% |
| 医学微调 | 几乎全部**掉点**，知识型微调损害泛化 |
| 模型规模 | 7B→70B 大幅提升，**70B 以后边际收益骤减** |
| 鲁棒性补测 | 打乱源顺序、去掉 CoT 影响不显著；few-shot 略升但差距仍在 |

### 关键发现
模型不像人类专家：面对**冲突证据**会崩（一致性低时准确率腰斩）、面对**低质量证据**缺乏怀疑、面对**不确定结论**倾向过度自信地硬答。这套"跨源合成 + 批判性怀疑"恰好逃逸出当前的 scaling 范式——测试时算力、参数规模、领域微调三条路都没能补上。

## 亮点与洞察
- **问题剥离得极干净**：把"做系统综述"这件包含检索、筛选、写作的庞杂任务，收敛成"给定相同材料能否复现结论"的封闭问答，既保留了核心难度（跨源合成+怀疑），又换来可大规模自动判分，方法论上很漂亮。
- **source concordance 是点睛之笔**：用单源 LLM 答案与标准答案的一致性比例来量化"证据冲突度"，把一个抽象的"批判性推理"能力，变成可测、可分层、解释力极强的自变量。
- **结论反直觉且对部署有警示意义**：推理↑、规模↑、医学微调都不灵，而这些系统已经在临床上被使用——基准直接戳破了"上更大更专的模型就行"的乐观假设。
- **uncertain/insufficient 两类作为探针**：专门保留"承认不知道"的答案类别，精准抓住了 LLM 过度自信、缺乏科学怀疑这一深层行为缺陷。

## 局限与展望
- **选择偏倚**：只纳入"所有源都可得（全文或摘要）"的综述，可能系统性偏向某类研究。
- **不覆盖完整 SR 流程**：刻意隔离掉了文献检索、筛选、偏倚风险评估等环节，只考证据合成——是受控代价，也是天花板。
- **标签为单一综述结论**：未来可引入多专家共识，或用更新的研究刷新结论，提升基准可靠性。
- **judge 用 LLM 标注一致性**：source concordance 依赖 DeepSeek-V3 单源判定，本身可能带模型偏差。

## 相关工作与启发
- **vs MedREQAL / HealthFC**：二者要么不给原始来源（退化成知识召回）、要么只给已合成分析（退化成检索）；MedEvidence 强制"跨多篇未合成原始研究做推理"，更贴近真实 SR。
- **vs ConflictingQA / ClashEval / ConflictBank / KNOT**：这些冲突证据基准多取自 Wikipedia 或网络、用 factoid 题或人工扰动单源；本文用同行评议的真实医学文献、真实的证据冲突。
- **启发**：长上下文训练 ≠ 长上下文理解（token 越长越差再次印证）；RLHF 放大语言层面的过度自信；为循证 AI 设计评测时，"承认不确定"和"对低质量证据怀疑"应当成为一等公民指标，而非只看准确率。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — "受控复现专家结论"的问题设定和 source concordance 量化思路都很新，把难以评测的证据合成能力做成了可自动判分的探针。
- **实验充分度**: ⭐⭐⭐⭐⭐ — 25 个模型横跨 7B–671B、推理/非推理/医学微调，叠加 token 长度、证据等级、来源一致性、规模等多维分层与鲁棒性补测，证据链扎实。
- **写作质量**: ⭐⭐⭐⭐ — 动机层层递进、与相关工作的边界讲得清楚，图表（Figure 4–7）支撑有力。
- **价值**: ⭐⭐⭐⭐⭐ — 直接质疑了 LLM 辅助系统综述工具已被 FDA/临床部署的现状，提供可追踪进展的金标准基准，循证医学 AI 方向的重要参照。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Cancer-Myth: Evaluating Large Language Models on Patient Questions with False Presuppositions](cancer-myth_evaluating_large_language_models_on_patient_questions_with_false_pre.md)
- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of Large Language Models in Mental Health Question Answering](counselbench_a_large-scale_expert_evaluation_and_adversarial_benchmarking_of_lar.md)
- [\[ICLR 2026\] Knowledgeable Language Models as Black-Box Optimizers for Personalized Medicine](knowledgeable_language_models_as_black-box_optimizers_for_personalized_medicine.md)
- [\[ICLR 2026\] Can SAEs Reveal and Mitigate Racial Biases of LLMs in Healthcare?](can_saes_reveal_and_mitigate_racial_biases_of_llms_in_healthcare.md)
- [\[ACL 2026\] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](../../ACL2026/medical_nlp/beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)

</div>

<!-- RELATED:END -->
