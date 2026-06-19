---
title: >-
  [论文解读] The Distracting Effect: Understanding Irrelevant Passages in RAG
description: >-
  [ACL 2025][信息检索/RAG][RAG] 本文提出了一个形式化的段落干扰效应（Distracting Effect）度量方法，并开发了多种获取高干扰段落的技术（包括偏斜检索和分类合成），证明了该度量跨LLM的鲁棒性，最终通过用高干扰段落微调LLM，在问答准确率上实现了最高7.5%的提升。 领域现状：检索增强生成（R…
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "RAG"
  - "干扰段落"
  - "检索增强生成"
  - "数据增强"
  - "鲁棒性微调"
---

# The Distracting Effect: Understanding Irrelevant Passages in RAG

**会议**: ACL 2025  
**arXiv**: [2505.06914](https://arxiv.org/abs/2505.06914)  
**代码**: 无  
**领域**: NLP理解 / 检索增强生成  
**关键词**: RAG, 干扰段落, 检索增强生成, 数据增强, 鲁棒性微调

## 一句话总结

本文提出了一个形式化的段落干扰效应（Distracting Effect）度量方法，并开发了多种获取高干扰段落的技术（包括偏斜检索和分类合成），证明了该度量跨LLM的鲁棒性，最终通过用高干扰段落微调LLM，在问答准确率上实现了最高7.5%的提升。

## 研究背景与动机

**领域现状**：检索增强生成（RAG）是让LLM解决知识密集型任务的关键方法。通过在prompt中添加检索到的段落，可以有效减少幻觉。然而，检索并非总是成功，检索结果中经常混入干扰段落——这些段落与查询语义相关但不包含正确答案，可能误导LLM。

**现有痛点**：（1）对干扰段落的理解停留在简单的二元分类（完全不相关 vs 干扰），缺乏量化度量；（2）现有获取干扰段落的方法局限于标准检索的top结果，在小语料库或特定查询上可能找不到足够的干扰段落；（3）随着检索器变得更强，其返回的不相关结果反而更具干扰性——这个问题会随时间加剧。

**核心矛盾**：更强的检索器本应带来更好的RAG性能，但不相关结果通过检索器的更严格筛选后，反而对LLM更具迷惑性。同时，缺乏系统的方法来量化和利用这种干扰效应。

**本文目标** (1) 如何形式化度量一个段落对特定查询的干扰效应；(2) 如何系统地获取高干扰段落；(3) 如何利用高干扰段落来提升RAG系统的鲁棒性。

**切入角度**：将干扰效应定义为LLM在仅给定查询和该段落时不选择弃权（输出"NO-RESPONSE"）的概率，这是一个既简单又有效的量化指标。

**核心 idea**：通过量化段落的干扰效应得分，结合多种获取方法（标准检索+偏斜检索+分类合成），构建高干扰训练集来微调LLM以增强RAG鲁棒性。

## 方法详解

### 整体框架

整体分为三个部分：（1）定义和计算干扰效应度量；（2）通过检索和生成两类方法获取高干扰段落；（3）用获取的高干扰段落构建训练集，微调LLM增强问答鲁棒性。

### 关键设计

1. **干扰效应度量 (Distracting Effect, DE)**:

    - 功能：量化一个不相关段落对LLM关于特定查询的干扰程度
    - 核心思路：构建prompt让LLM根据段落p回答查询q，如果段落不含答案则输出"NO-RESPONSE"。干扰效应 $DE_q(p) = 1 - p^{LLM}(\text{NO-RESPONSE}|q,p)$，即LLM不选择弃权的概率。该得分介于0到1之间，越高表示段落越容易让LLM"上当"。不需要生成完整回答，只需检查第一个token的概率即可，计算成本低。
    - 设计动机：该度量利用LLM自身识别相关信息的能力，不依赖额外参考模型，不需要假设模型的参数化记忆，且适用于问答之外的任何RAG任务

2. **答案偏斜检索 (Answer-Skewed Retrieval)**:

    - 功能：检索与查询相关但与答案无关的段落
    - 核心思路：修改dense retriever的查询嵌入，从中减去答案的信息。两种变体：减法 $E^{sub}(q,a) = E_Q(q) - \lambda E_D(a)$ 直接减去答案嵌入，投影 $E^{proj}(q,a) = E_Q(q) - \lambda \frac{\langle E_Q(q), E_D(a) \rangle E_D(a)}{\|E_D(a)\|^2}$ 投影去除答案方向分量。超参数λ控制排除答案信息的强度。检索结果再通过NLI模型排除包含正确答案的段落。
    - 设计动机：标准检索返回的top结果可能包含正确答案或高度相关段落，偏斜检索主动寻找与查询主题相关但不包含答案的段落，增加了获取干扰段落的多样性

3. **分类合成干扰段落 (Categorized Generation)**:

    - 功能：通过LLM生成不同类型的干扰段落，覆盖检索无法触及的场景
    - 核心思路：定义四种干扰段落类型，各用few-shot prompt指导Claude 3.5 Sonnet生成：(1) **Related Topic ($G^{rel}$)**：讨论高度相关主题但不含答案（如问Lincoln生日→给出其儿子Robert的生日）；(2) **Hypothetical ($G^{hypo}$)**：在假设情境中给出不同答案（如"在古罗马时代..."）；(3) **Negation ($G^{neg}$)**：以否定形式提供错误答案（如"普遍的误解是..."）；(4) **Modal Statement ($G^{modal}$)**：以不确定语气提供错误答案（如"金字塔可能是通过..."）。
    - 设计动机：在小语料库或特定主题查询中，检索可能找不到干扰段落。合成方法可以为任何查询生成干扰段落，且不同类型覆盖了不同的干扰机制

### 损失函数 / 训练策略

使用标准的指令微调loss对Llama-3.2-3B和Llama-3.1-8B进行微调。训练集构建策略"Hard"：50%的样本包含1个相关段落+4个最高干扰的段落，50%包含5个高干扰段落（无相关段落）。五个段落随机打乱顺序。对比基线"Retrieve"和"Rerank"使用标准检索的top-5结果。

## 实验关键数据

### 主实验

| 测试集 | 微调策略 | Llama-3.2-3B acc | Llama-3.1-8B acc |
|-------|---------|-----------------|-----------------|
| NQ | None (无微调) | 37.9 | 40.3 |
| NQ | Retrieve | 40.7 | 46.9 |
| NQ | Rerank | 39.7 | 47.0 |
| NQ | **Hard** | **42.8** | **49.4** |
| TriviaQA | None | 67.8 | 73.5 |
| TriviaQA | Retrieve | 67.6 | 78.7 |
| TriviaQA | **Hard** | **74.5** | **82.0** |
| WebQA | None | 41.9 | 40.6 |
| WebQA | Retrieve | 42.1 | 48.0 |
| WebQA | **Hard** | **49.7** | **51.0** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Gold段落 + 弱干扰(DE<0.2) | 准确率下降0.5-4.4% | 弱干扰段落影响有限 |
| Gold段落 + 强干扰(DE>0.8) | 准确率下降6-11% | 强干扰段落显著降低性能 |
| 跨LLM DE相关性 | Spearman相关系数很高 | 干扰效应是段落的内在属性 |
| 各方法最优占比 | R+st 52%, Gmodal ~15% | 联合使用覆盖更多查询 |
| Ungrounded实例 Hard vs Retrieve | +5.3-16.1% (3B) | 高干扰训练对无金标段落场景提升巨大 |

### 关键发现

- 干扰效应跨LLM高度一致：不同LLM（3B到70B参数）的DE得分呈现强Spearman相关性，说明干扰效应主要取决于段落本身而非模型
- 更强的检索器+重排序器返回的不相关段落更具干扰性，标准检索+重排(R+st)的top-1不相关段落对LLM的干扰效应最高
- 联合使用所有方法（检索+偏斜+合成）可以为约48%的查询找到比标准检索更具干扰性的段落
- Modal类型合成段落($G^{modal}$)平均干扰效应最高，Related Topic类($G^{rel}$)最低
- Hard微调在无金标段落（ungrounded）场景下提升最为显著（3B模型提升5.3-16.1%），因为此时模型完全依赖参数化记忆，更容易被干扰段落误导

## 亮点与洞察

- 将干扰效应从二元分类提升为连续度量，提供了更精细的理解工具
- 发现"更强检索器→更具干扰性的不相关结果"这一反直觉现象，对RAG系统设计有重要启示
- 偏斜检索的设计巧妙：在嵌入空间中减去答案方向，用向量运算表达"与查询相关但与答案无关"的语义
- 四种干扰段落类型（Related/Hypothetical/Negation/Modal）的分类为理解LLM脆弱性提供了有价值的视角

## 局限与展望

- 四种合成段落类型可能未覆盖所有干扰形式，分类法有待扩展
- 仅在问答任务上验证，在摘要、对话等其他RAG场景的适用性未知
- 仅在英文基准上实验，多语言泛化性未验证
- 训练数据仅使用800个NQ查询构建，规模较小
- 偏斜检索的λ超参数需要调优，不同查询的最优值可能不同
- 合成段落使用Claude 3.5 Sonnet生成，成本较高

## 相关工作与启发

- Cuconasu et al. (2024) 首先区分了random和distracting段落，本文将distraction量化为连续值
- Jin et al. (2024) 观察到强检索器的不相关结果更具干扰性，本文提供了更深入的分析和解决方案
- Yoran et al. (2024) 和Lin et al. (2024) 已探索用检索结果微调来增强RAG鲁棒性，本文证明了高干扰段落的额外价值
- Self-RAG (Asai et al., 2024) 通过自反思判断段落相关性的方法与本文的DE度量在哲学上相通
- 该方法对RAG系统的retriever和generator联合优化有重要参考价值

## 评分

- **新颖性**: 8/10 — 干扰效应的形式化定义和偏斜检索的设计都有创新
- **技术深度**: 8/10 — 度量设计有理论支撑，多方法体系完整
- **实验充分性**: 8/10 — 4个数据集7个LLM，分析深入且统计检验充分
- **写作质量**: 9/10 — 论述逻辑严密，例子直观，结构清晰
- **应用价值**: 9/10 — 对RAG系统鲁棒性提升有直接且实际的价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] From Ambiguity to Accuracy: The Transformative Effect of Coreference Resolution on RAG Systems](from_ambiguity_to_accuracy_the_transformative_effect_of_coreference_resolution_o.md)
- [\[ACL 2025\] PersonaBench: Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data](personabench_evaluating_ai_models_on_understanding_personal_information_through_.md)
- [\[ACL 2025\] Contradiction Detection in RAG-Based Chatbots](contradiction_detection_in_rag-based_chatbots.md)
- [\[ACL 2026\] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](../../ACL2026/information_retrieval/videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)
- [\[ACL 2025\] RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework](rageval_scenario_specific_rag_evaluation_dataset_generation_framework.md)

</div>

<!-- RELATED:END -->
