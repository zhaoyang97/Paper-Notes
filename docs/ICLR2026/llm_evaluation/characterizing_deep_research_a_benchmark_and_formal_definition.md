---
title: >-
  [论文解读] Characterizing Deep Research: A Benchmark and Formal Definition
description: >-
  [ICLR2026][LLM评测][深度研究] 这篇论文给"深度研究（Deep Research, DR）"这个被各家模型抢着用、却从没被严格定义过的任务下了一个形式化定义——核心不是"输出长报告"而是"搜索过程中对概念的高扇出（high fan-out）"，并据此构造了 100 道开放网络任务的 benchmark LIVEDRBENCH，用基于 claim 的精确率/召回率做客观打分，发现当前最强的 OpenAI DR 也只有 0.55 的平均 F1，系统普遍只覆盖了约一半的必要搜索查询。
tags:
  - "ICLR2026"
  - "LLM评测"
  - "深度研究"
  - "形式化定义"
  - "claim 评测"
  - "问题反演"
  - "信息合成"
---

# Characterizing Deep Research: A Benchmark and Formal Definition

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=5EmpOCq1Ql](https://openreview.net/forum?id=5EmpOCq1Ql)  
**代码**: https://github.com/microsoft/LiveDRBench  
**领域**: LLM 评测 / Deep Research / Agent benchmark  
**关键词**: 深度研究, 形式化定义, claim 评测, 问题反演, 信息合成

## 一句话总结
这篇论文给"深度研究（Deep Research, DR）"这个被各家模型抢着用、却从没被严格定义过的任务下了一个形式化定义——核心不是"输出长报告"而是"搜索过程中对概念的高扇出（high fan-out）"，并据此构造了 100 道开放网络任务的 benchmark LIVEDRBENCH，用基于 claim 的精确率/召回率做客观打分，发现当前最强的 OpenAI DR 也只有 0.55 的平均 F1，系统普遍只覆盖了约一半的必要搜索查询。

## 研究背景与动机
**领域现状**：2024 年底到 2025 年，Google、OpenAI、Perplexity、Grok 等纷纷推出"深度研究"产品，宣称能替人类专家完成需要数小时的调研、写报告、回答大海捞针式的问题。开源界也跟进了 HuggingFace DR、WebThinker、DeerFlow 等一批 agentic 实现。

**现有痛点**：尽管热度很高，"深度研究"到底是什么任务从来没人说清楚。人们模糊地把它理解成"写报告且耗时较长的任务"，并默认它比多跳问答（multi-hop QA）难得多，但这种理解既不精确也不可度量。更麻烦的是，任务的难度还取决于检索语料：像"列出所有由女性作家著作改编的奥斯卡获奖电影"看似是硬核调研任务，但只要网上恰好有一个网页直接列出了答案，它就退化成一次普通检索。缺少形式化定义，导致根本无法客观评估 DR 模型、衡量进展。

**核心矛盾**：现有 DR 评测要么主观（用 LLM 当裁判给报告的"全面性/洞察力/可读性"打分，如 DeepResearch Bench），要么数据不公开（FutureSearch 的 Deep Research Bench），要么用静态语料、无法直接对比闭源和开源系统。而"报告质量"这个评测对象本身就混进了文风、措辞等与推理能力无关的因素。

**本文目标**：(1) 给 DR 任务一个能把它和其他推理密集型任务区分开的形式化定义；(2) 构造一个能客观、可复现、可持续更新、且覆盖广泛领域的 benchmark。

**切入角度**：作者主张 DR 真正的难点不在"生成长报告"这一步，而在前置的**信息合成**——从语料里把一堆相关信息单元找出来、处理好、组合起来。于是把 DR 输出抽象成一个由 claim 构成的中间表示，把"推理挑战"从"表层报告生成"里剥离出来单独评测。

**核心 idea**：用"搜索强度（search intensity）× 推理强度（reasoning intensity）"两个维度来定义 DR，并用一个嵌套的 claim 列表作为可客观打分的中间输出表示。

## 方法详解

### 整体框架
论文做的是"定义 + benchmark"两件事，没有训练任何模型。整体逻辑是：先把 DR 任务形式化为一个二维空间里的极端点（同时高搜索、高推理），把它的输出抽象成 DAG 上合成出来的一组 claim；再设计一套基于 claim 的精确率/召回率指标，让"对不对"可以被客观打分；然后用"问题反演（problem inversion）"技术把现成的长文档推理题倒过来批量造出 DR 题目，构成 LIVEDRBENCH 的 100 道题、8 个类别；最后拿这套 benchmark 去测 OpenAI/Perplexity/Gemini 的 DR 产品和若干带搜索的 LLM 基线，并通过分析推理轨迹（reasoning trace）量化它们的搜索覆盖度与深度。

整篇是一个 benchmark/定义型工作，没有可画成 pipeline 的多模块算法，因此不强加框架图；下面把四个核心设计点讲清。

### 关键设计

**1. DR 的形式化定义：搜索强度 × 推理强度，而非"输出长度"**

作者直接反对"DR = 写长报告"这种通俗理解，把 DR 重新刻画为多跳 RAG 的极端版本。给定语料 $C$ 和查询 $q$，定义里有两个必要条件：(a) **搜索强度**——回答它需要处理的"信息单元（information unit）"数量很大；(b) **推理强度**——在"找到这些信息单元 / 处理它们 / 组合它们"这三个子任务里，至少有一个需要人类专家的非平凡推理。这里特意先定义了"信息单元"：它是一段落级别的原子信息（检索系统里通常实现为一个 chunk），所以一篇大文档可以包含多个信息单元——这样像"从一份 200 页财报的多个章节取数"也能算多跳。论文给出可操作的经验阈值：人类专家用网络搜索等工具求解 > 10 分钟即算推理密集；搜索强度约为 20 个信息单元（通过 ≥ 10 次查询）。

这个定义的价值在于它能把一堆容易混淆的任务摆到正确位置（论文的 Figure 1a）：HotpotQA 这类多跳 QA 搜索和推理都低；CURIE、CUAD 这类科学/法律 QA 推理高但搜索少；ELI5 这类解释任务有推理但检索集中在单一概念；只有 GAIA、Humanity's Last Exam 以及本文关注的"写政策建议/学术综述/分析报告"这类信息合成任务才同时落在高搜索、高推理的角落。一个反例很说明问题："这篇论文用的材料的属性是什么"（低搜索）和"NeurIPS 2024 接收论文列表"（低推理）都不算 DR。

**2. 基于 claim 的中间表示与精确率/召回率：把客观评测从主观报告里剥出来**

为了客观打分，作者把 DR 的解形式化为一个 ⟨query, list of claims⟩ 的嵌套结构：理想解要把所有 claim 及其（递归的）subclaim 都答对。每个顶层 claim 是一个字典，subclaim 是它的键。评测时，给每个 claim $A_i$ 一个与 ground-truth 的一致性分数 $s(A_i)$（通常是 $\{0,1\}$ 的二值分，本文用 GPT-4o 来判定），然后精确率/召回率定义为：

$$\text{Prec}(A) = \frac{\sum_{A_i} w_i\, s(A_i)\, \text{Prec}(A_i)}{\sum_{A_i} 1}, \qquad \text{Rec}(A) = \frac{\sum_{A_i} w_i\, s(A_i)\, \text{Rec}(A_i)}{\sum_{A_i^*} 1}$$

其中 $\text{Prec}(A_i)$、$\text{Rec}(A_i)$ 是该 claim 下所有 subclaim 一致性分数的平均（当 claim 是原子的、没有 subclaim 时取 1），$w_i$ 是可选的 claim 权重。关键的设计取舍是：**一个 claim 的得分要乘上它 subclaim 的精确率/召回率**——也就是说 subclaim 错了会把整个 claim 的信用拉低甚至归零（论文还给了一个 subclaim 一错就判零的严格版本）。这背后有两个动机：用户本来就会用 subclaim 去核验 claim 的可信度；而且这样能奖励"真的去搜语料找到答案"的系统，惩罚"靠记忆背出来"的系统。这套指标把不同形态的 DR 任务（列实体、找数据集、找材料）统一到了同一个可比的尺度上。

**3. 问题反演（problem inversion）：把长文档推理题倒过来，可扩展地造 DR 题**

要在"模型能访问整个 Web"的前提下造题非常棘手——网上随时可能冒出一篇正好回答某题的文章，把 DR 任务塌缩成单次检索。作者列了 5 条 desiderata（不能被单一网页回答、需要广泛多源、客观可复现、可周期性更新、覆盖多领域多用户），并借鉴 OpenAI BrowseComp 的思路——造"答案难找但易核验"的题——把它**模板化**成三步反演流程：① 找一篇信息密集的长文档（长上下文推理题尤其合适，因为它本身就自带可用来定位实体的问题）；② 抽出文档里描述的、带独特特征的实体或概念类；③ 反过来出一道题，要求在不知道原文档的情况下，找到具备这些特征的实体并定位佐证来源。

举例来说，标准的长上下文推理题是"给定一篇关于某实验的论文和一组键，按文档填出每个键的值"；反演后变成"给定符合某个唯一实体的键值对，识别出这个实体并定位佐证它的来源"。由于科学论文、公开报告、榜单都在持续更新，这套模板天然支持 benchmark 的持续刷新（防污染）；而且它允许造出答案包含多个实体/多条 claim 的题。作者会人工核验每道题是否有唯一答案，若不唯一就补充扩展 ground-truth。

**4. LIVEDRBENCH 的任务构成：100 题、8 类，覆盖科学、创新与世界事件**

最终 benchmark 含 100 道题、8 个类别，刻意覆盖科学家、信息工作者和普通网民三类用户、并让搜索/推理需求有梯度：**SCIFACTS**（31 题，源自 CURIE）分 Materials（17 题，按测量属性找材料，claim 是材料名、subclaim 是来源论文标题）和 Geo（19 题，找用了给定地理数据集的论文）；**NOVELDS**（20 题，基于数据集论文）分 Identification（按特征找数据集，subclaim 含年份/会议/链接）、Identification and Extraction（还要从论文里抽具体结果，如读图）、Peer Retrieval（找同问题空间的同类数据集论文）；**PRIORART**（17 题，把 ICLR2025/ICML2025 里 2-3 篇论文的想法手工拼成一段新摘要，要求识别其中的关键想法和源论文，模拟专利/创新查新）；**FLIGHTS**（7 题，基于官方航空事故报告反演，给一段高层描述去定位具体航班事故）；**ENTITIES**（20 题，围绕全球文化/事件——如奥斯卡——要求给出满足一组细则的实体穷举列表）。每一类都能用新论文、新事件持续扩充。

### 损失函数 / 训练策略
本文不训练模型，无损失函数。评测协议要点：claim 一致性分数由 GPT-4o 判定，最终精确率/召回率/F1 按上面的公式（Equation 1）计算；推理轨迹分析（覆盖度、依赖度、分支/回溯）也用 GPT-4o 完成。

## 实验关键数据

### 主实验
作者测了三家闭源 DR 产品（OpenAI DR、Perplexity DR、Gemini DR with 2.5 Pro）、一个开源 DR agent（DeepResearcher + DS-Qwen-32B），外加 3 个推理基线和 3 个非推理基线（均开了网络搜索）。所有 DR 模型在 8 类上的 F1 区间为 0.0–0.72，OpenAI DR 平均最强。

| 子类 | OpenAI DR (F1) | Perplexity DR (F1) | Gemini DR (F1) | DeepResearcher (F1) |
|------|------|------|------|------|
| SCIFACTS Materials | 0.314 | 0.150 | 0.022 | 0.000 |
| SCIFACTS Geo | **0.721** | 0.186 | 0.316 | 0.000 |
| NOVELDS Identification | 0.667 | 0.633 | 0.400 | 0.167 |
| NOVELDS Id.&Extraction | 0.470 | 0.333 | 0.345 | 0.023 |
| NOVELDS Peer Retrieval | 0.585 | 0.311 | 0.338 | 0.042 |
| PRIORART | 0.539 | 0.419 | 0.082 | 0.199 |
| ENTITIES | 0.603 | 0.447 | 0.338 | 0.076 |
| FLIGHTS | 0.540 | 0.276 | 0.090 | 0.090 |
| **平均** | **0.555** | 0.355 | 0.263 | 0.075 |

开源 DeepResearcher 除了 PRIORART（0.199，反超 Gemini DR）外几乎全面落后，凸显了闭源与开源 DR 系统之间的巨大差距。

### 消融 / 对比实验
在最难的 NOVELDS Identification and Extraction 子类上，把 DR 模型和带搜索的普通 LLM 基线放一起对比，普通模型（即便是推理增强的）表现都很差，说明 benchmark 确实难、且"DR 系统"相对"搜索 + 推理的 LLM"有实质增量：

| 模型 | Precision | Recall | F1 |
|------|------|------|------|
| OpenAI DR | 0.526 | 0.448 | 0.470 |
| Perplexity DR | 0.325 | 0.349 | 0.333 |
| Gemini DR | 0.406 | 0.329 | 0.345 |
| OpenAI o4-mini（推理基线） | 0.203 | 0.146 | 0.168 |
| Gemini 2.5 Pro（推理基线） | 0.186 | 0.130 | 0.142 |
| Gemini 2.5 Flash（非推理） | 0.211 | 0.097 | 0.111 |
| OpenAI GPT-4.1（非推理） | 0.126 | 0.078 | 0.088 |
| DeepResearcher + DS-Qwen-32B | 0.045 | 0.015 | 0.023 |
| Sonar Reasoning | 0.015 | 0.003 | 0.005 |

### 关键发现
- **覆盖度只有约一半**：作者用 GPT-4o 分析推理轨迹，先生成回答问题所必需的"特征查询 + 抽取查询"，再看 DR 模型轨迹里覆盖了多少。OpenAI DR 必要查询覆盖率 66.0%，Perplexity 52.0%、DeepResearcher 53.4%、Gemini 46.8%——即便最强系统也只搜到约三分之二的必要查询，普遍只覆盖约一半，改进空间巨大。
- **闭源走得更宽更深**：闭源模型平均发 24–64 条查询、其中 15–39 条是依赖前序结果的"dependent query"（深度的代理指标），而 DeepResearcher 只发 5–6 条、dependent 仅 5 条。OpenAI DR 的分支数（branching）也最高，呼应它最高的覆盖率，说明**广度是 DR 答案准确性的关键**；回溯（backtracking）上各家差异不显著。
- **claim 和 subclaim 都对才算对，是最难的地方**：模型在 SCIFACTS Materials、NOVELDS Id.&Extraction、FLIGHTS 上最差，因为要跨论文/报告的多个章节同时抽出主 claim 和 subclaim。以 SCIFACTS Materials 为例，OpenAI DR 单看论文标题 F1 有 0.735、单看材料名 0.504，但要求两者都对时整体只剩 0.314——精确抽取的难度被这套 subclaim 评测如实暴露了。

## 亮点与洞察
- **把"DR 是什么"从口号变成可度量的二维定义**：用搜索强度 × 推理强度刻画 DR，并给出 ">10 分钟、≥20 信息单元、≥10 次查询"的经验阈值，第一次让"这道题到底算不算 DR"有了可操作判据；"难度依赖语料"这个常被忽视的点（一个网页就可能让 DR 塌缩成检索）也被明确指出。
- **claim 评测把推理能力从文风里剥离**：报告评测天然主观，作者用嵌套 claim + "subclaim 错则 claim 归零"的乘法式打分，既客观可复现，又奖励"真去搜"而非"背记忆"，这套思路可迁移到任何长文本/信息合成任务的评测。
- **问题反演是可持续 benchmark 的工程亮点**：把现成的长上下文推理题倒过来批量造"答案难找易核验"的 DR 题，天然支持随新论文/新事件刷新题库、对抗数据污染，比"请专家手写题"可扩展得多。
- **轨迹分析直接指出改进方向**：覆盖度仅约 50%、广度比深度更决定准确性——这把"DR 该往哪改"从玄学变成了可量化的诊断（先把搜索广度做上去）。

## 局限与展望
- **claim 一致性用 GPT-4o 判定**：精确率/召回率、覆盖度、查询依赖度都依赖同一个 LLM 裁判，裁判本身的偏差/噪声会传导进所有指标；论文未充分讨论裁判一致性。
- **只评信息合成、不评最终报告质量**：作者明确把"给定 claim 写报告"剥离出去当作独立的长文本生成问题，因此 benchmark 不反映报告的组织、论证、可读性——一个 claim 全对但报告写得很差的系统在这里仍是满分。
- **DR 还包含本文未覆盖的能力**：computer use、写代码、调外部工具等也是 DR 的一部分，本文聚焦信息合成，结论不能直接外推到这些维度。
- **经验阈值偏启发式**：">10 分钟 / 20 信息单元 / 10 查询"是实用近似，边界附近的任务归类会有主观性。
- **改进展望**：对 ENTITIES 这种"算法简单但搜索繁琐"的任务，作者建议交错使用程序控制和模型控制；对 SCIFACTS 这种"答案对但 grounding 错"的情况，靠更好的训练来修；轨迹分析则可用于指导广度/深度的权衡。

## 相关工作与启发
- **vs DeepResearch Bench (Du et al., 2025)**：它用 LLM 裁判给报告的全面性/洞察力/可读性打主观分；本文改用基于 claim 的客观打分，把推理对错和文风分开，且支持可更新题库。
- **vs FutureSearch Deep Research Bench**：它评的是有经济价值的任务（数值答案、证据收集、claim 校验），但数据集不公开；本文提供公开 benchmark，覆盖大海捞针到广度枚举两端，并用整个 Web 当语料以便直接对比闭源/开源。
- **vs BrowseComp (Wei et al., 2025a)**：本文借用了它"造答案难找易核验的题"的反演思想，但把人工逐题构造模板化、并扩展到答案含多个实体/多条 claim 的情形。
- **vs 多跳 QA（HotpotQA 等）与 Mind2Web**：本文用搜索强度 × 推理强度把 DR 与这些低搜索或低推理的任务在定义上区分开，并显式建模了广度（搜索覆盖）与深度（claim 正确性）两个之前工作未显式刻画的维度。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一个给"深度研究"下可度量形式化定义、并配套客观 claim 评测的工作
- 实验充分度: ⭐⭐⭐⭐ 覆盖 3 家闭源 + 1 开源 DR 与 6 个基线、8 类任务，并做了推理轨迹的覆盖/深度/分支分析
- 写作质量: ⭐⭐⭐⭐ 定义—指标—构造—评测的逻辑链清晰，图表（任务空间、问题反演、F1 分类别）到位
- 价值: ⭐⭐⭐⭐⭐ 给一个正在爆发却缺乏共识定义的方向立了客观标尺，对推动 DR 评测标准化价值很大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] DRBench: A Realistic Benchmark for Enterprise Deep Research](drbench_a_realistic_benchmark_for_enterprise_deep_research.md)
- [\[ICLR 2026\] An Open-Ended Benchmark and Formal Framework for Adjuvant Research with MLLM](an_open-ended_benchmark_and_formal_framework_for_adjuvant_research_with_mllm.md)
- [\[ICLR 2026\] DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents](deepresearch_bench_a_comprehensive_benchmark_for_deep_research_agents.md)
- [\[ICLR 2026\] ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents](researchrubrics_a_benchmark_of_prompts_and_rubrics_for_evaluating_deep_research_.md)
- [\[ICLR 2026\] LiveResearchBench: A Live Benchmark for User-Centric Deep Research in the Wild](liveresearchbench_a_live_benchmark_for_user-centric_deep_research_in_the_wild.md)

</div>

<!-- RELATED:END -->
