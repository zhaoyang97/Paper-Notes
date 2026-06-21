---
title: >-
  [论文解读] ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents
description: >-
  [ICLR 2026][LLM评测][深度研究智能体] ResearchRubrics 用 2800+ 小时人工，给 101 个真实开放式研究 prompt 配上 2593 条专家手写、带权重的细粒度评分细则（rubric），再用 LLM-as-Judge 按细则逐条打分，评测主流 Deep Research 系统，发现连 Gemini DR、OpenAI DR 这类最强 agent 的平均细则达成率都不到 68%，瓶颈集中在隐含需求推断和多源信息综合。
tags:
  - "ICLR 2026"
  - "LLM评测"
  - "深度研究智能体"
  - "人工评分细则"
  - "LLM-as-Judge"
  - "任务复杂度"
  - "开放式评测"
---

# ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ErnvfmSX0P](https://openreview.net/forum?id=ErnvfmSX0P)  
**数据**: https://huggingface.co/datasets/ScaleAI/researchrubrics  
**领域**: LLM 评测 / Deep Research Agent / Benchmark  
**关键词**: 深度研究智能体, 人工评分细则, LLM-as-Judge, 任务复杂度, 开放式评测

## 一句话总结
ResearchRubrics 用 2800+ 小时人工，给 101 个真实开放式研究 prompt 配上 2593 条专家手写、带权重的细粒度评分细则（rubric），再用 LLM-as-Judge 按细则逐条打分，评测主流 Deep Research 系统，发现连 Gemini DR、OpenAI DR 这类最强 agent 的平均细则达成率都不到 68%，瓶颈集中在隐含需求推断和多源信息综合。

## 研究背景与动机
**领域现状**：Deep Research（DR）是一类新兴的 LLM agent 应用——给定一个开放式问题，agent 自主进行多步网络检索、定向取证、跨文档综合，最后产出一份有证据支撑的长文报告。OpenAI、Google、Perplexity 都已上线商用 DR 系统，并在 HLE 等基准上刷出不错的分数。

**现有痛点**：但 DR 的输出又长又杂、往往没有唯一正确答案，现有评测方法都接不住。传统 QA 基准（HLE、HotpotQA 等）只考短答案、可机械核对的事实题，比如「哪种材料带隙 0.9 eV、位错密度 $4\times10^8\,\text{cm}^{-2}$」→「氮化镓」，完全无法度量长文、多源综合的质量。而近期专门给 DR 设计的基准又各有硬伤：有的（DeepResearch Bench）用 LLM 自动生成评分细则、甚至用 LLM 生成的参考报告做评测，存在循环论证和监督失灵的风险；有的（DeepScholar-Bench、ReportBench）把范围窄化到「写 Related Work」这种单一技术任务，覆盖不了用户真实会问的从商业报告到消费决策的五花八门话题。

**核心矛盾**：评测 DR 既要**域多样性**（贴近真实用户的杂问题），又要**专家手写的细粒度细则**（避免 LLM 自评的锚定偏差和粗糙重叠指标），现有工作总是顾此失彼——要么细则是机器造的、要么任务被压窄到方便评的学术域。

**本文目标**：造一个「真实开放式 prompt × 专家手写细则」的标准基准，能逐条、原子级地诊断 DR agent 到底在哪一类要求上失败。

**切入角度**：作者认为 DR 任务的难度不是一维的，而应沿三条正交轴刻画——概念广度、逻辑嵌套深度、探索性（开放/欠定程度）。用这套坐标既能保证基准任务分布均衡，又能精确分析 agent 在「广度 vs 深度 vs 模糊度」上各自栽在哪。

**核心 idea**：用纯人工、带权重（含负分）、区分必选/可选的细则，把开放式长文评测拆成大量可逐条判定的原子标准，再交给 LLM-as-Judge 三值打分，从而把「主观长文质量」变成可量化、可复现、可归因的细则达成率。

## 方法详解

### 整体框架
ResearchRubrics 本质是一套**数据集 + 评测协议**，而非一个模型方法。它由三块拼成：（1）101 个单轮开放式 prompt，覆盖 9 大领域；（2）每个 prompt 配 20–43 条、共 2593 条专家手写带权重的评分细则；（3）一套把 agent 报告映射到「逐条细则达成度」的 LLM-as-Judge 评测公式。整条管线是：专家三人接力造出 prompt 与细则 → 给每个任务标注（广度、深度、探索）三元复杂度 → 待测 DR 系统对 prompt 产出报告 → LLM 判官对每条细则给「满足/部分满足/不满足」三值裁决 → 按权重归一化算出该任务的合规分数，并按细则类别拆解失败来源。因为这是 benchmark 论文、流程是线性的人工流水线而非多模块协同算法，不额外画框架图，关键全在「细则怎么设计、分数怎么算」。

### 关键设计

**1. 三专家接力的数据构建流水线：用多轮人审顶替 LLM 自动生成，去掉锚定偏差**

这一设计针对的痛点是「LLM 生成细则会带来循环论证和锚定偏差」。作者定义的「专家」是有扎实 STEM 背景、擅长任务设计与评测的人（不要求是每个 prompt 领域的专科专家），且每人只做自己熟悉的领域。流程是三人各司其职：Expert 1 先起草一个 prompt 加一组细则；交给 Expert 2 审阅，两人来回迭代直到 Expert 2 认可；最后 Expert 3 独立终审并做最后微调。原始 prompt 的灵感取自用户论坛、问答站和头脑风暴，再改写成 DR agent 会遇到的研究型问题。关键在于**所有 prompt 和细则都是人写人审、没有任何一条由 LLM 种子或生成**，这正是它和 LiveResearchBench（细则 LLM 生成、仅人审）这类基准的根本区别——后者人只是「审阅」机器初稿，仍会被机器措辞锚定。

**2. 三轴任务复杂度框架：把开放式难度拆成广度/深度/探索三个正交维度**

DR 任务难在哪并不统一——有的要广博知识、有的要深链推理、有的本身欠定需要厘清目标。作者把每个任务沿三条正交轴打标签：**概念广度（Conceptual Breadth）**分 Simple / Moderate / High，看涉及多少个、多分散的主题或信息源（如「分析亚洲可再生能源采纳的环境、经济、政治因素」就是跨 >5 源的 High）；**逻辑嵌套深度（Logical Nesting）**分 Shallow / Intermediate / Deep，看需要几步相互依赖的推理（4+ 步、含「分析→综合→评估→修订」层级规划算 Deep）；**探索性（Exploration）**分 Low / Medium / High，看任务欠定程度（3+ 个关键因素未指定、需要 agent 自己厘清目标的算 High，如「我想转行到有前景的方向，该考虑什么」）。每个任务被标成一个 (广度, 深度, 探索) 三元组，从而既能保证基准任务分布均衡，又能在评测时分维度归因——比如发现模型随逻辑嵌套加深而单调掉分，但对概念广度相对没那么敏感。

**3. 带权重、分必选/可选、含负分的细则体系：把长文质量拆成可逐条判定的原子标准**

为把主观长文质量变成可量化的东西，每条细则锚定六个评测轴之一：显式需求（Explicit）、隐式需求（Implicit）、信息综合（Synthesis）、引用使用（References）、表达质量（Communication）、指令遵循（Instruction Following）。每条细则被赋一个 $[-5, 5]$ 的整数权重：$\pm4$ 或 $\pm5$ 是**必选（mandatory）**标准、即合格答案的最低要求，$[-3, 3]$ 是**可选（optional）**标准、即区分优秀与「够用」的加分项；正权重奖励有价值的属性，负权重惩罚事实错误、跑题、冗长这类常见失败模式。负细则的措辞使得「一旦命中就把负权重计入总和」。这些权重对齐一套六级人类偏好量表（从 Critically Detrimental 到 Critically Important），以提升人机打分一致性。把必选与可选分开、并显式给负分，正是补上了现有基准「无法区分核心缺陷与缺点缀」的空白。

**4. 三值 LLM-as-Judge 打分公式：用部分得分支持开放式答案的灰度评判**

DR 答案常常「部分满足」某条细则，非黑即白会失真。作者让 LLM 判官对每条细则输出三值裁决 $m_{r_i}\in\{1, 0.5, 0\}$（满足 / 部分满足 / 不满足），任务 $k$ 的最终合规分数为正负权重的加权和、再用正权重之和（理论满分）归一化：

$$S_k = \frac{\sum_{r_i\in C} w_{r_i} m_{r_i}}{\sum_{r_i\in C,\, w_{r_i}>0} w_{r_i}}, \qquad m_{r_i} = \mathrm{Judge}(P_k, \mathrm{Res}, r_i) = \begin{cases} 1, & \text{满足} \\ 0.5, & \text{部分满足} \\ 0, & \text{不满足} \end{cases}$$

负细则同样走这套打分、只是权重为负。为定位失败来源，作者还定义了**按类别的失败率** $\bar{F}_c = \frac{1}{|T_c|}\sum_{t\in T_c} \frac{n_{\text{fail},c,t}}{n_{\text{fail},t}}$，即在每个出现过类别 $c$ 的任务里，类别 $c$ 贡献的「不满足」细则占该任务全部失败细则的比例，再跨任务平均——它回答的是「当细则失败时，哪类细则贡献了最多失败」，且因为只在出现该类别的任务上平均，各类别失败率不必加和为 1。值得注意的是评测里也对比了二值（把「部分满足」并入「不满足」），用于度量严格合规。

## 实验关键数据

### 主实验
在 RESEARCHRUBRICS 上评测三个商用 DR 系统，三值与二值两种打分下的总体合规分数（越高越好）：

| 系统 | 三值合规 | 二值合规 |
|------|---------|---------|
| Gemini DR | **0.677** | **0.615** |
| OpenAI DR | 0.664 | 0.597 |
| Perplexity DR | 0.566 | 0.487 |

核心结论：**没有任何系统三值合规超过 70%**，最强的 Gemini DR 也只有 67.7%（二值 61.5%）。这与 LiveResearchBench（最强系统 <74%）、DeepResearch Bench（<50%）的结论一致，暗示这是架构层面的根本局限而非某个基准的特例。

按细则六轴拆解失败率（部分代表值，%）：

| 评测轴 | Gemini | ChatGPT | Perplexity |
|--------|--------|---------|-----------|
| 显式需求 | 低（<20） | 低 | 低 |
| 隐式需求 | 49.2 | 49.0 | 48.6 |
| 信息综合 | 28.9 | 28.1 | 25.7 |
| 表达质量 | 20.0 | 19.0 | 16.5 |

**隐式推理 + 信息综合两轴合计占全部失败的 45–50%**，而显式事实检索和表达质量的失败率都低于 20%。也就是说 agent 擅长「照搬明说的事实、把话写漂亮」，却普遍栽在「推断没明说的需求、把多文档证据整合成连贯论证」上。

### 消融实验
细则设计对人机一致性（Macro F1）的影响：

| 配置 | Gemini DR（二值 F1） | 说明 |
|------|---------------------|------|
| 例子细节 Low | 0.733 | 细则只给基础描述 |
| 例子细节 High | **0.760** | 细则内嵌简短示例，一致性 +3~4% |
| LLM 增广 Absent | 0.760 | 原始人写细则 |
| LLM 增广 Present | 0.564 | LLM 自动改写细则，一致性灾难性 −15~20% |

二值 vs 三值打分的人机一致性：二值达 0.72–0.76 Macro F1，比三值高约 20 个百分点（三值仅 0.53–0.57）。Gemini-2.5-Pro 是最可靠判官（二值 0.76）。

### 关键发现
- **隐式推理是「优秀 vs 够用」的分水岭**：必选细则的失败主要发生在显式需求和信息综合上，而隐式推理的失败主要来自**可选**细则——说明系统能满足基本隐式要求，却普遍漏掉区分专业级与凑合级的细微质量点。
- **二值打分反而更可靠**：从三值切到二值，人机一致性涨约 20 个百分点，说明「部分满足」引入了歧义却没带来更强的区分力。
- **LLM 自动增广细则是毒药**：让 LLM 自动扩写/改写细则会把人机一致性砸掉 15–20%，反向印证了「细则必须人写」这一核心主张；而给细则内嵌简短示例（如具体政策名、引文）则稳定提升一致性 2–4%。
- **广度-精度权衡**：Gemini DR 产出 111 条引用但准确率 81%，Perplexity 只产 31 条引用却准确率 90%——追求覆盖面会牺牲精度，二者都没能很好处理「源的相关性与权威性」这一隐式判断。
- **难度随逻辑嵌套单调上升**：浅层推理任务处理得好，多步分析/评估类任务急剧掉分；概念广度也相关但没那么陡，系统处理多域综合反而比长推理链更稳。

## 亮点与洞察
- **纯人工细则是这篇最硬的底气**：在「LLM 评 LLM」满天飞的当下，作者花 2800+ 小时坚持 prompt 与细则全人写人审，并用消融证明「LLM 增广细则会砸掉一致性」，把「为什么不能图省事用机器造细则」量化成了实锤。
- **三轴复杂度框架可迁移**：广度/深度/探索这套正交坐标不只用于建基准，更是诊断工具——任何开放式 agent 评测都能借它把「模型到底弱在哪」拆清楚，而不是只给一个笼统总分。
- **必选/可选 + 负分细则的设计很实用**：一个 60% 的总分到底是「核心要求出了危险缺口」还是「只是缺点打磨」，靠必选/可选拆分一眼可辨，对部署决策直接有用。
- **「二值比三值更可靠」反直觉但有数据**：直觉上部分得分更细更好，实验却显示它引入歧义、拉低人机一致性——提醒做评测的人别为了「更细」而牺牲「可复现」。

## 局限与展望
- **只有 101 个单轮 prompt**：相对 2593 条细则，任务量偏小且全是单轮，覆盖不了多轮交互式 deep research 的真实场景。
- **只评了三个商用系统**：OpenAI / Gemini / Perplexity，开源 DR agent、不同 retrieval 架构未纳入，结论的「架构性局限」推断样本有限。
- **细则本身的天花板**：人写细则虽避免了 LLM 锚定，但仍依赖专家对「好答案」的预设，对真正新颖或超纲的优质回答可能误判；且「专家」定义为 STEM 通才而非领域专科，专业域细则的深度存疑。
- **动态信息源带来可复现性问题**：DR 依赖实时网络，今天能查到的证据明天可能变，长期重测时合规分数的可比性会打折。
- **改进方向**：扩展到多轮/可澄清的交互式任务、纳入开源 agent、研究如何把「隐式需求推断」这一最大短板显式建模进 agent 的规划阶段。

## 相关工作与启发
- **vs DeepResearch Bench**: 都做 PhD 级长文报告评测，但它用 LLM 生成细则 + LLM 生成参考报告做评测，存在循环论证；本文细则全人写人审，且用消融证明 LLM 增广细则有害。
- **vs ExpertLongBench**: 最接近本文——同样九域、专家细则、长文任务，但它依赖高质量既有参考、用 CLEAR 框架评测，把 prompt 限在临床/法律/分子等高度专业域；本文同时纳入普通消费者研究查询，域更宽、每任务平均细则数（26）更高。
- **vs LiveResearchBench / Mind2Web2**: 它们追求真实 prompt 但细则仍是 LLM 生成、仅人审，会有锚定偏差；本文从源头就人写人审。
- **vs HealthBench**: 借鉴了它的 Macro F1 验证 LLM 判官、以及「worst-at-k」式的能力下界分析思路，但把范围从医疗扩展到 DR 通用开放式研究，且本文的人机一致性（0.76）超过 HealthBench 的 0.709。

## 评分
- 新颖性: ⭐⭐⭐⭐ 三轴复杂度框架 + 纯人工带权细则的组合在 DR 评测里是首个，但单项技术（rubric、LLM-as-Judge）已有人做。
- 实验充分度: ⭐⭐⭐⭐ 三系统 × 三判官 × 二值/三值 × 多维度拆解 + 细则设计消融，相当扎实；惜系统与任务数偏少。
- 写作质量: ⭐⭐⭐⭐ 动机、复杂度框架、打分公式讲得清楚，失败归因分析有洞见。
- 价值: ⭐⭐⭐⭐⭐ 开源全部 prompt/细则/评测代码，填补了「真实域多样 + 专家细则」的 DR 评测空白，对推动可信研究助手很有用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Towards Personalized Deep Research: Benchmarks and Evaluations](towards_personalized_deep_research_benchmarks_and_evaluations.md)
- [\[ICLR 2026\] DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents](deepresearch_bench_a_comprehensive_benchmark_for_deep_research_agents.md)
- [\[ICLR 2026\] DRBench: A Realistic Benchmark for Enterprise Deep Research](drbench_a_realistic_benchmark_for_enterprise_deep_research.md)
- [\[ICLR 2026\] LiveResearchBench: A Live Benchmark for User-Centric Deep Research in the Wild](liveresearchbench_a_live_benchmark_for_user-centric_deep_research_in_the_wild.md)
- [\[ICLR 2026\] From Reproduction to Replication: Evaluating Research Agents with Progressive Code Masking](from_reproduction_to_replication_evaluating_research_agents_with_progressive_cod.md)

</div>

<!-- RELATED:END -->
