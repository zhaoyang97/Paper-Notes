---
title: >-
  [论文解读] DRBench: A Realistic Benchmark for Enterprise Deep Research
description: >-
  [ICLR 2026][LLM评测][深度研究] DRBench 构造了第一个面向企业场景的 deep research 基准——要求 Agent 同时从公开网页和私有企业数据（邮件、聊天、PPT、表格、PDF）中挖掘并综合关键洞察，用 Insight Recall / Factuality / Distractor Avoidance / Report Quality 四个维度评估，揭示出当下 Agent 在企业洞察召回上严重不足（最强的 GPT-5 也只有 ~37%）。
tags:
  - "ICLR 2026"
  - "LLM评测"
  - "深度研究"
  - "企业 Agent"
  - "基准测试"
  - "Insight Recall"
  - "多源检索"
  - "LLM-as-a-judge"
---

# DRBench: A Realistic Benchmark for Enterprise Deep Research

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=IGYQ4c92e2](https://openreview.net/forum?id=IGYQ4c92e2)  
**代码**: [https://github.com/ServiceNow/drbench](https://github.com/ServiceNow/drbench)  
**领域**: LLM 评测 / Agent 评测 / Deep Research  
**关键词**: 深度研究, 企业 Agent, 基准测试, Insight Recall, 多源检索, LLM-as-a-judge  

## 一句话总结
DRBench 构造了第一个面向企业场景的 deep research 基准——要求 Agent 同时从公开网页和私有企业数据（邮件、聊天、PPT、表格、PDF）中挖掘并综合关键洞察，用 Insight Recall / Factuality / Distractor Avoidance / Report Quality 四个维度评估，揭示出当下 Agent 在企业洞察召回上严重不足（最强的 GPT-5 也只有 ~37%）。

## 研究背景与动机
- **领域现状**：deep research（高层战略提问 → 拆子问题 → 检索评估材料 → 产出有据可循的总结）正成为 LLM Agent 的热门落地方向，已有 Local Deep Researcher、DeepResearcher、OpenHands 等一批模块化 Agent 架构。
- **现有痛点**：现有 deep research 基准（Deep Research Bench、DeepResearch Bench、GAIA、Mind2Web 2 等）几乎都是**纯网页检索**任务，评的也只是答案准确率、文档召回或报告 factuality 这类窄维度；企业类环境基准（TheAgentCompany、OSWorld、WorkArena）则聚焦于 computer-use 操作，不涉及深度研究。
- **核心矛盾**：真实企业的有价值洞察散落在 PDF、表格、邮件、内部聊天等异构私有系统里，**还要和公开网页信息交叉印证**，但现有基准没有一个同时考察"私有+公开"双源检索、洞察召回、抗干扰和报告质量。
- **本文目标**：建一个把公开网页检索与私有企业数据融合、贴近真实工作流、可复现的 deep research 评测床。
- **核心 idea**：**「needle-in-a-haystack 式洞察注入 + 多源企业环境 + 原子洞察级评估」**——把人工核验过的 groundtruth 洞察注入到分布在多个企业应用里的合成文件中，再混入似是而非的干扰洞察，最后在原子洞察粒度上度量 Agent 能否召回真洞察、避开干扰、并正确引用。

## 方法详解

### 整体框架
DRBench 由三部分构成：(1) 一条**五阶段任务生成流水线**（LLM 大规模生成 + 三名标注员人工核验），产出 100 个任务、1093 条 groundtruth 洞察、覆盖 Sales/Cybersecurity/Compliance 等 10 个领域；(2) 一个**可复现的企业搜索环境**，把生成数据灌进 Nextcloud（云盘）、Mattermost（聊天）、RoundCube（邮件）等真实应用，让 Agent 既能爬公网又能调 API 访问私有数据；(3) 一套**四维评估框架**（Insight Recall、Distractor Avoidance、Factuality、Report Quality），全部基于 LLM-as-a-judge。论文还附带了一个基线 Agent（DRBA）作为参考实现。

```mermaid
flowchart LR
    A[S1 公司+人设生成] --> B[S2 公开源/洞察采集]
    B --> C[S3 研究问题生成]
    C --> D[S4 内部洞察生成<br/>含干扰项]
    D --> E[S5 文件映射与生成<br/>needle-in-a-haystack]
    E --> F[(企业环境<br/>云盘/聊天/邮件)]
    F --> G[DRBA Agent<br/>规划→检索→综合]
    G --> H[结构化报告]
    H --> I[四维评估]
```

### 关键设计

**1. 五阶段「人在环中」任务生成流水线：把真实感和可控难度同时握在手里。** 任务不是手写也不是纯 LLM 拍脑袋，而是分五步逐层接力、每步都过人工核验。S1 用 LLM 生成合成公司画像（行业、产品、市场地位、竞品）和跨部门人设（如合规事务经理），经专家精修后构成 Task Context $C$；S2 在该公司背景下检索候选 URL，**只限定带日期、来自期刊或行业报告的权威站点**以保证洞察"时间不变"，标注员选定一个作为 Task URL 并抽取公开洞察 $I_p$；S3 用公司画像+人设+Task URL+$I_p$ 让 LLM 提开放式研究问题 $Q$，标注员筛选精修并确保该 URL 至少能部分支撑作答；S4 基于 $I_p$ 和 $Q$ 生成与企业业务对齐的内部洞察 $I_l$，**同时生成似是而非但无关的干扰洞察 $I_d$**；S5 把每条洞察分配到某种模态（邮件/聊天/PDF/docx 等）。难度由洞察数量、文件类型数、应用类型数共同调节，分 easy/medium/hard。

**2. needle-in-a-haystack 文件合成：让真洞察淹没在逼真噪声里。** S5 的文件生成模块对每条要注入的洞察走三步——先按模态搭出文件骨架（文档结构或聊天配置），再把该条相关/干扰洞察插进合适的章节，最后用**逼真但无关的内容**把文件其余部分填满。这样每个文件就成了一个"草垛"，Agent 必须在大量分散注意力的细节中精准找到那根"针"。标注员还要抽检确保文件连贯、无自相矛盾。一个任务通常跨 2–4 个应用、3–16 个支撑文件，逼真复刻企业数据生态的碎片化。

**3. 原子洞察级四维评估：诊断式而非黑盒打分。** 评估刻意不把报告当成单一整体输出，而是先用 LLM 把报告拆成原子洞察，再逐条比对 groundtruth，从而支持"部分得分"和故障定位。**Insight Recall**：每条预测洞察经 LLM Judge 与 groundtruth 匹配，命中即计入召回；为防 Agent 把文件原文整段抄进来刷分，Judge **只看前 $k=$（groundtruth 洞察数 $+5$）条**洞察。**Distractor Avoidance** $=1-\text{distractor recall}$，度量精度（是否误收干扰项）。**Factuality** 沿用 FactScore 思路：洞察若无引用或引用了不存在的源即判为不实，否则用 text-embedding-3-large 从被引文档取 top-5 片段，再让 Judge 判断证据是否支撑该 claim。**Report Quality** 让 Judge 在分析深度、相关性、人设一致性、连贯性、无矛盾、完整性六个维度各打 1–10 分取平均。

**4. DRBA 基线 Agent：四阶段企业研究工作流参考实现。** 为基准提供可比的起点，作者实现了 DRBA——按 research planning → action planning → 自适应研究循环 → report writing 组织。研究规划支持两档：**CRP（Complex Research Planning）**生成含调查领域/预期信息源/成功判据的结构化计划，**SRP（Simple Research Planning）**只做轻量子问题拆解；研究循环里的 **AAP（Adaptive Action Planning）**会迭代地选工具执行、把内容存进向量库、再根据研究缺口自适应生成新动作，直到完成或触顶最大迭代数。报告以"洞察+引用"的结构化格式输出（而非自由叙述的 raw report）以便评估。

## 实验关键数据

### 主实验（FullBenchmark，100 任务，GPT-4o 为 backbone 与 Judge）
不同规划模块组合下 DRBA 的表现（数值越高越好）：

| 配置 | Insight Recall | Factuality | Distractor Avoid. | Report Quality | 调和均值 |
|------|----------------|-----------|-------------------|----------------|---------|
| Base DRBA | 13.18 | 58.04 | 95.76 | 88.23 | 34.82 |
| + SRP | 13.42 | **62.11** | 96.62 | 89.74 | 35.68 |
| + CRP | 13.31 | 59.53 | **97.14** | 87.92 | 35.21 |
| + AAP | **15.97** | 60.37 | 96.48 | **90.08** | **39.74** |
| + SRP + AAP | 14.83 | 55.29 | 96.55 | 88.96 | 37.34 |
| + CRP + AAP | 14.19 | 52.08 | 96.47 | 87.54 | 35.89 |

### 消融：不同 backbone 模型（FullBenchmark 子集）

| Backbone | Planning | Insight Recall | Factuality | Distractor Avoid. | Report Quality | 调和均值 |
|----------|----------|----------------|-----------|-------------------|----------------|---------|
| GPT-5 | - | 36.52 | **72.11** | 93.22 | 93.41 | **63.81** |
| GPT-5 | CRP | **37.48** | 62.33 | 91.71 | 92.03 | 62.02 |
| Llama-3.1-405B | CRP | 18.33 | 65.72 | 95.04 | 89.01 | 43.70 |
| DeepSeek-V3.1 | CRP | 28.21 | 67.09 | 93.96 | 85.57 | 55.03 |
| Qwen-2.5-72B | CRP | 24.39 | 55.74 | 95.12 | 87.51 | 49.46 |

### 关键发现
- **洞察召回是整个领域的硬伤**：即便最强的 GPT-5 也只召回约 37% 的 groundtruth 洞察，开源模型多在 16–28%，说明 Agent 普遍依赖先验知识或网页内容，而非真正去挖企业文件里的关键事实。
- **抗干扰反而很强**（Distractor Avoidance 普遍 >93%），说明 Agent "不被误导"没问题，问题在"找不到决定性洞察"。
- **AAP（自适应动作规划）增益最明显**，同时提升召回与报告质量；但 CRP/SRP 与 AAP 叠加并无明显增益、有时反而拉低 factuality，说明重叠的规划策略会带来冗余/不稳定。
- **公网检索几乎全军覆没**：当所需信息不在私有文件、应该去网上查时，没有任何 Agent 成功从公网取到相关内容——它们发的是"杂货店顾客信任"这类宽泛查询，而非针对 FSMA 204 法规的精准搜索，暴露出"缺失知识检测 + 问题界定"能力的缺失。
- **定性分析**：强模型不仅能取出数字，还能把数字绑定到正确的时间/业务上下文（如"截至 Q2 2024 的 85%"），弱模型常只复述孤立数字而丢掉限定语，导致表面正确但召回为 0。

## 亮点与洞察
- **第一个真正"公开+私有"双源、企业落地的 deep research 基准**，填补了现有基准非纯网页即纯 computer-use 的空白。
- **needle-in-a-haystack 注入 + 干扰洞察**的设计很巧妙：把召回与精度对立起来，既能测"找得到"又能测"不乱抓"。
- **原子洞察级、诊断式评估**比端到端打分信息量大得多，能定位"漏掉/无据/无关"三类具体失败，并通过 $k=$groundtruth$+5$ 的截断巧妙堵住"整篇抄进来刷召回"的漏洞。
- 结论给出明确的"关键路径"：**自适应规划 + 缺失知识检测**是推进企业 deep research 的两大抓手。

## 局限与展望
- **只测召回不测精度的"有用性"**：未匹配上 groundtruth 的洞察是否仍对回答问题有价值，作者承认难以自动判定、被一刀切忽略，可能低估了真实有用发现的广度。
- **$+5$ 缓冲是经验值**，虽能防刷分但本身略显 arbitrary，对"应当包含多少额外洞察"缺乏原则性界定。
- **生成数据为合成**：公司、人设、内部洞察、文件全部 LLM 合成 + 人工抽检，与真实企业数据的分布差异、潜在偏置仍待验证。
- **评估强依赖 LLM Judge（GPT-4o）**，存在 judge 与被测模型同源的潜在偏置（虽作者称原子 claim 使 judge 方差很小）。
- 展望：把"缺失知识检测→主动公网检索"做成显式能力、扩展更多领域与模态、引入真实脱敏企业数据，都是自然的延伸方向。

## 相关工作与启发
- **deep research 基准**（Deep Research Bench、DeepResearch Bench、DeepResearchGym、ResearcherBench、Mind2Web 2、GAIA）：DRBench 与它们的核心差异是首次要求"公开+私有"双源、并提供可交互企业环境。
- **企业/computer-use 环境**（TheAgentCompany、OSWorld、WorkArena、CRMArena-Pro）：提供了真实应用环境，但只考核操作执行，不评深度研究综合能力。
- **deep research Agent**（Local Deep Researcher、Deep-Searcher、DeepResearcher、OpenHands/OpenManus/smolagents）：DRBench 把它们放进企业语境下系统性地暴露短板。
- **评估方法学**：Insight Recall 思路承接 DeepResearch Bench，Factuality 借鉴 FactScore + TREC-RAG，Report Quality 受 G-Eval 启发——是把多条已有评估范式整合到一个统一床上的好范例。
- **启发**：做 Agent 评测时，"诊断式原子级评估 + 防刷分截断 + 干扰项注入"这套组合拳值得复用；同时本文清晰指出"缺失知识检测"是当前 deep research Agent 的真正瓶颈，对做 Agent 系统的人很有指导意义。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 第一个公开+私有双源、企业落地的 deep research 基准，问题定义和环境构造都有明确增量。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 100 任务/10 领域、跨 GPT/Llama/Qwen/DeepSeek 多 backbone、多规划策略消融，并有定性分析与误差报告；但合成数据的真实性验证可再加强。
- **写作质量**: ⭐⭐⭐⭐ 流水线、环境、评估三块层次清晰，图 1–3 把 pipeline/Agent 架构讲得很直观。
- **价值**: ⭐⭐⭐⭐ 给企业 deep research 提供了可复现评测床，并明确指出"洞察召回弱、公网检索失败、缺失知识检测缺位"三大待解问题，对后续研究有清晰指引。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents](deepresearch_bench_a_comprehensive_benchmark_for_deep_research_agents.md)
- [\[ICLR 2026\] LiveResearchBench: A Live Benchmark for User-Centric Deep Research in the Wild](liveresearchbench_a_live_benchmark_for_user-centric_deep_research_in_the_wild.md)
- [\[ICLR 2026\] Characterizing Deep Research: A Benchmark and Formal Definition](characterizing_deep_research_a_benchmark_and_formal_definition.md)
- [\[ICLR 2026\] ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents](researchrubrics_a_benchmark_of_prompts_and_rubrics_for_evaluating_deep_research_.md)
- [\[ICLR 2026\] Towards Personalized Deep Research: Benchmarks and Evaluations](towards_personalized_deep_research_benchmarks_and_evaluations.md)

</div>

<!-- RELATED:END -->
