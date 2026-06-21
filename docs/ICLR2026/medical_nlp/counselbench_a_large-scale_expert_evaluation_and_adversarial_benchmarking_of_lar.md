---
title: >-
  [论文解读] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of Large Language Models in Mental Health Question Answering
description: >-
  [ICLR2026][医疗 LLM][心理健康问答] 作者联合 100 位持证心理咨询专业人士，构建了一个面向**开放式心理健康问答**的双组件基准 CounselBench：一组是 2000 条专家逐维度打分 + 跨度标注的评估集（CounselBench-Eval），一组是 120 道临床医生手写、专门用来诱发特定失败模式的对抗题（CounselBench-Adv），系统揭示了 LLM 在心理咨询场景下"高分但仍有安全隐患"的现状，并实证了 LLM-as-Judge 在该高风险领域不可靠。
tags:
  - "ICLR2026"
  - "医疗 LLM"
  - "心理健康问答"
  - "专家评测"
  - "对抗基准"
  - "LLM-as-Judge"
  - "开放式生成"
---

# CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of Large Language Models in Mental Health Question Answering

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=8MBYRZHVWT](https://openreview.net/forum?id=8MBYRZHVWT)  
**代码**: https://github.com/llm-eval-mental-health/CounselBench  
**领域**: NLP理解 / 医学QA / LLM评测  
**关键词**: 心理健康问答、专家评测、对抗基准、LLM-as-Judge、开放式生成

## 一句话总结
作者联合 100 位持证心理咨询专业人士，构建了一个面向**开放式心理健康问答**的双组件基准 CounselBench：一组是 2000 条专家逐维度打分 + 跨度标注的评估集（CounselBench-Eval），一组是 120 道临床医生手写、专门用来诱发特定失败模式的对抗题（CounselBench-Adv），系统揭示了 LLM 在心理咨询场景下"高分但仍有安全隐患"的现状，并实证了 LLM-as-Judge 在该高风险领域不可靠。

## 研究背景与动机
**领域现状**：医学问答基准（MedQA、MedMCQA 等）绝大多数是选择题或事实型任务，衡量的是模型的事实召回能力。但真实患者问的是开放式问题——没有唯一正确答案、表述模糊、还混杂着症状描述、治疗顾虑和情感需求。

**现有痛点**：心理健康问答尤其特殊。一方面它高度主观、依赖语境，好的回答要同时兼顾共情、可执行建议和职业边界；另一方面 CounselChat、同伴支持论坛、EHR 消息、NOCD 这类数字心理服务大多是**单轮异步**交互，缺乏后续追问，因此一句越界的用药建议、一个轻率的语气都可能立刻造成伤害。可现有评测要么用选择题代理绕开了开放式生成的歧义，要么用小规模专家组或 LLM-as-Judge，前者样本太少，后者会漏掉临床上真正要命的失败。

**核心矛盾**：要在临床上靠谱地评测 LLM 的心理咨询回答，必须有**大规模 + 临床专家深度参与**的评测协议——但专家标注昂贵难扩，这正是过去基准两难的地方。

**本文目标**：(1) 给开放式心理健康 QA 定义一套临床立得住的评测维度；(2) 大规模收集专家对真实回答的评分与理由；(3) 主动构造对抗题，把模型的系统性失败模式"逼"出来，而不是事后复盘。

**切入角度**：作者的关键观察是——既然真实平台都是单轮问答、且 CounselChat 上有持证治疗师对匿名患者问题的公开回答，那就可以把"模型回答 vs 真人治疗师回答"放在**同一批患者问题**上做盲评，再让临床医生针对已观察到的失败模式反向出题。

**核心 idea**：用"100 位临床专家 × 多维度盲评 + 临床医生反向出对抗题"替代选择题代理和 LLM 自评，建立一个 practitioner-anchored（以从业者为锚）的心理健康 QA 评测框架。

## 方法详解

### 整体框架
CounselBench 不是一个模型，而是一个**两组件基准 + 一套评测协议**。整体可以理解为四步流水线：① 从公开论坛 CounselChat 选 100 道真实患者问题（覆盖 20 个常见心理主题），每题配 GPT-4 / LLaMA-3.3 / Gemini-1.5-Pro 三个 LLM 回答 + 1 个真人治疗师回答；② 设计六维临床评测量规，招募 100 位持证/受训专业人士盲评，得到 2000 条带跨度标注和书面理由的评估（CounselBench-Eval）；③ 让 9 个 LLM 当 judge，用同一套量规重评同一批回答，对比人机判断差异；④ 从 Eval 暴露的失败中提炼出细粒度失败模式，请 10 位临床医生反向写 120 道对抗题，收集 9 个模型的 1080 条回答并让专家逐条标注是否触发目标失败（CounselBench-Adv）。

这是一篇 benchmark/数据集论文，核心价值在**数据怎么造、维度怎么定、评测怎么做**，所以不画 pipeline 框架图，下面把四个关键设计讲清。

### 关键设计

**1. 六维临床评测量规：把"好回答"拆成质量与安全两组可打分维度**

开放式回答最难的就是"没有唯一答案，怎么打分"。作者基于临床心理学文献和专家咨询，把回答质量拆成六个有临床依据的维度：**Overall Quality（整体质量）**、**Empathy（共情）**、**Specificity（针对性，是否贴合用户具体语境而非泛泛而谈）**、**Factual Consistency（事实一致性）**、**Medical Advice（是否给出只应由持证专业人士提供的诊疗建议）**、**Toxicity（有害/贬低/污名化语言）**。其中前三维和共情、针对性绑定了治疗联盟（therapeutic alliance）这一疗效预测指标，后三维对准安全风险。多数维度用 5 点 Likert（1 最差、5 最好），Factual Consistency 用 4 点制，Medical Advice 则做成二元（Yes/No）外加"我不确定"选项——因为越界医疗建议的严重度细分很难一致判断。关键在于，Medical Advice 维度还要求标注者**抽出具体的建议跨度并写理由**，这让后续能对"推荐了什么药/什么疗法"做事后细分分析。

**2. 大规模盲评协议：100 位专家、每条回答 5 人独立标、可比性优先**

为了既扩规模又保临床效度，作者通过 Upwork 招募并逐一核验了 100 位持证或受训的心理从业者（覆盖 32 类执照学位、43 个专业方向，人口结构与美国咨询行业全国统计吻合）。标注设计上有几个讲究：每位标注者拿到 5 道题、每题 4 个回答（1 真人 + 3 LLM，顺序随机以消除位置偏差），共 20 个 QA 对；**同一题的 4 个回答由同一组人评**，从而支持模型间的直接公平比较；每个 QA 对由 5 位不同专家独立评，支撑评分者间一致性。标注者全程**盲源**，且不被告知有些回答是 LLM 生成的。最终得到 $100 \times 4 \times 5 = 2000$ 条标注。除评分外还收集开放式理由（合并所有理由后中位长度 576.5 词）和跨度抽取。一致性上，各维度 Krippendorff's $\alpha \geq 0.7$（多数 0.82 左右），说明专家判断高度一致——这正是大规模专家评测相对小专家组/LLM 自评的核心优势。

**3. LLM-as-Judge 复测：用同一量规检验"模型能不能给自己当裁判"**

专家评测虽是金标准但昂贵难扩，于是作者直接检验 LLM 能否替代人类评判。做法是让 9 个先进 LLM 用**与人类专家完全相同的 QA 对和评测标准**重评一遍，再把分数和排名跟专家对齐比较。结论很尖锐：① LLM judge 普遍**给高分**，尤其 Factual Consistency 几乎一律打满，不管回答实际内容如何；② Toxicity 上所有 LLM judge 几乎一致给最低毒性分，即便专家已标出潜在有害内容——说明它们在安全评估上灵敏度极差；③ 排名上 LLM judge 与人类偏好大幅背离（除 GPT-5 外），最典型的是被专家评为最差的 Gemini-1.5-Pro，却被每个 LLM judge 在整体质量上排到 GPT-4 之上；④ 在句级问题文本抽取上（抽医疗建议/事实错误/毒性句），大多数模型一句都抓不到，少数仅能抓到有限的越界医疗建议。这组实验从正反两面坐实了 LLM-as-Judge 在心理健康这种高风险主观领域不可靠。

**4. CounselBench-Adv：从已观察失败反向构造对抗题，主动逼出模型短板**

Eval 暴露的失败类别（如"缺乏个性化"）太宽泛，没法精确探测。作者先做更深一层的理由复盘，把它细化成六个具体失败模式，并按模型族归类：GPT-4 易给**具体药物**（1. medication）和**具体疗法技术**（2. therapy）；LLaMA-3.3 易**臆测医学症状**（3. symptoms）和**带评判**（4. judgmental）；Gemini-1.5-Pro 易**冷漠**（5. apathetic）和**基于无依据的假设**（6. assumptions）。然后**重新雇回** 10 位参与过 Eval 的临床医生，给他们每个失败模式配上定义和来自真实 flagged 回答的范例，请他们各写"每个失败模式 2 道、共 12 道"的现实题，合计 120 道。关键设计在于：这些题本身**不包含**失败，而是被精心设计成**容易诱发**模型犯对应错误——是一种高精度的脆弱性探针。最后用与 Eval 相同的 prompting 对 9 个 LLM 各生成 1 条回答（$120 \times 9 = 1080$ 条），由另外 5 位（与出题者不同的）专家以"yes/no/not sure"分类标注是否触发目标失败。

## 实验关键数据

### 主实验：四类回答的专家六维评分（CounselBench-Eval）

| 回答来源 | Overall↑(1-5) | Empathy↑ | Specificity↑ | Medical Advice(%Yes) | Factual↑(1-4) | Toxicity↓ |
|----------|------|------|------|------|------|------|
| GPT-4 | 3.28 | 3.37 | 3.46 | 0.07 | 3.53 | 1.78 |
| **LLaMA-3.3** | **4.29** | **4.22** | **4.63** | 0.14 | **3.70** | **1.36** |
| Gemini-1.5-Pro | 3.26 | 2.76 | 3.50 | 0.08 | 3.52 | 1.64 |
| 在线真人治疗师 | 2.60 | 2.72 | 3.29 | 0.17 | 2.92 | 2.56 |

LLaMA-3.3 在六维中的五维领先，整体评分最高；但它有 **14%** 的回答被标为提供越界医疗建议（如推荐疗法技术），是显著安全隐患。相比之下 GPT-4 更常带安全免责声明，约三分之一输出会明确拒答并建议咨询真人。值得注意的是真人治疗师在多数质量维度反而最低——但这与论坛回答信息简短、风格各异有关，不能简单读成"LLM 比治疗师好"。

### 对抗实验：各模型在 6 类失败模式上的触发率（CounselBench-Adv，5 位专家标注）

| 失败模式 | GPT-3.5 | GPT-4 | GPT-5 | Llama-3.1 | Llama-3.3 | Claude-3.5 | Claude-3.7 | Gemini-1.5 | Gemini-2.0 |
|----------|------|------|------|------|------|------|------|------|------|
| 1. 用药 | 0.05 | 0 | **0.47** | 0.05 | 0.10 | 0 | 0 | 0 | 0 |
| 2. 疗法技术 | 0.20 | 0.20 | **0.85** | 0.55 | 0.65 | 0.45 | 0.50 | 0.20 | 0.26 |
| 3. 臆测症状 | 0.15 | 0.45 | 0.60 | 0.45 | 0.45 | 0.50 | 0.37 | 0.26 | 0.25 |
| 4. 评判语气 | 0.25 | 0.25 | 0.05 | 0.11 | 0.10 | 0.05 | 0.10 | 0.20 | 0.10 |
| 5. 冷漠 | **0.70** | 0.20 | 0.15 | 0.15 | 0.15 | 0.05 | 0.20 | 0.40 | 0.30 |
| 6. 无依据假设 | 0.40 | 0.35 | 0.15 | 0.25 | 0.25 | 0.35 | 0.25 | 0.40 | 0.35 |

对抗题有效诱发了目标失败：疗法建议在 GPT-5 高达 0.85，LLaMA 系 0.55–0.65；用药建议几乎所有模型都罕见（0–0.10），唯独 GPT-5 异常高（0.47）；冷漠语气在 GPT-3.5-Turbo 最突出（0.70）。

### 关键发现
- **失败模式有"模型族"特征**：同族模型（LLaMA / Gemini / Claude）失败分布相似，而 GPT 族模式独特，说明失败在族内相对稳定、但会随大版本升级显著漂移。
- **LLM judge 在对抗集上同样不靠谱**：即便给了明确定义和 in-context 范例，最好的 Claude-3.7-Sonnet 失败模式检测 F1 也只有 **0.50**（各模型 Acc. 0.63–0.74、F1 0.35–0.50），人机判断仍存在实质性鸿沟。
- **高分 ≠ 安全**：模型在质量维度能拿高分，却反复出现非建设性反馈、过度泛化、缺乏个性化和越界医疗建议——质量与安全是两条需要分开看的轴。

## 亮点与洞察
- **"反向出题"的对抗范式**：不像传统 red-teaming 用文献预定义的失败模式，CounselBench-Adv 的失败模式是从真实专家标注里**经验性**提炼出来的，再由临床医生写成"看似普通但易诱发错误"的题——这种 empirically-grounded 的对抗构造比 literature-driven 覆盖更贴近实战脆弱性。
- **质量/安全双轴 + 跨度标注**：把"好不好"和"危不危险"拆开，且要求抽出具体问题跨度并写理由，使数据既能训对齐也能训句级安全检测器，复用价值高。
- **对 LLM-as-Judge 的冷水**：在高风险心理领域，LLM 自评不仅给高分，还系统性漏掉安全问题——这提醒任何想用 LLM judge 替代人评的场景都要先做人类校验，尤其安全维度。
- **可迁移的评测协议**：六维量规 + 标注协议本身不绑定 CounselChat，可直接套到未来更大、更新的心理健康数据集上，做一致的模型横评。

## 局限与展望
- **单轮设定**：基准只覆盖单轮异步问答，未涉及多轮对话中的上下文追踪、连贯性与一致性；作者把多轮（含模拟患者 agent、多轮 red-teaming）列为未来方向。
- **数据源单一且偏论坛**：真人回答取自 CounselChat 高票答案，论坛内容质量参差且偏简短，导致"真人治疗师得分最低"这一结论需谨慎解读，不能等同于真实临床面诊水平。
- **专家招募与平台限制**：标注者来自 Upwork，虽逐一核验资质，但与全职临床环境仍有差异；部分 LLM（如 Gemini-1.5-Pro、Claude-3.5-Sonnet）在跑 Adv 的 judge 实验时已下线，导致两组实验的模型集不完全一致。
- **公开临床数据稀缺**：受隐私保护，带临床医生回答的公开数据极少，限制了基准向更广临床场景扩展——这既是局限也是该工作选 CounselChat 的现实原因。

## 相关工作与启发
- **vs MultiMedQA / HealthBench**：它们用医生设计的多维量规、规模也大，但聚焦结构化医学知识或复杂难落地的评分方案；本文专攻真实心理健康 QA，强调安全、共情与语境敏感，并把临床医生深度嵌入设计与标注。
- **vs 选择题型心理健康基准（Racha 等）**：这类用客观答案键绕开了开放式生成的歧义；CounselBench 直面 free-text 评测，并把标注扩到 100 位专业人士保证临床效度。
- **vs 既有 red-teaming（Grabb、Schoene & Canca 等）**：它们的失败模式预定义、文献驱动，覆盖有限；CounselBench-Adv 由 10 位临床医生从 Eval 实测失败出发**前瞻性**出题，更能逼出实践中真实出现的脆弱性。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个把 100 位临床专家深度嵌入的大规模开放式心理健康 QA 基准，"反向出对抗题"范式有新意。
- 实验充分度: ⭐⭐⭐⭐⭐ 2000 条专家评估 + 1080 条对抗标注 + 9 模型 LLM-judge 双重对比，覆盖质量与安全两轴，证据扎实。
- 写作质量: ⭐⭐⭐⭐ 结构清晰、维度与协议交代细致，附录支撑充分。
- 价值: ⭐⭐⭐⭐⭐ 数据与协议公开，可直接用于对齐训练、安全检测器、LLM-judge 校验，对高风险医疗 LLM 评测有长期价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](counselbench_llm_mental_health_qa.md)
- [\[ICLR 2026\] MedAraBench: Large-scale Arabic Medical Question Answering Dataset and Benchmark](medarabench_large-scale_arabic_medical_question_answering_dataset_and_benchmark.md)
- [\[ICLR 2026\] Cancer-Myth: Evaluating Large Language Models on Patient Questions with False Presuppositions](cancer-myth_evaluating_large_language_models_on_patient_questions_with_false_pre.md)
- [\[ICLR 2026\] Can Large Language Models Match the Conclusions of Systematic Reviews?](can_large_language_models_match_the_conclusions_of_systematic_reviews.md)
- [\[ACL 2026\] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](../../ACL2026/medical_nlp/mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)

</div>

<!-- RELATED:END -->
