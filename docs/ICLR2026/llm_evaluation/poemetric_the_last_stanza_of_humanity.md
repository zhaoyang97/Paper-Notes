---
title: >-
  [论文解读] POEMetric: The Last Stanza of Humanity
description: >-
  [ICLR 2026][LLM评测][诗歌生成] 本文提出 POEMetric——第一个系统评估诗歌生成的框架，用 10 个维度（基础指令遵循 + 高级创作能力 + 总体评价）、203 首人类定型诗 + 30 个 LLM 生成的 6090 首诗，通过「规则算法 + LLM 评委 + 人类专家」三方互证，量化地证明了：顶级 LLM 在格律和主题上已逼近满分，但在创意、个性、情感共鸣、意象与修辞这些「诗之所以为诗」的能力上仍远不及人类诗人。
tags:
  - "ICLR 2026"
  - "LLM评测"
  - "诗歌生成"
  - "LLM 评测"
  - "创意能力"
  - "LLM-as-a-judge"
  - "人类专家校验"
---

# POEMetric: The Last Stanza of Humanity

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=9VkJ058cTa](https://openreview.net/forum?id=9VkJ058cTa)  
**代码**: https://github.com/Bingru-Li/POEMetric  
**领域**: LLM 评测 / 创意写作基准  
**关键词**: 诗歌生成、LLM 评测、创意能力、LLM-as-a-judge、人类专家校验

## 一句话总结
本文提出 POEMetric——第一个系统评估诗歌生成的框架，用 10 个维度（基础指令遵循 + 高级创作能力 + 总体评价）、203 首人类定型诗 + 30 个 LLM 生成的 6090 首诗，通过「规则算法 + LLM 评委 + 人类专家」三方互证，量化地证明了：顶级 LLM 在格律和主题上已逼近满分，但在创意、个性、情感共鸣、意象与修辞这些「诗之所以为诗」的能力上仍远不及人类诗人。

## 研究背景与动机
**领域现状**：LLM 在数学、代码、推理这类逻辑任务上表现亮眼，但艺术与人文方向少有人触及，尤其是文学写作。诗歌因其「在受限形式内同时压缩语言精度、情感共鸣与文化素养」的特点，被作者视为检验 LLM 生成能力的最佳棱镜——格律工整、篇幅短小，便于做可量化的诊断。

**现有痛点**：已有诗歌生成工作（ByGPT5、PoeLM、GPoet 等）大多只盯着「格律 / 押韵的形式准确率」，已经证明 LLM 能写出形式合规的诗；而已有评测要么停留在 meter/rhyme accuracy、BLEU、困惑度这类客观形式指标，要么用 fluency/coherence 这种泛泛的文本生成通用维度。真正构成诗歌灵魂的——创意、作者意图与情感、意象与修辞之美——几乎无人系统评估。

**核心矛盾**：诗歌质量的本质是「高级、主观、需要文学批评训练才能判断」的能力，而它恰恰最难被自动化量化；可一旦完全依赖人类专家逐首标注，又因专家稀缺、单首标注极耗时而无法规模化。形式合规 ≠ 好诗，但现有指标只会量形式。

**本文目标**：构造一个既覆盖诗歌高级创作维度、又能在 6000+ 首诗规模上跑得动、且结果可信的评测框架，并用它回答「SOTA LLM 离人类诗人还有多远」。

**切入角度**：把传统文学批评（尤其是「实用批评 / Practical Criticism」）里批评家真正关注的要素，蒸馏成一组可打分的维度；同时用「规则客观指标 + LLM 评委 + 小样本人类专家校验」做三角互证，在规模与可信之间取平衡。

**核心 idea**：用一套扎根文学理论的 10 维指标 + 三方互证的评测协议，把「诗写得好不好」这件主观的事变成可复现的基准，从而把 LLM 与人类诗人放在同一把尺子下对比。

## 方法详解

### 整体框架
POEMetric 不是一个模型，而是一套「数据集 + 指标体系 + 评测协议」三件套的诗歌评测基准。整体流程是：先构建一个标注精细的人类诗歌数据集（203 首、7 种定型诗）作为金标准与生成提示的来源；再把每首人类诗的「形式 + 格律 + 押韵 + 主题」塞进同一个 prompt 模板，让 30 个 LLM 各写一遍，得到 6090 首机器诗；然后用 10 个评测维度（分基础指令遵循、高级创作能力、总体评价三层）去衡量人类诗与 LLM 诗；最后用三种互不相同的评判手段——手写的规则算法、Gemini-2.5-Pro 充当的 LLM 评委、以及 7 位人类专家——分别打分并交叉验证，确认这套自动化评测可信。输入是「定型诗形式 + 主题」，输出是「人类 vs 各 LLM 在 10 个维度上的得分对比」。

整条管线没有可训练参数、也没有反馈回环，本质是「数据 → 多模型批量生成 → 多评委多维度打分 → 一致性校验」的线性评测流水线，因此不画 pipeline 图；关键全在数据集怎么标、10 个维度怎么定义、三方评测怎么互证。

### 关键设计

**1. 人类定型诗金标准数据集：用「可量化的形式约束」锚定主观评测**

作者刻意只收**定型诗（fixed-form poetry）**而非自由诗，理由直白：定型诗有明确的格律与押韵约束，能先建立一个可量化的客观基线，再在此之上去验证那些更主观的高级指标——否则一上来就评自由诗，主观维度无从校准。他们从 Poetry Foundation 与 Academy of American Poets 两个库共抓 1309 首，再用自写算法逐首检测 meter/rhyme，只保留真正符合某种韵律模式的诗，最终留下 203 首：95 首 ballad、71 首 sonnet、12 首 villanelle、9 首 ghazal、7 首 sestina、6 首 limerick、3 首 pantoum，共 7 种诗体，时间跨度从 1800s 至今、既有名家名作也有冷门新作。每首都标注了作者、标题、来源、诗体、格律模式、押韵模式、主题与意象。这个数据集既是评测的金标准，也是 LLM 生成的提示来源——保证人机同题同形式，对比才公平。

**2. POEMetric 十维指标：把文学批评蒸馏成可打分维度**

这是框架的核心贡献，10 个维度分三层。**基础指令遵循（2 维）**：form accuracy（是否按指定 meter/rhyme 写）和 theme alignment（是否切合给定主题）。**高级创作能力（6 维）**：creativity（是否新颖有创意）、lexical diversity（用词是否丰富）、idiosyncrasy（是否体现作者个人特质）、emotional resonance（是否引发情感共鸣）、use of literary devices（评 simile / metaphor / personification / allusion 四类修辞）、use of imagery（是否能唤起鲜明画面、调动读者感官）。**总体评价（2 维）**：overall poem quality（这首诗好不好）和 authorship estimation（判断是人写还是 LLM 写）。这 6 个高级维度是作者从「实用批评」里批评家分析诗歌时真正聚焦的要素中蒸馏出来的，正是以往评测缺失、却最能区分「合格句子」与「好诗」的部分。

**3. 三角互证评测协议：规则算法 + LLM 评委 + 人类专家校验**

为兼顾规模与可信，作者用三种异质评判互相印证。**规则算法**给客观维度兜底：手写流程自动检测每首诗的 meter/rhyme 以算 form accuracy；lexical diversity 用 MATTR（Moving Average Type-Token Ratio）按作者跨诗求均值；creativity 量化为「LLM 诗相对原人类诗的用词重复率」（重复越多越像模仿、创意越低）。**LLM 评委**负责大规模主观打分：先做 pilot 比较 Gemini-2.5-Pro / DeepSeek-R1 / GPT-4o，发现 Gemini-2.5-Pro 与人类一致性最高（PAo $=0.662$ vs. $0.548/0.438$）、在 overall quality 上区分度也最好（Std. Dev. $0.63$ vs. $0.20/0.22$），故选它作为唯一评委对全部诗按 10 维打分（5 点 Likert）。**人类专家校验**确保 LLM 评委可信：经 IRB 批准招募 7 位有诗歌研究 / 英文文学背景的专家（含职业诗人、博士、博后、教授），匿名评一个 58 首的代表性子集，做 10 道选择题 + 3 道开放题。三方一致性用 Proportion Agreement Observed 衡量：

$$\mathrm{PAo} = \frac{2A}{n_A + n_B}$$

其中 $A$ 为两评委一致的次数，$n_A, n_B$ 为各自评分总数。Gemini 与人类在 10 道题上 PAo 达 $0.662$，并辅以 Quadratic Weighted Kappa $\kappa=0.361$、Spearman $\rho=0.378$，与已有 LLM-人类一致性研究（如 $\rho\approx0.41\sim0.42$）相当，说明自动化评测结果稳健可用。

## 实验关键数据

### 主实验
30 个 LLM（7 家公司）各对 203 个提示生成，共 6090 首机器诗，与 203 首人类诗对比。以 Gemini-2.5-Pro 为评委（满分 5.00），代表性结果如下：

| 维度 | 人类 | 最佳 LLM | 说明 |
|--------|------|------|------|
| Form Accuracy | — | Gemini-2.5-Pro 4.26 | 顶级 LLM 形式准确率高；Llama-3.3-70B 仅 2.29 |
| Theme Alignment | — | Gemini-2.5-Pro 4.99 | 主题对齐普遍接近满分 |
| Creativity | **4.02** | DeepSeek-R1 3.31 | 人类显著领先 |
| Idiosyncrasy | **3.95** | DeepSeek-R1 2.17 | LLM 个性最弱，差距最大 |
| Emotional Resonance | **4.06** | DeepSeek-R1 3.53 | 人类领先 |
| Imagery | **4.49** | DeepSeek-R1 4.30 | 人类领先 |
| Literary Devices | **4.67** | DeepSeek-R1 4.38 | 人类领先 |
| Lexical Diversity | 3.82 | DeepSeek-R1 **3.85** | 唯一 LLM 反超人类的维度 |
| Overall Quality | **4.22** | DeepSeek-R1 3.20 | 人类大幅胜出 |

核心结论：LLM 在「基础指令遵循」上几乎追平甚至超越人类，但在全部高级创作维度（除词汇多样性外）上集体落后，overall quality 上人类以 4.22 完胜最佳 LLM 的 3.20。

### 规则评测与缩放分析

| 现象 | 数据 | 含义 |
|------|---------|------|
| 规则 form accuracy | Gemini-2.5-Pro 0.50、Claude-3.7 0.47 | 自动检测算法可区分模型形式能力 |
| MATTR | LLM > 人类 | LLM 词汇多样性更高 |
| 重复率 | LLM 远高于人类 | LLM 对人类原作有明显「模仿/复读」痕迹 |
| 参数规模 | 同族越大越好 | 但「思考型」未必更强（GPT-4o/GPT-4 > o1/o3-mini） |
| 蒸馏模型 | 普遍弱于原模型 | 例外：Distill-Llama-3.3-70B 反超原版 |

### 关键发现
- **idiosyncrasy（个性）是人机差距最大的维度**：LLM 普遍缺乏个人独特性与生命体验，最佳 LLM 仅 2.17 而人类 3.95——这暗示「个性」最难被模型习得。
- **形式易、灵魂难**：顶级 LLM 能把 meter/rhyme/theme 做到接近满分，却写不出有创意、有情感、有个人印记的诗，印证「合格句子 ≠ 好诗」。
- **评委可识别作者**：Gemini-2.5-Pro 在未告知作者的情况下，仍认出 203 首人类诗中的 80 首（39.4%，靠背诵原作或识别诗人风格）；人类专家认出原作更少，但几乎总能判断「这是人写的」。
- **思考型 ≠ 更会写诗**：推理增强并不必然带来更好的诗歌，DeepSeek-R1-Distill 多数比原模型差，说明创意能力与推理能力并非同一回事。

## 亮点与洞察
- **「重复率反推创意」的巧思**：把 creativity 量化为「LLM 诗相对人类原作的用词重复率」，用一个可计算的客观量去逼近一个高度主观的概念——重复越多越像模仿，越缺创意。这个代理指标廉价、可复现，可迁移到其他「原创性 vs 模仿」的生成评测。
- **三方互证而非单一评委**：规则算法、LLM 评委、人类专家三者既各管一摊（客观/规模/可信）又互相校验，并用 PAo + Kappa + Spearman 给出量化一致性，是把「主观艺术评测」做得可信的范式。
- **定型诗作为评测脚手架**：先用有硬约束的定型诗建立可量化基线，再去验证主观维度——这种「先锚定可量化、再外推到模糊」的思路，可推广到 free verse 乃至其他创意写作（小说、散文）评测。
- **「最后一节」的命题感**：标题「The Last Stanza of Humanity」呼应了核心发现——idiosyncrasy 与情感共鸣这些「人之为人」的能力，是当前 LLM 最难逾越的诗歌堡垒。

## 局限与展望
- **只评英语**：作者承认本文仅检验英语诗歌，虽声称 POEMetric 也适用于低资源语言，但未实证。
- **只评定型诗**：自由诗（free verse）这一现代诗歌主流形式被排除在外，作者将其留作未来工作；而定型诗上 LLM 的形式优势是否能推广到无固定约束的自由诗，尚未可知。
- **人类校验样本偏小**：专家只评了 58 首（部分分析甚至只 13 首人类诗），子集代表性与统计功效有限；高级创作维度的「人类金标准」本身也带主观性。
- **单一 LLM 评委的潜在偏置**：虽经 pilot 选出一致性最高的 Gemini-2.5-Pro，但用一个 LLM 评所有诗仍可能引入系统性风格偏好；且评委能「认出」近 40% 人类原作，意味着 authorship/quality 打分可能受「记忆泄漏」污染。
- **改进思路**：扩展到多语种与自由诗、引入多 LLM 评委集成去偏、对评委「认出原作」做去记忆化处理（如改写表层、控制训练集泄漏），都能提升基准的公平性。

## 相关工作与启发
- **vs 形式导向生成工作（ByGPT5 / PoeLM / GPoet）**: 它们把 meter/rhyme 等结构指标嵌进生成、关注「写得合不合形式」；本文不训练生成模型，而是建评测框架，关注「写得好不好诗」，把高级创作能力补进评测空白。
- **vs ProFTAP（图灵测试式评测）**: ProFTAP 用「能否与人类诗区分」做单一判据；POEMetric 把它拆成 10 个可解释维度（含 authorship estimation 这一维），既能区分作者又能定位差在哪个能力上。
- **vs Yu et al. 的 LLM-as-a-judge 诗歌评测**: 后者只评 fluency/meaning/coherence/relevance/aesthetics 这类较通用维度；本文新增 idiosyncrasy、emotional resonance、imagery、literary devices 等诗歌专有维度，并用规则算法 + 人类专家双重校验 LLM 评委，可信度更高。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一个把文学批评蒸馏成可量化 10 维、并三方互证的诗歌评测框架
- 实验充分度: ⭐⭐⭐⭐ 30 个模型 6090 首诗规模大，但人类校验子集偏小、仅英语定型诗
- 写作质量: ⭐⭐⭐⭐⭐ 动机扎实、文学理论与量化指标衔接自然，结论清晰有命题感
- 价值: ⭐⭐⭐⭐⭐ 为「LLM 创意写作离人类还有多远」提供了可复现的标尺与诊断维度

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Sci2Pol：评测与微调 LLM 的「科学→政策简报」生成能力](sci2pol_evaluating_and_fine-tuning_llms_on_scientific-to-policy_brief_generation.md)
- [\[ICLR 2026\] FinSearchComp: Towards a Realistic, Expert-Level Evaluation of Financial Search and Reasoning](finsearchcomp_towards_a_realistic_expert-level_evaluation_of_financial_search_an.md)
- [\[ICLR 2026\] DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents](deepresearch_bench_a_comprehensive_benchmark_for_deep_research_agents.md)
- [\[ICLR 2026\] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)
- [\[ICLR 2026\] LiveResearchBench: A Live Benchmark for User-Centric Deep Research in the Wild](liveresearchbench_a_live_benchmark_for_user-centric_deep_research_in_the_wild.md)

</div>

<!-- RELATED:END -->
