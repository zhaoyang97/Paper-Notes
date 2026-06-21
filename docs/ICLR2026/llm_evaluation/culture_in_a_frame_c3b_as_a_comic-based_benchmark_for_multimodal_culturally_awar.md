---
title: >-
  [论文解读] Culture In a Frame: C$^3$B as a Comic-Based Benchmark for Multimodal Culturally Awareness
description: >-
  [ICLR2026][LLM评测][文化感知] C3B（Comics Cross-Cultural Benchmark）用 2220 张漫画、18789 个 QA 对，把"识别文化物体 → 判断文化冲突 → 跨语言文化内容生成"三个递进难度的任务串成一条链，专门考查多模态大模型的文化感知能力；在 11 个开源 MLLM 上的评测显示它们离人类水平还差一大截。
tags:
  - "ICLR2026"
  - "LLM评测"
  - "文化感知"
  - "漫画"
  - "多模态评测"
  - "多语言"
  - "MLLM"
---

# Culture In a Frame: C$^3$B as a Comic-Based Benchmark for Multimodal Culturally Awareness

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=jvPdTOSTVl](https://openreview.net/forum?id=jvPdTOSTVl)  
**代码**: https://c3b-benchmark.github.io/  
**领域**: 多模态VLM / 评测Benchmark  
**关键词**: 文化感知, 漫画, 多模态评测, 多语言, MLLM

## 一句话总结
C3B（Comics Cross-Cultural Benchmark）用 2220 张漫画、18789 个 QA 对，把"识别文化物体 → 判断文化冲突 → 跨语言文化内容生成"三个递进难度的任务串成一条链，专门考查多模态大模型的文化感知能力；在 11 个开源 MLLM 上的评测显示它们离人类水平还差一大截。

## 研究背景与动机

**领域现状**：文化感知（cultural awareness）正成为多模态大语言模型（MLLM）的核心能力之一——用户普遍发现现有模型在西方文化语境下表现很好，但一到非西方语境就掉链子。为了系统衡量这个能力，社区已经做了不少 benchmark（CVQA、CulturalVQA、GIMMICK、ALM-bench 等）。

**现有痛点**：作者指出这些 benchmark 有三个共性短板。一是**几乎全用真实世界图片**，而一张真实照片通常只承载一种文化，导致单图文化密度极低、任务对模型来说太简单；二是**多数是"一图一问"的单一任务形式**，没法从多个维度交叉考查同一张图；三是**几乎都是单语言**，而语言本身是文化的载体，缺了多语言任务就漏掉了"同一概念在不同语言里无对应表达"这层复杂度。

**核心矛盾**：要把"文化感知"考难、考全，就需要单张图里同时塞进多种文化、对同一张图设计多层任务、再叠加多语言——但真实照片这个媒介天然做不到"一帧多文化"，这是现有 benchmark 上不去难度的根本原因。

**本文目标**：造一个同时满足**多文化（每张图就多文化）、多任务（每条样本多问题）、多语言、且任务带递进难度**的文化感知 benchmark，并给出第一批开源 MLLM 的基线。

**切入角度**：作者把媒介从真实照片换成**漫画**。漫画描绘的是虚构场景，不受真实世界"一地一文化"的约束，因此可以把多种文化浓缩进同一帧画面，天然制造出高文化密度的复杂语境。

**核心 idea**：用"漫画当画布 + 三层递进任务链 + 多语言生成"来把文化感知评测的难度和覆盖面同时拉满，逼出当前 MLLM 在小众文化和文化冲突理解上的短板。

## 方法详解

### 整体框架

C3B 的产出是一套结构化的评测数据集：**2220 张漫画图 + 18789 个 QA 对，覆盖 77 种文化**。整个 benchmark 围绕一条"由易到难"的任务链组织——

1. **Level 1 · Culture-aware Objects Extraction（Extraction@Culture）**：考基础视觉识别 + 基本文化理解。含 Q1（识别漫画页的背景文化，可多选）和 Q2（从选项里挑出恰好包含画面内全部文化代表物的那一项）。
2. **Level 2 · Cultural-conflict Objects Detection（Conflict@Culture）**：考文化冲突理解。含 Q3（判断画面里是否存在文化冲突——若一张图同时出现两种不同文化即判为冲突）和 Q4（若有冲突，指出 Q2 里哪些物体与 Q1 的背景文化相抵触）。
3. **Level 3 · Culturally-aligned Content Generation（Generation@Culture）**：考给定多模态文化语境下的多语言生成，具体形式选机器翻译，覆盖日语→英/德/俄/西/泰五个语向（JA-EN/DE/RU/ES/TH）。

数据来源是"两套图"：Extraction 和 Conflict 共用 1023 张**自动生成**的漫画（背后是一条 Doubao 文生图管线），Generation 用从 Manga109 里**人工挑选**的 1197 张真实日漫页。标注则按任务类型走不同管线（详见关键设计）。评测时对识别/冲突任务用准确率，对 Q4 用复合指标 CACC，对翻译任务用 BLEU/COMET/BLEURT。

### 关键设计

**1. 漫画作媒介：把多种文化压进同一帧，抬高文化密度**

这是 C3B 区别于所有前作的根本设计，直击"真实照片单图只有一种文化、太简单"的痛点。真实照片绑死在真实场景里，一张图基本只能反映一地一文化；而漫画画的是虚构场景，可以把"夏威夷草裙的德国游客在新西兰文化村和毛利武士起冲突"这种现实里几乎不可能同框的元素塞进一格画面。为量化这一优势，作者定义了三个文化多样性指标：

$$\text{CDPI}(D) = \frac{1}{|D|}\sum_{i=1}^{n}\text{CultureInImage}(I_i)$$

$$\text{CBI}(D) = \text{CDPI}(D)\times N_{\text{cultures}},\qquad \text{CAD}(D)=\text{CDPI}(D)\times\log_2(N_{\text{cultures}}+1)$$

其中 CDPI 是每图平均文化数，CBI 把密度乘上数据集总文化数，CAD 则对总文化数取对数缩放以抑制"文化列表越长奖励越虚高"。结果上 C3B 的 CDPI 达到 **2.28**（CVQA/CulturalVQA/GIMMICK 等全是 1.00），CAD 14.29 远超第二名 GIMMICK 的 7.18——即漫画确实把单图文化密度翻了一倍多，这是它能把任务做难的物理基础。

**2. 三层递进难度的任务链：从"看见"到"看懂冲突"到"会生成"**

C3B 不像多数 benchmark 那样一图一问，而是对同一批图设计**逻辑递进**的任务，逐级抬高对文化感知的要求。第一层只要"看见并归类"文化物体（视觉识别 + 基础文化常识）；第二层要在识别基础上**判断不同文化的并置是否构成冲突**，并把冲突物体精确指认出来（需要跨文化的关系推理）；第三层要在理解画面文化语境后**用目标语言生成**符合语境的翻译（需要文化感知 + 多语言能力）。其中 Q4 尤其设计成依赖 Q1、Q2 的答案——只有先答对"背景是什么文化""画里有哪些文化物体"，才能正确指出"哪个物体和背景冲突"，从而把任务的难度逐题累积，逼出模型在长链推理下的薄弱环节。

**3. 多 agent 自动化构建管线：生成、标注、翻译各走一套质控流程**

要在保证质量的同时规模化造数据，作者为不同任务搭了不同的半自动管线。**图像生成**用 Doubao API：先让模型按"描述文化冲突场景"的指令逐行生成 prompt，人工筛掉有害内容后再据此生成漫画页。**Extraction/Conflict 标注**：先人工编出"全部文化列表"和"全部文化代表物列表"，Q1 在正确文化外随机补足到 5 选项、Q2 通过对正确物体列表做"加一个/删一个/删两个"的扰动造干扰项；冲突标注则用 Deepseek-V3 先做自动冲突检测（逐个物体判断是否与背景文化抵触，输出格式化冲突描述或"No"），再人工复核纠错。**Generation 标注**用 Translator + Reviewer 双 agent：Translator 先出粗译，Reviewer 从上下文一致性、基础翻译错误、文化相关错误三类问题给修改建议，建议回灌进 prompt 让 Translator 重译，最后人工定稿。这套"模型生成 + 人工把关"的分工保证了数据既有规模又可控质量。

**4. CACC 复合指标：让依赖型任务的评分体现整条推理链**

由于 Q4 建立在 Q1、Q2 之上，单纯用 Q4 自身准确率会掩盖"前两题答错导致后面全错"的真实情况。作者为 Q4 设计了复合准确率：

$$\text{CACC}(Q4) = a\cdot\text{ACC}(Q1) + b\cdot\text{ACC}(Q2) + c\cdot\text{ACC}(Q4)$$

取 $a=0.3,\ b=0.3,\ c=0.4$——Q1、Q2 贡献大致相等，Q4 占主导。这样一个模型只有在"识别对背景文化、识别对文化物体、且正确指认冲突"三件事都做到时才能拿高分，指标因此能真实反映模型在整条文化推理链上的综合表现，而不是在某个孤立子题上的侥幸命中。生成任务侧则用 BLEU 为主、COMET 和 BLEURT 辅助，对齐当前 LLM 翻译研究的评测惯例。

### 一个完整示例

拿一张"美洲原住民帽子出现在中国背景"的漫画页走一遍三层任务：
- **Q1（背景文化识别）**：模型需从 `A.中国 B.爱尔兰 …` 5 个选项里选出画面背景文化"中国"（可多选）。
- **Q2（文化物体识别）**：从若干"物体集合"选项里挑出恰好等于画面全部文化物体的那一项，比如包含"美洲原住民帽子、…"的那个集合。
- **Q3（是否存在文化冲突）**：判断"是"——因为美洲原住民元素和中国背景并置。
- **Q4（描述冲突）**：进一步指出"美洲原住民帽子不应出现在中国"——这一步必须建立在 Q1/Q2 都答对的前提上，CACC 才会高。
- **Q5（机器翻译）**：换到 Manga109 的日漫页，把对白"殺しはせんよ…少し血をいただくだけさ…"翻成英语/西语等目标语言，评 BLEU/COMET/BLEURT。

这条链让"看见 → 看懂 → 生成"的能力逐级显形，也让 Q4 的低分清楚暴露出模型在长链文化推理上的崩溃点。

## 实验关键数据

### 主实验

评测了 11 个开源 MLLM（SPHINX、Monkey、MiniGPT-v2、mPLUG-Owl3、LLaVA 系列、InternLM-XC2.5、Llama3.2、Qwen2.5-VL、InternVL2 等）。

识别与冲突任务主结果（部分代表模型，ACC / CACC，单位 %）：

| 模型 | Q1 (Extraction) | Q2 (Extraction) | Q3 (Conflict) | Q4 ACC | Q4 CACC |
|------|------|------|------|------|------|
| LLaVA1.5-7B | 32.5 | 2.93 | 56.3 | 0.00 | 10.6 |
| LLaVA-NeXT | 16.5 | 39.8 | 0.88 | 0.00 | 16.9 |
| InternLM-XC2.5 | 46.0 | 50.9 | 68.5 | 1.94 | 29.8 |
| Llama3.2 | 46.0 | 59.0 | 44.9 | 2.76 | 32.6 |
| **Qwen2.5-VL** | **53.7** | **55.9** | 63.1 | 3.20 | **34.2** |
| InternVL2 | 46.0 | 50.9 | 68.5 | 0.01 | 29.1 |

Qwen2.5-VL 综合最强：Q1 比第二梯队（InternLM-XC2.5/Llama3.2/InternVL2）高 7.7 分，Q4 比第二名 Llama3.2 高 1.6 分。但**所有模型在 Q4 上的原始 ACC 都极低（多数 <4%，两个 LLaVA 直接 0.00）**，说明"指认文化冲突物体"这一步对当前 MLLM 几乎是盲区。

多语言生成任务（Generation@Culture，BLEU，部分模型）：

| 模型 | JA-EN | JA-DE | JA-RU | JA-ES | JA-TH |
|------|------|------|------|------|------|
| MiniGPT-v2 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 |
| InternVL2 | 7.20 | 3.52 | 3.33 | 5.32 | 0.74 |
| **Qwen2.5-VL** | **13.2** | **12.0** | 8.74 | **14.5** | **9.72** |

Qwen2.5-VL 在全部语向领先；所有模型**在 JA-TH（泰语）最差、JA-EN（英语）最好**，暴露出低资源语言下的文化语境翻译短板。

### 消融实验

Q4 对 Q1/Q2 答案的依赖性分析（CACC，11 个模型均值）：

| 配置 | CACC | 说明 |
|------|------|------|
| Base Prompt | 19.231 | 不给 Q1/Q2 答案 |
| + Q1 Answer | 19.234 | 仅注入 Q1 答案，几乎不变 (+0.003) |
| + Q2 Answer | 19.231 | 仅注入 Q2 答案，无变化 |
| + Q1&Q2 Answer | 19.764 | 同时注入，提升最大 (+0.533) |

Q1、Q2 与 Q4 的相关系数分别为 0.56 和 0.51（中等相关），但即便把答案直接喂进 prompt，提升也只有边际级别。作者据此判断：这种相关更多来自"相关问题间的内在一致性"，而非模型真的会利用前序答案——也就是说 MLLM 并没有把 Q1/Q2 的线索有效串进 Q4 的推理。

### 关键发现

- **文化冲突理解是最大盲区**：Q4 几乎全军覆没，且喂答案也救不回来，说明短板在推理而非信息缺失。
- **小众文化识别差距悬殊**：按文化分组评 Q1，代表性文化（柬埔寨、日本）识别可靠，小众文化（芬兰、索马里）错误率显著更高。
- **人机差距巨大**：分易/中/难三档各取 100 题做人类评测，人类在 Q3 上三档全是 100% 准确率，整体 CACC（易 74.7 / 中 59.4 / 难 50.0）远高于模型（易 23.8 / 中 18.6 / 难 15.9）。
- **若干典型失败模式**：LLaVA-NeXT 倾向描述图片而不答题（"Turn-a-deaf-ear"）、LLaVA1.5 一直答"A"（"Take-a-shot-in-the-dark"）、以及无视指令的"stubbornness"。

## 亮点与洞察
- **"换媒介"是这篇最聪明的一招**：不靠堆数据量或堆语言数，而是看准"真实照片单图只能一种文化"这个物理瓶颈，用漫画一举把单图文化密度从 1.0 提到 2.28，难度自然就上来了——这是一个可迁移的造 benchmark 思路（凡是受真实数据约束的评测，都可以想想能不能换一种"可控合成"的媒介）。
- **递进任务链 + 依赖型指标**让 benchmark 有了"诊断"功能：Q4 依赖 Q1/Q2、再配 CACC，能定位模型到底在"识别"还是"冲突推理"哪一环崩，比单题准确率信息量大得多。
- **"喂答案也不涨"的消融很有价值**：它把"相关性"和"因果利用"区分开，提醒大家高相关不等于模型真在用前序信息，这个分析范式可以套到任何多步依赖任务上。

## 局限与展望
- **合成漫画的文化真实性存疑**：Extraction/Conflict 用的图是 Doubao 生成的，文生图模型本身可能带文化刻板印象或失真，虽有人工筛查 prompt，但"模型造的文化场景是否地道"难以完全保证（作者在 Ethics 里也承认尽力缓解但无法根除偏见）。
- **冲突定义偏狭**：把"两种文化同框"就判为冲突（co-occurrence conflict 而非真实文化抵触），这是一种简化，可能把许多正常的跨文化共存误标为"冲突"。
- **生成任务等同于机器翻译**：用 JA→X 翻译来代理"文化内容生成"，覆盖面有限且偏向日语源；BLEU 对文化贴切度的刻画也较弱。
- **只测开源模型**：未纳入 GPT-4o/Gemini 等闭源强模型，基线的"上限"参照缺失。
- 可改进方向：引入真实多文化漫画+众包文化标注、扩展冲突类型与生成任务形式、补齐闭源模型基线。

## 相关工作与启发
- **vs CVQA / CulturalVQA**：它们用真实照片、单图单文化、单一任务形式（CVQA 覆盖 30 语言但仍一图一问），C3B 用漫画做到一图多文化、一图多任务、并叠加多语言生成，CDPI 从 1.0 提到 2.28。
- **vs GIMMICK / ALM-bench**：GIMMICK 虽含 6 个任务、ALM-bench 覆盖 100+ 语言 19 个文化域，但图片仍是真实世界图、单图文化密度低；C3B 在文化密度（CAD 14.29 vs GIMMICK 7.18）和递进难度设计上更进一步。
- **vs Manga109 / CoMix / eBDtheque 等漫画数据集**：这些以漫画为中心的数据集主打基础视觉任务（角色命名、说话人识别），并非为文化评测设计；C3B 复用 Manga109 做翻译任务，但把漫画首次系统用作"文化感知"评测媒介。

## 评分
- 新颖性: ⭐⭐⭐⭐ 用漫画当媒介提升单图文化密度、配递进任务链与依赖型 CACC，是一个角度清奇且实用的 benchmark 设计。
- 实验充分度: ⭐⭐⭐⭐ 11 个开源模型 + 三任务 + 人机对比 + 文化分组与依赖性消融，分析较完整；缺闭源模型基线。
- 写作质量: ⭐⭐⭐⭐ 任务/构建/指标讲得清楚，图表配合到位；个别合成数据真实性的讨论略浅。
- 价值: ⭐⭐⭐⭐ 暴露了 MLLM 在小众文化与文化冲突推理上的硬伤，为后续文化感知研究提供了有诊断力的标尺。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Access Denied Inc: The First Benchmark Environment for Sensitivity Awareness](../../ACL2025/llm_evaluation/access_denied_inc_the_first_benchmark_environment_for_sensitivity_awareness.md)
- [\[ICLR 2026\] Culture in Action: Evaluating Text-to-Image Models through Social Activities](culture_in_action_evaluating_text-to-image_models_through_social_activities.md)
- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](../../ACL2026/llm_evaluation/minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)
- [\[ACL 2025\] SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture](../../ACL2025/llm_evaluation/sanskriti_a_comprehensive_benchmark_for_evaluating_language_models_knowledge_of_.md)
- [\[ACL 2026\] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](../../ACL2026/llm_evaluation/multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)

</div>

<!-- RELATED:END -->
