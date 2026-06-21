---
title: >-
  [论文解读] Beyond Classification Accuracy: Neural-MedBench and the Need for Deeper Reasoning Benchmarks
description: >-
  [ICLR 2026][VLM Reasoning][医学 VLM] 本文指出现有医学 VLM 基准只考分类精度、制造了"评测幻觉"，提出"广度—深度"双轴评测框架，并构建神经科深度推理基准 Neural-MedBench（120 个多模态病例、200 个推理任务），实测发现 GPT-5、Claude-4、MedGemma 等顶尖模型在深度推理上集体崩盘，且失败主要源于推理而非感知。
tags:
  - "ICLR 2026"
  - "VLM Reasoning"
  - "医学 VLM"
  - "临床推理"
  - "神经科"
  - "评测幻觉"
  - "Two-Axis Evaluation"
---

# Beyond Classification Accuracy: Neural-MedBench and the Need for Deeper Reasoning Benchmarks

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=KKA59ai0x6](https://openreview.net/forum?id=KKA59ai0x6)  
**代码/数据**: [https://neuromedbench.github.io/](https://neuromedbench.github.io/)  
**领域**: 多模态医学推理评测 / VLM Benchmark  
**关键词**: 医学 VLM、临床推理、神经科、评测幻觉、Two-Axis Evaluation  

## 一句话总结
本文指出现有医学 VLM 基准只考分类精度、制造了"评测幻觉"，提出"广度—深度"双轴评测框架，并构建神经科深度推理基准 Neural-MedBench（120 个多模态病例、200 个推理任务），实测发现 GPT-5、Claude-4、MedGemma 等顶尖模型在深度推理上集体崩盘，且失败主要源于推理而非感知。

## 研究背景与动机
**领域现状**：近年视觉-语言模型（VLM）在 MedMNIST v2、MultiMedQA 等标准医学基准上刷出近人类甚至超人类的成绩，营造出"医学 VLM 即将临床可用"的乐观印象。

**现有痛点**：这些基准绝大多数停留在标签预测、图文对齐等浅层分类任务，几乎不触及真实诊断所需的多模态综合、不确定性消解和"给出符合临床逻辑的论证"。作者把这种"浅基准高分掩盖深推理短板"的现象命名为**评测幻觉（evaluation illusion）**——模型看起来很行，实则在高风险诊断推理上系统性翻车。

**核心矛盾**：基准的"广度"（数据规模、人群/病种覆盖）与"深度"（推理保真度）是两个**很可能不相关**的维度。DiagnosisArena、MetaMedQA 等近期工作已暗示：只在广度上评测会高估模型能力，而模型在挑战性病例上的元认知与自我修正能力极弱。光靠堆数据量无法暴露这种缺陷。

**本文目标**：把被长期忽视的"深度轴"显式地操作化（operationalize），造一把能专门测临床推理保真度的"压力测试"尺子，并用它实证广度与深度的脱节。

**核心 idea**：**[双轴评测]** 与其继续扩大基准规模，不如造一个"小而难"的深度基准——以推理密度而非数据体量取胜，模仿 OSCE（客观结构化临床考试）的命题方式，逼模型整合多序列 MRI、电子病历与临床叙述，给出带论证的诊断。

## 方法详解

### 整体框架
Neural-MedBench 的构建是一条"漏斗式"流水线：从 2000+ 候选病例池出发，经多源汇集→多阶段专家筛选与模型验证，浓缩出 120 个高诊断复杂度的神经科病例，衍生出 200 个推理任务；再用一套"临床校准的混合评分"管线（准确率 + 语义相似度 + LLM 评分器）评测一整队 VLM，并配上人类基线锚定难度。

```mermaid
flowchart LR
    A[多源病例池 2000+<br/>ADNI/OASIS/Radiopaedia/病例报告] --> B[① 初筛: 多模态完整性]
    B --> C[② 专家策展<br/>2 神经内科 + 1 神经放射]
    C --> D[③ 标注 ground-truth<br/>诊断/鉴别/病灶/推理]
    D --> E[④ 共识 + 挑战性验证<br/>baseline 过滤平凡病例]
    E --> F[120 病例 / 200 任务<br/>3 任务族 × 3 难度级]
    F --> G[16 个 VLM 零样本评测]
    G --> H[混合评分: pass@k + BERTScore + LLM Grader]
    H --> I[误差分类 + 人类基线锚定]
```

### 关键设计

**1. 双轴评测框架：把"深度"从广度里拆出来。** 论文的概念底座是把医学 AI 评测拆成两条正交的轴——广度轴用大规模数据集测统计泛化与人群覆盖（现有基准几乎全在这），深度轴则用"小而精、专家策展"的复杂病例测推理保真度，逼模型在多模态信号下做不确定性推理并给出结构化论证。作者明确假设两轴**基本不相关**（success on breadth ≠ competence on depth），并把验证这一假设作为全文的实证主线。Neural-MedBench 就是把深度轴在临床神经科上落地的具体产物。

**2. 以推理密度取胜的漏斗式策展。** 不追求数据量而追求"每个病例都值得推理"：从 2000+ 候选经四步漏斗筛到 120 例——初筛保留多模态完整（影像+神经心理评分+病史）的病例，再由两名资深神经内科医生加一名神经放射医生按可信度、诊断复杂度与教学价值人工策展，标注出**不是单一标签而是结构化叙述**的 ground-truth（最终诊断、鉴别诊断、病灶刻画、解释性推理），最后用 baseline 模型把"太简单"的平凡病例剔除，确保每个保留病例都构成有意义的诊断挑战。病种上刻意同时纳入常见病（阿尔茨海默、缺血性卒中、癫痫）与罕见/复杂病（自身免疫性脑炎、中枢神经系统感染）。

**3. 三任务族 × 三难度级的分层设计。** 200 个任务分成三大族——鉴别诊断（给影像+病史，输出带论证的排序假设）、病灶识别（判类型与位置，测多模态空间推理）、论证生成（为诊断选择生成解释，对标病例讨论/职业考试）。再按推理深度分三级：Level 1 直接诊断（单模态经典征象，测模式识别）、Level 2 复杂诊断（证据模糊/冲突，须整合≥2 个模态消解不确定性）、Level 3 迭代诊断（模拟多轮问诊，须随新信息动态修正推理链）。这把"从事实回忆到高阶临床推理"做成了可控的连续谱，让评测能定位模型在哪一层断裂。

**4. 临床校准的两阶段混合评分。** 深度推理评分既要可靠又要可规模化，论文用两阶段化解这对矛盾：阶段一**评分器校准**——以神经科专用 rubric 引导一个 LLM 评分器（GPT-4o），再让多名持证神经科医生独立按同一 rubric 打分，实测 LLM 评分与专家共识的相关性极高（Pearson $r > 0.9$），科学验证该自动评分器可作为专家判断的可靠代理；阶段二**自动化社区评测**——把这把经一次性密集校准的评分器随基准开源，任何人无需接触临床医生即可对新模型全自动打分。指标上组合了诊断准确率 pass@1/pass@5、语义保真度 BERTScore（仅作次要相似度指标）、LLM 推理评分，以及由神经科医生把错误手工归入五类（感知失败 / 推理失败 / 知识缺口 / 接地错误 / 视觉幻觉）的误差分类法。同时跑医学生（n=5）与资深医师（n=5）盲测建立人类基线，给所有模型分数一个现实锚点。

## 实验关键数据

### 主实验表格（pass@1 / pass@5，零样本，节选 16 个模型中代表项）

| 类别 | 模型 | 直接诊断 p@1 | 直接诊断 p@5 | 复杂病 p@1 | 复杂病 p@5 | 多轮 p@1 | 多轮 p@5 |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Base | GPT-5 | **36.7** | 43.3 | **28.3** | 45.0 | **19.5** | 27.5 |
| Base | GPT-4o | 20.0 | 36.7 | 8.3 | 40.0 | 8.5 | 16.5 |
| Base | Gemini 2.5-Pro | 30.0 | **50.0** | 15.0 | 38.3 | 11.5 | 19.5 |
| General | Claude 4.0-Sonnet | 16.7 | 43.3 | 13.3 | 31.6 | 6.5 | 18.0 |
| Medical | MedGemma-27B | 30.0 | 36.7 | 18.3 | 38.3 | 10.5 | 15.5 |
| Medical | Lingshu | 26.7 | 40.0 | 21.7 | 35.0 | 8.5 | 20.0 |
| Medical | RadFM | 0.0 | 20.0 | 3.3 | 13.3 | 2.5 | 6.0 |
| Human | 医学生 | 3.3 | — | 3.3 | — | 6.0 | — |
| Human | 资深医师 | **40.0** | — | **35.5** | — | 15.0 | — |

> 即便最简单的直接诊断任务，最强的医学专用模型 MedGemma 也只有 30% pass@1，落后资深医师 10 个百分点；复杂病上资深医师（35.5%）几乎是 MedGemma（18.3%）的两倍。没有任何模型 pass@5 能稳定越过 50%，意味着正确诊断常常根本不在其 top-5 候选里。

### 误差分析与评测效率

| 误差类型分布（100 个错误响应） | 占比 |
|---|:---:|
| **推理失败（Reasoning Failure）** | **51%** |
| 感知失败（Perceptual Failure） | 27% |
| 其余（知识缺口/接地错误/视觉幻觉） | 22% |

| GPT-4o 在不同基准上的成本对比 | 图像数 | 图像 token 成本 | 通过率 |
|---|:---:|:---:|:---:|
| GMAI-MMBench | 12K | \$30.00 | 53.96% |
| OmniMedVQA | 128K | \$320.00 | 29.74% |
| **Neural-MedBench** | **1K** | **\$2.50** | **9.67%** |

### 关键发现
- **广度≠深度，评测幻觉被实证**：在广度基准（MMLU-Pro、DrVD-Bench）上高分的模型，到 Neural-MedBench 上断崖式下跌，首次给出两轴不相关的直接证据。
- **瓶颈是认知不是感知**：51% 的错误是"看对了关键征象却综合不出正确诊断"的推理失败，几乎是感知失败（27%）的两倍——解释了"语言流畅度高、诊断准确率低"的反差。
- **锚定偏差 vs 元认知**：医学生在多轮对话里反而强于 VLM，因为能用新证据自我修正；VLM 有明显锚定偏差，拿到反证也不肯改初始假设。
- **通才 vs 专才**：pass@1 上医学专用 MedGemma 占优（领域微调对精度有用），pass@5 上大通才 Gemini 2.5-Pro 反超（生成广度更利于产生鉴别诊断列表）。
- **高信噪比省钱**：同样用 GPT-4o，Neural-MedBench 的图像 token 成本比 GMAI-MMBench 低一个数量级（\$2.5 vs \$30），却把通过率压到 9.67%，证明"小而难"在成本上也更划算。

## 亮点与洞察
- **命名一个真问题**："评测幻觉"这个词精准点出了医学 VLM 领域"刷榜近人类、临床仍不可用"的尴尬，比单纯发个新数据集更有概念冲击力。
- **方法论而非数据集**：双轴框架把"为什么要造小基准"上升到评测哲学层面——广度测泛化、深度测保真，两者互补而非替代，给后续基准设计指了方向。
- **评分器可信度被认真对待**：用 $r>0.9$ 的临床医生相关性校准 LLM 评分器，并把校准后的评分器开源，兼顾了"专家金标准"与"社区可规模化"，是该类工作里少见的扎实做法。
- **难度被人类基线锚定**：低分到底是基准坏了还是真难？用资深医师 vs 医学生双人类基线把这点钉死——资深医师也只有 35–40%，说明分低是任务真难而非评分苛刻。

## 局限与展望
- **规模偏小、单科**：120 病例 / 200 任务、仅神经科，统计功效有限，作者也把"小"作为深度轴的设计取舍，但跨科室泛化性待验证。
- **LLM 评分器仍依赖 GPT-4o**：评分器本身是 LLM，虽经校准，但其偏好与潜在偏差可能随基础模型升级而漂移，长期可靠性需持续验证。
- **零样本设定**：只测 zero-shot 内在推理，未探究 CoT/工具调用/检索增强等推理脚手架能把深度分提升多少，留白较大。
- **结论是"诊断"而非"药方"**：论文擅长暴露推理短板，但对"如何修复整合性临床推理"只给方向（专才+通才合成），无具体方法。展望上作者承诺开放 leaderboard 与扩展 roadmap。

## 相关工作与启发
- **与广度基准的关系**：定位为 MedMNIST v2、MultiMedQA、GMAI-MMBench、OmniMedVQA、DiagnosisArena、MedAgentsBench 等广度基准的**互补**而非替代，对应"广度统计泛化、深度推理保真"的分工。
- **方法论同源**：评分上承接 BERTScore、CheXbert、RadGraph-F1 等语义指标与 EVAL、LLMEval-Med 等 LLM-as-Judge 框架，并用临床医生校准把可靠性补足。
- **更广启发**：这套"小而难 + 误差归因 + 人类锚定 + 可规模化评分器"的范式不限于医学——任何"刷榜虚高、实战拉胯"的领域（法律、金融、Agent 推理）都可以借鉴双轴思路，专门造深度压力测试来戳破评测幻觉。

## 评分
- **新颖性**: ⭐⭐⭐⭐ "评测幻觉 + 双轴框架"的概念提炼很有冲击力，神经科多模态深度推理基准属空白填补；扣分在于"小而难基准"思路本身（如 OSCE 类、专家策展）并非首创。
- **实验充分度**: ⭐⭐⭐⭐ 16 个模型横跨通才/医学专用 + 双人类基线 + 误差五分类 + 成本对比，证据链完整有力；规模偏小、零样本单一设定略减分。
- **写作质量**: ⭐⭐⭐⭐⭐ 论证主线（幻觉→双轴→构建→实证脱节→误差归因）层层递进，图表与叙事配合清晰，概念命名记忆点强。
- **价值**: ⭐⭐⭐⭐⭐ 直击"医学 VLM 临床可信度"的核心痛点，开源基准+校准评分器+leaderboard 可被社区直接复用，对纠偏整个领域的评测导向有实际推动作用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Thyme: Think Beyond Images](thyme_think_beyond_images.md)
- [\[ICLR 2026\] More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models](more_thought_less_accuracy_on_the_dual_nature_of_reasoning_in_vision-language_mo.md)
- [\[CVPR 2026\] ChartR: Evaluating Reasoning Accuracy and Robustness in Chart Question Answering](../../CVPR2026/vlm_reasoning/chartr_evaluating_reasoning_accuracy_and_robustness_in_chart_question_answering.md)
- [\[CVPR 2026\] See Further, Think Deeper: Advancing VLM's Reasoning Ability with Low-level Visual Cues and Reflection](../../CVPR2026/vlm_reasoning/see_further_think_deeper_advancing_vlms_reasoning_ability_with_low-level_visual_.md)
- [\[CVPR 2026\] Deeper Thought, Weaker Aim: Understanding and Mitigating Perceptual Impairment during Reasoning in Multimodal Large Language Models](../../CVPR2026/vlm_reasoning/deeper_thought_weaker_aim_understanding_and_mitigating_perceptual_impairment_dur.md)

</div>

<!-- RELATED:END -->
