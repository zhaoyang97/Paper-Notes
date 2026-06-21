---
title: >-
  [论文解读] JointAVBench: A Benchmark for Joint Audio-Visual Reasoning Evaluation
description: >-
  [ICLR 2026][VLM Reasoning][Audio-Visual Reasoning] JointAVBench 是首个面向 Omni-LLM 的"音视频强相关"联合推理基准，覆盖 5 个认知维度、4 类音频信号、3 种场景跨度共 15 个任务，用半自动管线从电影里合成 2853 道必须音视频协同才能答对的选择题，最强模型也只到 65.3% 准确率。
tags:
  - "ICLR 2026"
  - "VLM Reasoning"
  - "Audio-Visual Reasoning"
  - "Omni-LLM"
  - "Benchmark"
  - "音视频强相关"
  - "多场景推理"
---

# JointAVBench: A Benchmark for Joint Audio-Visual Reasoning Evaluation

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Zg1YH8R5GG](https://openreview.net/forum?id=Zg1YH8R5GG)  
**代码**: [Project Page](https://roverx12345.github.io/jointavbench_webpage/)  
**领域**: 多模态评测 / 音视频联合推理 / Omni-LLM Benchmark  
**关键词**: Audio-Visual Reasoning, Omni-LLM, Benchmark, 音视频强相关, 多场景推理  

## 一句话总结
JointAVBench 是首个面向 Omni-LLM 的"音视频强相关"联合推理基准，覆盖 5 个认知维度、4 类音频信号、3 种场景跨度共 15 个任务，用半自动管线从电影里合成 2853 道必须音视频协同才能答对的选择题，最强模型也只到 65.3% 准确率。

## 研究背景与动机
- **领域现状**：理解视频天然需要同时对视觉和听觉信息做推理。新一代 Omni-LLM（Gemini、Qwen-Omni 等）已能联合处理音频与视频，但缺一个专门评估"联合推理能力"的综合基准，进展因此受限。
- **现有痛点**：已有基准在三个维度上各有缺口——纯视频基准（EgoSchema、Video-MME、MVBench）根本不含音频；音视频基准要么缺乏严格的音视频相关性控制（WorldSense 真正音视频相关比例仅 62.9%、偏重视觉任务），要么只用静态图像或简单视频（OmniBench、AV-Odyssey 主要是图+音），要么音频类型单一（多数只覆盖 1-3 类）。
- **核心矛盾**：几乎所有现有基准都忽略了**多场景推理**——把一个场景里说的话、另一个场景里出现的物体、跨场景的情节顺序关联起来，恰恰是人类认知的核心，也是当前评测的盲区。
- **本文目标**：构建一个**音视频严格相关**、能系统化覆盖多种音频类型与多层级场景的基准，逼模型真正做联合音视频推理而非靠单模态走捷径。
- **核心 idea**：**【强相关 + 三维分类法】** 用"视觉/音频单独都答不出"作为硬约束设计 15 个任务，沿认知维度×音频类型×场景跨度三轴系统编排；**【半自动合成】** 为规避人工标注的高成本，用 vision-LLM/audio-LLM/通用 LLM 三类模型协作合成 QA，再用人工把关，把标注难度和成本压下来。

## 方法详解

### 整体框架
JointAVBench 先确立"音视频强相关、高质量视频源、多维任务分类法"三条构建原则，再用一条三阶段半自动管线落地：从电影里切场景、为每个场景生成全模态字幕，按任务的模态/场景约束只喂必要信息来合成 QA，最后做多级质量控制并加人工验证。最终从 1046 部短电影中产出 2853 道经人工核验的选择题。

```mermaid
flowchart LR
    A[SF20K 电影<br/>切场景 PySceneDetect] --> B[Stage1 全模态字幕<br/>视频字幕+四类音频字幕]
    B --> C[Stage2 QA 合成<br/>按任务只喂必要模态/场景]
    C --> D[Stage3 质量控制<br/>通用→专用校验+干扰项]
    D --> E[人工验证<br/>保留 2853 题]
```

### 关键设计
**1. 三维分类法把"联合推理"拆解成可量化的 15 个任务：点出能力盲区。** 作者沿三条正交的轴系统编排任务——认知维度（时序 / 空间 / 情绪 / 情节 / 长程，共 5 类）、音频信号类型（语音 SPE / 声纹 VOT / 声音事件 SEV / 音乐 MUS，共 4 类）、场景跨度（单场景 / 跨场景 / 全场景，共 3 类）。每个交叉格子对应一个具体任务，例如"说话人情绪识别 SPER"是单场景×声纹×情绪，"多情节排序 MPO"是跨场景×语音声纹×情节。这种笛卡尔式编排让基准既能细粒度定位模型在哪一类音频、哪一种场景跨度上失败，又保证了对"联合推理"这一抽象能力的全面覆盖，相比此前 8-26 个零散任务更系统。

**2. 音视频强相关的硬约束：让单模态彻底失效。** 整个基准的灵魂是"问题必须同时依赖视觉和听觉才能答"。设计上，合成 QA 时严格只喂任务指定模态、指定场景的字幕——比如做"说话人空间定位 SPL"时，只给视频字幕加上某一个场景的声纹描述，把无关模态和无关场景的干扰彻底剔除。质量控制阶段再用 Modality Check 显式验证每道题确实两个模态缺一不可（"成年男性说话人情绪如何"若音频里只有一个男声就会被丢弃，因为单模态即可推断）。这套机制把基准的真实音视频相关比例做到 **93.5%**，远高于此前最好的 80.4%（OmniBench）和 99.0% 的 AV-Odyssey（但后者只是图+音、缺视频与多场景）。

**3. 三阶段半自动合成管线：用 LLM 协作压低标注成本。** Stage 1 全模态字幕生成——先按 Panda-70M 的做法用 PySceneDetect 切场景并合并语义相似的相邻片段保证场景内一致，再为每个场景生成视频字幕，并分别生成语音转写、声纹、声音事件与音乐四类音频字幕（因现有音频模型难区分声音事件与音乐，先合成后用 LLM judge 分离并消除幻觉）。Stage 2 QA 合成——对 LLM 不易把握的时序/情节类任务用人工设计的模板，对一般任务（如人物关系推断 CRI）放手交给 LLM 以保多样性。Stage 3 质量控制采用"通用→专用"的链式思维逐步过滤：通用校验做 Modality Check 与 Logic Check，专用校验按任务做序列检查、歧义检查、音频信号类型检查，最后为每题生成三个似是而非的干扰项构成选择题。

**4. 人工验证与标签精修：把自动产出锚定到高保真。** 自动管线产出 3974 道 MCQ 后，标注团队按答案正确性、信息正确性、音视频依赖性、问题难度四项打分，分成 Accepted（全过、直接保留）、Pending Review（答对但某项偏低、按评分二次筛选）、Discarded（答错、剔除）。最终保留 2853 题、留存率 71.8%，证明自动管线产出质量已足够高；其后再做一次只修正答案标签、不删样本不改规模的事后精修，进一步降低残余标签不一致。

## 实验关键数据

### 主实验（部分模型，准确率 %，Avg 为 15 任务平均）

| 类型 | 模型 | Size | SPER | MPO | PTG | CSA | Avg |
|---|---|---|---|---|---|---|---|
| Omni | Gemini2.5-Pro | - | 40.2 | 67.6 | 62.1 | 47.9 | **65.3** |
| Omni | Qwen3-Omni | 30B | 39.9 | 57.7 | 32.9 | 45.0 | 63.6 |
| Omni | Gemini2.5-Flash | - | 27.6 | 55.3 | 59.3 | 39.7 | 58.0 |
| Omni | Qwen2.5-Omni | 7B | 35.2 | 40.4 | 21.5 | 48.8 | 56.5 |
| Video | InternVL-2.5 | 8B | 31.9 | 44.2 | 27.5 | 40.8 | 51.7 |
| Video | GPT-4o | - | 18.8 | 17.3 | 14.8 | 39.7 | 45.0 |
| Audio | Kimi-Audio | 7B | 36.9 | 32.0 | 26.2 | 40.5 | 45.6 |
| Audio | Qwen2-Audio | 7B | 35.0 | 38.2 | 27.6 | 31.1 | 39.5 |

### 模态利用分析（A+V vs 单模态最优/最差）

| 模型 | 模态 | $N_o$↑ | $N_u$↓ | Avg |
|---|---|---|---|---|
| Qwen2.5-Omni | A+V | 9 | 1 | 56.5 |
| VideoLLaMA2 | A+V | 5 | 5 | 46.8 |
| OneLLM | A+V | 8 | 2 | 36.9 |

> $N_o$：A+V 超过单模态最优分的任务数；$N_u$：A+V 低于单模态最差分的任务数。能力越强的模型 $N_o$ 越高、$N_u$ 越低，说明融合越有效。

### 关键发现
- **整体偏低**：最强的 Gemini2.5-Pro 也只有 65.3%，Omni-LLM 系统性优于纯视频/纯音频模型，凸显原生模态融合的价值。
- **音频类型不均衡**：模型在声音事件、音乐上表现好（视觉对应强），但在语音、声纹上挣扎——SPL、SPER、MPO 是全局最差的任务，因多数音视频数据集忽略情绪/性别等声纹信息。
- **跨场景是真痛点**：单场景表现好、跨场景明显变差、全场景反而回升（侧重全局叙事而非细节）；场景数从 0-20 增到 60+ 时多场景任务准确率骤降约 20%。
- **情绪/空间反常**：Omni-LLM 在 15 个任务里 11 个领先，却在情绪任务上输给单模态模型（额外模态反成干扰），在 SOOG/SOER 空间任务上甚至不如 Video-LLM（依赖视觉空间信息、未有效整合音频线索）。

## 亮点与洞察
- **"强相关"这一硬约束是核心贡献**：93.5% 的真实音视频相关比例把基准从"能用单模态走捷径"中解放出来，真正逼出联合推理能力差距。
- **三维笛卡尔分类法可诊断**：不是堆任务数，而是让每个失败都能定位到"哪类音频×哪种场景跨度"，对后续模型改进极具指向性。
- **多场景推理的系统揭示**：用"场景数 vs 准确率"曲线量化了 Omni-LLM 跨场景关联能力的崩塌，指明了一个此前被忽略的研究方向。
- **半自动管线的工程价值**：三类 LLM 协作 + 通用→专用链式校验 + 人工把关，在 71.8% 留存率下规模化产出高质量音视频 QA，可复用到其他模态评测。

## 局限与展望
- **题型单一**：全部为四选一 MCQ，可能高估理解力（存在蒙对/排除法），缺开放式生成与定位类评测。
- **域偏窄**：视频源全部来自电影（SF20K），叙事性强但与教学、监控、第一人称等真实场景分布有差距，结论外推需谨慎。
- **规模有限**：2853 题分摊到 15 个任务后每任务样本不算多，细粒度结论的统计稳健性有待更大规模验证。
- **依赖现成模型合成**：字幕由现有 vision/audio-LLM 生成，其固有幻觉与盲区（如声纹理解弱）可能被引入基准；作者虽用 LLM judge 和人工缓解，但无法完全消除。
- **展望**：作者将其定位为"暴露差距"的诊断工具，明确指出跨场景推理与声纹/语音理解是 Omni-LLM 的下一步攻坚点。

## 相关工作与启发
- **对比纯视频基准**（EgoSchema / Video-MME / MVBench / LVBench）：它们音频类型数为 0、音视频相关比例为 0，JointAVBench 填补了"联合"这一维。
- **对比音视频基准**（Music-AVQA / OmniBench / AV-Odyssey / LongVALE / AVUT / WorldSense）：本文在音频类型数（4，最多）和真实音视频相关比例（93.5%，含视频与多场景的最高）上同时领先，且是少有的强调多场景的工作。
- **启发**：评测设计可以用"模态强相关 + 正交多维分类法"来系统性地榨出能力盲区；对模型侧而言，声纹/情绪理解和跨场景长程关联是当前 Omni-LLM 最值得投入的两个短板。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 首个把"音视频强相关 + 多场景 + 多音频类型"三者同时做严格的基准，三维分类法和强相关约束有清晰的差异化定位。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 Omni/Video/Audio 三类共 17 个主流模型，含音频类型、场景跨度、场景数、模态利用率等多角度细粒度分析，发现扎实。
- **写作质量**: ⭐⭐⭐⭐ 动机—缺口—方案逻辑清晰，分类表与对比表直观，管线图解完整。
- **价值**: ⭐⭐⭐⭐ 为 Omni-LLM 提供了诊断性强、指向明确的评测工具，跨场景与声纹理解的发现对社区有实际牵引价值；题型单一与电影域偏窄略限其上限。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] AV-Reasoner: Improving and Benchmarking Clue-Grounded Audio-Visual Counting for MLLMs](../../CVPR2026/vlm_reasoning/av-reasoner_improving_and_benchmarking_clue-grounded_audio-visual_counting_for_m.md)
- [\[ICLR 2026\] LENS: Multi-level Evaluation of Multimodal Reasoning with Large Language Models](lens_multi-level_evaluation_of_multimodal_reasoning_with_large_language_models.md)
- [\[ICLR 2026\] GIR-Bench: Versatile Benchmark for Generating Images with Reasoning](gir-bench_versatile_benchmark_for_generating_images_with_reasoning.md)
- [\[ICLR 2026\] MathNet: A Global Multimodal Benchmark for Mathematical Reasoning and Retrieval](mathnet_a_global_multimodal_benchmark_for_mathematical_reasoning_and_retrieval.md)
- [\[CVPR 2025\] Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation](../../CVPR2025/vlm_reasoning/beyond_final_answers_crystal_benchmark_for_transparent_multimodal_reasoning_eval.md)

</div>

<!-- RELATED:END -->
