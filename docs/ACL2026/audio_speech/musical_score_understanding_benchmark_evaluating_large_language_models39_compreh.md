---
title: >-
  [论文解读] MSU-Bench: Musical Score Understanding Benchmark
description: >-
  [ACL 2026][音频/语音][乐谱理解] MSU-Bench 是首个针对完整乐谱理解的人工标注基准，包含 150 首作品的 1800 个生成式 QA 对，覆盖四级难度，评估揭示了 LLM/VLM 在乐谱定位和幻觉方面的严重不足，而 ABC 记谱法的文本输入显著缓解了这些问题。 领域现状：LLM 和 VLM 在自然语言处…
tags:
  - "ACL 2026"
  - "音频/语音"
  - "乐谱理解"
  - "音乐信息检索"
  - "ABC记谱法"
  - "多模态基准"
  - "幻觉"
---

# MSU-Bench: Musical Score Understanding Benchmark

**会议**: ACL 2026  
**arXiv**: [2511.20697](https://arxiv.org/abs/2511.20697)  
**代码**: [https://github.com/Congren-Dai/MSU-Bench](https://github.com/Congren-Dai/MSU-Bench)  
**领域**: 多模态 / 音乐理解  
**关键词**: 乐谱理解, 音乐信息检索, ABC记谱法, 多模态基准, 幻觉

## 一句话总结

MSU-Bench 是首个针对完整乐谱理解的人工标注基准，包含 150 首作品的 1800 个生成式 QA 对，覆盖四级难度，评估揭示了 LLM/VLM 在乐谱定位和幻觉方面的严重不足，而 ABC 记谱法的文本输入显著缓解了这些问题。

## 研究背景与动机

**领域现状**：LLM 和 VLM 在自然语言处理上展现了强大能力，但对完整乐谱的推理能力未被充分探索。现有音乐理解基准通常局限于片段、短摘录或多选题形式，且多聚焦于单声部音乐。

**现有痛点**：VLM 处理完整乐谱时面临两个持续性挑战——(1) 定位失败：模型常常无法正确识别小节位置，而小节定位是回答和声、织体等高级问题的前提；(2) 幻觉：模型生成不以乐谱为依据的内容，且定位错误会加剧幻觉。

**核心矛盾**：完整乐谱理解需要整合音高、节奏、和声和大规模结构的推理，但现有基准未能系统评估这种综合能力。

**本文目标**：(1) 构建覆盖四级难度的完整乐谱理解基准；(2) 支持文本（ABC 记谱法）和视觉（PDF）双模态评估；(3) 系统评估主流 LLM/VLM 的能力。

**切入角度**：ABC 记谱法作为文本结构化格式显式编码了小节结构、音高、节奏等信息，提供了一种 LLM 友好的乐谱表示，可以大幅缓解定位和幻觉问题。

**核心 idea**：用四级层次结构（起始信息→记谱和音符→和弦和和声→织体和曲式）系统化评估乐谱理解，ABC 作为上界模态用于符号到理论的推理。

## 方法详解

### 整体框架

MSU-Bench 要解决的是"如何系统评估 LLM/VLM 对完整乐谱（而非片段或选择题）的理解"。它收录 150 首完整乐谱（巴赫、贝多芬、肖邦、德彪西等）并人工标注 1800 个生成式 QA 对，问题按从基础识别到高级分析的四级难度递进；同一份乐谱以两种模态喂入——ABC 记谱法（文本 → LLM）与 PDF 乐谱（图像 → VLM），最终由 LLM-as-judge 对生成答案自动打分，从而把定位、幻觉、模态差距等问题量化出来。

### 关键设计

**1. 四级层次评估框架：按教学梯度从识别到分析逐级加难**

完整乐谱理解需要整合音高、节奏、和声与大规模结构的推理，单一难度无法刻画模型到底卡在哪一层。本文据此把问题分成四级：Level 1 起始信息（调号、拍号、速度）→ Level 2 记谱与音符（特定小节的音符、演奏法）→ Level 3 和弦与和声（和弦识别、调性分析）→ Level 4 织体与曲式（主题动机、曲式结构），每首作品在每级各设 3 个问题。这一梯度对应音乐学本科课程的教学层次，既能定位模型能力边界，也意味着能答好这些题的模型可充当教学助手。

**2. 双模态评估：用 ABC 与 PDF 的对照量化模态差距**

VLM 处理完整乐谱时最大的瓶颈是小节定位失败，而定位错误又会放大幻觉。本文把同一乐谱分别以 ABC 记谱法的文本形式喂给 LLM、以 PDF 图像形式喂给 VLM：ABC 显式编码了小节结构、音高与节奏，相当于绕过了 VLM 的视觉 OCR 与定位难题，因此可作为符号到理论推理的能力上界。两路对照直接量化出"模态差距"，把视觉输入的定位困难对乐谱理解的拖累显性化。

**3. 联合 vs 顺序提问：检验层次推理能否被提问方式激活**

四级问题之间存在天然的推理依赖（高级和声、曲式分析以基础识别为前提），提问方式可能影响模型发挥。本文对比两种策略：联合提问（一次性给出全部四级问题）与顺序提问（逐级提问）。结果是联合提问性能更好，暗示模型能在一次推理中利用层次间的关联，这一发现也为实际使用给出了最佳提问策略。

### 损失函数 / 训练策略

基准本身不涉及训练；微调实验在 MSU-Bench 训练集上做标准指令微调（SFT），结果显示两种模态均显著提升且不损害模型的通用知识。

## 实验关键数据

### 主实验

**零样本评估（部分）**

| 模型 | 模态 | L1 | L2 | L3 | L4 | Avg |
|------|------|----|----|----|----|-----|
| GPT-4o | ABC | 高 | 中 | 低 | 低 | — |
| GPT-4o | PDF | 中 | 低 | 低 | 低 | — |
| Qwen3-72B | ABC | 较高 | — | — | — | — |
| Gemma-3 | PDF | 低 | — | — | — | — |

### 关键发现

- **显著模态差距**：ABC 文本输入一致性地大幅优于 PDF 视觉输入，证实定位困难是 VLM 的核心瓶颈
- 微调在两种模态上都显著提升性能，且不损害通用知识
- 联合提问优于顺序提问，表明模型能利用层次推理
- Level 1（基础信息）最容易，Level 3-4（和声、曲式）最难
- 即使最强模型在 Level 4 上也表现不佳，表明高级音乐分析仍是重大挑战

## 亮点与洞察

- 将音乐理解按教学层次结构化评估的设计理念清晰且实用
- ABC 记谱法作为"定位问题的银弹"的发现有直接应用价值——暗示未来的音乐 AI 应优先解决 OCR 和小节定位
- 首个覆盖完整乐谱（含多声部）的理解基准，填补了重要空白

## 局限与展望

- 仅覆盖西方古典音乐，未包含爵士、流行等风格
- QA 对由人工标注，标注成本限制了规模
- ABC 记谱法本身有局限，不如 MusicXML 丰富
- 评估使用 LLM judge，对音乐专业术语的评判准确性待验证

## 相关工作与启发

- **vs 现有音乐 NLP 基准**: 现有基准聚焦于片段或选择题，MSU-Bench 首次评估完整乐谱的生成式理解
- **vs OMR 系统**: OMR 关注识别，MSU-Bench 关注理解和推理

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个完整乐谱理解基准，跨学科（音乐学+NLP）
- 实验充分度: ⭐⭐⭐⭐ 15+ 模型评估、双模态、微调实验，但缺少人工评估对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机充分
- 价值: ⭐⭐⭐⭐ 为音乐 AI 研究提供了急需的评估基础设施

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](../../ICLR2026/audio_speech/mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [\[AAAI 2026\] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding](../../AAAI2026/audio_speech/hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)
- [\[ICML 2026\] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks](../../ICML2026/audio_speech/mecat_a_multi-experts_constructed_benchmark_for_fine-grained_audio_understanding.md)
- [\[CVPR 2026\] AMUSE: Audio-Visual Benchmark and Alignment Framework for Agentic Multi-Speaker Understanding](../../CVPR2026/audio_speech/amuse_audio-visual_benchmark_and_alignment_framework_for_agentic_multi-speaker_u.md)
- [\[ACL 2026\] Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner](full-duplex-bench-v2_a_multi-turn_evaluation_framework_for_duplex_dialogue_syste.md)

</div>

<!-- RELATED:END -->
