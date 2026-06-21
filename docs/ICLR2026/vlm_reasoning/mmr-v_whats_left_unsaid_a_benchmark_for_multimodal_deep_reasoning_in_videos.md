---
title: >-
  [论文解读] MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos
description: >-
  [ICLR 2026][VLM Reasoning][视频推理] MMR-V 是一个面向"视频深度推理"的评测基准，强调跨长程多帧的证据挖掘与"言外之意"的隐式推理，揭示出当前最强的 Gemini-2.5-pro 也只有 64.3% 准确率，且 CoT 与 test-time scaling 几乎无效。
tags:
  - "ICLR 2026"
  - "VLM Reasoning"
  - "视频推理"
  - "多帧推理"
  - "隐式推理"
  - "MLLM 评测"
  - "思维链"
---

# MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos

**会议**: ICLR 2026  
**项目主页**: [https://mmr-v.github.io/](https://mmr-v.github.io/)  
**数据集**: [https://huggingface.co/datasets/JokerJan/MMR-VBench](https://huggingface.co/datasets/JokerJan/MMR-VBench)  
**领域**: 多模态视频推理 / Benchmark  
**关键词**: 视频推理, 多帧推理, 隐式推理, MLLM 评测, 思维链  

## 一句话总结
MMR-V 是一个面向"视频深度推理"的评测基准，强调跨长程多帧的证据挖掘与"言外之意"的隐式推理，揭示出当前最强的 Gemini-2.5-pro 也只有 64.3% 准确率，且 CoT 与 test-time scaling 几乎无效。

## 研究背景与动机
**领域现状**：o1、DeepSeek-R1 把文本推理推上了新台阶，o3、GPT-5 又通过工具调用把图像推理做得有声有色，多模态推理成了热门方向。但绝大多数工作停留在图像，更难的视频推理鲜有人碰。

**现有痛点**：现有视频基准（如 MVBench、Video-MME）几乎都在测"感知与理解"——任务往往只要定位到问题里提到的那一帧（question frame）再看几帧相邻帧就能答对。本文把这些缺陷归纳为三点：(1) **帧上下文受限**，即使是长视频也只用到少数相邻帧，没真正利用视频的长程时序结构；(2) **缺乏推理**，很多问题靠直接感知即可作答；(3) **任务脱离现实**，简单的感知/相邻帧理解满足不了真实 AI 系统的能力需求。

**核心矛盾**：视频天然带时序、信息密度高，要求模型在**长程、跨多帧**之间寻找线索并做多模态推理——这恰恰是现有"感知型"基准测不到的盲区。一个自然的问题是：当前 MLLM 能不能像 o3 处理图像那样"用视频思考"（think with videos）？

**本文目标**：构造一个真正考查多模态深度推理的视频基准，让答案无法靠表层感知获得，必须跨帧挖掘证据、理解潜台词。

**核心 idea（双过程理论分类）**：受 Kahneman 双过程理论启发，作者把任务分成 **隐式推理（Implicit）** 与 **显式推理（Explicit）**——前者考查"言外之意/潜台词"（如棕色大衣象征父亲、房间号 7 象征好运），更像 EQ 与世界知识的快思考；后者考查基于视频中明确呈现的细节做严谨逻辑推理（如魔术揭秘、因果分析）。

## 方法详解

### 整体框架
MMR-V 的构建围绕三条原则展开：**P1 多帧**（必须参考长程多帧）、**P2 深度推理**（答案不可直接感知，需理解潜台词）、**P3 真实**（与真实用户理解一致、无个体认知偏差）。整条流水线是"人工策展视频 → 人工标注问题与正确答案 → 模型对抗式生成干扰项 → checklist 人工质检"，最终得到 317 个视频、1257 道平均约 10 选项的多选题。

```mermaid
flowchart LR
    A[YouTube 视频策展<br/>4 条 checklist] --> B[人工标注<br/>问题+正确答案]
    B --> C[干扰项标注<br/>Str.1/2/3]
    C --> D[checklist 人工质检<br/>5 名标注员]
    D --> E[MMR-V<br/>317 视频 / 1257 任务]
```

### 关键设计

**1. 隐式/显式双轨任务体系：把"言外之意"显式纳入推理评测**。作者没有把多模态推理窄化为数学题、谜题那种文本主导的推理，而是把它拓宽到"整合艺术风格、光照、景深等视觉证据"乃至跨长程多帧的证据链。任务被切成 10 个大类、33 个子类，其中隐式侧含隐喻理解（MU）、主题理解（TU）、情绪识别（ER）、评论匹配（CM）、隐式符号（IS），显式侧含因果推理（CAR）、时序结构推理（SSR）、反直觉推理（CIR）、跨模态迁移推理（CTR）、视频类型与意图（VTI）。这套分类让基准既能测"快思考"的潜台词理解，也能测"慢思考"的跨帧逻辑。

**2. 长程多帧的证据分布：从根上杜绝"看一帧就答"**。视频策展时刻意排除日常记录、体育直播等线性描述型内容，只挑创作者精心设计、主题考究的视频，并偏好评论活跃的高人气视频以对齐大众认知。统计上视频时长跨度 7~3771 秒、平均 277 秒，单题平均需要在约 12 帧上推理、覆盖约 60% 视频时长——这意味着证据帧往往远离问题帧，模型不得不真正做跨帧检索与分析，而非感知相邻帧。

**3. 对抗式干扰项标注：用模型自身的错误来造"陷阱选项"**。为保证干扰项的迷惑性，作者设计了三种策略并做对照实验。Str.1 是核心：让强模型 GPT-4o 直接作答人工标注的问题，**若它答错（人工核验）就把这个错误答案留作高质量干扰项**；Str.2 是给定题目和正确答案让 GPT-4o 生成干扰项；Str.3 是人工手写。100 题的对照（Table 1）显示 Str.1 的干扰项最难——GPT-4o 在 Str.1 上仅 59%、Qwen-VL-7B 仅 37%。更关键的是，GPT-4o 直接答这 100 题时人工核验准确率只有 17%，直接暴露了当前模型多模态推理的短板，也佐证了用模型错误来构造干扰项的有效性。

## 实验关键数据

### 主实验
评测覆盖 11 个闭源 + 10 个开源模型，主设置为 zero-shot 与 zero-shot + CoT。随机准确率约 10%。

| 模型 | Overall | 说明 |
|------|---------|------|
| Gemini-2.5-pro (1 fps) | **64.3%** | 全场最高 |
| GPT-5 (固定帧数) | 60.9% | 固定帧设置下最佳 |
| GPT-4o | 52.8% | |
| GPT-4o-mini | 34.8% | 同架构小模型显著下降 |
| Gemma-3-27b-it | 开源最佳 | 仍落后闭源 |
| Qwen2.5-VL-72B / 7B | 39.1% / 30.1% | 体现 scaling law |
| LLaVA-Video | 18.4% | |

### 推理增强策略消融

| 策略 | 平均增益 |
|------|----------|
| CoT prompting | +0.88% |
| "Thinking" 模型 | +2.4% |
| 增加音频模态（Gemini-2.0-Flash 等） | +1.0%~+1.4% |

### 关键发现
- **多模态推理难题**：在文本域有效的 CoT 和 test-time scaling 在 MMR-V 上几乎无效（CoT 仅 +0.88%）。采样分析发现 CoT 中视觉分析仅占约 10%，说明模型推理仍是"基于问题帧感知 + 文本推理"，缺少把跨帧证据挖掘嵌入推理链的能力——多模态 CoT 与文本 CoT 本质不同。
- **多模态有增益**：支持全模态的模型加上音频后稳定小幅提升（约 1%）。
- **人机鸿沟**：模型在文本推理上可达人类水平，但在视频多模态推理上与人类仍有显著差距。
- **scaling law**：同架构下大模型显著优于小模型（如 GPT-4o 对 GPT-4o-mini 相对增益约 18%）。

## 亮点与洞察
- **"言外之意"作为一等公民**：把隐喻、主题、情绪、文化符号这类需要 EQ 与世界知识的隐式推理纳入评测，是对"多模态推理=数学/谜题"窄化定义的有力纠偏。
- **用模型错误造干扰项**：Str.1 把强模型的失败转化为高质量陷阱选项，既提升难度又自然对齐"模型易混淆点"，比纯人工或纯生成都更有针对性。
- **直指 CoT 本质差异**：通过"视觉分析仅占 CoT 10%"的量化证据，点明当前 MLLM 的推理仍是文本主导，为后续"视觉思维链"研究指明了方向。

## 局限与展望
- **规模有限**：317 视频 / 1257 题相对偏小，且全多选题格式可能存在选项启发式捷径，需配合更鲁棒的评测协议。
- **标注主观性**：隐式推理（潜台词、情绪、象征）天然带主观，虽以热门评论交叉校验缓解，但"正确答案"的客观性仍可能受质疑。
- **未给出解法**：论文主要诊断问题（CoT 无效、缺视觉推理），但没有提出增强多模态 CoT 的训练/推理方法，留给后续工作。

## 相关工作与启发
- **视频理解基准**（MVBench、Video-MME 等）侧重感知与相邻帧理解，MMR-V 与之正交，补上了"长程跨帧深度推理"的缺口。
- **多模态推理**（o3、GPT-5 的图像 think-with-image）启发了本文向视频域的延伸；MMR-V 的结论也反过来说明视频域比图像域的证据挖掘难得多。
- **认知科学**（Kahneman 双过程理论、Polanyi 默会知识）为隐式/显式分类提供了理论根基，是 benchmark 设计借鉴心理学的一个范例。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把隐式/言外之意推理与长程多帧证据挖掘引入视频评测，定位清晰、切口新颖。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 21 个模型 + 人类实验 + 干扰项策略/音频/scaling 多维消融，诊断扎实。
- **写作质量**: ⭐⭐⭐⭐ 动机、任务定义与发现层层递进，图例丰富，"CoT 只有 10% 视觉分析"等洞察令人印象深刻。
- **价值**: ⭐⭐⭐⭐ 暴露了当前 MLLM "用视频思考"的根本短板，为多模态推理研究提供了高质量、有区分度的试金石。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] IV-Bench: A Benchmark for Image-Grounded Video Perception and Reasoning in Multimodal LLMs](iv-bench_a_benchmark_for_image-grounded_video_perception_and_reasoning_in_multim.md)
- [\[ICLR 2026\] MMR-Life: Piecing Together Real-life Scenes for Multimodal Multi-image Reasoning](mmr-life_piecing_together_real-life_scenes_for_multimodal_multi-image_reasoning.md)
- [\[ICLR 2026\] Agent-X: Evaluating Deep Multimodal Reasoning in Vision-Centric Agentic Tasks](agent-x_evaluating_deep_multimodal_reasoning_in_vision-centric_agentic_tasks.md)
- [\[ICLR 2026\] PuzzleWorld: A Benchmark for Multimodal, Open-Ended Reasoning in Puzzlehunts](puzzleworld_a_benchmark_for_multimodal_open-ended_reasoning_in_puzzlehunts.md)
- [\[ICLR 2026\] MathNet: A Global Multimodal Benchmark for Mathematical Reasoning and Retrieval](mathnet_a_global_multimodal_benchmark_for_mathematical_reasoning_and_retrieval.md)

</div>

<!-- RELATED:END -->
