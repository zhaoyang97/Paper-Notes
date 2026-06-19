---
title: >-
  [论文解读] VEU-Bench: Towards Comprehensive Understanding of Video Editing
description: >-
  [CVPR 2025][视频生成][视频编辑理解] 提出 VEU-Bench，首个全面评估视频大模型对视频编辑元素理解能力的基准，涵盖10个编辑维度、3个评估层级（识别/推理/判断）共19个细粒度任务，并训练专家模型 Oscars 超越开源SOTA 28.3%。 领域现状：互联网上广泛分享的视频大多是经过编辑的视频…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "视频编辑理解"
  - "视频大模型基准"
  - "剪辑理解"
  - "镜头分析"
  - "抽象推理"
---

# VEU-Bench: Towards Comprehensive Understanding of Video Editing

**会议**: CVPR 2025  
**arXiv**: [2504.17828](https://arxiv.org/abs/2504.17828)  
**代码**: 项目主页 (project page)  
**领域**: 视频理解 / 视频编辑  
**关键词**: 视频编辑理解, 视频大模型基准, 剪辑理解, 镜头分析, 抽象推理

## 一句话总结

提出 VEU-Bench，首个全面评估视频大模型对视频编辑元素理解能力的基准，涵盖10个编辑维度、3个评估层级（识别/推理/判断）共19个细粒度任务，并训练专家模型 Oscars 超越开源SOTA 28.3%。

## 研究背景与动机

**领域现状**：互联网上广泛分享的视频大多是经过编辑的视频。视频编辑涉及镜头构图（如景别、角色）、镜头运动（如推拉摇移）、剪切类型（如匹配剪切、跳切）和转场效果等多个维度。Video-LLMs（如Qwen2-VL、LLaVA-Video等）在通用视频理解任务上取得了显著进步。

**现有痛点**：三个核心问题。(1) 现有VEU基准（如AVE、MovieCuts、AutoTransition）主要聚焦于编辑元素的分类，缺乏对推理和判断层面的评估——比如不仅要识别"这是匹配剪切"，还要能解释为什么用匹配剪切以及它的叙事效果。(2) 现有基准覆盖的编辑维度不完整，通常只关注镜头设置或剪切类型中的某一个子集。(3) 缺乏对Video-LLMs在VEU任务上的系统性评估。

**核心矛盾**：视频编辑元素是抽象概念（如"匹配剪切"需要理解跨场景的形状/运动对齐模式），而非现实世界中直接可观察的物体或动作。这种抽象性使VEU成为评估模型抽象推理能力的天然场景，但当前Video-LLMs在这方面严重不足。

**本文目标**：(1) 构建全面覆盖视频编辑各维度各层级的基准；(2) 系统评估当前SOTA Video-LLMs的VEU能力；(3) 验证VEU数据对提升通用视频理解的价值。

**切入角度**：将视频编辑元素按帧内/帧间/镜头间三个粒度分类，在每个维度上从识别→推理→判断三个层级递进评估，并利用本体知识库实现高质量的自动标注。

**核心 idea**：构建基于本体知识库的标注pipeline，将VEU从简单分类扩展到推理和判断层级，用50K高质量数据训练的专家模型证明VEU数据能显著提升Video-LLMs的通用视频理解能力。

## 方法详解

### 整体框架

VEU-Bench包含30,000个视频和49,536个QA样本。数据来源于AVE、MovieCuts、AutoTransition等已有数据集，经过过滤、平衡和扩充。通过三层级任务设计（识别/推理/判断）× 十个编辑维度构成19个细粒度任务。Oscars专家模型基于Qwen2-VL-7B用LoRA在VEU-50K训练集上微调。

### 关键设计

1. **十维度三层级任务体系**:

    - 功能：系统化地覆盖视频编辑理解的各个方面
    - 核心思路：**帧内维度**(6个)——景别(Shot Size)、角度(Angle)、场景(Location)、主体(Subject)、类型(Type)、色彩(Color)，只需分析单帧。**帧间维度**(2个)——镜头运动(Motion)、播放速度(Speed)，需要分析同一场景内的多帧变化。**跨镜头维度**(2个)——剪切类型(Cut)、转场效果(Transition)，涉及不同场景之间的切换。三个评估层级：**识别**层（选择题分类）、**推理**层（识别+提供证据和原理）、**判断**层（评估编辑元素在具体视频中的功能和效果）。
    - 设计动机：专业视频编辑教程定义了这三个粒度的分类，三层级评估从"是什么"→"为什么这样"→"效果如何"逐步加深难度。

2. **基于本体知识库的自动标注Pipeline**:

    - 功能：将VEU任务从简单分类扩展到推理和判断，同时保证高质量标注
    - 核心思路：首先参考专业剪辑教程为每个编辑元素构建知识库——推理任务定义"关键属性"（描述各维度的抽象模式），判断任务定义"功能"（描述编辑元素在视频内容中的作用）。标注时MLLM根据视频内容选择最相关的属性/功能，然后将抽象术语（如"物体"、"场景"）替换为具体内容。例如"匹配剪切"的属性"连接两个形状相似的物体跨帧"被具体化为"匹配剪切将形状相似的骨头和太空船连接跨帧"。
    - 设计动机：直接让MLLM标注开放域推理任务效果差（即使GPT-4o也挣扎于VEU），通过将开放式推理简化为"基于知识库的改写"任务，降低了标注难度，保证了标注质量，同时确保回答包含正确的编辑模式知识。

3. **Pattern Matching评估机制**:

    - 功能：更准确地评估推理和判断任务的回答质量
    - 核心思路：评估分数由两部分组成——模式匹配分(PM)和信息匹配分(IM)。PM衡量回答与编辑模式本体的对齐度（video-agnostic的编辑知识），IM评估回答中具体视觉细节的准确性。最终开放题分数 $S_{oe} = (5 \times Acc + S_{match})/2$，其中 $S_{match} = (PM + IM)/2$。
    - 设计动机：直接用LLM评分容易被回答中正确的描述性信息（如正确识别场景物体）所"迷惑"，即使模型误判了编辑模式也可能得高分。PM正则化降低了这种偏差，使评分更聚焦于编辑理解能力本身。

### 损失函数 / 训练策略

Oscars基于Qwen2-VL-7B，使用LoRA微调（r=16, α=32）。学习率1e-4，weight decay 0.01，warmup比例0.05，AdamW优化器。视频采样1 fps，最大64帧。训练集45,154个样本，4块A100 GPU训练。

## 实验关键数据

### 主实验

| 模型 | Score_mc (识别) | Score_oe (推理+判断) | Score_all | vs开源SOTA提升 |
|------|----------------|---------------------|-----------|-------------|
| GPT-4o | **2.93** | **2.36** | **2.64** | - |
| Gemini-1.5-Pro | 2.71 | 2.11 | 2.44 | - |
| LLaVA-OV-7B (开源SOTA) | 2.27 | 1.69 | 1.98 | - |
| Qwen2-VL-7B (base) | 2.33 | 1.31 | 1.82 | - |
| **Oscars (ours)** | **2.85** | **2.23** | **2.54** | **+28.3%** |

### 消融实验

| 配置 | 通用基准提升 | 说明 |
|------|------------|------|
| Qwen2-VL → Oscars (VideoMME-Attribute) | +7.3% | 属性感知 |
| Qwen2-VL → Oscars (MVBench-State Change) | +5.5% | 状态变化 |
| Qwen2-VL → Oscars (TempCompass-Order) | +8.5% | 顺序理解 |
| Qwen2-VL → Oscars (平均9个推理任务) | +8.3% | 推理能力全面提升 |
| Simple prompt vs Context prompt (Qwen2-VL) | +13.7% | 加入编辑知识定义改善最大 |
| Simple prompt vs Context prompt (VideoLLaMA2) | +6.6% | 弱模型也受益 |
| 仅PM评分 vs PM+IM评分 (Spearman) | 更高对齐 | PM正则化提升人类对齐度 |

### 关键发现

- **Video-LLMs在VEU任务上表现远低于通用基准**：在Video-MME上80%+的模型在VEU-Bench上部分维度低于随机猜测。识别帧内编辑元素远比识别帧间/跨镜头元素容易。
- **推理和判断任务极具挑战性**：所有模型平均开放题分数不到2/5。判断任务（理解编辑意图）比推理任务（描述编辑模式）更难。
- **VEU数据显著提升通用视频理解**：仅用50K VEU数据微调，就在VideoMME、MVBench、TempCompass的推理相关任务上平均提升8.3%。这证明编辑理解训练确实增强了模型的抽象推理能力。
- **概念知识 vs 视觉感知的独立性**：模型对编辑概念的文本描述准确率不低(~2.7/3.0)，表明它们"知道"剪辑概念，但无法在视频中"看到"它们。问题出在语言模型内在知识与视觉感知模块的对齐上。
- **Context prompt提供编辑定义可大幅改善效果**：对Qwen2-VL提升13.7%，但对已经很强的Gemini效果不明显。

## 亮点与洞察

- **VEU作为抽象推理的训练数据**：编辑元素是抽象的专业概念，理解它们需要模式识别和推理能力。VEU训练数据意外地提升了通用视频理解中的推理任务，揭示了一种新的训练数据策略——使用领域特定的抽象任务数据来增强通用推理能力。
- **"知识有但看不到"的诊断实验**：通过文本概念测试区分了"缺知识"和"知识-视觉对齐差"两种失败模式，发现后者是主因。这为改进方向提供了清晰指引。
- **本体知识库驱动的标注pipeline**：将开放式推理标注简化为"选择+改写"两步，既保证了专业准确性又实现了规模化，这种策略可扩展到其他需要专业知识的标注场景。

## 局限与展望

- 视频主要来自已有数据集(AVE、MovieCuts、AutoTransition)，覆盖的视频类型和风格有限
- 当前只评估了短视频片段(1-60秒)，长视频中更复杂的编辑叙事理解未涉及
- Oscars虽然超越开源SOTA，但在Cut和Transition维度仍弱于GPT-4o，说明跨镜头理解仍是难点
- 评估中pattern matching的PM分数依赖LLM判断，可能存在系统性偏差
- 知识库目前基于通用编辑教程，未涵盖特定领域的编辑手法（如纪录片、MV等）

## 相关工作与启发

- **vs AVE/MovieCuts/AutoTransition**: 这些基准只做分类任务（识别镜头类型或剪切类型），VEU-Bench将评估扩展到推理和判断三个层级，覆盖更全面的编辑维度。
- **vs EditQA-2k**: EditQA探索了Video-LLMs分析编辑视频内容的能力，但仍局限于编辑效果而非编辑元素本身的理解。VEU-Bench提供了更系统化的编辑理解评估。
- **vs Video-MME/MVBench等通用基准**: 通用基准关注动作、事件、时序等"自然"视觉理解，忽略了视频编辑这种"人工"维度的理解。VEU-Bench填补了这个空白，且VEU训练数据可作为通用基准的有效补充。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统化提出视频编辑理解的三层级评估框架，本体知识库标注pipeline新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 11个模型全面评估，多维度深入分析，通用基准迁移验证充分
- 写作质量: ⭐⭐⭐⭐ 任务定义清晰，实验分析逻辑严谨，图表信息丰富
- 价值: ⭐⭐⭐⭐ VEU数据提升通用理解的发现有启发性，评估框架和数据集对社区有价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Mimir: Improving Video Diffusion Models for Precise Text Understanding](mimir_improving_video_diffusion_models_for_precise_text_understanding.md)
- [\[CVPR 2025\] Video-Bench: Human-Aligned Video Generation Benchmark](video-bench_human-aligned_video_generation_benchmark.md)
- [\[ICML 2026\] V2V-Bench: A Comprehensive Benchmark for Video-to-Video Generation Evaluation](../../ICML2026/video_generation/v2v-bench_a_comprehensive_benchmark_for_video-to-video_generation_evaluation.md)
- [\[CVPR 2025\] SketchVideo: Sketch-Based Video Generation and Editing](sketchvideo_sketch-based_video_generation_and_editing.md)
- [\[CVPR 2025\] Pathways on the Image Manifold: Image Editing via Video Generation](pathways_on_the_image_manifold_image_editing_via_video_generation.md)

</div>

<!-- RELATED:END -->
