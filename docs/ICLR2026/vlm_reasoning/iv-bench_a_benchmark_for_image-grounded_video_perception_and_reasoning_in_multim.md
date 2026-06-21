---
title: >-
  [论文解读] IV-Bench: A Benchmark for Image-Grounded Video Perception and Reasoning in Multimodal LLMs
description: >-
  [ICLR 2026][VLM Reasoning][图像锚定视频理解] IV-Bench 是首个"图像锚定视频感知与推理"基准——用一张**外部来源**的参考图作为视觉上下文去问视频里的问题，966 个视频配 2,560 条图文 query、13 类任务，结果发现最强 MLLM 也只拿到 28.9% 准确率（人类 88.8%），戳破了当前模型"看图理解视频"能力的虚胖。
tags:
  - "ICLR 2026"
  - "VLM Reasoning"
  - "图像锚定视频理解"
  - "多模态基准"
  - "视频感知"
  - "视频推理"
  - "MLLM 评测"
---

# IV-Bench: A Benchmark for Image-Grounded Video Perception and Reasoning in Multimodal LLMs

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=3di7ct0iOJ](https://openreview.net/forum?id=3di7ct0iOJ)  
**代码**: [https://github.com/multimodal-art-projection/IV-Bench](https://github.com/multimodal-art-projection/IV-Bench)  
**领域**: 多模态评测 / 视频理解 / VLM 推理  
**关键词**: 图像锚定视频理解, 多模态基准, 视频感知, 视频推理, MLLM 评测  

## 一句话总结
IV-Bench 是首个"图像锚定视频感知与推理"基准——用一张**外部来源**的参考图作为视觉上下文去问视频里的问题，966 个视频配 2,560 条图文 query、13 类任务，结果发现最强 MLLM 也只拿到 28.9% 准确率（人类 88.8%），戳破了当前模型"看图理解视频"能力的虚胖。

## 研究背景与动机
**领域现状**：MLLM 在图像、视频理解上进展飞快，配套的评测基准（Video-MME、MVBench、LongVideoBench 等）层出不穷。但这些基准几乎清一色用**纯文本 query** 去问视频内容。

**现有痛点**：语言本质上太抽象，无法精确指代细粒度视觉属性（某个设计上的微妙差异）或特定实例（某个人的脸）。现实里"以图搜物"的需求巨大——Pinterest Lens 早在 2018 年就月处理 6 亿次视觉搜索，62% 的年轻消费者明确想要视觉搜索能力，正是因为文字在这些场景下力不从心。然而"用一张图作为视觉锚点去理解视频"这件事，在现有评测框架里被系统性地忽视了。

**核心矛盾**：少数尝试图文 query 的基准（V2P-Bench、VideoRefer-Bench）有个致命缺陷——参考图直接**从视频帧里抠出来**，这让任务退化成"帧匹配"：模型靠感知相似度就能蒙对，存在信息泄漏，根本测不出泛化理解能力。而唯一用外部图的 Video-MMMU，视频又只是辅助、不答题也行。

**本文目标**：构建第一个真正考验"图像锚定视频感知与推理"的基准，要求**图和视频都不可或缺**，缺一就答不对。

**核心 idea**：**[外部图作为强制视觉锚点]** 参考图一律来自视频之外的网络来源，加上两轮严苛质控强制"图必需 + 视频必需 + 至少两个有效干扰项"，从根上杜绝帧匹配捷径和单模态泄漏，逼模型把图里的概念性知识真正落到视频内容上。

## 方法详解

### 整体框架
IV-Bench 不是一个模型，而是一套数据构建 + 评测协议。它围绕"图文 query → 视频"的配对组织：966 个时长均超 5 分钟的视频，覆盖知识、影视、体育、艺术表演、生活记录 5 大类；每个视频配多条 query，每条 query = 一张外部参考图 + 一段文本问题 + 多选项。13 类任务被分成 7 类感知任务（看得见、认得出）和 6 类推理任务（算得出、推得通）。整套数据经过"标注 → 第一轮质控（清晰性/正确性/分类）→ 第二轮质控（强制图必需、补足有效干扰项）"的流水线产出，最终用 32 帧均匀采样、`视频帧 + 图 + 问题` 的统一格式去评测 28 个 MLLM。

```mermaid
flowchart LR
    A[选 966 个视频<br/>5 类 · 时长>5min] --> B[人工标注<br/>外部图+文本问题+干扰项]
    B --> C[第一轮质控<br/>清晰/正确/分类校对]
    C --> D[第二轮质控<br/>删可单模态解出的样本<br/>补≥2个有效干扰项]
    D --> E[2560 条图文 query<br/>13 任务=7感知+6推理]
    E --> F[统一评测<br/>32帧+图+问题 → 多选准确率]
```

### 关键设计

**1. 外部来源参考图：把"帧匹配"拦在门外。** IV-Bench 与同类图文基准最本质的区别，是参考图**禁止从视频里截取**，必须由标注者从网络上另找一张与视频里某关键词、人物、主题相关的图。这一条看似简单却极关键：当图来自视频帧本身时，模型只要做视觉相似度对齐就能定位答案，评测沦为感知匹配；而外部图迫使模型先理解图里的概念（"这是某球员的球衣""这是某种器物"），再把这个概念性知识泛化到视频内容上，从而真正测出"图锚定的视频理解"。表 1 用 ImgSrc（图来源）、ImgNec（图必需）、VidNec（视频必需）三个维度量化了这一差异——IV-Bench 是唯一同时满足 Out-of-Video + 图必需 + 视频必需的基准。

**2. 两轮质控强制"双模态缺一不可"。** 高质量的核心在于第二轮质控：任何能仅凭常识或仅凭视频就答出的样本一律删除；任何无意中泄漏了图像视觉内容的文本 query 都被重写。最精妙的是**有效干扰项**的设计——每题至少塞两个"对当前图是错的、但若换一张图配同样问题就会变成正确答案"的干扰项。这等于构造了一组反事实选项：只有真正读懂了"这张特定的图"，才能在这些貌似都合理的选项里挑对，从机制上确保图对每个样本都是必需的。

**3. 13 任务的感知—推理双层分级。** 任务设计从"看见"到"推断"覆盖一条完整能力谱。7 类感知任务考"直接提取视觉信息"：Existence（图中物体是否出现在视频）、Reverse Existence（图中有而视频中无）、NLI（视频场景与图相似度）、Spatial Relationship（空间关系）、Keyframe Extraction（图中内容首次出现的时间戳）、Constrained OCR（受图约束的文字识别）、Detailed Events。6 类推理任务考"高阶认知"：Counting、Space-Time Computing（算时长/距离）、Summary、Instruction Understanding（理解物体功能/制作过程）、Attribute Change（属性变化）、Temporal Reasoning（结合世界知识推断事件起止时间）。这种分级让基准既能定位模型在哪一层断裂，也方便诊断"感知尚可、推理崩盘"这类系统性短板。

**4. 评测协议中的输入顺序与 token 分配探针。** 评测默认用 `视频帧 + 图 + 问题` 顺序、32 帧均匀采样。论文进一步把"图放在视频帧之后 vs 之前""帧数 vs 分辨率"做成可控变量，发现**图放在视频帧之后效果最好**、**性能对帧数比对分辨率更敏感**、且**只有大模型才能从图上下文获益、小模型几乎无提升**。这几条不是附带结论，而是该基准作为诊断工具的核心产出——它把"如何向 MLLM 喂入图像上下文"这一工程问题量化成了可操作的设计准则。

## 实验关键数据

### 主实验表格
28 个 MLLM（5 商业 + 23 开源）统一在 32 帧、`视频帧+图+问题` 格式下评测，准确率（%）：

| 模型 | Overall | P-Avg(感知) | R-Avg(推理) |
|------|---------|-------------|-------------|
| Human | **88.8** | 91.5 | 86.9 |
| Qwen2.5-VL-72B | **28.9** | — | — |
| InternVL2.5-78B | 28.6 | 33.4 | 21.9 |
| Gemini-2.0-Pro | 27.7 | — | 24.9 |
| Qwen2.5-VL-7B(<10B 最佳) | 18.5 | — | — |
| 随机猜测 | 11.11 | — | — |
| Llama-vid | 10.5 | 11.2 | 9.5 |

注：平均选项数 9，故随机基线约 11.11%。

### 消融实验表格

| 消融维度 | 关键发现 |
|----------|----------|
| 是否加入图像上下文 | 加图显著提升视频理解（验证图的必要性） |
| 模型规模 | 大模型能利用图上下文、小模型几乎无收益（能力随 scale 涌现） |
| 帧数 vs 分辨率 | 性能对**增加帧数**比对**提高分辨率**更敏感 |
| 图的位置 | 图放在**视频帧之后**效果最好 |

### 关键发现
- **人机鸿沟巨大**：人类 88.8% vs 最强模型 28.9%，相差近 60 个点，说明"图锚定视频理解"是当前 MLLM 的真实盲区，而非已被解决的能力。
- **感知 > 推理**：模型普遍在感知任务上好于推理任务（如 InternVL2.5-78B 感知 33.4% / 推理 21.9%），Temporal Reasoning 尤其崩。
- **小模型连图都用不起来**：10B 以下最佳的 Qwen2.5-VL-7B 仅 18.5%，利用视觉上下文的能力强依赖模型规模。

## 亮点与洞察
- **"外部图 + 双模态必需"是真正的方法论创新**：它把图文视频基准从"帧匹配测试"升级为"概念泛化测试"，反事实干扰项的设计尤其巧妙，从机制上堵死了刷分捷径。
- **诊断价值高**：感知/推理双层 + 13 细任务 + 输入顺序/token 分配的系统消融，给出的不只是一个排行榜，而是一份"怎样向 MLLM 喂图像上下文"的工程指南。
- **规模一致性强**：28 个模型、人类基线、9 个干扰项的难度设置，让"28.9% vs 88.8%"这个结论非常可信。

## 局限与展望
- **多选题格式的天花板**：全部任务为多选，虽便于自动评分，但无法考验开放式生成、定位框回归等更贴近真实交互的能力。
- **静态单图约束**：每条 query 只配单张外部图，未覆盖"多图序列锚定""图+视频混合检索"等更复杂场景。
- **只诊断不开方**：论文给出了输入顺序、帧数等经验准则，但未提出能显著缩小人机差距的建模方案，后续如何让模型真正"把图概念落到视频"仍是开放问题。
- 视频均 >5 分钟、32 帧采样，长视频下稀疏采样可能本身就漏掉关键帧，帧数敏感性结论也部分受此影响。

## 相关工作与启发
- **纯文本视频基准**（Video-MME、MVBench、LongVideoBench、MLVU、MMBench-Video）：奠定了视频理解评测范式，但 query 全是文本，无法考图锚定能力——IV-Bench 正是补这块空白。
- **图文视频基准**（V2P-Bench、VideoRefer-Bench）：引入了图 query，但图来自视频内部，退化为帧匹配；IV-Bench 用外部图破解了这一缺陷。
- **Video-MMMU**：唯一用外部图，但视频可有可无；IV-Bench 强制图与视频双必需。
- **启发**：对做多模态评测的研究者，"反事实干扰项 + 强制双模态必需"是一种可复用的防泄漏数据构建范式；对做 MLLM 建模的研究者，"图放视频帧之后、帧数优先于分辨率"是即拿即用的工程默认值。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个外部图锚定的视频感知推理基准，反事实干扰项 + 双模态必需的构造思路有真正的方法论增量，不是简单堆数据。
- 实验充分度: ⭐⭐⭐⭐ 28 个模型 + 人类基线 + 输入顺序/规模/帧数-分辨率多维消融，覆盖面足够支撑结论。
- 写作质量: ⭐⭐⭐⭐ 动机讲得有画面（视觉搜索的现实数据）、任务定义清晰、对比表把差异量化得很明白。
- 价值: ⭐⭐⭐⭐ 暴露了 MLLM 在图锚定视频理解上的真实短板（28.9% vs 88.8%），并给出可操作的工程准则，对评测和建模两端都有指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos](mmr-v_whats_left_unsaid_a_benchmark_for_multimodal_deep_reasoning_in_videos.md)
- [\[ICLR 2026\] GIR-Bench: Versatile Benchmark for Generating Images with Reasoning](gir-bench_versatile_benchmark_for_generating_images_with_reasoning.md)
- [\[CVPR 2026\] MMTIT-Bench: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation](../../CVPR2026/vlm_reasoning/mmtit-bench_a_multilingual_and_multi-scenario_benchmark_with_cognition-perceptio.md)
- [\[ICLR 2026\] ExpVid: A Benchmark for Experiment Video Understanding & Reasoning](expvid_a_benchmark_for_experiment_video_understanding_reasoning.md)
- [\[CVPR 2026\] CaST-Bench: Benchmarking Causal Chain-Grounded Spatio-Temporal Reasoning for Video Question Answering](../../CVPR2026/vlm_reasoning/cast-bench_benchmarking_causal_chain-grounded_spatio-temporal_reasoning_for_vide.md)

</div>

<!-- RELATED:END -->
