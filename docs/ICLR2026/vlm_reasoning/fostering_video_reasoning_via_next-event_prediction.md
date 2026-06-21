---
title: >-
  [论文解读] Fostering Video Reasoning via Next-Event Prediction
description: >-
  [ICLR 2026][VLM Reasoning][Next-Event Prediction] 本文提出 **Next-Event Prediction (NEP)** 这一学习任务——把视频切成"过去/未来"两段，让 MLLM 只看过去帧、预测未来事件的文字描述，用视频自带的未来内容当自监督信号来逼出时序推理能力；并配套构建了 33K 训练集 V1-33K 与评测基准 FutureBench。
tags:
  - "ICLR 2026"
  - "VLM Reasoning"
  - "Next-Event Prediction"
  - "视频时序推理"
  - "MLLM"
  - "自监督学习"
  - "FutureBench"
---

# Fostering Video Reasoning via Next-Event Prediction

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=8nUgzuvskm](https://openreview.net/forum?id=8nUgzuvskm)  
**代码**: 待确认  
**领域**: 多模态视频理解 / 时序推理  
**关键词**: Next-Event Prediction, 视频时序推理, MLLM, 自监督学习, FutureBench  

## 一句话总结
本文提出 **Next-Event Prediction (NEP)** 这一学习任务——把视频切成"过去/未来"两段，让 MLLM 只看过去帧、预测未来事件的文字描述，用视频自带的未来内容当自监督信号来逼出时序推理能力；并配套构建了 33K 训练集 V1-33K 与评测基准 FutureBench。

## 研究背景与动机
**领域现状**：MLLM 的视频理解能力近年突飞猛进，但主流的视频指令微调任务（视频问答 VQA、字幕生成 Captioning、时间戳定位 Grounding）本质上都是**观察性 (observational)** 的——做的是物体识别、事件识别、事实回忆这些"看见什么说什么"的感知活儿。

**现有痛点**：(1) 这些任务主要服务于跨模态对齐，却忽略了视频区别于静态图像的核心维度——**时间**：VQA 经常只靠几个关键帧就能答，Captioning 是逐帧映射成文字，都学不到动态事件的演进；(2) VQA 和时间戳定位往往需要**人工标注或更强 MLLM 蒸馏**，扩展性差、成本高。

**核心矛盾**：LLM 之所以会推理，靠的是 next-token prediction 这个又简单又能无限 scale 的自监督任务。可视频侧一直缺一个对应物——**到底用什么学习任务才能给 MLLM 灌入时序推理能力？**

**本文目标**：找到视频版的"next-token prediction"——一个自监督、可扩展、且专门逼出时序推理的学习任务。

**核心 idea**：**用视频的"未来"当监督信号**。把视频在某个因果转折点切开，模型只看前半段，被要求预测后半段会发生什么。因为目标文字描述的事件在输入里根本看不见，模型被迫从感知跨到预测——必须把视觉编码器的"看见的事实"和 LLM 里的"常识知识（物理规律、社会规范、人类典型行为）"结合起来做因果推断。

## 方法详解

### 整体框架
NEP 把视频帧序列 $V=[v_1,\dots,v_T]$ 在一个切分点 $t<T$ 处分成过去段 $V_{\le t}=[v_1,\dots,v_t]$ 和未来段 $V_{>t}=[v_{t+1},\dots,v_T]$，训练 MLLM 接收 $V_{\le t}$ 作为输入、生成描述未来段事件的文本 $Y$，本质是一个条件于视频帧的 seq-to-seq 语言建模问题。整个工作由三块组成：**NEP 任务形式化 → V1-33K 数据构造流水线 → 四种指令微调策略**，并配 FutureBench 做多跳时序推理评测。

```mermaid
flowchart LR
    A[原始视频+字幕] --> B[字幕分析<br/>LLM 找因果转折点]
    B --> C[Grounding 定位时间戳 t]
    C --> D[切成过去段/未来段<br/>字幕拆成 past/future]
    D --> E[可选: 推理+批判<br/>生成 reasoning trace]
    E --> F[V1-33K<br/>33K past-future 对]
    F --> G[MLLM 输入过去帧]
    G --> H[预测未来事件文本]
    H --> I[四种策略: SFT/CFT/Distill/Mix]
```

### 关键设计

**1. 用"未来段"做自监督，把感知任务变成预测任务**：NEP 的精髓在于监督目标描述的是输入里看不到的事件，这一点把模型从"感知"（物体检测、动作识别）逼到了"预测"。要预测合理的下一事件，MLLM 必须同时调用两路信息——视觉编码器感知到的视觉证据，以及 LLM 里的常识世界知识作为推理驱动力，把它们整合后向时间前方投射并保持叙事连贯。作者把这套任务类比成逻辑学的三种推理：VQA 是**归纳 (induction)**、NEP 是**演绎 (deduction)**、上一事件预测是**溯因 (abduction)**；实验证明演绎式的 NEP 在时序基准上收益最大，因为演绎需要刻意运用抽象逻辑原则、认知负荷更高，正好逼出推理能力。

**2. 低成本自动化数据流水线（V1-33K）**：流水线把视频转成训练样本只需三步——(i) **字幕分析**（可选）：用 LLM 解析视频字幕找出场景切换和"已发生→将发生"的因果转折点，得到文本切分点；(ii) **Grounding 与切分**：用 MLLM 把文本切分点对齐到视频时间轴得到时间戳 $t$，据此把视频切成过去/未来两段、字幕也对应拆成 past-caption 和 future-caption（监督目标）；(iii) **推理与批判**（可选）：用文本推理模型对 past-caption 生成预测与推理轨迹，再由另一个 LLM 批判，为后面 CFT/蒸馏准备数据。关键洞察是——整个信号构造**唯一的前置能力就是时间戳 grounding**，而这在早期微调阶段就已学会，于是 NEP 能产出"自动标注的自监督信号"，比 VQA 便宜得多（连 PhD 都觉得"出好问题"很难）。数据覆盖 YouTube、YouCook2、NextQA、Charades、ActivityNet 等，含物理事件、人际互动、体育等丰富场景。

**3. 控制变量式的任务对比**：为了干净地隔离"NEP 任务本身"的效果，作者刻意用**和 Captioning/QA 完全相同的视频数据源**来构造 NEP 数据，从而能在同模型、同视频、同数据量下只改任务形式来对比，把数据质量和来源偏差这两个混淆因素钉死。这种设计让"NEP 比 Captioning/MCQA/OEQA 更能涨时序能力"的结论更可信。

**4. 四种指令微调策略**：在固定 NEP 任务下进一步对比四种训练策略——**SFT**（直接用 ground-truth 未来字幕做交叉熵）、**CFT** 批判微调（用 GPT 生成的批判信号）、**Distill** 蒸馏（用 DeepSeek 的结构化推理轨迹）、**Mix** 混合（固定总预算下等比例混合三类监督，mini-batch 从统一混合池采样）。结论是 SFT 虽简单却最高效，CFT/Distill 也有效但要额外标注或辅助 LLM 反馈，性价比反而不如 SFT。

### FutureBench 评测设计
FutureBench 用多选 QA 形式评测，每个视频配一个从完整视频终态导出的"end anchor"（终点目标），模型要前向+后向推理出通往该结果的中间事件。分两种范式：**Extrapolation 外推**（按 1-hop/2-hop/3-hop 预测一串连续未来事件，跳数越多因果推理越深）和 **Interpolation 内插**（给定首个未来事件、锚点事件、终点事件，补出非连续的中间事件）。共 1056 条精心构造的 QA，干扰项设计成"常识上合理但与结果轨迹逻辑矛盾"。纯文本强推理模型 o4-mini 在去掉视觉输入后只有 **32.0%** 准确率，说明该基准确实强依赖视觉感知。

## 实验关键数据

### 主实验：NEP vs 其他视频指令微调任务（Qwen2.5-VL-7B，3K 样本）

| 任务 | 观察范围 | G-Avg.(通用) | T-Avg.(时序) | FutureBench |
|------|---------|-------------|-------------|-------------|
| Instruct(原始) | — | 60.3 | 49.7 | 52.6 |
| Captioning | 全视频 | 60.0 | 49.7 | 55.8 |
| MCQA | 全视频 | 58.5 | 47.7 | 60.3 |
| OEQA | 全视频 | 60.4 | 51.2 | 58.8 |
| **NEP** | 部分视频 | **60.9** | **53.5** | **61.3** |

NEP 在时序基准上显著领先（T-Avg. 53.5 vs 其他 ≤51.2），同时通用基准（VideoMME/MVBench/LongVideoBench）不掉点甚至略涨，说明它在不牺牲通用理解的前提下强化了时序推理。

### 消融：四种微调策略（部分结果）

| 模型 | 策略 | G-Avg. | T-Avg. |
|------|------|--------|--------|
| Qwen2.5-VL-3B | Instruct | 57.2 | 45.8 |
| | SFT | 56.3 | **48.2** |
| | Distill | 58.1 | 48.4 |
| | Mix | 57.9 | 48.5 |
| Qwen2.5-VL-7B | Instruct | 60.3 | 49.7 |
| | **SFT** | 59.7 | **52.6** |
| | Distill | 61.2 | 51.9 |
| | Mix | 59.9 | 53.3 |

SFT 这一最简单的策略就拿到了时序基准上的大幅提升（7B 上 T-Avg. 49.7→52.6），CFT/Distill 虽有效但依赖额外标注，性价比不如 SFT。

### 关键发现
- **NEP 涨时序不掉通用**：跨 TempCompass、TemporalBench、SEED-Bench-R1、FutureBench 四个时序基准一致提升，VideoMME/MVBench/LongVideoBench 保持稳定。
- **演绎 > 归纳/溯因**：把同一 3K 数据改成 VQA(归纳)、NEP(演绎)、上一事件预测(溯因)三种形式，演绎式 NEP 在时序基准上收益最大。
- **简单即有效**：SFT 这种最朴素的策略已能撬动主要收益，呼应了"任务能 scale 比策略花哨更重要"的 next-token prediction 哲学。

## 亮点与洞察
- **找到了视频版的"next-token prediction"**：把"未来段"当免费自监督信号，是个干净、可无限 scale、且专门对准时序推理的学习任务，理念上非常优雅。
- **控制变量做得扎实**：刻意复用 Captioning/QA 同源视频，让"任务形式"成为唯一变量，结论可信度高。
- **逻辑学三分类视角**：把 VQA/NEP/上一事件预测对应到归纳/演绎/溯因，为"为什么 NEP 更能涨推理"提供了认知科学层面的解释框架。
- **配套基准 FutureBench**：多跳外推+内插设计，o4-mini 纯文本仅 32% 证明了视觉感知不可或缺，是个有区分度的时序推理评测。

## 局限与展望
- **数据质量参差**：自动流水线产出的未来段监督难度差异大（有的 trivial 有的极难），作者承认这点并预期数据质量提升后 NEP 还能更强，但当前没有难度分层的精细控制。
- **依赖 grounding 能力**：整套自监督建立在"模型已会时间戳 grounding"的前提上，若早期微调阶段该能力不足，信号质量会受影响。
- **绝对增益有限**：通用基准基本持平、时序基准涨幅在几个点量级，NEP 更多是"任务设计层面的概念贡献"而非 SOTA 刷分。
- **未来方向**：把 NEP 推到更大数据规模验证 scaling law、结合 R1-style RL 进一步放大推理收益、向更长时序跨度的多跳预测扩展。

## 相关工作与启发
- **vs 视频指令微调**（Video-LLaVA / LLaVA-NeXT / Qwen-VL 系列）：这些模型的训练目标都是观察性的（描述/解释可见内容），本文换成预测性目标，对齐"建模世界动态"而非"理解静态帧"。
- **vs 计算机视觉的未来预测**（动作预测、未来帧预测、运动预测）：以往工作多在短时域（下一帧/下一动作）优化低层表示学习目标、目的是预训练视频编码器；本文聚焦**高层、语义级、自然语言表达**的未来事件预测，冻结视觉编码器、改进的是跨模态投影器和 LLM。
- **启发**：NEP 揭示了一条"用数据自带的时间结构造自监督信号"的通用思路，可迁移到机器人视频、自动驾驶预测等需要时序因果推理的具身场景；也提示 MLLM 训练应从"感知任务"转向"预测任务"来逼出更高阶能力。

## 评分
- 新颖性: ⭐⭐⭐⭐ — "用未来段当自监督信号"是个简洁有力的任务设计，逻辑学三分类视角也很有启发性，但 next-event/future prediction 的大方向并非首创。
- 实验充分度: ⭐⭐⭐⭐ — 控制变量对比扎实，覆盖 7 个基准 + 任务/策略/数据量三维消融，配套自建 FutureBench；不足是绝对增益有限、缺更大规模 scaling 验证。
- 写作质量: ⭐⭐⭐⭐ — 动机讲得清楚（对标 next-token prediction），图示和任务对比直观，三分类框架提升了说服力。
- 价值: ⭐⭐⭐⭐ — 为"如何给 MLLM 灌时序推理"提供了一个可扩展、低成本、可复现的范式与基准，对视频推理研究有方法论价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] OASIS: On-Demand Hierarchical Event Memory for Streaming Video Reasoning](../../CVPR2026/vlm_reasoning/oasis_on-demand_hierarchical_event_memory_for_streaming_video_reasoning.md)
- [\[ICLR 2026\] ExpVid: A Benchmark for Experiment Video Understanding & Reasoning](expvid_a_benchmark_for_experiment_video_understanding_reasoning.md)
- [\[CVPR 2026\] VGent: Visual Grounding via Modular Design for Disentangling Reasoning and Prediction](../../CVPR2026/vlm_reasoning/vgent_visual_grounding_via_modular_design_for_disentangling_reasoning_and_predic.md)
- [\[ICLR 2026\] VideoZoomer: Reinforcement-Learned Temporal Focusing for Long Video Reasoning](videozoomer_reinforcement-learned_temporal_focusing_for_long_video_reasoning.md)
- [\[ICLR 2026\] Vid-LLM: A Compact Video-based 3D Multimodal LLM with Reconstruction–Reasoning Synergy](vid-llm_a_compact_video-based_3d_multimodal_llm_with_reconstructionreasoning_syn.md)

</div>

<!-- RELATED:END -->
