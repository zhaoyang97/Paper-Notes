---
title: >-
  [论文解读] VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection
description: >-
  [CVPR 2025][LLM推理][视频链式思维] VideoEspresso 构建了一个20万+的大规模视频CoT推理数据集（包含空间bounding box和时间grounding标注），并提出VideoQA-SC混合框架——用1.5B轻量级模型选择平均2.36个核心帧，再用8B推理模型进行两阶段证据提取+答案生成，以仅1.8%的帧数和14.7%的计算量超越了GPT-4o和所有开源LVLM。
tags:
  - "CVPR 2025"
  - "LLM推理"
  - "视频链式思维"
  - "核心帧选择"
  - "视频QA"
  - "多模态推理"
  - "数据集"
---

# VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection

**会议**: CVPR 2025  
**arXiv**: [2411.14794](https://arxiv.org/abs/2411.14794)  
**代码**: [https://github.com/hshjerry/VideoEspresso](https://github.com/hshjerry/VideoEspresso)  
**领域**: LLM推理  
**关键词**: 视频链式思维, 核心帧选择, 视频QA, 多模态推理, 数据集

## 一句话总结

VideoEspresso 构建了一个20万+的大规模视频CoT推理数据集（包含空间bounding box和时间grounding标注），并提出VideoQA-SC混合框架——用1.5B轻量级模型选择平均2.36个核心帧，再用8B推理模型进行两阶段证据提取+答案生成，以仅1.8%的帧数和14.7%的计算量超越了GPT-4o和所有开源LVLM。

## 研究背景与动机

**领域现状**：LVLM在多模态理解上取得了显著进展，但在视频推理任务上仍有不足，主要受限于高质量大规模VideoQA数据集的稀缺。

**现有痛点**：(1) 手动标注成本高且粒度不够细；(2) 自动构造方法采用逐帧分析，计算昂贵且引入大量冗余信息；(3) 现有视频CoT工作主要在文本层面做推理，忽视了视觉grounding（物体在哪？是哪一帧？）；(4) 现有LVLM处理视频时通常均匀采样大量帧（如128帧），计算开销巨大但大部分帧是冗余的。

**核心矛盾**：视频中信息高度冗余，但复杂推理又需要精确定位关键帧和关键物体。均匀采样浪费计算资源，过少采样则可能丢失关键信息。

**本文目标** (1) 构建一个包含多模态CoT标注（空间+时间grounding）的大规模VideoQA数据集；(2) 设计高效的视频推理框架，用极少帧实现高质量推理。

**切入角度**：先用语义感知的方法去冗余和生成QA对，再用GPT-4o标注多模态CoT（核心帧+关键物体+证据链），最后训练一个"先选帧再推理"的混合框架。

**核心 idea**：用轻量模型选出2-3个核心帧，再用大模型做"提取证据→基于证据推理"的两阶段细粒度视频推理。

## 方法详解

### 整体框架

分数据构建和模型框架两部分：**数据构建**——从7个视频数据集收集视频，经自适应FPS采样 → InternVL2-8B帧描述 → BGE-M3语义去冗余 → GPT-4o生成QA对 → Claude/Gemini质量过滤 → GPT-4o标注多模态CoT（核心帧选择+关键物体提取+证据生成）→ GroundingDINO空间标注 + BGE-M3时间检索。**模型框架**（VideoQA-SC）——Stage 1: 轻量级Frame Selector（InternVL2-1B + Qwen-0.5B）选核心帧；Stage 2: 推理LVLM两阶段SFT——先训练证据提取，再训练基于证据的推理答案生成。

### 关键设计

1. **语义感知帧去冗余**:

    - 功能：从视频中高效去除冗余帧，保留关键信息
    - 核心思路：先自适应FPS采样（动态场景FPS 2-4，静态FPS 1），再用InternVL2-8B为每帧生成描述，通过BGE-M3模型计算相邻帧描述的语义相似度，去除相似度超过阈值τ的冗余帧（LIFO过滤策略）
    - 设计动机：基于文本语义而非像素级差异来判断冗余，更能捕捉内容层面的变化

2. **多模态CoT标注流水线**:

    - 功能：为每个QA对生成包含空间和时间grounding信息的推理链
    - 核心思路：三步走——(1) GPT-4o从帧描述中选择与问题最相关的核心帧描述；(2) 从核心帧描述中提取关键物体作为证据；(3) 将关键物体组织成自然语言推理链。空间标注用GroundingDINO标记bounding box并用CLIP验证一致性，时间标注用BGE-M3语义检索匹配原始帧
    - 设计动机：仅文本CoT缺乏视觉grounding，多模态CoT通过空间+时间定位让推理过程可追溯、可验证

3. **VideoQA-SC混合LVLM协作框架**:

    - 功能：高效精准的视频推理
    - 核心思路：**Frame Selector**——1B参数LVLM生成帧描述 + 0.5B参数LLM根据问题选择核心帧（平均仅2.36帧）。**两阶段推理LVLM**——Stage 1训练证据提取（"请提供有助于回答问题的证据"），Stage 2训练答案生成（"请结合证据回答问题"）。两阶段渐进式训练确保多模态信息的逐步整合
    - 设计动机：分离帧选择和推理，用极小的模型做帧选择（节省计算），让大模型只处理最关键的2-3帧；两阶段SFT防止模型跳过证据直接猜答案

### 损失函数 / 训练策略

两阶段SFT均使用LoRA微调：lr分别为2e-5和1e-5，batch=16，8×A100，LoRA rank=16, alpha=32。输入分辨率224×224，max token 6144，1个epoch，cosine decay。

## 实验关键数据

### 主实验

| 模型 | 参数 | 帧数 | TFLOPs | Avg Acc (14 tasks) |
|------|-----|------|--------|-------------------|
| GPT-4o | - | FPS3 | - | 26.4% |
| Qwen-VL-Max | - | FPS3 | - | 26.0% |
| InternVL2-8B | 8B | FPS1 | 73.2 | 28.7% |
| Qwen2-VL-7B | 7B | FPS1 | 64.6 | 28.5% |
| LongVA-DPO-7B | 7B | 128帧 | 465.4 | 24.4% |
| **VideoEspresso** | **8.5B** | **2.36帧** | **9.26** | **34.1%** |

主观评价（10分制总分）：

| 模型 | 逻辑性 | 事实性 | 准确性 | 简洁性 | 综合 |
|------|--------|--------|--------|--------|------|
| GPT-4o | 73.2 | 63.1 | 61.7 | 70.0 | 66.1 |
| InternVL2 | 70.6 | 56.3 | 54.5 | 66.8 | 60.1 |
| **VideoEspresso** | **72.3** | **61.3** | **59.7** | **75.7** | **65.8** |

### 消融实验

| 配置 | Accuracy | 变化 |
|------|----------|------|
| Full model | 34.13% | - |
| GT-CoT（oracle证据） | 72.95% | +38.82% |
| w/o Bbox（去除空间标注） | 33.14% | -0.99% |
| w/o CoT（去除推理链） | 31.32% | -2.81% |

核心帧选择器泛化性：

| 模型 | 均匀采样 | + Selector | 变化 |
|------|---------|-----------|------|
| GPT-4o (16帧) | 26.9% | 29.5% (2.36帧) | +2.6% |
| InternVL2 (16帧) | 28.6% | 30.0% (2.36帧) | +1.4% |

### 关键发现

- **仅2.36帧即可超越128帧方法**：VideoEspresso用LongVA-DPO 1.8%的帧数和2%的FLOPs就超出了9.7%的准确率，证明了"精选少量帧"远优于"暴力多帧"
- **GT-CoT的巨大提升空间**：使用ground truth证据可从34.1%飙升至73.0%（+38.8%），说明当前模型的证据提取能力还有极大改进空间
- **核心帧选择器可作为plug-and-play模块**：直接插入GPT-4o可在帧数减少85%的同时提升2.6%准确率
- **简洁性上甚至超越GPT-4o**：主观评价中简洁性得分75.7 vs GPT-4o的70.0，说明精选核心帧减少了冗余输出

## 亮点与洞察

- **"少即是多"的哲学**在视频推理中效果惊人——2.36帧 > 128帧。核心帧选择器通过语义理解而非均匀采样实现了信息的极致压缩
- **两阶段"先证据后推理"的训练策略**非常巧妙——避免了模型走"看题猜答案"的捷径，强制其先显式提取视觉证据
- **数据集构建流水线的自动化程度高**：InternVL2帧描述→BGE-M3去冗余→GPT-4o生成QA→多LLM交叉验证→GroundingDINO空间标注，整体pipeline具有很好的可扩展性

## 局限与展望

- GT-CoT实验表明自动标注的CoT质量仍有巨大提升空间（34.1% vs 72.9%）
- 测试集仅1382条，规模偏小，可能存在统计方差
- 选择器在某些模型上反而轻微降低性能（如LongVA -0.6%），说明选择器的泛化性还需提升
- 14个任务分布不均（因果推理8.7万 vs 烹饪步骤276），可能影响评估公平性

## 相关工作与启发

- **vs LongVA-DPO**: LongVA用128帧暴力处理视频，VideoEspresso用2.36帧就大幅超越（+9.7%），且FLOPs仅为2%
- **vs MVBench**: MVBench关注基础视频理解（"是什么物体"），VideoEspresso强调复杂推理（"为什么"、"推断"），在数据集设计层面就有根本区别
- **vs 文本CoT方法**: 纯文本CoT缺乏视觉grounding，VideoEspresso引入空间bbox和时间定位使推理链可追溯

## 评分

- 新颖性: ⭐⭐⭐⭐ 多模态CoT标注和混合选帧-推理框架设计新颖，但各子模块（帧描述、语义检索等）是已有技术的组合
- 实验充分度: ⭐⭐⭐⭐ 14任务9模型的全面对比+消融+跨模型泛化性验证充分
- 写作质量: ⭐⭐⭐⭐ 流水线描述清晰，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 20万高质量视频CoT数据集+高效推理框架，对视频理解研究具有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought](argus_vision-centric_reasoning_with_grounded_chain-of-thought.md)
- [\[CVPR 2025\] Interleaved-Modal Chain-of-Thought](interleaved-modal_chain-of-thought.md)
- [\[CVPR 2026\] E-comIQ-ZH: A Human-Aligned Dataset and Benchmark for Fine-Grained Evaluation of E-commerce Posters with Chain-of-Thought](../../CVPR2026/llm_reasoning/e-comiq-zh_a_human-aligned_dataset_and_benchmark_for_fine-grained_evaluation_of_.md)
- [\[CVPR 2025\] Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation](enhancing_video-llm_reasoning_via_agent-of-thoughts_distillation.md)
- [\[NeurIPS 2025\] VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning](../../NeurIPS2025/llm_reasoning/videorft_incentivizing_video_reasoning_capability_in_mllms_via_reinforced_fine-t.md)

</div>

<!-- RELATED:END -->
