---
title: >-
  [论文解读] Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity
description: >-
  [CVPR 2025][视频理解][视频异常理解] 本文提出 Holmes-VAU，构建了包含 70k+ 多粒度标注的视频异常理解基准 HIVAU-70k，并设计异常聚焦时序采样器（ATS）让多模态 VLM 集中关注异常密集区域，在长视频异常检测和推理任务上大幅超越现有方法。 1. 领域现状：视频异常理解（VAU）是视频监控…
tags:
  - "CVPR 2025"
  - "视频理解"
  - "视频异常理解"
  - "多粒度标注"
  - "异常聚焦采样"
  - "多模态大语言模型"
  - "层次化指令数据"
---

# Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity

**会议**: CVPR 2025  
**arXiv**: [2412.06171](https://arxiv.org/abs/2412.06171)  
**代码**: [https://github.com/pipixin321/HolmesVAU](https://github.com/pipixin321/HolmesVAU)  
**领域**: 图像分割  
**关键词**: 视频异常理解, 多粒度标注, 异常聚焦采样, 多模态大语言模型, 层次化指令数据

## 一句话总结

本文提出 Holmes-VAU，构建了包含 70k+ 多粒度标注的视频异常理解基准 HIVAU-70k，并设计异常聚焦时序采样器（ATS）让多模态 VLM 集中关注异常密集区域，在长视频异常检测和推理任务上大幅超越现有方法。

## 研究背景与动机

1. **领域现状**：视频异常理解（VAU）是视频监控、暴力内容分析、自动驾驶等应用的核心任务。传统方法主要做帧级异常评分，将异常检测当作一个闭集预测问题。近期多模态方法开始结合视觉和文本信息，利用 VLM 进行异常相关的指令微调和文本生成。

2. **现有痛点**：现有 VAU 数据集通常只在单一时间粒度上提供标注——要么是短片段级别（clip-level），要么是视频级别（video-level）。这导致模型只能理解瞬时异常（如爆炸、打斗）或需要长期上下文的复杂事件（如盗窃、纵火），无法兼顾。此外，现有方法在处理长视频时通常采用均匀采样，容易遗漏关键异常帧或引入过多冗余计算。

3. **核心矛盾**：缺乏层次化的多粒度异常标注数据，使得模型无法同时在短期感知和长期推理层面理解异常；均匀采样策略将相同的注意力分配给异常帧和正常帧，对长视频不友好。

4. **本文目标** (1) 构建多粒度层次化异常理解数据集；(2) 设计高效的长视频异常采样策略。

5. **切入角度**：作者观察到异常帧通常包含更多信息且变化更大，因此应自适应地在异常密集区域采样更多帧。同时，通过半自动标注引擎将 LLM 与人工分割结合，高效生成多级标注。

6. **核心 idea**：用半自动引擎构建 clip/event/video 三级异常指令数据，结合异常聚焦时序采样器让 VLM 高效处理长视频异常。

## 方法详解

### 整体框架

Holmes-VAU 的整体流程为：输入一个长视频，首先通过冻结的视觉编码器（InternVL2 的 ViT）提取每帧的视觉 token；然后通过异常聚焦时序采样器（ATS）自适应选择 N 个关键帧；选中帧的视觉 token 经投影器映射到语言特征空间，与文本提示拼接后输入预训练大语言模型，最终生成异常描述和分析文本。训练分两步：第一步用帧级标签训练异常评分器，第二步用 HIVAU-70k 的全部指令数据通过 LoRA 微调 VLM。

### 关键设计

1. **HIVAU-70k 半自动标注引擎**:

    - 功能：高效构建包含 70,000+ 多粒度标注的视频异常理解基准
    - 核心思路：分三步完成。(1) 层次化视频解耦——人工标定异常事件时间边界，将事件进一步切分为随机长度的 clip，共得到 5443 个视频、11076 个事件、55806 个 clip。(2) 层次化自由文本标注——用 LLaVA-Next-Video 对每个 clip 生成详细描述，再用 LLM 将 clip 描述汇总为事件级摘要（包括判断、描述、分析三部分），最后汇总为视频级摘要。(3) 层次化指令构建——将自由文本与预设的异常相关问题模板配对，形成 QA 形式的指令数据。整个标注过程仅需 5 名标注员约 20 小时完成视频分割。
    - 设计动机：纯人工标注成本过高且不可扩展，而纯自动标注质量不可控。通过将"分割"交给人工、将"描述和推理"交给 LLM 再人工审核的流程，兼顾了质量和效率。

2. **异常聚焦时序采样器（ATS）**:

    - 功能：从长视频的 T 帧中自适应选择 N 个关键帧，使 VLM 集中在异常密集区域
    - 核心思路：ATS 由两个组件构成。(a) 异常评分器 $\phi_s$：基于 UR-DMU 的轻量 VAD 网络，对每帧的 CLS token 预测异常分数 $s_i$。(b) 密度感知采样器：将异常分数视为概率质量函数，计算累积分布函数 $S_{cumsum}(t) = \sum_{i=1}^{t}(s_i + \tau)$，然后沿累积轴均匀取 N 个点，映射回时间轴得到采样帧索引。$\tau=0.1$ 控制采样均匀度——$\tau$ 越大采样越接近均匀，$\tau$ 越小越聚焦异常区域。
    - 设计动机：均匀采样会遗漏关键异常帧，Top-K 采样会丢失上下文信息（只关注局部异常帧），而 ATS 通过概率密度的方式在异常密集区域分配更多采样点，同时保留正常区域的时间上下文，平衡了覆盖性和聚焦性。

3. **指令微调与 LoRA 适配**:

    - 功能：在保持 VLM 原有通用能力的前提下注入异常理解知识
    - 核心思路：采用 InternVL2-2B 作为基座模型，冻结视觉编码器和投影器参数。用 LoRA（$r=64, \alpha=128$）微调语言模型，batch size 512 训练 1 个 epoch，使用 AdamW + cosine decay 优化。$r$ 的选择通过消融实验确定——$r$ 过大会损害通用视频理解能力。
    - 设计动机：全量微调会破坏 LLM 原有能力且计算成本高，LoRA 是当前最成熟的高效微调方案，$r=64$ 在 VAU 专项能力和通用能力之间取得最佳平衡。

### 损失函数 / 训练策略

训练分两阶段。第一阶段用 HIVAU-70k 的帧级标签训练异常评分器，损失函数为标准二分类交叉熵 $\mathcal{L}_{AS} = -\sum_{i=1}^{T}(s_i \log(\hat{y}_i) + (1-s_i)\log(1-\hat{y}_i))$。第二阶段固定异常评分器，用全部指令数据通过交叉熵损失微调 VLM 的 LoRA 参数。注意在评估 UCF-Crime 和 XD-Violence 的异常检测性能时，只用对应训练集训练以确保公平。

## 实验关键数据

### 主实验

**异常检测性能对比（表1）**：

| 方法 | 类型 | XD-Violence AP(%) | UCF-Crime AUC(%) |
|------|------|-------------------|-------------------|
| UR-DMU | 弱监督 | 81.66 | 86.97 |
| VadCLIP | 弱监督 | 84.51 | 88.02 |
| LAVAD | 可解释多模态 | 62.01 | 80.28 |
| **Holmes-VAU** | **可解释多模态** | **87.68** | **88.96** |

**异常推理性能对比（表2），Video-level**：

| 方法 | 参数量 | BLEU↑ | CIDEr↑ | METEOR↑ | ROUGE↑ |
|------|--------|-------|--------|---------|--------|
| InternVL2 | 8B | 0.145 | 0.035 | 0.101 | 0.122 |
| QwenVL2 | 7B | 0.155 | 0.044 | 0.112 | 0.137 |
| **Holmes-VAU** | **2B** | **0.566** | **1.437** | **0.165** | **0.355** |

### 消融实验

**层次化指令数据消融（表3）**：

| 训练数据 | Clip BLEU | Event CIDEr | Video CIDEr |
|----------|-----------|-------------|-------------|
| C only | 0.984 | 0.120 | 0.106 |
| E only | 0.508 | 1.183 | 0.872 |
| C+E | 0.889 | 1.285 | 0.889 |
| **C+E+V** | **0.913** | **1.519** | **1.437** |

**采样策略消融（表4，N=16）**：

| 采样方法 | Video BLEU↑ | Video CIDEr↑ |
|----------|-------------|-------------|
| Top-K | 0.476 | 1.302 |
| Uniform | 0.511 | 1.345 |
| **ATS** | **0.566** | **1.437** |

### 关键发现

- 三级层次化数据各有贡献：clip 级提升视觉感知，event 级提升事件判断，video 级提升长程推理分析。三者联合使用效果最佳。
- ATS 在所有帧数设置（8/16/32）下均优于 Uniform 和 Top-K 采样，且推理延迟可接受。
- LoRA 的 $r$ 值存在 sweet spot：$r=64$ 时 VAU 能力和通用能力最优平衡，$r$ 过大会显著损害 Video-MME 上的通用性能。
- 仅用 2B 模型即大幅超越 7-8B 通用 VLM，说明领域特定指令数据的重要性远超模型规模。

## 亮点与洞察

- **密度感知采样思想**：将异常分数当作概率密度函数，通过累积分布函数做非均匀采样，是一种优雅的自适应采样策略。这个思想可以迁移到任何需要"重要性采样"的序列处理场景，如视频摘要、关键帧提取等。
- **半自动标注引擎的分层设计**：将标注任务分解为"人工做粗粒度分割 + LLM 做细粒度文本标注 + 人工审核"，是一种通用的大规模数据集构建方法论，可迁移到其他视频理解任务。
- **小模型 + 好数据 > 大模型**：2B 的 Holmes-VAU 在异常理解上大幅超越 7-8B 的通用 VLM，证明了领域指令数据的价值。

## 局限与展望

- 数据集基于 UCF-Crime 和 XD-Violence，场景以监控视频为主，缺乏自动驾驶、医疗等更广泛的异常场景。
- 异常评分器基于 UR-DMU 架构，需要帧级标注训练，在完全无标注场景下无法直接使用。
- ATS 的 $\tau$ 需要手动设置，未来可以探索自适应调节。
- 仅使用 InternVL2-2B 作为基座，更大模型（如 7B/13B）可能获得更强的推理能力。
- 对于极长视频（数小时），采样帧数 N=16 可能仍然不够，需要更高效的处理方式。

## 相关工作与启发

- **vs LAVAD**: LAVAD 是 training-free 方法，直接用 LLM 做异常评分和解释，不需要领域数据微调。Holmes-VAU 通过 HIVAU-70k 指令微调大幅提升了异常理解的准确性，AP 从 62.01 提升到 87.68。
- **vs UCA**: UCA 提供了视频级的异常因果分析数据集，但缺乏多粒度标注。HIVAU-70k 的三级结构更全面。
- **vs Video-ChatGPT/Video-LLaMA**: 这些通用视频 VLM 缺陷异常领域知识，在长程推理上表现有限。Holmes-VAU 证明了领域指令微调的必要性。

## 评分

- 新颖性: ⭐⭐⭐⭐ 多粒度异常理解框架和 ATS 采样策略有新意，但个别组件（异常评分器、LoRA 微调）是已有技术的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 在检测和推理两个维度上与大量方法对比，消融实验覆盖数据粒度、采样策略和微调参数
- 写作质量: ⭐⭐⭐⭐ 结构清晰，motivation 推导自然，但部分技术细节需参考附录
- 价值: ⭐⭐⭐⭐ HIVAU-70k 数据集和 ATS 采样器对视频异常理解领域有较好实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity](../../NeurIPS2025/video_understanding/vadtree_explainable_training-free_video_anomaly_detection_via_hierarchical_granu.md)
- [\[CVPR 2025\] DrVideo: Document Retrieval Based Long Video Understanding](drvideo_document_retrieval_based_long_video_understanding.md)
- [\[CVPR 2025\] ETAP: Event-based Tracking of Any Point](etap_event-based_tracking_of_any_point.md)
- [\[CVPR 2026\] Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding](../../CVPR2026/video_understanding/question-guided_visual_compression_with_memory_feedback_for_long-term_video_unde.md)
- [\[CVPR 2025\] MLVU: Benchmarking Multi-task Long Video Understanding](mlvu_benchmarking_multi-task_long_video_understanding.md)

</div>

<!-- RELATED:END -->
