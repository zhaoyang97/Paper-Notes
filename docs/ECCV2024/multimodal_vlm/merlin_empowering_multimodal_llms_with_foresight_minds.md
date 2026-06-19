---
title: >-
  [论文解读] Merlin: Empowering Multimodal LLMs with Foresight Minds
description: >-
  [ECCV 2024][多模态VLM][多模态大语言模型] 提出 Foresight Pre-Training (FPT) 和 Foresight Instruction-Tuning (FIT) 两阶段训练范式，通过轨迹建模赋予多模态大语言模型"前瞻性思维"能力，使模型能够基于当前观察预测未来事件并进行推理。
tags:
  - "ECCV 2024"
  - "多模态VLM"
  - "多模态大语言模型"
  - "未来推理"
  - "轨迹预测"
  - "前瞻性思维"
  - "视觉理解"
---

# Merlin: Empowering Multimodal LLMs with Foresight Minds

**会议**: ECCV 2024  
**arXiv**: [2312.00589](https://arxiv.org/abs/2312.00589)  
**代码**: [GitHub](https://ahnsun.github.io/merlin)  
**领域**: 多模态VLM  
**关键词**: 多模态大语言模型, 未来推理, 轨迹预测, 前瞻性思维, 视觉理解

## 一句话总结

提出 Foresight Pre-Training (FPT) 和 Foresight Instruction-Tuning (FIT) 两阶段训练范式，通过轨迹建模赋予多模态大语言模型"前瞻性思维"能力，使模型能够基于当前观察预测未来事件并进行推理。

## 研究背景与动机

人类能够基于当前观察预测未来事件，神经科学中称之为"预测性加工"（predictive processing）。然而，现有的多模态大语言模型（MLLMs）如 GPT-4V 和 Bard 虽然在图像理解和逻辑推理方面表现出色，但缺乏基于当前图像观察预测未来事件的能力。即使提供多帧图像序列，现有 MLLM 仍难以分析和推断目标的具体行为（如预测物体运动或交互）。

作者将人类预见未来的过程分解为两个阶段：

**观察动态线索**：观察目标的运动和状态变化

**分析行为模式并推理**：基于观察分析行为模式，推断可能发生的事件

现有 MLLM 的 LLM 部分已具备良好的逻辑推理能力（第二阶段），关键挑战在于第一阶段——如何让 MLLM 从多图像观察中正确获取时空动态信息。直接建模下一帧（如重建下一帧图像）存在视觉信息冗余问题，难以直接提取动态线索。为此，作者提出使用**轨迹**（trajectory）作为学习目标——轨迹作为高度结构化的表示，能够连接过去和未来的时间上下文。

## 方法详解

### 整体框架

Merlin 由三个核心组件组成：

1. **图像编码器**：采用预训练的 CLIP ViT-L/14，输入图像分辨率为 448×448，生成 1024 个编码 token
2. **大语言模型解码器**：采用开源的 Vicuna-7B v1.5
3. **模态对齐投影器**：使用 3×3 的 2D 卷积层（stride=2, padding=1）实现维度投影和 token 聚合

选择 2D 卷积而非 1D 线性层或交叉注意力层作为连接器的原因：
- 2D 卷积能在空间尺度上聚合局部视觉 token，高效实现空间到通道信息的转换
- 相比交叉注意力，2D 卷积具有更好的收敛特性，为两阶段训练打下坚实基础
- 能有效压缩 token 数量，支持高分辨率和多帧输入

### 关键设计

#### Foresight Pre-Training (FPT) — 前瞻性预训练

FPT 的核心思想是因果建模与多帧图像交织的时间轨迹，使 MLLM 具备感知跨帧动态线索的能力。

**建模方式**：给定视频片段中的多帧图像，模型以第一帧中目标的观察（描述或位置）作为查询，需要预测该目标在整个视频中的完整轨迹：

$$P(Y|X) \sim P(Y|\{X_1, X_2, ...\}, O_{first})$$

其中 $X_i$ 为第 $i$ 帧，$O_{first}$ 为第一帧观察，$Y$ 为目标的轨迹。

**数据构建**遵循三个原则：

1. **精确定义任务提示和答案格式**：使用任务提示告知 MLLM 具体任务（检测或跟踪），并在问题中指定答案格式，使不同类型任务能灵活组织且不损害通用语言能力
2. **清晰指示多模态信息**：为每组图像 token 添加特殊帧指示器（如 `frame1:<image>`、`frame2:<image>`），帮助 MLLM 更好地关注对应图像
3. **帧与观察的交织组织**：对同一身份目标，将其出现的帧与位置观察交织排列，并用 ID token（`<Idi>` 和 `</Idi>`）包裹构建轨迹

**观察类型**分为三种：位置描述、外观描述和动作描述，随机选择其一作为查询。

**训练策略**：不同于以往先进行模态对齐再进行多任务预训练的分离做法，Merlin 将两者合并为一个阶段，**解冻所有模块**进行预训练。混合使用 10M 图像文本对数据和约 5M 来自多种数据源的问答数据进行多任务学习。

#### Foresight Instruction Tuning (FIT) — 前瞻性指令微调

FPT 赋予模型观察多帧动态线索的能力，但仅此不足以实现真正的"前瞻性思维"。FIT 在 FPT 基础上引入 **Trajectory Chain-of-Thought (T-CoT)**，将轨迹建模作为逻辑推理链中的桥梁。

**核心公式**：

$$P(Z|X,Y) \sim P(Z|\{X_1, X_2, ...\}, O_{first}, Y)$$

其中 $Z$ 为未来观察（可以是动作、事件、趋势或可能性），由多帧图像、首帧观察和轨迹共同作为条件，使 MLLM 因果预测未来。

**T-CoT 的工作方式**：当用户查询某个目标的未来时，Merlin 首先展示该目标的观察轨迹，再展示其他相关目标的轨迹，最后基于这些轨迹推理可能的未来事件。例如，在足球场景中，Merlin 先输出红衣球员轨迹，再输出白衣球员轨迹，推断白衣球员可能会铲球导致双方倒地。

**FIT 数据构建**：利用 GPT-4 基于 MultiSports、TITAN 和 STAR 三个场景数据集的轨迹和动作信息生成约 30K T-CoT 对话数据。

### 损失函数 / 训练策略

**两阶段训练配置**：

| 配置项 | 预训练 (FPT) | 指令微调 (FIT) |
|--------|-------------|---------------|
| 视觉编码器 | 解冻 | 冻结 |
| 投影器 | 解冻 | 解冻 |
| LLM | 解冻 | 解冻 |
| 学习率 | 5e-5 | 5e-5 |
| 全局 batch size | 2048 | 256 |
| 训练步数 | 7k | 3k |
| 优化器 | AdamW (β₂=0.95) | AdamW (β₂=0.95) |
| 学习率调度 | cosine decay | cosine decay |
| 精度 | bfloat16 | bfloat16 |

**数据组成**：
- FPT 阶段：10M 图像文本对 (LAION) + 5M 多任务问答数据
- FIT 阶段：665K LLaVA 指令数据 + 30K T-CoT 对话 + 40K FPT 采样数据

训练在 64 块 NVIDIA A800 GPU 上进行，预训练约 12 小时，指令微调约 3 小时。

## 实验关键数据

### 主实验

**表1：未来推理能力（MMBench 子任务）**

| 方法 | LLM | Dev Avg | OL | PPR | FR | IR | FP | Test Avg |
|------|-----|---------|---|----|----|----|----|----|
| mPLUG-Owl | 7B | 41.0 | 18.5 | 18.7 | 66.7 | 86.7 | 14.3 | 45.9 |
| Shikra | 7B | 51.5 | 32.1 | 30.7 | 63.0 | 88.9 | 42.9 | 60.0 |
| Kosmos-2 | 1.6B | 54.4 | 38.3 | 33.3 | 56.8 | 91.1 | 52.4 | 58.2 |
| LLaVA-1.5 | 7B | 59.6 | 43.2 | 52.0 | 71.6 | 93.3 | 38.1 | - |
| **Merlin** | **7B** | **64.4** | **42.0** | **54.7** | **72.8** | **97.8** | **54.8** | **66.5** |

Merlin 在 10 项指标中 8 项取得最优，综合得分显著领先。

**表2：目标跟踪评估**

| 方法 | LaSOT Success | GOT10k AO | GOT10k SR₀.₅ | GOT10k SR₀.₇₅ |
|------|--------------|-----------|-------------|--------------|
| SiamFC | 33.6 | 34.8 | 35.3 | 9.8 |
| SiamRPN++ | 49.6 | 51.8 | 61.8 | 32.5 |
| LLaVA-1.5 (跟踪训练) | 19.4 | 23.5 | 20.2 | 9.7 |
| **Merlin** | **39.8** | **51.4** | **55.9** | **42.8** |

Merlin 是首个能执行跟踪任务的 MLLM，且仅用少量跟踪数据即可达到与专家模型可比的性能。

**表3：幻觉评估 (POPE)**

| 方法 | Random Acc | Popular Acc | Adversarial Acc |
|------|-----------|------------|----------------|
| LLaVA (7B) | 72.16 | 61.37 | 58.67 |
| LLaVA-1.5 (7B) | 83.29 | 81.88 | 78.96 |
| Qwen-VL (7B) | 84.73 | 84.13 | 82.26 |
| **Merlin (7B)** | **91.58** | **89.53** | **84.10** |

Merlin 在所有场景下均大幅超越现有方法，"yes" 比率接近 50%，展现出卓越的视觉感知能力。

### 消融实验

**表4：FPT 与 FIT 策略消融**

| 预训练（ITP+FPT） | 微调（ITD+FIT） | GOT10K AO | 推理 Avg |
|:---:|:---:|:---:|:---:|
| ITP only | ITD only | - | 59.5 |
| ITP only | ITD+FIT | - | 60.7 |
| FPT only | ITD+FIT | 15.5 | 52.8 |
| ITP+FPT | ITD only | 51.4 | 61.2 |
| ITP+FPT | ITD+FIT | **51.4** | **64.4** |

**表5：模型配置消融**

| 分辨率 | 投影器 | 编码器 | Token数 | 推理 | GOT10K |
|--------|--------|--------|---------|------|--------|
| 448x | Conv2d | 解冻 | 256 | **64.4** | **51.4** |
| 336x | Conv2d | 解冻 | 256 | 59.8 | 47.3 |
| 336x | MLP | 解冻 | 576 | 58.1 | 23.5 |
| 448x | Conv2d | 冻结 | 256 | 60.8 | 28.4 |

### 关键发现

1. **FPT 和 FIT 互补**：FPT 提供动态线索感知，FIT 通过 T-CoT 激活前瞻推理能力，两者缺一不可
2. **图像文本对数据不可或缺**：预训练中缺少图像文本对会严重损害模型通用能力
3. **高分辨率有利于精确定位**：448x 相比 336x 在定位和跟踪任务上提升显著
4. **Conv2d 投影器优于 MLP**：有效压缩 token 数，支持多图像输入且不降低性能
5. **前瞻学习意外减少幻觉**：通过学习轨迹对应关系，模型获得更精确的目标关注能力，避免误识别
6. **精确任务描述至关重要**：缺少精确任务描述会导致跟踪性能从 51.4% 骤降至 28.4%

## 亮点与洞察

1. **轨迹作为结构化学习目标的洞察**：相比直接预测下一帧图像，轨迹提供了高度抽象且结构化的时空表示，有效避免了视觉冗余问题
2. **Trajectory Chain-of-Thought 的创新**：将 CoT 思想扩展到视觉轨迹领域，轨迹作为推理链条的"桥梁"，连接观察和未来预测
3. **多任务统一的对话格式设计**：通过精确的任务定义和格式规范，在一个模型中统一处理检测、跟踪、引用理解、未来推理等多种任务
4. **前瞻学习的溢出效应**：训练模型预测轨迹不仅提升了未来推理能力，还意外增强了通用视觉理解和减少幻觉，为 MLLM 训练提供新思路

## 局限与展望

1. **长视频处理限制**：当前仅支持 ≤8 帧输入，因依赖图像编码器而非视频编码器，无法处理长程视频序列
2. **评估基准不完善**：现有未来推理评估基准并不全面，基于 MMBench 子任务构建的基准仅为初步尝试
3. **视频编码效率**：需要开发更高效的长程视频 tokenizer
4. **轨迹表示的局限**：当前轨迹仅使用边界框表示，未考虑更细粒度的空间信息（如姿态、细粒度动作）
5. **对话数据规模**：T-CoT 数据仅 30K，扩大规模可能进一步提升推理质量

## 相关工作与启发

- **与 LLaVA-1.5 的关系**：Merlin 基于 LLaVA 架构，用 Conv2d 替代 MLP 投影器，并引入多帧和轨迹建模
- **与 Shikra 的关系**：Shikra 率先在 MLLM 中引入空间坐标对话能力，Merlin 将其扩展到时间维度
- **启发意义**：
  1. 轨迹可作为跨模态"语言"，连接视觉和语言模态
  2. 前瞻性学习可作为一种通用的视觉表示增强策略
  3. 多任务预训练与指令微调可合并为一个阶段

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4.5 | 首次将"前瞻性思维"系统性引入MLLM，轨迹CoT是创新贡献 |
| 技术深度 | 4.0 | 两阶段训练范式设计合理，数据构建方法详尽 |
| 实验充分性 | 4.0 | 多任务评估全面，但未来推理基准相对简单 |
| 表达清晰度 | 4.0 | 写作流畅，图示丰富，方法论述清晰 |
| **总体** | **4.0** | 在MLLM未来推理方向的先驱工作，思路新颖，验证充分 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MERLIN: Building Low-SNR Robust Multimodal LLMs for Electromagnetic Signals](../../CVPR2026/multimodal_vlm/merlin_building_low-snr_robust_multimodal_llms_for_electromagnetic_signals.md)
- [\[ECCV 2024\] Genixer: Empowering Multimodal Large Language Model as a Powerful Data Generator](genixer_empowering_multimodal_large_language_model_as_a_powerful_data_generator.md)
- [\[ECCV 2024\] AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization](addressclip_empowering_vision-language_models_for_city-wide_image_address_locali.md)
- [\[ECCV 2024\] Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation](eyes_closed_safety_on_protecting_multimodal_llms_via_image-to-text_transformatio.md)
- [\[ICML 2026\] CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception](../../ICML2026/multimodal_vlm/cvsearch_empowering_multimodal_llms_with_cognitive_visual_search_for_high-resolu.md)

</div>

<!-- RELATED:END -->
