---
title: >-
  [论文解读] R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios
description: >-
  [AAAI 2026][视频理解][音视频推理] 提出首个面向复杂音视频场景的细粒度时空推理数据集 R-AVST（5K+未裁剪视频、27K物体、100类音视频事件），定义三个核心推理任务，并基于 GRPO 训练 AVST-Zero 模型，通过多维奖励函数直接优化音视频时空推理能力。 领域现状：多模态大语言模型（MLLM）在视…
tags:
  - "AAAI 2026"
  - "视频理解"
  - "音视频推理"
  - "时空定位"
  - "强化学习"
  - "GRPO"
  - "Video-LLM"
---

# R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios

**会议**: AAAI 2026  
**arXiv**: [2511.16901](https://arxiv.org/abs/2511.16901)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 音视频推理, 时空定位, 强化学习, GRPO, Video-LLM

## 一句话总结

提出首个面向复杂音视频场景的细粒度时空推理数据集 R-AVST（5K+未裁剪视频、27K物体、100类音视频事件），定义三个核心推理任务，并基于 GRPO 训练 AVST-Zero 模型，通过多维奖励函数直接优化音视频时空推理能力。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLM）在视频理解任务中取得了快速进展，如 InternVL-2.5、Qwen2.5-VL、VideoLLaMA3 等模型已展现出强大的视频理解能力。但当前研究主要集中于**简单视频场景**，未能反映真实世界中音视频事件的复杂性和多样性。

**现有痛点**：

**数据集层面**：现有音视频数据集（如 AVE、UnAV-100、PU-VALOR）主要关注**时间维度**理解，忽视了可发声物体的**空间属性**；而时空定位数据集（如 VidSTG、HC-STVG、V-STaR）提供了时空标注，但未充分捕捉真实音视频动态，且物体类型有限。

**模型层面**：LLaVA-ST、GroundingGPT 等模型已逐步扩展时空建模能力，但依赖大规模高质量标注数据，缺乏足够的探索能力。VideoChat-R1、Video-R1 等 RL 模型虽开始使用 GRPO 增强推理，但其奖励设计对音视频时空推理的支持有限，且缺乏针对复杂音视频场景的专用任务。

**核心矛盾**：缺少一个同时具备**细粒度时空标注**和**丰富音视频事件覆盖**的数据集来推动视频理解模型在真实复杂场景中的时空推理能力。

**切入角度**：从音视频场景出发，构建首个包含细粒度时空标注的音视频推理数据集，并利用基于规则奖励的 GRPO 训练范式，绕过中间监督信号直接优化行为策略。

## 方法详解

### 整体框架

R-AVST 项目包含两部分：（1）R-AVST 数据集构建；（2）AVST-Zero 模型训练。数据集构建遵循"收集过滤→字幕分析→边界框标注→QA生成→质量控制"五步流程。模型基于 GRPO 在 R-AVST 上训练，使用多维奖励函数引导策略更新。

### 关键设计

#### 1. **R-AVST 数据集构建**

**数据收集与过滤**：从 UnAV-100 收集未裁剪 YouTube 视频，经三步过滤：按时长分组（短/中/长）、限制每个视频最多3个音视频事件（平衡事件数分布）、剔除事件占比低于 0.08 的视频。最终得到 5,237 个高质量视频。

**字幕分析**：使用 GPT-4o-mini 作为分析器 LLM，从事件字幕中提取名词物体并标注其**听觉和视觉属性**（"可见且可听"或"仅可见"）。通过精心设计的提示词强调"可听性"在物体-声音关系中的定义，提高分析准确率。共标注 27,253 个物体，其中 50.88% 为"可见且可听"。

**空间标注**：利用 Grounded-SAM2 进行自动化逐帧物体标注，以降低大规模视频标注成本。设置 BOX_THRESHOLD=0.4、TEXT_THRESHOLD=0.3。

**QA 自动生成**：定义三类问题对应三个任务——
- 时间推理："When is the moment [objects] make sound and are visible?"
- 空间推理："What objects make sound between [start] and [end], and where are they?"
- 时空推理："When is the moment [objects] make sound and are visible, and where are they?"

训练集包含 2,663 时间+2,666 空间+1,204 时空 QA，测试集 663+664+306 = 1,633 QA。

#### 2. **三个核心推理任务**

- **音视频时间推理（AVTR）**：给定可见且可听的物体，推断该物体发声并可见的时间段。
- **音视频空间推理（AVSR）**：给定时间区间，定位场景中发声物体和静默物体的空间位置。
- **音视频时空推理（AVSTR）**：同时推断物体的时间段和空间位置，最接近人类感知机制。

#### 3. **AVST-Zero 模型与多维奖励**

**训练方式**：基于 Qwen2.5-VL-7B（或 Qwen2.5-Omni-7B）使用 **GRPO（Group Relative Policy Optimization）** 进行全 RL 微调，无需 SFT 阶段中间监督。

**四维奖励设计**：

- **格式奖励** $R_{\text{format}}$：检查是否包含正确的标签对（`<answer>`, `<object>`, `<when>`, `<where>`）。
- **物体奖励** $R_{\text{object}}$：使用 Word2Vec 计算预测与真实物体名称的语义相似度，超过阈值 $\tau$ 则奖励为 1。
$$R_{\text{object}} = \begin{cases} 1, & \text{if } \text{sim}(V_{\text{pred}}, V_{\text{gt}}) \geq \tau \\ 0, & \text{otherwise} \end{cases}$$
- **时间奖励** $R_{\text{temporal}}$：预测时间段与真实时间段的 IoU。
$$R_{\text{temporal}} = \frac{|I_{\text{pred}} \cap I_{\text{gt}}|}{|I_{\text{pred}} \cup I_{\text{gt}}|}$$
- **空间奖励** $R_{\text{spatial}}$：重叠时间区间内预测边界框与真实边界框的平均 2D IoU。
$$R_{\text{spatial}} = \frac{1}{N} \sum_{t=T_{\text{start}}}^{T_{\text{end}}} \text{IoU}(t)$$

**最终奖励**：$R = \lambda_f R_{\text{format}} + \lambda_t R_{\text{temporal}} + \lambda_o R_{\text{object}} + \lambda_s R_{\text{spatial}}$，其中 $\lambda_f = 1$，其他参数根据任务类型而定。

### 损失函数 / 训练策略

使用 GRPO 标准目标函数，对每个问题采样 G=6 组输出，计算组内相对优势 $A_i$，通过 clipped surrogate objective + KL 正则化更新策略。训练使用 4 块 NVIDIA RTX A6000，单 epoch，batch size = 1/GPU，max_prompt_length=512，max_completion_length=1024。

## 实验关键数据

### 主实验

**时间推理任务（AVTR）**：

| 方法 | m_tIoU | R1@0.3 | R1@0.5 | R1@0.7 |
|------|--------|--------|--------|--------|
| Qwen2.5-VL(7B) | 36.05 | 46.40 | 34.38 | 16.22 |
| Video-LLaMA3(7B) | 37.17 | 50.30 | 35.29 | 22.67 |
| VideoChat-R1(7B) | 43.17 | 60.81 | 46.70 | 25.68 |
| **AVST-Zero(7B)** | **47.96** | **71.13** | **51.43** | 23.91 |

**空间推理任务（AVSR）**：

| 方法 | Obj Acc | m_vIoU | AP@0.3 | AP@0.5 |
|------|---------|--------|--------|--------|
| Qwen2.5-VL(7B) | 1.91 | 2.31 | 0.90 | 0.15 |
| VideoChat-R1(7B) | 15.54 | 1.99 | 3.11 | 0.36 |
| **AVST-Zero(7B)** | 14.34 | 2.27 | **3.12** | **0.87** |
| **AVST-Zero-Omni(7B)** | **19.48** | **3.87** | **4.47** | **2.17** |

**时空推理任务（AVSTR）**：

| 方法 | m_tIoU | m_vIoU | AP@0.3 | AP@0.5 |
|------|--------|--------|--------|--------|
| VideoChat-R1(7B) | 41.81 | 2.15 | 3.21 | 0.60 |
| **AVST-Zero(7B)** | **46.04** | 8.59 | 10.38 | 3.83 |
| **AVST-Zero-Omni(7B)** | 35.97 | **17.74** | **22.90** | **12.26** |

### 消融实验

| 配置 | AVTR m_tIoU | AVSR Obj Acc | AVSR m_vIoU | AVSTR m_tIoU | AVSTR m_vIoU |
|------|------------|-------------|-------------|-------------|-------------|
| SFT | 42.84 | 9.52 | 3.42 | 38.40 | 4.26 |
| AVST-Zero | **48.17** | 20.72 | **4.62** | **46.93** | **10.87** |
| w/o 时间奖励 | 46.67 | **23.95** | 4.54 | 45.82 | 8.31 |
| w/o 空间奖励 | 47.03 | 23.17 | 3.28 | 44.29 | 9.23 |

### 关键发现

1. **RL 优于 SFT**：直接使用 GRPO 在所有三个任务上比 SFT 带来更显著的提升，说明 RL 更适合细粒度时空推理任务。
2. **AVST-Zero-Omni 在空间维度表现优于 AVST-Zero**（得益于基座模型的音视频联合感知），但在时间维度较弱。
3. **时空维度奖励存在交叉效应**：移除时间奖励会影响空间指标，反之亦然，表明时空推理具有相互依赖性。
4. **跨数据集泛化**：在 AVE 和 AVSBench-V1 上也取得了有竞争力的表现（空间维度 m_vIoU 6.84% 最优）。

## 亮点与洞察

1. **首个音视频时空推理数据集**：R-AVST 填补了现有数据集在音视频场景下缺乏细粒度时空标注的空白，覆盖 100 类事件和 5.2 个平均物体/视频。
2. **全 RL 训练无需 SFT**：利用任务的规则化特性，设计多维奖励直接用 GRPO 训练，避免了对大规模高质量标注数据的依赖。
3. **物体属性分析的创新思路**：通过 LLM 分析字幕中物体的"可听/可见"属性，为音视频场景理解提供了新的标注范式。
4. **自动化标注流水线**：LLM 分析 + Grounded-SAM2 自动标注 + 程序化 QA 生成，大幅降低标注成本。

## 局限与展望

1. 空间推理的绝对性能仍较低（AP@0.5 在空间任务中最高仅 2.17%），说明音视频场景下的精细空间定位仍极具挑战。
2. 数据集基于 UnAV-100 构建，视频来源相对单一，可扩展到更多领域（如自动驾驶、工业检测）。
3. 当前仅使用 7B 规模模型，更大模型可能带来显著提升。
4. 未探索音频特征的直接编码（Omni 版本通过多模态基座间接利用音频），可考虑更显式的音频建模。

## 相关工作与启发

- DeepSeek-R1/GRPO 的规则化 RL 训练范式被成功迁移到视频时空推理任务，启示在于**将任务目标分解为可计算的奖励维度**是 RL 在视觉任务中成功的关键。
- Grounded-SAM2 的自动标注能力为大规模视频数据集构建提供了可行路径。
- 与 VideoChat-R1 相比，AVST-Zero 在时间和时空推理上的提升说明**任务特定的多维奖励设计**比通用奖励更有效。

## 评分

- 新颖性: ⭐⭐⭐⭐ （首个音视频时空推理数据集+全 RL 训练，有较强原创性）
- 实验充分度: ⭐⭐⭐⭐ （多任务对比、消融、跨数据集验证、定性分析齐全）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，数据集构建流程详尽）
- 价值: ⭐⭐⭐⭐ （为音视频时空推理开辟新方向，数据集有长期价值）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] OmniGround: A Comprehensive Spatio-Temporal Grounding Benchmark for Real-World Complex Scenarios](../../CVPR2026/video_understanding/omniground_a_comprehensive_spatio-temporal_grounding_benchmark_for_real-world_co.md)
- [\[ICML 2026\] AVTrack: Audio-Visual Tracking in Human-centric Complex Scenes](../../ICML2026/video_understanding/avtrack_audio-visual_tracking_in_human-centric_complex_scenes.md)
- [\[AAAI 2026\] FineTec: Fine-Grained Action Recognition Under Temporal Corruption via Skeleton Decomposition and Sequence Completion](finetec_fine-grained_action_recognition_under_temporal_corruption_via_skeleton_d.md)
- [\[AAAI 2026\] FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding](finevau_a_novel_human-aligned_benchmark_for_fine-grained_video_anomaly_understan.md)
- [\[CVPR 2026\] Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos](../../CVPR2026/video_understanding/mistake_attribution_fine-grained_mistake_understanding_in_egocentric_videos.md)

</div>

<!-- RELATED:END -->
