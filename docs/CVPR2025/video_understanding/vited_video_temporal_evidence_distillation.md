---
title: >-
  [论文解读] ViTED: Video Temporal Evidence Distillation
description: >-
  [CVPR 2025][视频理解][视频问答] ViTED提出一个自动生成时间定位证据链的框架，将证据收集、时间基准定位和问答推理统一到单一视频语言模型中，通过证据蒸馏提升复杂视频问答能力。 视频问答（VideoQA）是视频理解的核心任务。现有视频大模型的两个关键限制： - 均匀采样导致关键帧遗漏：模型以固定间隔采样固定数量…
tags:
  - "CVPR 2025"
  - "视频理解"
  - "视频问答"
  - "证据链推理"
  - "时间定位"
  - "链式思维"
  - "知识蒸馏"
---

# ViTED: Video Temporal Evidence Distillation

**会议**: CVPR 2025  
**arXiv**: [2503.12855](https://arxiv.org/abs/2503.12855)  
**代码**: 无  
**领域**: Video Understanding  
**关键词**: 视频问答, 证据链推理, 时间定位, 链式思维, 知识蒸馏

## 一句话总结

ViTED提出一个自动生成时间定位证据链的框架，将证据收集、时间基准定位和问答推理统一到单一视频语言模型中，通过证据蒸馏提升复杂视频问答能力。

## 研究背景与动机

视频问答（VideoQA）是视频理解的核心任务。现有视频大模型的两个关键限制：

- **均匀采样导致关键帧遗漏**：模型以固定间隔采样固定数量的帧，可能遗漏视频中不均匀分布的关键证据（如某个短暂的挥手动作）
- **缺乏时间定位和多步推理能力**：模型无法将证据与视频中的具体时间段关联，无法进行"先找证据→再推理→再回答"的多步推理

例如回答"婴儿为什么把手放进嘴里？"需要:（1）定位妈妈用勺子喂食的片段；（2）观察婴儿不适的表情；（3）推理出婴儿试图把食物弄出来。现有模型无法完成这种链式推理。

现有的时间基准定位模型虽然能定位特定描述的时间段，但需要**提前知道要定位什么**——它们无法从问题出发自主识别和定位相关证据。

## 方法详解

### 整体框架

ViTED包含三个阶段：(1) **证据池生成**：将视频分为多层级多粒度片段，用VLM为每个片段生成与问题相关的描述；(2) **证据链搜索**：用LLM通过beam search在证据池中搜索最能支持正确答案的证据链序列；(3) **证据蒸馏训练**：将搜索到的证据链加入训练数据，训练模型同时生成证据链和答案。

### 关键设计

**1. 多层级证据池生成（Hierarchical Evidence Pool）**

- **功能**：从视频中全面提取不同时间尺度的潜在证据，覆盖全局上下文到细粒度局部动作
- **核心思路**：将视频在 $N=5$ 个层级进行非均匀分割，$(L,S) \in \{(1/16, 1/16), (1/8, 1/16), (1/4, 1/8), (1/2, 1/4), (1, 1)\}$。对每个片段用LLaMA-3.2-Vision-11B生成与问题相关的描述，形成证据池 $E = \{(t_s, t_e, \epsilon)_i\}$
- **设计动机**：证据在视频中不均匀分布（一个全局活动 vs 一个短暂动作），需要多粒度覆盖。非均匀分割比均匀采样更不容易遗漏关键信息

**2. 证据链搜索与精炼（Evidence Chain Search）**

- **功能**：从大量噪声证据中找到最能推导出正确答案的证据链序列
- **核心思路**：先用LLM缩小证据池 $E \rightarrow E^*$（保留top-K），再进行beam search：初始化宽度 $W=K/2$ 的beam，迭代添加新证据到链中，计算 $P(A|Q, C_i \oplus ev_j)$ 并保留top-W链。收敛后对最优链进行LLM摘要，使其具有时序因果连贯性。最后过滤保留能正确推导答案的链
- **设计动机**：单个证据只提供部分信息，需要组合多条证据形成推理链。Beam search在效率和质量间取得平衡

**3. 课程式证据蒸馏训练**

- **功能**：将证据链的生成和推理能力蒸馏到单一VLM中
- **核心思路**：两阶段训练——Stage-1标准指令微调（Q→A），Stage-2证据蒸馏（Q→Evidence Chain + A）。训练时使用next token prediction交叉熵损失。推理时模型先生成带时间戳的证据链，再基于证据回答问题
- **设计动机**：课程学习避免一开始就学习复杂任务。先学基础问答能力，再学证据推理

### 损失函数

标准的next token prediction交叉熵损失，分别在Stage-1（答案token）和Stage-2（证据链+答案token）上优化。

## 实验关键数据

### 主实验：VideoQA基准对比（7B模型）

| 方法 | CinePile | PerceptionTest | NExT-QA | STAR | NExT-GQA |
|------|----------|---------------|---------|------|----------|
| LLaMA-3.2V (11B) | 39.55 | 52.65 | 67.58 | 45.62 | 11.64 |
| LLaVA-OneVision | 46.42 | - | - | - | - |
| SeViLA (4B) | - | - | 73.8 | 64.9 | 16.6 |
| **ViTED** | **48.2** | **64.8** | **80.1** | **66.2** | **22.4** |

### 消融实验：证据蒸馏的影响

| 训练方式 | NExT-QA | NExT-GQA |
|---------|---------|----------|
| 无CoT | 75.3 | 14.2 |
| + "step-by-step" prompt | 76.1 | 15.1 |
| + 证据蒸馏 (ViTED) | **80.1** | **22.4** |

### 关键发现

- ViTED在NExT-GQA（时间定位问答）上以零样本方式超越GPT-4驱动的Agent方法，证明蒸馏后的模型内化了时间定位能力
- 人类评估显示ViTED生成的证据链质量远高于基线VLM的推理解释
- NExT-QA中54%的问题需要定位和推理一个或多个时间窗口，证据链方法在这类问题上优势最大
- 证据链平均包含2-3个hop，需要在视频的不同时间位置收集不同粒度的线索
- 简单的"step-by-step"提示效果有限（+0.8%），真正的证据蒸馏带来显著提升（+4.8%）

## 亮点与洞察

1. **将CoT从文本域扩展到视频域**：不是简单的文本推理链，而是带时间戳的视频证据链，每条证据关联到视频的具体时间段
2. **自动化数据生成流水线**：无需人工标注即可从现有VideoQA数据集生成高质量证据链训练数据
3. **单模型替代Agent系统**：通过蒸馏将多模块Agent的能力（证据收集+定位+推理）压缩到单次前向传播中

## 局限与展望

- 证据池生成依赖外部VLM（LLaMA-3.2-Vision），其质量直接影响下游效果
- beam search的计算开销较大，不适合实时应用
- 当前仅处理已有QA对的数据增强，未探索开放式问题
- 未来可探索在线证据搜索（推理时动态搜索而非依赖训练时蒸馏的知识）

## 相关工作与启发

- **与SeViLA的关系**：SeViLA用帧选择器定位关键帧，ViTED更进一步定位时间窗口并生成文本证据
- **与Agent方法的关系**：Agent系统（如VIP等）通过多次API调用进行推理，ViTED蒸馏为单模型单次推理
- **启发**：视频理解从"感知"向"推理"的升级需要显式的证据支持，而非端到端的黑盒预测

## 评分

⭐⭐⭐⭐

首次将带时间定位的证据链推理引入视频问答，自动化数据生成流水线设计精巧。在多个benchmark上SOTA，特别是NExT-GQA上超越GPT-4 Agent。技术路线完整且可复现。主要局限在于训练和数据生成的计算开销。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding](beyond_single-sample_reliable_multi-sample_distillation_for_video_understanding.md)
- [\[ICML 2026\] Foresee-to-Ground: From Predictive Temporal Perception to Evidence-Driven Reasoning](../../ICML2026/video_understanding/foresee-to-ground_from_predictive_temporal_perception_to_evidence-driven_reasoni.md)
- [\[ACL 2025\] From Teacher to Student: Tracking Memorization Through Model Distillation](../../ACL2025/video_understanding/from_teacher_to_student_tracking_memorization_through_model_distillation.md)
- [\[CVPR 2025\] On the Consistency of Video Large Language Models in Temporal Comprehension](on_the_consistency_of_video_large_language_models_in_temporal_comprehension.md)
- [\[CVPR 2025\] STOP: Integrated Spatial-Temporal Dynamic Prompting for Video Understanding](stop_integrated_spatial-temporal_dynamic_prompting_for_video_understanding.md)

</div>

<!-- RELATED:END -->
