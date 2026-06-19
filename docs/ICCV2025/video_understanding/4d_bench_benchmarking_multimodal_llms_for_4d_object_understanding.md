---
title: >-
  [论文解读] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding
description: >-
  [ICCV 2025][视频理解][4D理解] 本文提出 4D-Bench，首个评估多模态大模型 (MLLM) 在 4D 物体（动态 3D 物体）理解能力的基准，包含 4D 物体问答和 4D 物体描述两大任务，揭示了即使是 GPT-4o 在简单 4D 物体上也仅达 63% 准确率（人类基线 91%），尤其在物体计数和时序理解上表现薄弱。
tags:
  - "ICCV 2025"
  - "视频理解"
  - "4D理解"
  - "多模态大模型评测"
  - "时空推理"
  - "基准测试"
  - "视频问答"
---

# 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding

**会议**: ICCV 2025  
**代码**: [https://4dbench.github.io/](https://4dbench.github.io/)  
**领域**: 视频理解  
**关键词**: 4D理解, 多模态大模型评测, 时空推理, 基准测试, 视频问答

## 一句话总结

本文提出 4D-Bench，首个评估多模态大模型 (MLLM) 在 4D 物体（动态 3D 物体）理解能力的基准，包含 4D 物体问答和 4D 物体描述两大任务，揭示了即使是 GPT-4o 在简单 4D 物体上也仅达 63% 准确率（人类基线 91%），尤其在物体计数和时序理解上表现薄弱。

## 研究背景与动机

1. **领域现状**：多模态大模型 (MLLM) 如 GPT-4o、Qwen2-VL 在 2D 图像和视频理解上已取得了显著进展，但 4D（3D + 时间）物体理解领域几乎未被系统评估。4D 数字资产在数字孪生、增强现实、游戏等领域有着重要应用，理解和交互的需求日益增长。

2. **现有痛点**：目前不存在公开的标准化基准来评估 MLLM 的 4D 物体理解能力。现有的 3D 语言理解基准（如 ScanQA）专注于静态 3D 场景，忽略运动信息；2D 视频基准忽略多视角理解。两者都无法捕捉 4D 物体理解所需的多视角空间-时序联合推理能力。

3. **核心矛盾**：4D 物体理解要求同时具备多视角空间理解（从不同角度观察物体以解决遮挡和歧义）和时序理解（跟踪物体动态变化），这对现有仅在 2D 图像/视频上训练的 MLLM 提出了全新挑战。且大规模 4D-文本配对数据极其稀缺，难以直接训练 4D 理解模型。

4. **本文目标**：构建一个高质量的 4D 物体理解基准，系统评估现有 MLLM 在 4D 理解各个维度上的能力和不足，为未来改进提供方向。

5. **切入角度**：不构建专门的 4D 理解模型，而是通过将 4D 物体渲染为多视角视频，直接利用现有 MLLM 的图像/视频理解能力进行 4D 理解。同时包含合成的反事实数据（如 6 条腿的蜘蛛），用于检验模型是否真正理解输入而非依赖先验知识。

6. **核心 idea**：通过精心设计的需要多视角-时序联合推理的问题和人工标注的高质量描述，构建一个能深入诊断 MLLM 4D 理解能力各维度（外观、动作、计数、空间关系、时序关系）的综合基准。

## 方法详解

### 整体框架

4D-Bench 的构建流程包含四个阶段：(1) 4D 数据收集——从 Objaverse-XL 渲染数万个动态 3D 物体的多视角视频；(2) 数据清洗——通过运动分析和视觉质量评估过滤低质量样本；(3) 标注——为 QA 任务设计需要多视角时序理解的问题（MLLM 辅助 + 人工验证），为 Caption 任务提供 5 份独立人工标注；(4) 评估——采用传统指标 + GPT-4o 评分的综合评估框架。

### 关键设计

1. **4D 物体问答任务 (4D Object QA)**:

    - 功能：评估 MLLM 在 5 个子任务上的表现——外观 (Appearance)、动作 (Action)、物体计数 (Object Counting)、空间关系 (Spatial Relationship)、时序关系 (Temporal Relationship)。
    - 核心思路：共 751 个四选一问题，涵盖 736 个 4D 物体。问题经过精心设计，确保必须融合多视角和时序信息才能回答。标注流程采用 MLLM 生成 + 多轮过滤（Qwen2-VL 7B 格式检查 → 纯文本盲测过滤 → 人工审查）的混合方案。
    - 设计动机：每个子任务针对 4D 理解的不同维度，如计数任务要求跨视角融合解决遮挡，时序任务要求追踪物体随时间的演变。

2. **4D 物体描述任务 (4D Object Captioning)**:

    - 功能：评估 MLLM 生成 4D 物体外观和动作描述的能力。
    - 核心思路：从约 8000 个候选中精选 580 个代表性 4D 物体，每个物体由 5 位独立标注员分别撰写一段描述。评估时使用传统 n-gram 指标（BLEU、ROUGE 等）、嵌入指标（BERTScore）和 GPT-4o 评分（分 GPT-Appearance 和 GPT-Action 两个维度，各 0-5 分）。
    - 设计动机：描述任务要求综合多视角外观信息和时序动作信息，比 QA 更全面但评估更复杂。

3. **反事实数据与鲁棒性评估**:

    - 功能：检验 MLLM 是否真正理解视觉输入，还是依赖先验知识作答。
    - 核心思路：数据集包含合成的违反常识的 4D 物体（如 6 条腿的蜘蛛），所有测试的 MLLM（包括 GPT-4o）都错误回答了 8 条腿，说明模型依赖训练数据中的知识而非真正理解输入。还进行了输入顺序鲁棒性（时序优先 vs 视角优先）和时间戳信息的消融。
    - 设计动机：揭示 MLLM "看起来理解但实际上只是记忆" 的根本问题。

### 损失函数 / 训练策略

本文为基准测试工作，不涉及模型训练。评估采用 K=3 个视角，每个视角 N=6 帧的标准输入配置。

## 实验关键数据

### 主实验

| 模型 | 物体计数(%) | 时序关系(%) | 动作(%) | 空间关系(%) | 外观(%) | 总体(%) |
|------|-----------|-----------|--------|-----------|--------|--------|
| GPT-4o | 44.09 | 59.29 | 63.55 | 69.40 | 77.21 | **62.98** |
| Gemini 1.5 Pro | 46.46 | 58.57 | 59.35 | 64.18 | 68.38 | 59.52 |
| Qwen2-VL 72B | 45.67 | 55.71 | 58.41 | 61.19 | 72.06 | 58.72 |
| LLaVA-Video 7B | 42.52 | 55.00 | 52.80 | 56.72 | 78.68 | 56.86 |
| **人类基线** | **88.98** | **89.29** | **94.39** | **91.04** | **89.71** | **91.08** |
| 模型平均 | 37.29 | 49.29 | 49.37 | 53.57 | 63.92 | 50.69 |

### 消融实验

| 配置 | GPT-Appearance | GPT-Action | GPT-Eval |
|------|---------------|------------|----------|
| GPT-4o Captioning | 3.507/5 | 3.258/5 | 3.382/5 |
| 人类 Captioning | 3.772/5 | 3.879/5 | 3.826/5 |
| 1 视角 → 3 视角 | 2.79 → 2.98 | - | ↑0.19 |
| 1 帧 → 6 帧 | - | - | 2.48 → 2.96 |

### 关键发现

- **物体计数是最大弱点**：所有模型在计数任务上平均仅 37.29%，远低于其他子任务，反映了 MLLM 跨视角融合和对应关系建立的困难
- **外观理解 >> 动作理解**：模型在外观和空间理解上表现尚可（63.92%、53.57%），但在动作和时序理解上明显薄弱（49.37%、49.29%）
- **开源模型在外观上接近闭源，但动作理解差距明显**：如 LLaVA-Video 7B 外观达 78.68% 超越 GPT-4o，但动作仅 52.80%
- **CoT 反而降低性能**：Chain-of-Thought 提示导致 Qwen2-VL 7B 准确率下降 9.72%，传统语言 CoT 不适用于视觉推理

## 亮点与洞察

- **首个 4D 理解基准**：填补了 MLLM 评估在 4D 维度上的空白，提供了系统化的能力诊断框架，对理解模型的多视角时空推理能力有重要价值。
- **反事实数据的巧妙设计**：用违反常识的合成物体检验模型是否真正"看懂"还是"猜对"，这种 OOD 评估思路可迁移到其他基准设计中。
- **视角和帧数的影响分析**：发现增加到 3 个视角和 6 帧后性能趋于饱和甚至下降，说明模型的长上下文处理和信息选择能力是瓶颈。

## 局限与展望

- 数据集规模有限（751 QA + 580 Caption），统计显著性可能受影响
- 仅使用多视角视频作为 4D 表示，未探索点云等原生 3D 表示的 MLLM 理解能力
- 合成数据与真实场景有分布偏移，评估结果可能不完全反映真实 4D 理解能力
- 未来可探索视觉 CoT、自适应视角选择、点云+视频多模态输入等改进方向

## 相关工作与启发

- **vs VSI-Bench**：VSI-Bench 评估空间智能，4D-Bench 增加了时序维度，两者互补
- **vs MVBench/Video-MME**：这些 2D 视频基准不需要多视角推理，4D-Bench 要求跨视角信息融合

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个系统性 4D 物体理解基准
- 实验充分度: ⭐⭐⭐⭐⭐ 14 个模型全面评测 + 多角度分析
- 写作质量: ⭐⭐⭐⭐ 动机清晰，分析深入
- 价值: ⭐⭐⭐⭐ 揭示了 MLLM 在 4D 理解上的显著不足

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics](hermes_temporal-coherent_long-form_understanding_with_episodes_and_semantics.md)
- [\[ICCV 2025\] DisTime: Distribution-based Time Representation for Video Large Language Models](distime_distribution-based_time_representation_for_video_large_language_models.md)
- [\[ICCV 2025\] Aligning Effective Tokens with Video Anomaly in Large Language Models](aligning_effective_tokens_with_video_anomaly_in_large_language_models.md)
- [\[ICCV 2025\] Multi-modal Multi-platform Person Re-Identification: Benchmark and Method](multi-modal_multi-platform_person_re-identification_benchmark_and_method.md)
- [\[ICCV 2025\] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding](dynimg_key_frames_with_visual_prompts_are_good_representation_for_multi-modal_vi.md)

</div>

<!-- RELATED:END -->
