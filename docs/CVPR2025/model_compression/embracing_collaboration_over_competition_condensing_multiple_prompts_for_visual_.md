---
title: >-
  [论文解读] Embracing Collaboration Over Competition: Condensing Multiple Prompts for Visual In-Context Learning
description: >-
  [CVPR 2025][模型压缩][Visual ICL] 提出 Condenser 将多个 Visual ICL 的 prompt 候选通过 Patch-wise 跨注意力凝聚为单一 prompt，实现多 prompt 协作而非竞争选择，在分割/检测/上色等任务上以 16 个 prompt 输入达到 46.63 mIoU（vs 单 prompt 44.14），推理速度比逐一评估快 15×。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "Visual ICL"
  - "提示学习"
  - "跨注意力融合"
  - "GAN"
---

# Embracing Collaboration Over Competition: Condensing Multiple Prompts for Visual In-Context Learning

**会议**: CVPR 2025  
**arXiv**: [2504.21263](https://arxiv.org/abs/2504.21263)  
**代码**: [https://github.com/gimpong/CVPR25-Condenser](https://github.com/gimpong/CVPR25-Condenser)  
**领域**: 模型压缩  
**关键词**: Visual ICL、prompt凝聚、跨注意力融合、MAE-VQGAN、多prompt协作

## 一句话总结
提出 Condenser 将多个 Visual ICL 的 prompt 候选通过 Patch-wise 跨注意力凝聚为单一 prompt，实现多 prompt 协作而非竞争选择，在分割/检测/上色等任务上以 16 个 prompt 输入达到 46.63 mIoU（vs 单 prompt 44.14），推理速度比逐一评估快 15×。

## 研究背景与动机

**领域现状**：Visual ICL 通过提供"输入图像-标签"prompt 对让模型在推理时学习新任务。现有方法从候选 prompt 池中选择"最佳单一 prompt"，但不同 prompt 可能包含互补的信息。

**现有痛点**：(1) 单一 prompt 选择可能遗漏其他 prompt 中的有用信息。(2) 逐一评估 K 个 prompt 进行选择的计算开销大（K× 推理时间）。(3) 简单平均多个 prompt 丢失了空间局部信息。

**核心矛盾**：更多 prompt 提供更多信息，但当前范式只能选一个。合并多个 prompt 需要在保持空间一致性的同时聚合异质信息。

**本文目标** 设计一个轻量模块将 K 个 prompt 凝聚为 1 个高质量 prompt，使多 prompt 协作成为可能。

**切入角度**：Patch-wise 跨注意力——query 图像的每个 patch 只与 K 个 prompt 中同一空间位置的 patch 做注意力聚合。这种局部性保持了空间一致性，同时聚合了多 prompt 的互补信息。

**核心 idea**：用 Patch-wise Cross-Attention 将 K 个候选 prompt 凝聚为 1 个，实现"多 prompt 协作"替代"单 prompt 竞争"。

## 方法详解

### 整体框架
K 个候选 prompt（图像+标签对）→ 每个 prompt 内先自注意力识别信息性 patch → Patch-wise Cross-Attention（PCA）让 query 的每个位置跨 K 个 prompt 同位置聚合 → 凝聚后的单一 prompt（图像+标签）→ 送入冻结的 MAE-VQGAN backbone 推理。

### 关键设计

1. **Patch-wise Cross-Attention（PCA）**:

    - 功能：在保持空间一致性的前提下跨 prompt 聚合信息
    - 核心思路：query 图像的位置 $(i,j)$ 的 patch 仅与 K 个 prompt 中位置 $(i,j)$ 的 patch 做注意力聚合。与全局 cross-attention 相比（44.87 mIoU），PCA（46.63 mIoU）更好因为局部一致性对视觉任务至关重要
    - 设计动机：全局 cross-attention 会将不同空间位置的信息混合，破坏 prompt 的空间结构

2. **图像/标签分离凝聚**:

    - 功能：prompt 图像和标签分别凝聚，共享 key 来源
    - 核心思路：两个 PCA 模块——一个凝聚 prompt 图像 $\mathbf{F}_{c*}^I$，一个凝聚 prompt 标签 $\mathbf{F}_{c*}^L$。两者共享来自图像空间的 key，确保标签凝聚也是基于图像内容的
    - 设计动机：标签和图像需要保持对应关系，共享 key 自然地保证了这种对应

3. **端到端训练（Token Prediction Loss）**:

    - 功能：通过 backbone 的端到端梯度优化凝聚质量
    - 核心思路：$\mathcal{L}_{TP}$（Token Prediction）让凝聚后 prompt 通过冻结 backbone 的输出尽量匹配 ground truth。$\mathcal{L}_{PA}$（Pre-Alignment）用余弦相似度预对齐凝聚特征与目标特征。消融显示去掉 $\mathcal{L}_{TP}$ 后 mIoU 从 46.63 崩溃到 8.66
    - 设计动机：无端到端反馈凝聚模块完全无法学习有意义的融合

### 损失函数 / 训练策略
$\mathcal{L} = \mathcal{L}_{TP} + \lambda \cdot \mathcal{L}_{PA}$。MAE-VQGAN backbone 完全冻结，只训练 Condenser 模块。

## 实验关键数据

### 主实验

| 方法 | K | 分割 mIoU | 检测 mIoU | 推理时间 |
|------|---|----------|----------|---------|
| 单 prompt | 1 | 44.14 | 43.22 | 66ms |
| Prompt-SelF (选择) | 16 | - | - | 989ms |
| **Condenser** | **16** | **46.63** | **44.64** | **66ms** |

### 消融实验

| 融合方式 | 分割 mIoU |
|---------|----------|
| 平均池化 | 17.23 |
| 全局 cross-attention | 44.87 |
| **Patch-wise cross-attention** | **46.63** |
| 无 $\mathcal{L}_{TP}$ | 8.66 |

### 关键发现
- **多 prompt 协作 >> 单 prompt 选择**：K=16 时 46.63 vs K=1 时 44.14（+2.49 mIoU）
- **几乎零推理开销**：凝聚后推理时间与单 prompt 相同（66ms），比逐一选择快 15×
- **性能随 K 持续提升**：K=1→2→4→8→16→32 性能单调上升
- **简单平均完全不行**（17.23 mIoU）：凝聚不是简单的特征平均

## 亮点与洞察
- **"协作替代竞争"的范式转变**：从"选最好的一个"到"融合所有成一个"，思路简洁且有效
- **Patch-wise 设计的空间一致性**是关键设计——视觉任务中空间信息必须被保持

## 局限与展望
- Condenser 模块本身需要训练，对新任务可能需要重新训练
- K 增大时 Condenser 的输入增多但推理不增加——说明过大 K 边际收益递减
- 仅在 MAE-VQGAN backend 上验证

## 相关工作与启发
- **vs Prompt-SelF**：逐一评估选择最佳 prompt，15× 慢且最终只用 1 个。Condenser 用所有 K 个且更快
- **vs InMeMo**：43.14 mIoU，Condenser K=1 就超越（44.14），K=16 更达 46.63

## 评分
- 新颖性: ⭐⭐⭐⭐ 多 prompt 凝聚的概念新颖，PCA 设计合理
- 实验充分度: ⭐⭐⭐⭐ 分割/检测/上色三任务，K值消融，融合方式对比
- 写作质量: ⭐⭐⭐⭐ 协作 vs 竞争的叙事清晰
- 价值: ⭐⭐⭐⭐ 对 Visual ICL 领域有重要贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition](../../ACL2025/model_compression/teamlora_boosting_low-rank_adaptation_with_expert_collaboration_and_competition.md)
- [\[CVPR 2025\] ECVC: Exploiting Non-Local Correlations in Multiple Frames for Contextual Video Compression](ecvc_exploiting_non-local_correlations_in_multiple_frames_for_contextual_video_c.md)
- [\[CVPR 2025\] Mamba-Adaptor: State Space Model Adaptor for Visual Recognition](mamba-adaptor_state_space_model_adaptor_for_visual_recognition.md)
- [\[CVPR 2025\] MobileMamba: Lightweight Multi-Receptive Visual Mamba Network](mobilemamba_lightweight_multi-receptive_visual_mamba_network.md)
- [\[CVPR 2026\] Balanced Dataset Distillation via Modeling Multiple Visual Pattern Distribution](../../CVPR2026/model_compression/balanced_dataset_distillation_via_modeling_multiple_visual_pattern_distribution.md)

</div>

<!-- RELATED:END -->
