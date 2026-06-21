---
title: >-
  [论文解读] EmoPrefer: Can Large Language Models Understand Human Emotion Preferences?
description: >-
  [ICLR 2026][人体理解][描述性情感识别] 针对描述性多模态情感识别（DMER）评估代价高的痛点，提出 EmoPrefer——首个情感偏好数据集与基准，系统探索 MLLM 是否能替代人工标注者完成情感偏好判断，最佳方案（Qwen2.5-Omni）达到 67.21% 两类 WAF，仍留有提升空间。
tags:
  - "ICLR 2026"
  - "人体理解"
  - "描述性情感识别"
  - "情感偏好学习"
  - "多模态大模型评判"
  - "偏好数据集"
  - "基准测试"
---

# EmoPrefer: Can Large Language Models Understand Human Emotion Preferences?

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=EhA4znYsuG](https://openreview.net/forum?id=EhA4znYsuG)  
**代码**: [zeroQiaoba/AffectGPT/EmoPrefer](https://github.com/zeroQiaoba/AffectGPT/tree/master/EmoPrefer)  
**领域**: 情感计算 / 人体情感理解  
**关键词**: 描述性情感识别、情感偏好学习、多模态大模型评判、偏好数据集、基准测试

## 一句话总结
针对描述性多模态情感识别（DMER）评估代价高的痛点，提出 EmoPrefer——首个情感偏好数据集与基准，系统探索 MLLM 是否能替代人工标注者完成情感偏好判断，最佳方案（Qwen2.5-Omni）达到 67.21% 两类 WAF，仍留有提升空间。

## 研究背景与动机

**领域现状**：描述性多模态情感识别（DMER）用自由形式自然语言描述情感状态，相比传统分类范式（如 6 类基本情感）更精细、更可解释，近年借助 MLLM 取得快速进展。

**现有痛点**：DMER 的评估极为困难。基于 ground-truth 描述的评估方法依赖人工标注的"黄金描述"，但情感天然与多模态行为（表情、手势、音调等）深度绑定，高质量参考描述既昂贵又不完整。而直接收集人工偏好标注（给定两段描述让人选更好的那个）虽更可行，但 M 个模型 N 样本所需比较数为 C(M,2)×N，代价同样巨大。

**核心矛盾**：是否能用 MLLM 本身作为偏好判官（judge），从而低成本地替代人工偏好标注？

**本文目标**：构建首个以人类情感为中心的多模态偏好数据集与基准，系统评估当前 MLLM 在情感偏好预测上的能力边界。

**核心 idea**：用多模态 LLM 替代人工标注员完成"哪段情感描述更好"的偏好判断，降低 DMER 评估成本。

## 方法详解

### 整体框架

EmoPrefer 由三个模块构成：EmoPrefer-Data（高质量人工偏好数据集）、EmoPrefer-Bench（多 MLLM 零样本评测基准）、以及两套评估指标（识别性能 + 顺序一致性）。整体流程如下：

```mermaid
flowchart LR
    A[两个 DMER 数据集\nMERR-Fine & MER-Caption+] -->|取交集1368样本| B[EmoPrefer-Data\n专家三人一致标注]
    B --> C[EmoPrefer-Bench\n12+ MLLM × 4种提示策略]
    C --> D[识别性能 WAF/ACC]
    C --> E[顺序一致性 Swap Consistency]
    D & E --> F[模型众包聚合\ntop-k投票]
    F --> G[实际应用: Bradley-Terry\n模型排行榜]
```

### 关键设计

**1. EmoPrefer-Data：高置信度偏好标注**

数据来源于两个已有 DMER 数据集（MERR-Fine 和 MER-Caption+）的交集视频，共 1,368 个样本，每个样本含同一视频的两段描述（分别来自两个数据集）。招募情感计算方向硕士生作为标注员，先用 12 个公认共识样本做资格测试（准确率需≥75%），最终保留 3 名合格标注员。每个样本三人独立判断（描述 A 更好 / 描述 B 更好 / 持平），只保留三人全部一致的样本，确保高置信度。人工一致性上界在去掉"持平"选项后约 69.31%（两类），包含"持平"后降至 59.23%，说明"持平"样本天然歧义更大。

**2. 四种提示策略：从端到端到链式分解**

为系统探索 MLLM 的偏好判断能力，设计四种递进式提示策略：
- **S1（直接判断）**：视频 + 两段描述同时输入 MLLM，直接让模型选更好的描述。
- **S2（两步、同一 MLLM）**：先让 MLLM 生成视频的情感描述，再以此为参照判断哪段更符合。
- **S3（两步、外部 LLM 做裁判）**：MLLM 只负责生成视频描述，判断步骤交给外部纯文本 LLM（Qwen2.5-7B），规避 MLLM 多模态微调后文本理解能力退化的问题。
- **S4（S3 + 额外推理）**：在 S3 的判断步骤前加显式推理过程，验证 CoT 是否有效。

实验发现大多数 MLLM 在 S3/S4 下表现更好，说明其文本理解能力受多模态训练影响；但 Qwen2.5-VL、Qwen2.5-Omni 等训练流程更完善的模型反而在 S1/S2 下更优，因为更长的推理链会引入误差累积。

**3. 顺序一致性：衡量鲁棒性的第二维度**

除识别性能外，本文引入 **swap consistency** 指标：对同一样本分别以正序和逆序输入 MLLM，计算两次预测一致的比例。理想的偏好判官应不受输入顺序影响。实验表明两类指标（识别性能与顺序一致性）相关性较弱，说明它们衡量不同维度的能力，优秀模型需在两者上均表现良好。

**4. 模型众包：聚合多 MLLM 提升可靠性**

对多个 MLLM 预测结果进行多数投票（model-based crowdsourcing）。将模型按识别性能排序，取 top-k 模型聚合。实验发现 k 过大引入噪声、k 过小效果有限；限定为开源模型且 k=3 或 4 时，可在不损失明显性能的前提下大幅降低成本（无需调用付费闭源 API）。

## 实验关键数据

### 主实验（EmoPrefer-Bench 最优策略）

| 模型 | 策略 | 两类 WAF ↑ | 两类 ACC ↑ | 顺序一致性 ↑ |
|------|------|-----------|-----------|------------|
| VideoChat | S4 | 51.79 | 52.31 | 40.85 |
| Video-LLaVA | S4 | 54.53 | 54.62 | 42.86 |
| LLaVA-Next-Video | S4 | 56.41 | 56.57 | 53.92 |
| VITA-1.5 | S4 | 60.08 | 60.12 | 59.06 |
| Qwen2-Audio | S4 | 63.17 | 63.23 | 61.50 |
| **Qwen2.5-VL** | S1 | 64.43 | 65.28 | 62.02 |
| **Qwen2.5-Omni** | S2 | **67.21** | 67.32 | **79.09** |
| GPT-4o（闭源） | S1 | 59.28 | 59.41 | 64.55 |
| GPT-4.1（闭源） | S1 | 60.75 | 60.75 | **80.84** |
| Gemini-1.5-Flash（闭源） | S1 | 64.64 | 65.19 | 72.04 |
| **人工一致性上界** | — | ~69.31 | — | — |

开源 Qwen2.5-Omni 超越所有闭源模型识别性能，逼近人工上界，说明情感偏好判断能力已部分转移至开源模型。

### 关键发现

- 多数模型 S3/S4（外部 LLM 判断）优于 S1/S2；但文本能力强的模型（Qwen2.5-VL/Omni）用 S1/S2 更好，因更长推理链反而累积误差。
- 更大的外部 LLM（Qwen3-14B）不一定优于 Qwen2.5-7B，说明 LLM 规模与情感偏好对齐并非单调关系。
- 顺序一致性与识别性能的 Pearson 相关较低（仅弱相关），需同时考量两个维度才能全面评估。
- 模型众包（top-k=3 开源模型）兼顾效果与成本，正序+逆序组合（normal-swapped）在众包场景下效果不稳定，不推荐默认使用。

## 亮点与洞察

- **评估任务转化视角新颖**：将 DMER 评估从"与 ground-truth 计算相似度"转化为"偏好学习"，再进一步转化为"用 MLLM 自动完成偏好判断"，每一步都有清晰的动机和代价分析。
- **首个全模态情感偏好数据集**：现有 judge 基准几乎全是文本模态，EmoPrefer-Data 首次覆盖文本+图像+音频+视频四模态，且专注情感领域，填补空白。
- **实践应用落地**：结合 Bradley-Terry 算法，直接应用于真实竞赛（MER2025）的模型排行，验证了实际可用性。
- **开源模型与闭源模型的代价-性能权衡**：论文明确给出"开源 top-3 众包"的推荐配置，工程实践价值高。

## 局限与展望

- EmoPrefer-Data 规模仅 1,368 样本，且视频主要来自 MER2024（中文场景、正面单人），跨文化泛化能力未验证。
- 最佳 MLLM 仍低于人工一致性上界（67.21% vs 69.31%），情感偏好解码仍是开放问题。
- 仅评估零样本能力，情感感知微调（emotion-aware RLHF）的潜力尚未探索。
- "持平"类别的预测效果显著弱于二分类，如何处理模糊偏好样本值得深入研究。

## 相关工作与启发

- **vs MT-Bench / JudgeBench**：这些基准专注文本模态的一般任务（写作、数学、推理），EmoPrefer 将 judge 能力扩展至多模态情感专域。
- **vs MLLM-as-a-Judge**：该工作使用图像序列，不含音频；EmoPrefer 引入完整视频+音频，更贴近真实情感场景。
- **vs preference-driven DMER 评估**（Lian et al., 2025b）：本文正是对其偏好标注成本问题的直接响应，用 MLLM 替代人工。
- **启发**：情感感知奖励模型的训练可以用 EmoPrefer-Data 作为种子数据，在 MLLM 的 RLHF 流程中引入情感偏好对齐信号。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个多模态情感偏好数据集与基准，任务定义清晰、填补空白
- 实验充分度: ⭐⭐⭐⭐ 覆盖 12+ MLLM × 4 策略，消融系统，附众包与实际应用验证
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表丰富，评估体系完整
- 价值: ⭐⭐⭐⭐ 为 DMER 评估降本提效，数据集和基准对情感 RLHF 研究有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] LINK: Learning Instance-level Knowledge from Vision-Language Models for Human-Object Interaction Detection](link_learning_instance-level_knowledge_from_vision-language_models_for_human-obj.md)
- [\[ICML 2025\] Scaling Large Motion Models with Million-Level Human Motions](../../ICML2025/human_understanding/scaling_large_motion_models_with_million-level_human_motions.md)
- [\[CVPR 2025\] ChatGarment: Garment Estimation, Generation and Editing via Large Language Models](../../CVPR2025/human_understanding/chatgarment_garment_estimation_generation_and_editing_via_large_language_models.md)
- [\[CVPR 2026\] Bridging Facial Understanding and Animation via Language Models](../../CVPR2026/human_understanding/bridging_facial_understanding_and_animation_via_language_models.md)
- [\[CVPR 2025\] Pose Priors from Language Models](../../CVPR2025/human_understanding/pose_priors_from_language_models.md)

</div>

<!-- RELATED:END -->
