---
title: >-
  [论文解读] E-SAM: Training-Free Segment Every Entity Model
description: >-
  [ICCV 2025][语义分割][实体分割] E-SAM 是一个无需额外训练的框架，通过三个级联模块——多层级掩码生成（MMG）、实体级掩码精炼（EMR）和欠分割修复（USR）——系统性地解决 SAM 自动掩码生成（AMG）中的过分割和欠分割问题，在基准指标上超越现有实体分割方法 **+30.1 分**。
tags:
  - "ICCV 2025"
  - "语义分割"
  - "实体分割"
  - "SAM"
  - "训练免调"
  - "自动掩码生成"
  - "过分割"
  - "欠分割"
  - "NMS"
---

# E-SAM: Training-Free Segment Every Entity Model

**会议**: ICCV 2025  
**arXiv**: [2503.12094](https://arxiv.org/abs/2503.12094)  
**代码**: 未公开（论文中提及但链接）  
**领域**: 实体分割 / 基础模型  
**关键词**: 实体分割, SAM, 训练免调, 自动掩码生成, 过分割, 欠分割, NMS

## 一句话总结

E-SAM 是一个无需额外训练的框架，通过三个级联模块——多层级掩码生成（MMG）、实体级掩码精炼（EMR）和欠分割修复（USR）——系统性地解决 SAM 自动掩码生成（AMG）中的过分割和欠分割问题，在基准指标上超越现有实体分割方法 **+30.1 分**。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**：实体分割（Entity Segmentation, ES）旨在分割图像中所有视觉可辨别的实体而不依赖预定义类别标签。与传统语义/实例/全景分割不同，ES 是类别无关的，更贴近人类视觉感知方式。代表方法有 EntitySeg、CropFormer 等，但它们依赖大量标注数据、高训练成本，且泛化能力有限。

**SAM 的潜力与不足**：

### 现有痛点

**现有痛点**：潜力**：SAM 在超过 10 亿掩码上训练，具备强大的零样本分割能力。其 AMG 模式通过均匀点采样生成全图分割，理论上适合 ES 任务。

### 核心矛盾

**核心矛盾**：不足**：AMG 为每个采样点生成 3 层掩码（object/part/subpart），然后用简单 NMS 去冗余。这种策略导致严重的**过分割**（保留太多重叠掩码）和**欠分割**（错误移除关键掩码，遗漏实体或细节）。

**核心矛盾**：SAM 的 AMG 生成大量多粒度掩码，但缺乏有效的后处理策略将其整理为准确的实体级分割图。简单的 NMS 无法处理 multi-granularity 掩码之间的复杂重叠关系。

**本文要解决**：如何在不做任何额外训练的前提下，高效地从 SAM 的 AMG 输出中获得高质量的实体级分割？

## 方法详解

### 整体框架

E-SAM 冻结 SAM 的所有组件（Image Encoder $E_{img}$、Prompt Encoder $E_{prompt}$、Mask Decoder $D_{mask}$），仅在推理阶段通过三个级联模块优化 AMG 输出：

$\text{输入图像} \xrightarrow{AMG} \text{多粒度掩码} \xrightarrow{MMG} \text{分层掩码} \xrightarrow{EMR} \text{实体级掩码} \xrightarrow{USR} \text{最终ES图}$

### 关键设计

1. **Multi-level Mask Generation (MMG)**：

    - 功能：将 AMG 输出的掩码按面积和置信度分级，应用不同 NMS 策略
    - 核心思路：
        - 均匀生成 32 个点提示/边（比 SAM 默认更密集）
        - 根据掩码面积将 SAM 返回的掩码分为 object、part、subpart 三个层级
        - 对 object 级别掩码应用严格的 NMS（高 IoU 阈值）保留高置信度的大掩码
        - 对 part/subpart 级别掩码在密集区域保留更多备选掩码
    - 设计动机：一刀切的 NMS 阈值无法同时处理不同粒度的掩码。大物体需要严格去冗余，小细节需要保留更多候选。

2. **Entity-level Mask Refinement (EMR)**：

    - 功能：将 object 级掩码精炼为准确的实体级掩码，解决重叠和冗余
    - 步骤：
        - (a) **增加采样密度**：用更多均匀点构建 mask gallery（高置信度掩码候选库）
        - (b) **分离重叠掩码**：识别 object 级掩码中的重叠区域，利用 mask gallery 中的掩码将其分离为独立的相邻掩码
        - (c) **合并相似掩码**：构建掩码间的相似度矩阵，利用 mask gallery 判断实体级一致性，将高相似度的掩码合并
    - 设计动机：AMG 的 object 级掩码可能重叠（同一区域被多个点覆盖），也可能过度碎片化。EMR 通过"先分后合"的策略系统地整理为互不重叠的实体掩码。

3. **Under-Segmentation Refinement (USR)**：

    - 功能：修复 EMR 输出中的欠分割区域
    - 核心思路：
        - 使用超像素中心点（superpixel centroids）作为额外 prompt
        - 利用 part/subpart 级掩码的中心点作为补充 prompt
        - 将这些 prompt 送入 SAM 生成额外的高置信度掩码
        - 与 EMR 输出融合，确保未被覆盖的实体得到分割
    - 设计动机：EMR 处理后可能仍有遗漏——一些实体可能因初始均匀采样点未覆盖、或在 NMS 中被错误移除。USR 通过多源 prompt 生成补充掩码来弥补。

### 完全训练免调

- 所有三个模块都是基于规则的后处理策略
- 不修改 SAM 的任何权重
- 唯一的超参数是各阶段的 NMS 阈值、面积分级阈值和相似度合并阈值

## 实验关键数据

### 主实验

在 EntitySeg 基准上：

- E-SAM 超越此前 SOTA 实体分割方法 **+30.1 分**（在基准指标上）
- 在相同 backbone 大小下，E-SAM 性能一致地超过 SAM AMG 的 **2 倍以上**

### 与不同方法的对比

| 对比方法 | 类型 | E-SAM 优势 |
|---------|------|-----------|
| SAM AMG | 基础模型 AMG | 解决过分割/欠分割，性能提升 2x+ |
| Semantic-SAM | 增强 SAM | 无需额外训练，更好的实体级精度 |
| EntitySeg/CropFormer | 专用 ES 模型 | 无需训练数据，零样本泛化更强 |

### 泛化性实验

- 在未见过的数据集（open-world scenarios）上，E-SAM 展示了强泛化能力（Figure 7），验证了训练免调方法的优势——因为没有在特定分布上过拟合

### 消融实验

- **MMG 的贡献**：分层 NMS vs. 统一 NMS → 分层策略显著减少过分割
- **EMR 的贡献**：分离+合并步骤缺一不可，去掉任一步骤性能都下降
- **USR 的贡献**：USR 主要解决大面积未覆盖区域，对完整实体发现至关重要
- **三模块协同**：单独任何一个模块的提升有限，三者级联配合才能发挥最大效果

### 关键发现

- SAM 的 AMG 输出虽然质量参差不齐，但包含了足够丰富的多粒度信息，问题在于如何组织和筛选
- 训练免调方法在 ES 任务上可以大幅超越需要训练的专用方法，说明 SAM 的基础能力在恰当的后处理下可以充分释放
- 过分割和欠分割需要不同的策略——分别由 EMR（去冗余/合并）和 USR（补漏）处理

## 亮点与洞察

- **"不训练胜过有训练"的强力证据**：E-SAM 完全训练免调，却超越了需要大量标注和高训练成本的 CropFormer 等方法。这说明 SAM 这类大规模预训练基础模型已经包含了足够的分割能力，关键在于如何有效地调用和组织其输出。
- **系统性地分析 AMG 缺陷并逐一解决**：MMG 处理过分割、EMR 精炼实体级别、USR 弥补欠分割——每个模块针对一个具体问题，设计清晰，逻辑自洽。
- **"先分后合"的掩码精炼策略**：EMR 先分离重叠掩码再根据实体一致性合并，比直接合并或直接分离更鲁棒。这种策略借鉴了传统图像分割中分水岭算法的"先过分割再合并"思路。
- **多源 prompt 的互补性**：USR 同时利用超像素中心、part 中心和 subpart 中心作为 prompt，三种来源在空间尺度上互补，最大化覆盖遗漏区域。
- **强泛化性**：训练免调的最大优势在于不受训练数据分布限制，E-SAM 在未见数据集上的强表现验证了这一点。

## 局限与展望

- **推理效率较低**：三个级联模块都需要多次调用 SAM 的编码器和解码器（特别是 EMR 中的 mask gallery 构建和 USR 中的额外 prompt 生成），推理时间远高于单次 AMG 或训练好的端到端方法
- **超参数敏感性**：多个阈值（NMS 阈值、面积分级、相似度合并阈值等）需要手动调优，不同数据集可能需要不同设置
- **对 SAM 版本的依赖**：E-SAM 完全依赖 SAM 的分割质量，如果底层 SAM 在某些场景失效（如极端遮挡、透明物体），E-SAM 也无法弥补
- **缺乏视频/时序扩展**：仅针对单帧图像，未考虑视频中时序一致性带来的实体追踪和分割改进
- **掩码精度的上限受限于 SAM**：E-SAM 只能从 SAM 已生成的掩码中筛选和组合，无法生成 SAM 从未产出的掩码形状

## 亮点与洞察

## 局限与展望

## 相关工作与启发

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree](sam2long_enhancing_sam_2_for_long_video_segmentation_with_a.md)
- [\[AAAI 2026\] SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model](../../AAAI2026/segmentation/saq-sam_semantically-aligned_quantization_for_segment_anything_model.md)
- [\[ICCV 2025\] OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation](omnisam_omnidirectional_segment_anything_model_for_uda_in_panoramic_semantic_seg.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [\[NeurIPS 2025\] SAM-R1: Leveraging SAM for Reward Feedback in Multimodal Segmentation via Reinforcement Learning](../../NeurIPS2025/segmentation/sam-r1_leveraging_sam_for_reward_feedback_in_multimodal_segmentation_via_reinfor.md)

</div>

<!-- RELATED:END -->
