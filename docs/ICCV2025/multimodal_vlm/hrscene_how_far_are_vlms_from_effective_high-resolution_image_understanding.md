---
title: >-
  [论文解读] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?
description: >-
  [ICCV 2025][多模态VLM][高分辨率图像理解] 提出 HRScene 基准，涵盖 25 个真实场景和 2 个诊断数据集（分辨率 1K-35K），评估 28 个 VLM 后发现：当前最强模型在真实高分辨率任务上平均准确率仅约 50%，且存在显著的区域差异和 lost-in-middle 问题。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "高分辨率图像理解"
  - "VLM基准"
  - "视觉语言模型"
  - "多模态评估"
  - "Needle-in-a-Haystack"
---

# HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?

**会议**: ICCV 2025  
**arXiv**: [2504.18406](https://arxiv.org/abs/2504.18406)  
**代码**: [项目主页](https://yszh8.github.io/hrscene/)  
**领域**: 多模态VLM  
**关键词**: 高分辨率图像理解, VLM基准, 视觉语言模型, 多模态评估, Needle-in-a-Haystack

## 一句话总结

提出 HRScene 基准，涵盖 25 个真实场景和 2 个诊断数据集（分辨率 1K-35K），评估 28 个 VLM 后发现：当前最强模型在真实高分辨率任务上平均准确率仅约 50%，且存在显著的区域差异和 lost-in-middle 问题。

## 研究背景与动机

高分辨率图像 (HRI) 理解在病理学、自动驾驶、文档理解等领域至关重要。虽然 Gemini、Claude、GPT 等 VLM 声称支持高分辨率输入，但存在严重的评估缺口：

**缺乏基准**: 主流 VLM 报告中评估的基准（MMMU、VQAv2、AI2D 等）平均分辨率低于 1K，不适合 HRI 评估

**场景单一**: 现有 HRI 数据集仅关注特定场景（如远距离图像）或特定分辨率（如 8K）

**诊断不足**: 现有 multi-modal NIAH 测试主要关注长文本或低分辨率多图像，缺乏对 HRI 区域利用能力的诊断

本文动机：构建一个**统一、全面、易用**的 HRI 基准，系统评估 VLM 的高分辨率理解能力并发现其核心缺陷。

## 方法详解

### 整体框架

HRScene 包含两大部分：
- **25 个真实场景数据集**: 分辨率 1K-35K，覆盖 8 大类别
- **2 个合成诊断数据集**: 用于精确定位 VLM 缺陷

### 关键设计

1. **场景分类体系 (Taxonomy)**:

    - **8 大类别**: 日常照片、城市规划、扫描文档、艺术品、多子图、遥感、医学诊断、研究理解
    - **25 个具体场景**: 从显微镜到射电望远镜，覆盖各种相机类型
    - **多种能力测试**: 计数、时序/语义推理、整体判断、视觉检索、空间关系、小目标检测
    - 6 个数据集需要领域专家知识，19 个属于通用领域

2. **数据收集与重标注**:

    - 从 25 个现有数据源收集，8 个由 10 名研究生级标注员重标注
    - 所有图像分辨率 ≥ 1024×1024
    - 为 6 个数据集构建干扰选项（每个样本至少 4 个选项）
    - 数字答案通过随机偏移自动生成选项
    - 进一步收集 750 样本的人类表现作为上界

3. **WhiteBackground NIAH 诊断**:

    - 将 VQAv2 图像（needle）放置在 N×N 白色网格（haystack）的不同行列位置
    - 评估 VLM 在不同网格位置的性能差异，检测**区域差异 (Regional Divergence)**
    - 测试 1×1 到 10×10 网格大小

4. **ComplexGrid NIAH 诊断**:

    - 用图像检索工具找到与 needle 最相似的图像作为 distractor
    - 组合成更大网格，要求模型指出 needle 所在行列
    - 评估 VLM 在多个干扰图像中检索正确图像的能力

### 数据规模与划分

| 统计项 | 数量 |
|--------|------|
| 总样本 | 7,068 |
| 重标注 | 2,005 |
| 从零标注 | 384 |
| val | 750 (= 人类标注) |
| testmini | 1,000 |
| test | 5,323 |

## 实验关键数据

### 主实验 (表格)

**真实世界数据集总体表现**:

| 模型 | Art | Daily | Medical | Paper | Remote | Research | Sub-Img | Urban | Avg |
|------|-----|-------|---------|-------|--------|----------|---------|-------|-----|
| Qwen2-VL 7B | 69.46 | 64.20 | 40.40 | 64.62 | 50.60 | 36.69 | 71.42 | 40.17 | 56.65 |
| InternVL2 40B | 74.35 | 62.67 | 38.10 | 70.89 | 44.16 | 43.15 | 74.10 | 44.40 | 58.45 |
| Qwen2-VL 72B | 75.85 | 66.20 | 43.69 | 78.13 | 52.48 | 39.36 | 74.89 | 44.66 | 61.85 |
| Gemini2.0 Flash | 76.46 | 62.27 | 51.94 | 75.12 | 47.59 | 34.85 | 68.62 | 44.54 | 59.82 |
| GPT-4o | 69.13 | 55.90 | 22.63 | 66.80 | 44.05 | 35.38 | 65.13 | 41.72 | 52.91 |
| **Human** | **75.33** | **77.75** | **23.81** | **88.75** | **58.33** | **48.50** | **90.00** | **55.25** | **64.72** |
| **28模型平均** | 61.54 | 53.18 | 36.64 | 58.17 | 41.75 | 36.08 | 60.60 | 37.84 | **49.68** |

### 消融实验 (表格)

**WhiteBackground NIAH 诊断 — 区域差异分析**:

| 模型 | 1×1 Perf | 3×3 Perf | 3×3 Region↓ | 5×5 Perf | 5×5 Region↓ | 10×10 Perf | 10×10 Region↓ |
|------|----------|----------|-------------|----------|-------------|------------|---------------|
| Qwen2-VL 7B | 85.93 | 84.22 | 5.30 | 83.14 | 6.52 | 79.91 | 10.56 |
| Qwen2-VL 72B | 84.13 | 84.51 | 5.62 | 84.04 | 6.62 | 84.56 | **9.61** |
| GPT-4o-mini | 68.66 | 60.69 | 13.77 | 52.53 | 19.59 | 32.94 | 33.65 |
| DeepSeek-VL2 | 72.06 | 49.71 | 15.75 | 34.29 | 23.37 | 23.95 | 23.30 |
| InternVL2 40B | 84.53 | 83.42 | 4.57 | 80.02 | 8.84 | 74.95 | 13.18 |

（Region 指标 = 不同位置性能的标准差，越低越好）

### 关键发现

- **整体差距显著**: 28 个 VLM 的平均准确率仅 49.68%，最强模型 Qwen2-VL 72B 也仅 61.85%
- **领域差异大**: Medical（36.64%）和 Research（36.08%）最差，Paper（58.17%）和 Art（61.54%）较好
- **人类 vs 机器**: 人类平均 64.72% 但 Medical 仅 23.81%（需专家知识），Sub-Img 人类 90% 但模型仅 60%
- **Regional Divergence**: 随网格增大，大多数模型性能显著下降（如 GPT-4o-mini 从 1×1 的 68.66% 降至 10×10 的 32.94%），但 Qwen2-VL 72B 几乎不受影响
- **参数量不总是关键**: Qwen2-VL 7B 在多数指标上优于许多更大模型（如 Llava-Next 34B）
- **Lost-in-middle 现象**: VLM 对网格中间位置的图像识别能力弱于边缘位置

## 亮点与洞察

- HRScene 是目前最全面的 HRI 基准：25 个场景、分辨率跨 4 个数量级、覆盖专家+通用领域
- 两个诊断数据集的设计精巧，定量揭示了 VLM 的两个核心缺陷（区域差异和 lost-in-middle）
- 发现 Qwen2-VL 72B 在 WhiteBackground NIAH 中几乎不受分辨率影响（Region 指标始终低于 10），暗示其内部分辨率处理策略可能更优
- 人类在 Medical 上仅 23.81% 但模型平均 36.64%，说明某些专家领域模型已超越非专家人类

## 局限与展望

- 基准以多选题为主，对开放式回答的评估有限
- 诊断数据集仅用白色背景和视觉相似干扰，更自然的复合场景可进一步挑战模型
- 部分数据集样本量较少（如 Galaxy 和 Grass），统计显著性待加强
- 未涉及视频高分辨率理解（如高分辨率视频帧序列）
- test 集答案不公开，需通过在线平台提交，可能限制快速迭代研究

## 相关工作与启发

- 与 MME-Realworld 和 HR-Bench 互补，提供了更广泛场景覆盖
- NIAH 测试从文本/多图像扩展到高分辨率单图像，是一个自然而重要的推广
- Qwen2-VL 的分块处理策略值得深入研究，其抗分辨率衰减能力显著优于其他模型
- 对 VLM 高分辨率处理架构（双编码器 vs 分块策略）的比较有助于指导模型设计

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个大规模统一 HRI 基准，诊断数据集设计巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ 28 个模型、27 个数据集、人类性能对比、诊断分析
- **写作质量**: ⭐⭐⭐⭐ 分类体系清晰，发现的问题表述明确
- **价值**: ⭐⭐⭐⭐⭐ 为 VLM 高分辨率理解提供了急需的系统性评估工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)
- [\[ICCV 2025\] FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers](falcon_resolving_visual_redundancy_and_fragmentation_in_high.md)
- [\[ICML 2026\] CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception](../../ICML2026/multimodal_vlm/cvsearch_empowering_multimodal_llms_with_cognitive_visual_search_for_high-resolu.md)
- [\[ICCV 2025\] Training-free Generation of Temporally Consistent Rewards from VLMs](training-free_generation_of_temporally_consistent_rewards_from_vlms.md)
- [\[ECCV 2024\] FlexAttention for Efficient High-Resolution Vision-Language Models](../../ECCV2024/multimodal_vlm/flexattention_for_efficient_high-resolution_vision-language_models.md)

</div>

<!-- RELATED:END -->
