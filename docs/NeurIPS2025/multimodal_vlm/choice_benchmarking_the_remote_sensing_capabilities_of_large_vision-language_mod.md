---
title: >-
  [论文解读] CHOICE: Benchmarking the Remote Sensing Capabilities of Large Vision-Language Models
description: >-
  [NeurIPS 2025][多模态VLM][遥感] 提出 CHOICE，一个面向遥感领域的大规模多层级 VLM 基准，包含 10,507 道全新采集题目，覆盖感知与推理 2 大维度、6 个子维度、23 个叶任务，首次实现对 VLM 遥感能力的系统化与客观化评估。 遥感领域的 VLM 评估面临三个核心问题： 分散的评估范围：…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "遥感"
  - "VLM基准"
  - "多层级评估"
  - "多选题"
  - "视觉推理"
---

# CHOICE: Benchmarking the Remote Sensing Capabilities of Large Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2411.18145](https://arxiv.org/abs/2411.18145)  
**代码**: [GitHub](https://github.com/xiaoan-whu/CHOICE)  
**领域**: 多模态VLM  
**关键词**: 遥感, VLM基准, 多层级评估, 多选题, 视觉推理

## 一句话总结

提出 CHOICE，一个面向遥感领域的大规模多层级 VLM 基准，包含 10,507 道全新采集题目，覆盖感知与推理 2 大维度、6 个子维度、23 个叶任务，首次实现对 VLM 遥感能力的系统化与客观化评估。

## 研究背景与动机

遥感领域的 VLM 评估面临三个核心问题：

**分散的评估范围**: 现有评估依赖单一数据集（如 UCM-Caption 仅评估图像描述、DIOR-RSVG 仅评估视觉定位），缺乏统一的多维度评估框架

**碎片化的基准**: LHRS-Bench、VRSBench 等虽提供多任务评估，但维度粗糙、样本量少，缺少像素级和多时相任务

**数据泄露**: 大量基准复用 DOTA、DIOR 等公开数据集，这些数据可能已参与 VLM 训练，导致评估结果不客观

CHOICE 通过全新采集遥感图像（来自全球 50 个城市）、设计多选题格式（消除评分偏差）、构建 23 个叶任务的层级体系来解决上述问题。

## 方法详解

### 整体框架

CHOICE 的能力分类体系采用三层结构：

- **L-1 (2个维度)**: 感知 (Perception) 与 推理 (Reasoning)
- **L-2 (6个子维度)**: 图像级理解(ILC)、单实例识别(SII)、跨实例辨别(CID)、属性推理(AttR)、评估推理(AssR)、常识推理(CSR)
- **L-3 (23个叶任务)**: 从场景分类到灾害判别的全覆盖

### 关键设计

**数据覆盖**: 来自全球 6 大洲 50 个随机选取城市（基于 Oxford Economics GCI 前 1000 大城市），使用 Landsat-8、Sentinel-1/2、Google Earth Engine 等多源卫星数据，空间分辨率从 0.1m/pixel 到 30m/pixel。

**题目构建三种方式**:

1. **标签驱动构建**: 预定义标签（如场景类别、季节标签），从 Google Earth Engine 采集对应图像，从其他样本标签中选取干扰项
2. **基础模型驱动构建**: 利用视觉基础模型提取实例属性（旋转边界框坐标、颜色等），经人工验证后构建细粒度问题
3. **人机协作构建**: 人工标注员撰写精确描述，GPT-4 生成干扰选项，再经人工核验

**评估策略**:

- LLM-based VLM: 直接输出 A/B/C/D 选项，准确率计算
- CLIP-based VLM: 将问题和选项转换为陈述句，计算与图像的相似度
- Visual Grounding: IoU > 0.5 为正确

### 质量控制

题目构建格式统一为 $P_i = [Q_i, C_i, I_i, L_i]$，其中 $Q_i$ 为问题, $C_i$ 为 $n$ 个选项 ($2 \leq n \leq 4$), $I_i$ 为遥感图像, $L_i$ 为正确标签。共招募 12 名遥感/计算机视觉背景的硕士和博士生参与质量保证。

## 实验关键数据

### 主实验

**L-2 维度评估结果**:

| 模型 | ILC | SII | CID | AttR | AssR | CSR |
|------|-----|-----|-----|------|------|-----|
| GPT-4o-2024-11-20 | 0.845 | 0.616 | **0.591** | **0.536** | 0.277 | **0.900** |
| GPT-4o-mini | 0.800 | 0.588 | 0.448 | 0.494 | **0.474** | 0.876 |
| Gemini-1.5-Pro | **0.867** | 0.585 | — | — | — | — |
| Qwen2-VL-70B | — | — | — | — | — | — |
| GeoChat (RSVLM) | — | — | — | — | — | — |

### 消融实验

**关键对比维度**:

| 分析维度 | 发现 |
|---------|------|
| 通用 VLM vs RSVLM | RSVLM 在需要专业遥感知识的任务上占优，但在通用知识整合方面不及通用 VLM |
| 开源 vs 闭源 | Qwen2-VL-70B、InternVL2-40B 在特定任务上可匹敌甚至超越 GPT-4o |
| ILC vs SII 难度 | 图像级理解准确率约 80%+，单实例细粒度感知准确率显著更低 |
| 感知 vs 推理 | AssR（评估推理）是所有模型最弱的维度，GPT-4o-mini 仅 0.474 |

### 关键发现

1. **RSVLM 并无一致优势**: 遥感专用 VLM 在需要专业知识的任务上表现更好，但忽略了通用知识整合，导致整体表现不如通用 VLM
2. **细粒度感知与推理是核心挑战**: 涉及复杂场景、社会属性和遥感特定特征的推理任务对所有 VLM 都极具挑战
3. **开源 VLM 可替代闭源**: 最新开源模型在遥感任务上展现出巨大潜力，可媲美甚至超越 GPT-4o

## 亮点与洞察

- **全新数据保证客观性**: 完全避免使用公开数据集，从根本上解决数据泄露问题
- **23 个叶任务的层级分类体系**: 远超现有基准的维度覆盖（LHRS-Bench 仅 11 维，VRSBench 仅 3 维）
- **全球 50 城市覆盖**: 解决遥感图像的区域内类间差异问题
- **多选题格式**: 消除自由文本评估中指标不一致和主观性问题
- 首次纳入像素级任务（RES）和多时相分析（变化检测），填补现有基准空白

## 局限性

- 评估集中于光学遥感，未涉及 SAR、LiDAR 等多模态遥感数据
- 多选题格式虽然客观但限制了对模型生成能力的评估
- 部分任务（如 Visual Grounding）样本量较少（600 题），统计显著性可能受限
- 缺少对模型推理过程的分析，仅关注最终答案正确率

## 相关工作与启发

- **MMBench / MMStar**: 通用 VLM 基准，其层级评估思想启发了 CHOICE 的三层分类体系
- **LHRS-Bot / GeoChat**: 遥感专用 VLM，CHOICE 揭示了它们在通用知识整合方面的不足
- **CLIP / RemoteCLIP**: CHOICE 设计了专门的 CLIP 评估策略，将问题转化为文本-图像匹配
- 启发：遥感 VLM 的下一步发展方向应是在保持领域专业性的同时加强通用推理能力

## 评分

- ⭐ 新颖性: 4/5 — 首个全新采集、全球覆盖的遥感 VLM 层级基准
- ⭐ 实验充分度: 5/5 — 24 个模型、23 个任务、10,507 道题的大规模评估
- ⭐ 写作质量: 4/5 — 分类体系清晰，图表丰富
- ⭐ 价值: 4/5 — 为遥感 VLM 社区提供了急需的标准化评估工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] VLM4RSDet: Collaborative Optimization with Vision-Language Model for Enhancing Remote Sensing Object Detection](../../CVPR2026/multimodal_vlm/vlm4rsdet_collaborative_optimization_with_vision-language_model_for_enhancing_re.md)
- [\[ICCV 2025\] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves](../../ICCV2025/multimodal_vlm/ideator_jailbreaking_and_benchmarking_large_visionlanguage_m.md)
- [\[ICLR 2026\] CityLens: Evaluating Large Vision-Language Models for Urban Socioeconomic Sensing](../../ICLR2026/multimodal_vlm/citylens_evaluating_large_vision-language_models_for_urban_socioeconomic_sensing.md)
- [\[NeurIPS 2025\] MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly](mmlongbench_benchmarking_longcontext_visionlanguage_models_e.md)
- [\[NeurIPS 2025\] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios](mme-videoocr_evaluating_ocr-based_capabilities_of_multimodal_llms_in_video_scena.md)

</div>

<!-- RELATED:END -->
