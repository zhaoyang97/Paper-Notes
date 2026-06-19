---
title: >-
  [论文解读] OpenHOI: Open-World Hand-Object Interaction Synthesis with Multimodal Large Language Models
description: >-
  [NeurIPS 2025 Oral][多模态VLM][手物交互] 提出 OpenHOI 框架，利用多模态大语言模型（MLLM）的常识推理能力来推断陌生物体的接触区域和抓取类型，实现开放世界的手物交互合成，无需针对每个物体收集训练数据。 领域现状：手物交互（HOI）合成在 VR/AR、机器人抓取、动画等领域至关重要…
tags:
  - "NeurIPS 2025 Oral"
  - "多模态VLM"
  - "手物交互"
  - "开放世界"
  - "MLLM"
  - "接触推理"
  - "抓取合成"
---

# OpenHOI: Open-World Hand-Object Interaction Synthesis with Multimodal Large Language Models

**会议**: NeurIPS 2025 Oral  
**arXiv**: [2505.18947](https://arxiv.org/abs/2505.18947)  
**代码**: 有  
**领域**: 多模态VLM / 手物交互  
**关键词**: 手物交互, 开放世界, MLLM, 接触推理, 抓取合成

## 一句话总结

提出 OpenHOI 框架，利用多模态大语言模型（MLLM）的常识推理能力来推断陌生物体的接触区域和抓取类型，实现开放世界的手物交互合成，无需针对每个物体收集训练数据。

## 研究背景与动机

**领域现状**：手物交互（HOI）合成在 VR/AR、机器人抓取、动画等领域至关重要。现有方法通常在封闭物体集上训练，泛化到新物体时效果差。

**现有痛点**：(1) 需要大量手物接触数据，采集困难；(2) 对新物体形状缺乏先验知识；(3) 基于学习的方法在训练分布外的物体上严重退化。

**核心矛盾**：封闭集训练 vs 开放世界应用——如何在不见过的物体上生成合理的抓取姿态？

**切入角度**：MLLM（如 GPT-4V）具有丰富的物体常识——它知道"杯子的把手适合用力抓"、"鸡蛋表面光滑需要精细抓"——利用这种常识推理来指导抓取合成。

**核心 idea**：MLLM 推理接触区域 + 抓取类型 → 条件化的抓取姿态生成 → 物理优化确保物理合理。

## 方法详解

### 整体框架

输入物体图像/描述 → MLLM 推理（接触区域、抓取类型、力度）→ 条件化扩散模型生成手部姿态 → 物理后处理（穿透消除、接触优化）。

### 关键设计

1. **MLLM 接触推理**

    - 功能：推断物体的可接触区域、适合的抓取类型和力度
    - 核心思路：将物体图像和文字描述提供给 MLLM，通过精心设计的提示引导其输出结构化的接触信息
    - 设计动机：MLLM 的常识知识可以弥补缺乏训练数据的问题——它"知道"杯子怎么拿

2. **条件化抓取生成**

    - 功能：根据 MLLM 推理的接触条件生成手部 MANO 参数
    - 核心思路：条件扩散模型，以接触热力图和抓取类型嵌入为条件，生成手部姿态参数
    - 设计动机：扩散模型能生成多样化的合理姿态，而非单一确定性输出

3. **物理后处理**

    - 功能：消除手物穿透、优化接触质量
    - 核心思路：迭代优化——检测穿透 → 沿法线方向推手 → 优化接触面积
    - 设计动机：纯学习方法无法保证物理合理，后处理修复剩余问题

### 损失函数 / 训练策略

扩散模型训练：去噪损失 $\|ε - ε_\theta(x_t, t, c)\|^2$。接触条件 $c$ 包含 MLLM 推理的区域热力图和类型嵌入。

## 实验关键数据

### 主实验

| 方法 | 穿透深度↓ | 接触面积↑ | 物理稳定性↑ | 新物体泛化 |
|------|---------|---------|-----------|---------|
| GraspTTA | 3.2mm | 12.5cm² | 78% | ✗ 差 |
| ContactOpt | 2.8mm | 15.3cm² | 82% | ✗ 差 |
| MLLM baseline | 4.5mm | 8.7cm² | 65% | ✓ 有 |
| **OpenHOI** | **1.5mm** | **18.2cm²** | **91%** | **✓ 好** |

### 消融实验

| 配置 | 穿透深度 | 物理稳定性 | 说明 |
|------|---------|-----------|------|
| 无 MLLM 推理 | 2.8mm | 82% | 无接触先验 |
| 有 MLLM，无物理后处理 | 2.1mm | 85% | 有先验但有穿透 |
| **完整 OpenHOI** | **1.5mm** | **91%** | **MLLM+物理** |

### 关键发现

- MLLM 接触推理将穿透深度从 2.8mm 降至 2.1mm，物理后处理进一步降至 1.5mm
- 在训练集外的新物体上，OpenHOI 显著优于封闭集方法
- 多样性指标高——同一物体可生成多种合理抓取姿态
- MLLM 的抓取类型推理与人类标注一致率达 85%+

## 亮点与洞察

- **常识驱动**：利用 MLLM 的物体常识替代训练数据，是一种全新的泛化思路。可迁移到机器人操控中的开放世界抓取规划。
- **模块化设计**：MLLM 推理、扩散生成、物理优化三层解耦，每层可独立替换和改进。
- **实用性强**：对 VR/AR 中的虚拟手物交互有直接应用价值。

## 局限与展望

- MLLM 推理的延迟较高（数秒），不适合实时应用
- 对极端形状物体（非典型几何）的常识推理可能不准确
- 仅考虑单手交互，双手协作未涉及
- 物理后处理可能改变 MLLM 推理的接触位置

## 相关工作与启发

- **vs GraspTTA**：GraspTTA 需要目标物体类型在训练集中出现；OpenHOI 真正开放世界
- **vs ContactGen**：ContactGen 学习通用接触模式，但缺乏物体特定的常识驱动

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ MLLM 常识驱动的手物交互是新颖思路
- 实验充分度: ⭐⭐⭐⭐ 定量+定性评估充分
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰
- 价值: ⭐⭐⭐⭐⭐ 开放世界HOI是重要应用场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] HanDyVQA: A Video QA Benchmark for Fine-Grained Hand-Object Interaction Dynamics](../../CVPR2026/multimodal_vlm/handyvqa_a_video_qa_benchmark_for_fine-grained_hand-object_interaction_dynamics.md)
- [\[ICCV 2025\] On Large Multimodal Models as Open-World Image Classifiers](../../ICCV2025/multimodal_vlm/on_large_multimodal_models_as_open-world_image_classifiers.md)
- [\[ACL 2025\] CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World](../../ACL2025/multimodal_vlm/craftext_benchmark_advancing_instruction_following_in_complex_multimodal_open-en.md)
- [\[CVPR 2025\] Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models](../../CVPR2025/multimodal_vlm/img-diff_contrastive_data_synthesis_for_multimodal_large_language_models.md)
- [\[NeurIPS 2025\] Adapting Vision-Language Models for Evaluating World Models](adapting_visionlanguage_models_for_evaluating_world_models.md)

</div>

<!-- RELATED:END -->
