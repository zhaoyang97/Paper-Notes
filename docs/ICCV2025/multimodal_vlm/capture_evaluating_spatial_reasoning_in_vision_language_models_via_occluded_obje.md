---
title: >-
  [论文解读] CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting
description: >-
  [ICCV 2025][多模态VLM][VLM evaluation] 本文提出CAPTURe基准，通过要求VLM在遮挡场景中对规律排列的物体进行"模态补全计数"（amodal counting），系统评估VLM的空间推理和世界模型构建能力，发现即使最强的GPT-4o在遮挡场景下也有14.75%的计数误差，而人类几乎无误差。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "VLM evaluation"
  - "spatial reasoning"
  - "amodal completion"
  - "occlusion"
  - "counting benchmark"
---

# CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting

**会议**: ICCV 2025  
**arXiv**: [2504.15485](https://arxiv.org/abs/2504.15485)  
**代码**: [https://github.com/atinpothiraj/CAPTURe](https://github.com/atinpothiraj/CAPTURe)  
**领域**: 多模态VLM  
**关键词**: VLM evaluation, spatial reasoning, amodal completion, occlusion, counting benchmark

## 一句话总结
本文提出CAPTURe基准，通过要求VLM在遮挡场景中对规律排列的物体进行"模态补全计数"（amodal counting），系统评估VLM的空间推理和世界模型构建能力，发现即使最强的GPT-4o在遮挡场景下也有14.75%的计数误差，而人类几乎无误差。

## 研究背景与动机
- **领域现状**：VLM在多种视觉推理任务上取得了突破性进展，但其是否能像人类一样理解遮挡场景、推断不可见物体仍是开放问题
- **现有痛点**：（1）现有VLM评测忽略了遮挡推理能力；（2）模态补全（amodal completion）通常通过像素级预测评估，不适用于文本输出的VLM；（3）缺少客观可量化的遮挡推理指标
- **核心矛盾**：人类视觉系统能轻松推断遮挡物后的物体并计数，但VLM是否具备类似的"世界模型"能力是未知的
- **本文要解决的问题**：设计一个客观、可量化的VLM遮挡推理评测基准
- **切入角度**：利用物体的规律排列模式（如网格、圆形等）使遮挡后的计数问题有唯一确定答案，以计数准确度作为评估指标
- **核心idea**：模式+遮挡+计数 = 可测量的世界模型评测，同时测试VLM的模式识别、空间推理和计数三维能力

## 方法详解

### 整体框架
CAPTURe是一个evaluation benchmark而非方法论文。基准设计包含两个子集：CAPTURe$^{\text{real}}$（924张真实图像，92种物体类别）和CAPTURe$^{\text{synthetic}}$（1250张合成图像，可控变量）。每张图像包含按规律排列的物体，部分区域被黑色方块遮挡，VLM需要推断遮挡下的物体并报告总数。

### 关键设计
1. **CAPTURe$^{\text{real}}$数据集**:

    - 功能：提供真实场景下的模态补全计数评测
    - 核心思路：从FSC-147数据集筛选包含规律排列物体的图像，经GPT-4o初筛+人工验证过滤出924张，手动覆盖黑色遮挡块。保留遮挡和非遮挡两个版本用于对比
    - 设计动机：在自然场景中测试VLM，覆盖92种物体类型，平均每张图有61.45个物体、13.97个被遮挡

2. **CAPTURe$^{\text{synthetic}}$数据集**:

    - 功能：提供完全可控的诊断性评测
    - 核心思路：生成简单图形（圆点、方块）按不同模式排列的图像，系统变化物体数量（5-15）、排列形状（矩形/圆形/三角形）、位置（5种）、颜色（5种）
    - 设计动机：排除背景干扰、纹理变化等混淆因素，精确定位VLM的失败原因

3. **辅助信息实验（Oracle & Prediction）**:

    - 功能：通过提供额外信息诊断VLM错误来源
    - 核心思路：（1）All Object Coordinate Oracle：提供所有物体坐标，仅需计数文本坐标；（2）Visible Object Coordinate Oracle：提供可见物体坐标，仍需推断遮挡物体；（3）Inpainting Pipeline：用FLUX.1-Fill修复遮挡区域再送入VLM
    - 设计动机：分离"视觉计数能力"和"世界模型/遮挡推理能力"，明确错误根源

### 评估指标
- 主指标：sMAPE（对称平均百分比误差），范围0-100%，越低越好
- $\text{sMAPE} = 100 \cdot \frac{1}{n}\sum_{i=1}^{n}\frac{|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}$
- 无法给出答案的响应按最大误差100%计算

## 实验关键数据

### 主实验

| 模型 | CAPTURe$^{\text{real}}$ 无遮挡 | CAPTURe$^{\text{real}}$ 有遮挡 | Δ | CAPTURe$^{\text{syn}}$ 无遮挡 | CAPTURe$^{\text{syn}}$ 有遮挡 | Δ |
|------|------|------|-----|------|------|-----|
| GPT-4o | 13.34% | 14.75% | +1.41 | 5.90% | 9.71% | +3.81 |
| InternVL2 | 26.17% | 32.90% | +6.73 | 16.44% | 17.57% | +1.13 |
| Molmo | 25.90% | 32.49% | +6.59 | 8.40% | 17.73% | +9.33 |
| Qwen2VL | 18.96% | 29.33% | +10.37 | 6.63% | 11.74% | +5.11 |
| 6 VLM平均 | 21.95% | 27.59% | +5.64 | 11.89% | 15.64% | +3.75 |
| **人类** | - | **3.79%** | - | - | **0.92%** | - |

### 消融实验（辅助信息对CAPTURe$^{\text{real}}$遮挡集的影响）

| 模型 | 原始遮挡 | +所有坐标 | +可见坐标 | +修复图像 |
|------|---------|----------|----------|----------|
| GPT-4o | 14.75% | 2.93% (-11.82) | 9.20% (-5.55) | 15.89% (+1.14) |
| InternVL2 | 32.90% | 17.48% (-15.42) | 25.13% (-7.77) | 31.12% (-1.78) |
| Qwen2VL | 29.33% | 9.62% (-19.71) | 17.70% (-11.63) | 22.64% (-6.69) |
| 3 VLM平均 | 25.66% | 10.01% (-15.65) | 17.34% (-8.32) | 23.22% (-2.44) |

### 关键发现
- 所有VLM在遮挡和非遮挡条件下均存在显著计数误差，且遮挡一致导致性能下降
- 人类在遮挡条件下误差极低（3.79%/0.92%），VLM表现比人类差7-14倍
- 提供所有物体坐标后误差大幅下降（平均-15.65%），说明VLM的一大瓶颈是视觉计数本身
- 图像修复对改善VLM表现效果有限（平均-2.44%），说明扩散模型也不是完美的世界模型
- 模型能较好识别排列模式（准确率>80%），但在遮挡下准确率下降约11%
- 遮挡物体数量越多，误差越大；但总物体数量对误差影响较小
- CountGD（检测模型）在非遮挡条件下远优于VLM，但无法处理遮挡

## 亮点与洞察
- 评测设计巧妙：利用模式+遮挡+计数三要素，将世界模型构建能力转化为客观可量化的指标
- 实验分析深入：通过oracle实验精确分离了"视觉计数"和"遮挡推理"两类错误来源
- 发现VLM的一个基本弱点：即使在无遮挡条件下，图像中的计数对VLM也是困难任务
- Hybrid VLM+CountGD系统的尝试表明，将专用检测模型的输出送给VLM可改善表现

## 局限与展望
- 仅评测了4-6个VLM，未覆盖最新模型（如GPT-4.5、Gemini等）
- CAPTURe$^{\text{real}}$中的物体多数来自FSC-147，数据多样性受限
- 答案提取依赖Llama 3.1 8B，虽验证100%准确但增加了流程复杂度
- 仅考虑了规律排列的物体，未涉及不规则排列场景
- 遮挡块形状固定为矩形，未考虑不规则遮挡
- 基准聚焦于"是否能做"而非"如何改进"，缺乏针对性的改进方法

## 相关工作与启发
- **FSC-147**：密集计数数据集，CAPTURe$^{\text{real}}$的图像来源
- **CountGD**：SOTA目标检测计数方法，作为VLM的对比基准
- **SpartQA**：空间推理VQA基准，但仅测试可见物体之间的关系
- **FLUX.1-Fill**：扩散修复模型，用于提供"预测的世界模型"辅助
- **启发**：VLM评测应更多关注"看不见"的部分（遮挡推理、常识推断），而非仅评测可见信息的处理

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将模态补全计数作为VLM空间推理和世界模型能力的测试，视角独特
- 实验充分度: ⭐⭐⭐⭐ 多VLM对比+人类基线+Oracle+修复管道+因素分析，实验设计系统全面
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，实验分析层层递进，图表丰富
- 价值: ⭐⭐⭐⭐ 揭示了VLM在视觉计数和遮挡推理上的根本不足，为改进VLM提供了明确方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models](multiverse_a_multi-turn_conversation_benchmark_for_evaluating_large_vision_and_l.md)
- [\[ICCV 2025\] VisNumBench: Evaluating Number Sense of Multimodal Large Language Models](visnumbench_evaluating_number_sense_of_multimodal_large_language_models.md)
- [\[CVPR 2026\] Geometrically-Constrained Agent for Spatial Reasoning](../../CVPR2026/multimodal_vlm/geometrically-constrained_agent_for_spatial_reasoning.md)
- [\[ICCV 2025\] Spatial Preference Rewarding for MLLMs Spatial Understanding](spatial_preference_rewarding_for_mllms_spatial_understanding.md)
- [\[ICCV 2025\] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)

</div>

<!-- RELATED:END -->
