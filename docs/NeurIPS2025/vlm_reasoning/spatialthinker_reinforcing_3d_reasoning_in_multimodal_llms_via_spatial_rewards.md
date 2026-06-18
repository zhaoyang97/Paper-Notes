---
title: >-
  [论文解读] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards
description: >-
  [NeurIPS 2025][多模态VLM][空间推理] 提出 SpatialThinker，通过在线 RL 结合多目标密集空间奖励（格式→计数→准确性→空间定位的字典序门控）训练 MLLM 构建场景图并进行结构化空间推理，仅用 7K 样本超越 GPT-4o 在 3DSRBench 上 12.1%。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "空间推理"
  - "场景图"
  - "强化学习"
  - "密集奖励"
  - "3D理解"
---

# SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards

**会议**: NeurIPS 2025  
**arXiv**: [2511.07403](https://arxiv.org/abs/2511.07403)  
**代码**: [https://github.com/SpatialThinker](https://github.com/SpatialThinker)  
**领域**: 多模态VLM  
**关键词**: 空间推理, 场景图, 强化学习, 密集奖励, 3D理解

## 一句话总结
提出 SpatialThinker，通过在线 RL 结合多目标密集空间奖励（格式→计数→准确性→空间定位的字典序门控）训练 MLLM 构建场景图并进行结构化空间推理，仅用 7K 样本超越 GPT-4o 在 3DSRBench 上 12.1%。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**：空间理解是 MLLM 的核心短板，现有方法要么需要海量数据（SpatialVLM 用 2B 样本）、要么需要显式 3D 输入（深度图/点云）、要么用稀疏奖励做 RL

**现有痛点**：稀疏奖励（仅最终答案是否正确）对视觉定向推理提供的指导不足；SFT 学静态模式而非推理策略

**核心 idea**：

### 现有痛点

**现有痛点**：场景图引导推理**：模型先构建问题相关的子场景图（物体+边界框+关系），再在此结构上推理

### 核心矛盾

**核心矛盾**：密集空间奖励**：四组件奖励（格式+计数+准确性+CIoU空间定位）配字典序门控，避免奖励 hacking

## 方法详解

### 关键设计
1. **推理模板**：`<observe>`场景描述 → `<scene>`JSON场景图（物体ID+bbox+关系三元组）→ `<think>`推理 → `<answer>`答案
2. **密集空间奖励 + 字典序门控**：

    - 格式奖励 $R_f$（$w=0.1$）：JSON可解析+字段完整
    - 计数奖励 $R_c$（$w=0.2$）：物体和关系数匹配度
    - 准确性奖励 $R_a$（$w=0.5$）：最终答案正确
    - 空间奖励 $R_s$（$w=0.2$）：仅当答案正确时激活，用匈牙利匹配+CIoU 评估定位精度
    - 门控：$R_{total} = \mathbb{I}[R_f=1] \cdot (w_f R_f + w_c R_c + w_a R_a + \mathbb{I}[R_a=1] \cdot w_s R_s)$
3. **STVQA-7K 数据集**：从 Visual Genome 场景图生成的 7.5K 高质量空间 VQA，覆盖 9 类空间推理

### 训练策略
基于 Qwen2.5-VL-7B，用 GRPO（无需 Critic 网络），4×H100 训练 15 小时。

## 实验关键数据

### 主实验

| 模型 | 3DSRBench | CV-Bench | BLINK Avg. |
|------|-----------|---------|-----------|
| GPT-4o | 44.3 | 79.4 | 80.4 |
| Qwen2.5-VL-7B (base) | 48.4 | 68.6 | 68.2 |
| + Sparse RL | 52.4 (+4.0) | 72.1 | 72.8 |
| + **SpatialThinker (Dense)** | **55.6 (+7.2)** | **75.3** | **76.8** |

### 消融实验

| 奖励配置 | 3DSRBench | Δ vs base |
|----------|-----------|-----------|
| 无 RL | 48.4 | - |
| 稀疏奖励 | 52.4 | +4.0 |
| +格式+计数 | 53.8 | +5.4 |
| +空间奖励(无门控) | 54.1 | +5.7 |
| +空间奖励(有门控) | **55.6** | **+7.2** |

### 关键发现
- 密集奖励的收益是稀疏奖励的 1.8 倍（+7.2 vs +4.0）
- 字典序门控至关重要——无门控时模型会生成过多物体来骗取空间奖励
- 仅 7K 样本超越百万级 SFT 数据集的方法

## 亮点与洞察
- **"observe→localize→think→answer"的推理模板**模拟了人类空间推理：先观察场景，再定位关键物体，然后推理，最后回答
- **密集空间奖励在仅 7K 样本上的效果超越百万级 SFT**——证明了"指导什么学"比"学什么多"更重要
- 字典序门控是避免奖励 hacking 的关键设计——直接可迁移到其他多目标 RL

## 局限与展望
- 场景图构建质量依赖训练数据的标注精度；对极复杂场景子图提取可能不完整

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 场景图+密集RL的空间推理首创
- 实验充分度: ⭐⭐⭐⭐⭐ 12个基准全面评估+详细消融
- 写作质量: ⭐⭐⭐⭐⭐ 奖励设计推导清晰
- 价值: ⭐⭐⭐⭐⭐ 用极少数据解决空间推理难题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](../../ICCV2025/multimodal_vlm/mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)
- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[CVPR 2026\] STAR-R1: Multi-View Spatial TrAnsformation Reasoning by Reinforcing Multimodal LLMs](../../CVPR2026/multimodal_vlm/star-r1_multi-view_spatial_transformation_reasoning_by_reinforcing_multimodal_ll.md)
- [\[NeurIPS 2025\] VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents](vagen_reinforcing_world_model_reasoning_for_multi-turn_vlm_agents.md)
- [\[NeurIPS 2025\] Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs](struct2d_a_perception-guided_framework_for_spatial_reasoning_in_mllms.md)

</div>

<!-- RELATED:END -->
