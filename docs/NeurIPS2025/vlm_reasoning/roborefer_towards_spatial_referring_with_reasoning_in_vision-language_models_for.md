---
title: >-
  [论文解读] RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics
description: >-
  [NeurIPS 2025][多模态VLM][空间指代] 提出 **RoboRefer**，一个 3D 感知的推理型 VLM，通过 **SFT + RFT** 两阶段训练策略（含度量敏感的过程奖励函数），在空间指代任务中实现精确的单步空间理解和多步空间推理，在 RefSpatial-Bench 上超越 Gemini-2.5-Pro 达 17.4%。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "空间指代"
  - "视觉语言模型"
  - "多步推理"
  - "强化微调"
  - "机器人操作"
---

# RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics

**会议**: NeurIPS 2025  
**arXiv**: [2506.04308](https://arxiv.org/abs/2506.04308)  
**代码**: [zhoues.github.io/RoboRefer](https://zhoues.github.io/RoboRefer)  
**领域**: 多模态VLM  
**关键词**: 空间指代, 视觉语言模型, 多步推理, 强化微调, 机器人操作  
**arXiv**: [2506.04308](https://arxiv.org/abs/2506.04308)  
**代码**: 无  
**领域**: 多模态VLM  

## 一句话总结

提出 **RoboRefer**，一个 3D 感知的推理型 VLM，通过 **SFT + RFT** 两阶段训练策略（含度量敏感的过程奖励函数），在空间指代任务中实现精确的单步空间理解和多步空间推理，在 RefSpatial-Bench 上超越 Gemini-2.5-Pro 达 17.4%。

## 研究背景与动机

**领域现状**：空间指代（spatial referring）是具身AI的基础能力——机器人需要理解"把杯子放在键盘和笔筒之间，与杯子logo对齐"这样的空间约束指令。

**现有痛点**：现有VLM方法要么需要昂贵的3D重建造成模态差距，要么将深度当RGB输入共享编码器导致模态干扰，且几乎所有方法只能处理单步空间理解（如"哪个更近"），无法进行多步推理。

**核心矛盾**：(1) 现有数据集只覆盖15种空间关系，缺乏多步推理标注；(2) SFT训练倾向于记忆答案而非泛化推理能力；(3) 缺乏评估多步空间推理的基准。

**本文目标**：构建一个同时具备精确3D空间感知和多步空间推理能力的VLM，能实际应用于机器人操控和导航。

**切入角度**：独立的深度编码器避免模态干扰 + 强化微调产生可泛化的推理策略。

**核心 idea**：SFT学空间感知、RFT学空间推理，用度量敏感的过程奖励指导中间推理步骤的精度。

## 方法详解

### 整体框架（图2）

RoboRefer使用独立的RGB编码器和深度编码器分别提取特征，通过各自的投影器对齐到LLM空间。训练分三步：(1) 深度对齐，(2) SFT增强空间理解，(3) RFT增强多步推理。

### 关键设计 1：独立深度编码器

- **功能**：为深度模态提供专用的编码路径，避免与RGB编码器的干扰。
- **核心思路**：深度编码器和深度投影器从RGB对应部分初始化。联合训练时RGB编码器不受深度输入影响，深度编码器独立更新。
- **设计动机**：共享编码器方案（如SpatialRGPT）需要2倍以上的RGB数据补偿模态干扰，且退化预训练图像编码器的性能。Tab. 4证明独立编码器在通用VQA上几乎无损。

### 关键设计 2：SFT两步训练

1. **深度对齐**：仅训练深度投影器，用RefSpatial的RGB-D标注对齐深度空间到文本空间
2. **空间理解增强**：微调所有参数，同时用RGB和RGB-D输入训练，强制图像编码器也学习超越深度线索的空间理解。数据包含单步标注 + 多步推理过程数据（作为RFT的"冷启动"）

### 关键设计 3：RFT + 度量敏感过程奖励

- **功能**：用 GRPO 强化学习在 SFT 模型基础上进一步提升多步推理泛化能力。
- **四种奖励函数**：
    - **Outcome Format Reward** $R_{OF}$：格式正确性奖励
    - **Point L1 Reward** $R_P$：最终预测点是否在目标范围内（0/1）
    - **Process Format Reward** $R_{PF}$：中间步骤格式 "[Perception Type] [Target Object]:"
    - **Accuracy Reward** $R_{Acc}$：对每个关键步骤，按感知类型度量预测误差（如坐标用L1距离）
- **总奖励**：$r_i = R_{OF}(a_i) + R_P(a_i) + \alpha R_{PF}(a_i) + \alpha R_{Acc}(a_i)$，$\alpha = 0.25$
- **关键设计细节**：过程奖励是序不变的（order-invariant），不约束推理轨迹的固定顺序
- **设计动机**：SFT倾向记忆答案，RFT通过探索（采样N个响应）和奖励信号学到更泛化的推理策略

### 关键设计 4：RefSpatial 数据集

- **规模**：2.5M 样本，20M QA 对（2×先前最大数据集）
- **数据源三角**：2D网络图像（空间概念+广泛深度感知）→ 3D具身视频（精细室内空间理解）→ 仿真数据（多步推理过程标注）
- **覆盖 31 种空间关系**（vs 先前15种），最多5步推理
- **精细标注**：每个物体有层次化描述（"杯子"→"离相机最近的杯子"），支持在杂乱场景中无歧义指代

## 实验关键数据

### 主实验：单步空间理解（Table 1）

| Method | Input | CV-Bench Avg | BLINK Avg | RoboSpatial | SAT |
|---|---|---|---|---|---|
| Gemini-2.5-Pro | RGB | 91.74 | 89.17 | 77.24 | 70.59 |
| SpatialRGPT-8B | RGB-D | 89.77 | 85.32 | 66.67 | 64.00 |
| **RoboRefer-8B-SFT** | **RGB-D** | **96.24** | **92.18** | **84.55** | **86.67** |

### 主实验：多步空间指代（Table 2）

| Method | RefSpatial-Bench-L. | RefSpatial-Bench-P. | RefSpatial-Bench-U. |
|---|---|---|---|
| Gemini-2.5-Pro | 46.96 | 24.21 | 27.14 |
| Molmo-72B | 45.77 | 14.74 | 21.24 |
| RoboRefer-2B-SFT | 47.00 | 48.00 | 33.77 |
| **RoboRefer-2B-RFT** | **52.00** | **54.00** | **41.56** |

**在未见过的空间关系组合上（Unseen），RFT相比SFT提升9.1%。**

### 2D指代任务（Table 3）

| Method | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|---|---|---|---|
| Qwen2.5-VL-72B (B.→P.) | 95.4 | 91.5 | 92.5 |
| **RoboRefer-8B-SFT** | **96.6** | **91.9** | **94.3** |

### 消融实验（Table 7）

| 配置 | CV-Bench | BLINK |
|---|---|---|
| 无2D数据 | 84.17 | 74.48 |
| 无3D数据 | 81.83 | 74.61 |
| 无仿真数据 | 83.96 | 75.10 |
| 无深度编码器 | 91.24 | 85.27 |
| **全部** | **94.77** | **89.27** |

### 关键发现

- **数据配方至关重要**：三种数据源缺一不可；2D数据移除严重影响室外场景(BLINK)，3D数据移除严重影响室内(CV-Bench)
- **深度编码器在多步推理中增益更大**：因为中间步骤的误差会累积放大深度线索的价值
- **过程奖励带来5个百分点的提升**（Tab. 7 RFT部分）
- **真实机器人（UR5, G1人形）上的验证**：成功完成需要多步空间推理的长horizon动态任务（Tab. 5/6）

## 亮点与洞察

1. **SFT + RFT 的渐进式训练范式**：SFT提供空间感知的"冷启动"，RFT释放泛化推理能力，避免了纯SFT的记忆倾向
2. **度量敏感的过程奖励是关键创新**：不同于仅看最终答案的outcome reward，过程奖励能精确指导每个推理步骤的中间预测精度
3. **点预测而非框**：对机器人来说，单点预测比2D框更自然——可直接映射为3D坐标、无遮挡问题、统一抓取/放置/导航
4. **2B模型超越Gemini-2.5-Pro 17.4%**：在多步空间推理这个特定能力上，专用小模型可以显著超越通用大模型

## 局限与展望

1. **仅2D点预测**：需要额外的深度-3D映射步骤，直接预测3D点是未来方向
2. **人类意图理解不足**：人类指令常常简短模糊，模型需要更好的意图推断能力
3. **RFT仅验证了2B模型**：受计算限制未在8B模型上做RFT，扩展效果未知
4. **定量空间关系有限**：主要处理定性关系（左/右/近/远），精确度量推理（如"距离桌边10cm"）仍有挑战
5. **仿真到真实的迁移**：虽在真实场景验证了可行性，但仿真数据的推理标注模式可能存在domain gap

## 相关工作与启发

- **与 RoboPoint 的区别**：RoboPoint仅用图像中的基本空间线索，不支持多步推理也缺乏3D深度感知
- **与 SpatialRGPT 的区别**：后者处理更简单的VQA任务且需要外部mask/检测工具输入，RoboRefer直接从文本指令定位
- **对具身AI的启发**：空间指代可以统一操控和导航——同一个VLM同时为抓取和行走提供目标点

## 评分

⭐⭐⭐⭐⭐ (5/5)

非常完整的工作：新任务定义（多步空间指代）、新数据集（RefSpatial, 20M QA）、新训练范式（SFT+RFT+过程奖励）、新基准（RefSpatial-Bench）、真实机器人验证。2B模型大幅超越Gemini-2.5-Pro令人印象深刻。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics](../../CVPR2025/multimodal_vlm/robospatial_teaching_spatial_understanding_to_2d_and_3d_vision-language_models_f.md)
- [\[NeurIPS 2025\] SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning](ssr_enhancing_depth_perception_in_vision-language_models_via_rationale-guided_sp.md)
- [\[CVPR 2025\] ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models](../../CVPR2025/multimodal_vlm/espire_a_diagnostic_benchmark_for_embodied_spatial_reasoning_of_vision-language_.md)
- [\[NeurIPS 2025\] SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models](sd-vlm_spatial_measuring_and_understanding_with_depth-encoded_vision-language_mo.md)
- [\[NeurIPS 2025\] Sherlock: Self-Correcting Reasoning in Vision-Language Models](sherlock_selfcorrecting_reasoning_in_visionlanguage_models.md)

</div>

<!-- RELATED:END -->
