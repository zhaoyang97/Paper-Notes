---
title: >-
  [论文解读] FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving
description: >-
  [NeurIPS 2025 Spotlight][自动驾驶][视觉CoT] FSDrive让VLA"用视觉思考"——先作为世界模型生成融合了未来车道线、3D检测框和场景预测的统一视觉CoT帧，再作为逆动力学模型基于当前观测和视觉CoT进行轨迹规划，用极少数据(约0.3%)即可激活MLLM的视觉生成能力。
tags:
  - "NeurIPS 2025 Spotlight"
  - "自动驾驶"
  - "视觉CoT"
  - "轨迹规划"
  - "世界模型"
  - "VLA"
  - "未来帧预测"
---

# FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.17685](https://arxiv.org/abs/2505.17685)  
**代码**: [GitHub](https://github.com/MIV-XJTU/FSDrive)  
**领域**: 自动驾驶 / VLA  
**关键词**: 视觉CoT, 轨迹规划, 世界模型, VLA, 未来帧预测

## 一句话总结

FSDrive让VLA"用视觉思考"——先作为世界模型生成融合了未来车道线、3D检测框和场景预测的统一视觉CoT帧，再作为逆动力学模型基于当前观测和视觉CoT进行轨迹规划，用极少数据(约0.3%)即可激活MLLM的视觉生成能力。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：现有VLA自动驾驶模型大多使用文本CoT（如场景描述、坐标文字）作为推理中间步骤，但存在关键问题：

1. **模态鸿沟**：将连续视觉信息压缩为离散文本是有损的，丢失精细时空关系
2. **语义断裂**：文本描述中的坐标和场景关系与原始视觉输入形成模态断层
3. **信息不足**：文本CoT难以表达动态场景的时间演变和空间结构

人类驾驶员的思维更接近"在脑中模拟未来场景"而非"用语言描述",因此应让模型也用视觉方式"思考"。

## 方法详解

### 整体框架

两阶段训练：
- **预训练阶段**：统一视觉理解(VQA) + 视觉生成(未来帧预测)，渐进式从结构先验到完整场景
- **微调阶段**：场景理解 + 基于视觉CoT的轨迹规划

### 关键设计

1. **视觉时空CoT**:
    - 功能：生成一张融合了多种未来信息的统一图像帧作为推理中间步骤
    - 核心思路：将未来车道线(红色标注)、3D检测框、和预测场景整合到一张图中；车道线表示空间可行驶区域，检测框表示关键物体运动，场景图表示时间演变
    - 设计动机：统一为图像格式避免跨模态转换的语义损失，同时编码空间(车道线+检测框)和时间(场景演变)两个维度的未来信息

2. **统一预训练范式**:
    - 功能：在现有MLLM基础上同时激活视觉理解和视觉生成能力
    - 核心思路：将VQ-VAE的图像码本扩展到MLLM的文本词表中，使模型可以自回归预测视觉token
    - 设计动机：仅需约0.3%的数据量（相比从头训练的方法），不修改MLLM架构，直接激活潜在的视觉生成能力
    - 渐进式生成：先生成车道线token $Q_l$（静态物理约束）→ 3D检测框token $Q_d$（动态物理约束）→ 完整未来帧token $Q_f$

### 损失函数 / 训练策略

- **预训练**：VQA交叉熵 + 视觉token自回归预测损失联合训练
- **微调**：DriveLM GVQA场景理解 + nuScenes轨迹规划，使用统一视觉CoT
- 从Qwen2-VL-2B初始化，预训练32 epochs，微调12 epochs
- VQ-VAE使用MoVQGAN编解码器

## 实验关键数据

### 主实验（表格）

nuScenes轨迹规划（ST-P3 metrics）:

| 方法 | LLM | L2 (1s) ↓ | L2 (2s) ↓ | L2 (3s) ↓ | Col. (1s) ↓ | Col. (2s) ↓ | Col. (3s) ↓ |
|------|-----|-----------|-----------|-----------|-------------|-------------|-------------|
| VAD | - | 0.54 | 1.15 | 1.98 | 0.04 | 0.39 | 1.17 |
| OmniDrive | ✓ | 0.51 | 1.04 | 1.70 | - | - | - |
| **FSDrive** | ✓ | **较优** | **较优** | **较优** | **更低** | **更低** | **更低** |

FSDrive在L2位移误差和碰撞率上均优于基线，同时在DriveLM场景理解和未来帧FID上达到竞争性能。

### 消融实验

- 视觉CoT vs 文本CoT vs 无CoT：视觉CoT显著优于文本CoT，后者又优于无CoT
- 渐进式生成 vs 直接生成：渐进式生成的未来帧更符合物理规律
- VQA预训练的必要性：去除VQA预训练导致场景理解任务大幅下降
- MoVQGAN码本大小：较大码本提升生成质量但增加推理开销

### 关键发现

- 仅用Qwen2-VL-2B即可实现有竞争力的轨迹规划，说明视觉CoT的推理增益
- 视觉CoT比文本CoT提供更丰富的时空信息，直接降低碰撞率
- 渐进式生成是关键——直接生成完整未来帧容易违反物理规律
- 用极少数据量即可激活MLLM的视觉生成能力，不需要从头训练

## 亮点与洞察

- **范式创新**：让VLA"用视觉思考"而非"用文字思考"，更贴近人类驾驶认知
- **成本极低**：仅需0.3%数据量激活视觉生成，无需修改MLLM架构
- **渐进式物理先验**：先推理车道线/检测框作为物理约束，再生成完整场景，确保物理可行性
- 将世界模型（未来预测）和逆动力学模型（轨迹规划）统一在同一个VLA中

## 局限与展望

- 仅在nuScenes上验证，该数据集场景相对单一
- VQ-VAE的视觉token缺乏语义信息，可能影响理解任务
- 当前仅预测1帧未来，未探索多帧或长时间跨度的视觉推理
- 2B模型的推理速度是否满足实时性要求未验证

## 相关工作与启发

- 与EMMA（纯文本CoT）和CoT-VLA（图文混合CoT）的区别：FSDrive用纯视觉统一CoT
- 借鉴了视觉提示工程（如红圈引导注意力）的思想来标注车道线和检测框
- 世界模型(生成)+逆动力学(规划)的双用途VLA为自动驾驶VLM提供了新范式

## 评分

⭐⭐⭐⭐ — 视觉CoT理念新颖且合理，渐进式物理先验设计巧妙，低成本激活视觉生成的方案实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] HybridDriveVLA: Vision-Language-Action Model with Visual CoT reasoning and ToT Evaluation for Autonomous Driving](../../CVPR2026/autonomous_driving/hybriddrivevla_vision-language-action_model_with_visual_cot_reasoning.md)
- [\[NeurIPS 2025\] How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning](how_different_from_the_past_spatio-temporal_time_series_forecasting_with_self-su.md)
- [\[ICCV 2025\] SparseLaneSTP: Leveraging Spatio-Temporal Priors with Sparse Transformers for 3D Lane Detection](../../ICCV2025/autonomous_driving/sparselanestp_leveraging_spatio-temporal_priors_with_sparse_transformers_for_3d_.md)
- [\[AAAI 2026\] Rethinking the Spatio-Temporal Alignment of End-to-End 3D Perception](../../AAAI2026/autonomous_driving/rethinking_the_spatio-temporal_alignment_of_end-to-end_3d_perception.md)
- [\[AAAI 2026\] RAST: A Retrieval Augmented Spatio-Temporal Framework for Traffic Prediction](../../AAAI2026/autonomous_driving/rast_a_retrieval_augmented_spatio-temporal_framework_for_traffic_prediction.md)

</div>

<!-- RELATED:END -->
