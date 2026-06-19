---
title: >-
  [论文解读] GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding
description: >-
  [CVPR 2025][3D视觉][open-vocabulary affordance] 提出 GREAT 框架，通过多头 Affordance Chain-of-Thought (MHACoT) 微调 InternVL 推理交互图像中的物体几何属性和潜在交互意图，形成 affordance 知识字典，并通过跨模态自适应融合模块（CMAFM）将知识注入点云和图像特征，实现开放词汇 3D 物体 affordance 定位。同时构建最大规模 3D affordance 数据集 PIADv2（15K 图像 + 38K 点云）。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "open-vocabulary affordance"
  - "chain-of-thought"
  - "MLLM (InternVL)"
  - "点云"
  - "跨模态"
  - "PIADv2 dataset"
---

# GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding

**会议**: CVPR 2025  
**arXiv**: [2411.19626](https://arxiv.org/abs/2411.19626)  
**代码**: [项目页面](https://yawen-shao.github.io/GREAT/)  
**领域**: 3D视觉  
**关键词**: open-vocabulary affordance, chain-of-thought, MLLM (InternVL), point cloud, cross-modal fusion, PIADv2 dataset

## 一句话总结

提出 GREAT 框架，通过多头 Affordance Chain-of-Thought (MHACoT) 微调 InternVL 推理交互图像中的物体几何属性和潜在交互意图，形成 affordance 知识字典，并通过跨模态自适应融合模块（CMAFM）将知识注入点云和图像特征，实现开放词汇 3D 物体 affordance 定位。同时构建最大规模 3D affordance 数据集 PIADv2（15K 图像 + 38K 点云）。

## 研究背景与动机

**领域现状**: 3D 物体 affordance grounding 旨在定位物体上支持特定交互的区域（如杯子的把手可以"抓握"），是机器人感知与操作的桥梁。已有方法通过图像或文本引入外部交互先验来引导 3D affordance 定位。

**现有痛点**:
1. **有限语义空间**: 现有方法（IAGNet, LASO, OpenAD）依赖训练集中出现的 affordance 类别，对未见 affordance（如训练了"grasp"但测试"pour"）泛化失败
2. **忽略不变几何**: 拥有相同 affordance 的不同物体往往共享几何属性（如可倒水的物体都有壶嘴状结构），但未被利用
3. **缺乏类比推理**: 人类能从一个交互联想到其他可能交互，但模型缺乏这种 brainstorming 能力

**核心矛盾**: Affordance 的动态性和多样性使得仅靠数据驱动的模式匹配难以泛化到开放词汇场景。

**本文目标**: 在任意指令下定位 3D 物体的 affordance 区域，特别是对未见物体和未见 affordance 的泛化。

**切入角度**: 模拟人类认知——用 MLLM 的世界知识进行多步推理（CoT），挖掘几何属性和交互意图，形成可迁移的 affordance 知识。

**核心 idea**: 用 MHACoT 微调 MLLM 挖掘物体几何属性（为什么这个部位能交互）和交互意图（还能怎样交互），将推理出的知识注入点云特征实现开放词汇 affordance 定位。

## 方法详解

### 整体框架

1. **特征提取**: ResNet18 提取图像特征 $\mathbf{F}_i$，PointNet++ 提取点云特征 $\mathbf{F}_p$
2. **MHACoT 推理**: 微调 InternVL 对交互图像进行四步 CoT 推理
3. **知识编码与整合**: Roberta 编码推理文本 → cross-attention 关联物体知识和 affordance 知识
4. **CMAFM**: 将几何知识注入点云特征，将意图知识融合到图像特征
5. **解码器**: 融合特征输出逐点 affordance 热力图

### 关键设计

#### 1. Multi-Head Affordance Chain-of-Thought (MHACoT)

分两个"头"四步推理：

**Object-Head（几何推理）**:
- Prompt 1: "指出图像中物体与人交互的部位" → 定位交互区域
- Prompt 2: "从几何结构角度解释为什么这个部位能进行交互" → 提取几何属性

**Affordance-Head（意图推理/Brainstorming）**:
- Prompt 3: "描述物体与人之间的交互过程" → 细粒度交互描述
- Prompt 4: "列出两个该物体与人的其他常见交互方式" → 类比推理潜在 affordance

微调策略：仅训练 LoRA 适配器（rank=16），冻结 InternVL 主体参数，10 个 epoch，lr=4e-5。

#### 2. 知识编码与整合

- Roberta 编码 Object-Head 输出为几何知识 $\mathbf{T}_o \in \mathbb{R}^{N_o \times C}$
- Roberta 编码 Affordance-Head 输出为意图知识 $\mathbf{T}_a \in \mathbb{R}^{N_a \times C}$
- Cross-attention 关联两类知识：$\bar{\mathbf{T}}_o = f_\delta(f_m(\mathbf{T}_o, \mathbf{T}_a))$，$\bar{\mathbf{T}}_a = f_\delta(f_m(\mathbf{T}_a, \mathbf{T}_o))$

#### 3. Cross-Modal Adaptive Fusion Module (CMAFM)

将几何知识 $\bar{\mathbf{T}}_o$ 注入 PointNet++ 最深层编码器：
- Cross-attention: 点云特征为 query，几何知识为 key/value
- 特征再表示 + 池化扩展 → 拼接融合 → 1×1 卷积
- 意图知识 $\bar{\mathbf{T}}_a$ 直接 reshape 后与图像特征拼接融合

### 损失函数

$$\mathcal{L}_{total} = \mathcal{L}_{focal} + \mathcal{L}_{dice}$$

- Focal loss: 处理正负样本不均衡
- Dice loss: 优化区域级重叠
- 不使用 affordance 类别标签，直接监督逐点热力图

## 实验关键数据

### 主实验表（PIADv2 数据集）

| 方法 | Seen AUC ↑ | Unseen Obj AUC ↑ | Unseen Aff AUC ↑ |
|------|-----------|-------------------|-------------------|
| Baseline | 87.04 | 72.74 | 58.09 |
| IAGNet | 89.03 | 73.03 | 62.29 |
| LASO | 90.34 | 73.32 | 64.07 |
| **GREAT** | **91.99** | **79.57** | **69.81** |

- Unseen Affordance 分区：GREAT 的 aIOU（12.05）相比 LASO（8.37）提升 **44.0%**
- Unseen Object 分区：AUC 提升 8.5%，aIOU 提升 25.6%

### 消融表

| 配置 | Seen AUC | Unseen Obj AUC | Unseen Aff AUC |
|------|----------|----------------|----------------|
| **完整 GREAT** | **91.99** | **79.57** | **69.81** |
| ✗ AffCoT | 90.88 | 74.58 | 67.18 |
| ✗ ObjCoT | 90.13 | 75.87 | 64.69 |
| ✗ CMAFM | 89.52 | 78.42 | 63.00 |
| ✗ MLLM 微调 | 88.75 | 77.83 | 66.49 |

- AffCoT 对 Unseen Object 贡献最大（移除后 AUC 降 5 点）
- ObjCoT 对 Unseen Affordance 贡献最大（移除后 AUC 降 5 点，aIOU 降 3.2 点）
- CMAFM 对 Unseen Affordance 的 aIOU 影响最大（12.05 → 6.24）

### 关键发现

1. Seen→Unseen Object→Unseen Affordance，所有方法性能逐步下降，验证了 OV 的困难程度
2. AffCoT 的 brainstorming 使模型能类比推理出训练中未见的 affordance
3. ObjCoT 让模型关注几何属性而非物体类别，提升跨物体泛化
4. 注意力可视化：有 ObjCoT 时模型聚焦刀刃/壶嘴，无 ObjCoT 则关注整个物体
5. MLLM 微调是必要的——原始 InternVL 不理解 affordance 概念

## 亮点与洞察

1. **CoT 推理用于 affordance**: 首次将链式推理引入 3D affordance 任务，模拟人类认知过程
2. **双头设计精妙**: Object-Head 解决"为什么能交互"（几何层面），Affordance-Head 解决"还能怎样交互"（意图层面），互补互促
3. **数据集贡献大**: PIADv2（15K 图 + 38K 点云 + 24 affordance + 43 物体类别）是目前最大的 3D affordance 数据集
4. **不依赖 affordance 类别标签**: 损失函数直接监督热力图，天然适配开放词汇设置
5. **可视化解释性强**: 注意力图清晰展示了 CoT 各步骤对模型关注区域的影响

## 局限与展望

1. MLLM 推理需要为每张交互图像运行四次 prompt，效率较低
2. PointNet++ 和 ResNet18 作为特征提取器较为基础，换用更强的 backbone 可能效果更好
3. PIADv2 的图像和点云不是一对一配对（从不同实例采样），可能引入配对噪声
4. 仅对 InternVL 做了 LoRA 微调，未探索其他 MLLM（如 LLaVA、GPT-4V）
5. Prompt 设计依赖手工设计，未探索自动化 prompt optimization

## 相关工作与启发

1. **IAGNet** (Yang et al., 2023): 首次用 2D 交互图像引导 3D affordance，GREAT 的直接基线
2. **LASO** (Li et al., 2024): 用文本条件 affordance query 分割 afforded 区域
3. **InternVL** (Chen et al., 2024): GREAT 微调的基础 MLLM
4. **Chain-of-Thought** (Wei et al., 2022): CoT 推理的来源

**启发**: 用 MLLM 做"知识挖掘"而非直接做最终任务预测，是一种有效的利用大模型的方式。CoT 推理可以为计算机视觉任务提供可解释的中间推理步骤。Affordance 理解是连接感知与操作的关键环节，对机器人领域意义重大。

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**: ⭐⭐⭐⭐ — MHACoT 双头推理和几何-意图协同推理设计新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三分区评测、详细消融、注意力可视化、数据集贡献
- **论文写作**: ⭐⭐⭐⭐ — 动机清晰，图示丰富，方法描述完整
- **实用价值**: ⭐⭐⭐⭐ — 开放词汇 affordance 对机器人操作具有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions](grounding_3d_object_affordance_with_language_instructions_visual_observations_an.md)
- [\[CVPR 2025\] SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](seeground_see_and_ground_for_zero-shot_open-vocabulary_3d_visual_grounding.md)
- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)
- [\[CVPR 2025\] Masked Point-Entity Contrast for Open-Vocabulary 3D Scene Understanding](masked_point-entity_contrast_for_open-vocabulary_3d_scene_understanding.md)
- [\[CVPR 2025\] Mosaic3D: Foundation Dataset and Model for Open-Vocabulary 3D Segmentation](mosaic3d_foundation_dataset_and_model_for_open-vocabulary_3d_segmentation.md)

</div>

<!-- RELATED:END -->
