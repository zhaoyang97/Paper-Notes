---
title: >-
  [论文解读] Describe, Adapt and Combine: Empowering CLIP Encoders for Open-set 3D Object Retrieval
description: >-
  [ICCV 2025][3D视觉][开放集3D检索] 提出 DAC 框架，通过 "描述-适配-组合" 三步策略协同 CLIP 与多模态大语言模型 (MLLM)，仅使用多视图图像即可在开放集 3D 物体检索任务上大幅超越此前使用全模态（点云+体素+图像）的 SOTA 方法，平均 mAP 提升超过 +10%。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "开放集3D检索"
  - "CLIP"
  - "MLLM"
  - "LoRA"
  - "多视图"
---

# Describe, Adapt and Combine: Empowering CLIP Encoders for Open-set 3D Object Retrieval

**会议**: ICCV 2025  
**arXiv**: [2507.21489](https://arxiv.org/abs/2507.21489)  
**代码**: [GitHub](https://github.com/wangzhichuan123/DAC)  
**领域**: 3D视觉  
**关键词**: 开放集3D检索, CLIP, MLLM, LoRA, 多视图

## 一句话总结

提出 DAC 框架，通过 "描述-适配-组合" 三步策略协同 CLIP 与多模态大语言模型 (MLLM)，仅使用多视图图像即可在开放集 3D 物体检索任务上大幅超越此前使用全模态（点云+体素+图像）的 SOTA 方法，平均 mAP 提升超过 +10%。

## 研究背景与动机

开放集 3D 物体检索 (Open-set 3DOR) 要求从仓库中检索训练时 **未见过的类别** 的 3D 物体，面临两大挑战：

**域外泛化困难**: 现有方法假设训练与测试共享相同类别/域，在开放集场景中性能急剧下降

**3D 数据稀缺导致过拟合**: 训练数据有限，模型容易对已知类别过拟合，无法泛化到未见类别

此前 SOTA 方法 HGM2R 需要使用所有模态（点云、体素、多视图图像）并引入测试数据参与训练，复杂且不实际。

**核心动机**: CLIP 经过大规模图文对比预训练，天然具备泛化特征表示能力。作者发现：
- 一个简单的 "多视图 CLIP" 基线（直接聚合 CLIP 视觉编码器的逐视图特征）已能取得不错效果
- 但人类面对未知物体时，除了视觉分析，还会通过语言描述进行推理（如 "像马但有黑白条纹"）
- 因此引入 MLLM 提供互补的文本线索，弥补纯视觉的不足

## 方法详解

### 整体框架

DAC = Describe + Adapt + Combine，三步流程：

1. **Describe**: 用 MLLM（InternVL）生成文本描述
2. **Adapt**: 用 AB-LoRA 微调 CLIP 适配多视图投影图像
3. **Combine**: 融合视觉与文本特征用于检索

### 关键设计

1. **Describe — MLLM 双重用途**:

    - **训练时**: 用 MLLM 为每个已知类别生成丰富描述（prompt: "Describe in one sentence what [cls] should look like"），替代简单的 "a photo of [cls]" 模板，更好对齐 CLIP 的对比学习目标
    - **推理时**: 将多视图图像输入 MLLM，生成外观和语义描述（prompt: "There are images of an object from different angles. Describe this object in one sentence."），为未知类别物体提供 out-of-box 文本知识
    - 任何现成 MLLM 均可使用，主要采用 InternVL-4B

2. **Adapt — Additive-Bias LoRA (AB-LoRA)**:

    - CLIP 预训练于自然图像，与多视图投影图像存在域差距，需要微调适配
    - 标准 LoRA 的权重更新 $\Delta\mathbf{W}\mathbf{z}$ 直接累积来自已知类别的输入 $\mathbf{z}$，容易过拟合
    - **核心创新**: 在 LoRA 中添加可学习偏置项 $\mathbf{\Phi}$:
    $\mathbf{o} = \mathbf{W}_o\mathbf{z} + \gamma\mathbf{BA}\mathbf{z} + \mathbf{\Phi}$
    - 偏置项打破了权重更新与输入的紧密耦合，起到正则化作用，减缓过拟合
    - $\mathbf{A}$ 用标准正态初始化，$\mathbf{B}$ 和 $\mathbf{\Phi}$ 初始化为零，确保起始不扰动原始权重
    - 应用于 CLIP 视觉和文本编码器的 self-attention（$W_q, W_k, W_v$）

3. **训练目标**:

    - 类别描述经 CLIP 文本编码器生成分类权重 $\mathbf{c}_i = \mathcal{T}(t_i)$
    - 多视图图像经 CLIP 视觉编码器 + 均值池化得全局特征 $\mathbf{g}_k = \frac{1}{M}\sum\mathbf{f}_{k,m}$
    - 交叉熵损失:
    $\mathcal{L}_{CE} = -\frac{1}{N_t}\sum_{k=1}^{N_t}\log\frac{\exp(\mathbf{g}_k \cdot \mathbf{c}_y / \tau)}{\sum_{i=1}^{L}\exp(\mathbf{g}_k \cdot \mathbf{c}_i / \tau)}$

4. **Combine — 特征融合**:

    - 将适配后的视觉全局特征 $\mathbf{g}$ 与文本特征 $\mathbf{f}_t$ 加权融合：
    $\mathbf{h} = \tanh(\mathbf{g} + \alpha\mathbf{f}_t)$
    - 使用 cosine 相似度进行检索
    - 简单的加法融合效果优于拼接（使 R的融合优于拼接 (+7.1% mAP)

### 损失函数 / 训练策略

- 训练损失: 交叉熵对比损失，对齐多视图视觉特征与类别描述文本特征
- SGD 优化器，lr=2e-4，batch size=4，cosine scheduler，30 epochs
- LoRA rank=8，dropout=0.25
- 2× NVIDIA RTX 4090 训练

## 实验关键数据

### 主实验

在四个开放集 3DOR 基准上的表现（Open-set Setup, mAP%）：

| 方法 | 模态 | OS-ESB-core | OS-NTU-core | OS-MN40-core | OS-ABO-core |
|------|------|-------------|-------------|--------------|-------------|
| HGM2R | P+I+V | 51.74 | 44.88 | 64.20 | 63.39 |
| DAC (B/32) | **I 仅** | **58.70** | **59.21** | 62.40 | **66.10** |
| DAC (L/14) | **I 仅** | **57.80** | **65.83** | **68.98** | **70.74** |

- 仅用多视图图像，DAC (L/14) 在四个数据集上平均 mAP 超越 HGM2R 约 **+10%**
- DAC 不使用测试数据训练，不需要点云和体素，更简单且实用

### 消融实验

AB-LoRA 效果（OS-MN40-core, ViT-B/32）:

| 配置 | mAP↑ | NDCG↑ | ANMRR↓ |
|------|------|-------|--------|
| 无 LoRA | 55.39 | 68.08 | 45.96 |
| 标准 LoRA | 59.85 | 70.25 | 41.75 |
| AB-LoRA | **62.40** | **72.63** | **39.82** |

各模块贡献（ViT-B/32, 格式 mAP/NDCG/ANMRR）:

| InternVL | AB-LoRA | OS-ESB-core | OS-MN40-core |
|----------|---------|-------------|--------------|
| ✗ | ✗ | 53.93/23.00/49.70 | 49.60/65.71/50.88 |
| ✓ | ✗ | 56.16/23.58/48.39 | 55.39/68.08/45.96 |
| ✗ | ✓ | 57.45/23.96/47.13 | 59.35/71.89/42.72 |
| ✓ | ✓ | **58.70/24.27/45.67** | **62.40/72.63/39.82** |

### 关键发现

- **偏置的惊人效果**: 仅在 LoRA 中加一个简单的偏置项即可提升 +2.55% mAP，有效缓解过拟合
- 直接用 InternVL 的嵌入做检索仅 38.20% mAP，远低于 CLIP 的 53.93%，说明 CLIP 的判别性特征更适合检索
- 加法融合优于拼接融合：加法使两种模态在同一特征空间中直接互补

## 亮点与洞察

- **极简但有效**: 仅用多视图图像+现成 MLLM，无需点云/体素，无需测试数据参与训练
- **MLLM 的创造性双用**: 训练时增强文本监督，推理时提供外部语义线索，设计优雅
- **AB-LoRA 的通用价值**: 在 LoRA 中加偏置来缓解小数据过拟合，思路可迁移到其他领域

## 局限与展望

- 对高属性机械零件（OS-ESB-core）InternVL 描述能力有限，改进空间大
- 灰度投影图像丢失了颜色信息，可能影响颜色敏感类别的检索
- 未探索更强的多视图聚合策略（如 attention pooling）

## 相关工作与启发

- **MV-CLIP**: 简单的多视图 CLIP 基线，但需要类别信息做视图选择，不适用于开放集
- **ULIP-2/OpenShape**: 对齐语言-图像-3D 嵌入，但受限于 3D 数据量
- **CoOp/CLIP-Adapter**: 轻量适配策略，DAC 的 AB-LoRA 是更优的替代

## 评分

- 新颖性: ⭐⭐⭐⭐ (MLLM+CLIP 协同 + AB-LoRA)
- 技术深度: ⭐⭐⭐ (方法简洁易懂)
- 实验充分度: ⭐⭐⭐⭐⭐ (4个数据集、12个对比方法、充分消融)
- 实用价值: ⭐⭐⭐⭐ (仅需图像输入，部署简单)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DINO Eats CLIP: Adapting Beyond Knowns for Open-set 3D Object Retrieval](../../CVPR2026/3d_vision/dino_eats_clip_adapting_beyond_knowns_for_open-set_3d_object_retrieval.md)
- [\[ICCV 2025\] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)
- [\[ICCV 2025\] 3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation](3d_gaussian_map_with_openset_semantic_grouping_for_visionlan.md)
- [\[ICCV 2025\] LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities](llava-3d_a_simple_yet_effective_pathway_to_empowering_lmms_with_3d_capabilities.md)
- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](open-vocabulary_octree-graph_for_3d_scene_understanding.md)

</div>

<!-- RELATED:END -->
