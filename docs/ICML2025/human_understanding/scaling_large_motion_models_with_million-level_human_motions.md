---
title: >-
  [论文解读] Scaling Large Motion Models with Million-Level Human Motions
description: >-
  [ICML 2025][人体理解] 本文提出 MotionLib（首个百万级运动数据集，120 万条序列）、MotionBook（无损特征 + 2D 无查找运动分词器）和 Being-M0（大型运动模型），首次在运动生成领域展示了数据和模型规模的 scaling law。 文本到运动生成（T2M）是一个新兴领域…
tags:
  - "ICML 2025"
  - "人体理解"
---

# Scaling Large Motion Models with Million-Level Human Motions

**会议**: ICML 2025  
**arXiv**: [2410.03311](https://arxiv.org/abs/2410.03311)  
**领域**: 人体理解  

## 一句话总结

本文提出 MotionLib（首个百万级运动数据集，120 万条序列）、MotionBook（无损特征 + 2D 无查找运动分词器）和 Being-M0（大型运动模型），首次在运动生成领域展示了数据和模型规模的 scaling law。

## 研究背景与动机

文本到运动生成（T2M）是一个新兴领域，在游戏、电影和机器人等方面有广泛应用。然而，当前方法受限于数据规模：

- **数据量差距巨大**：最大的运动数据集 Motion-X 仅有 ~8 万序列，而视觉文本数据（如 ImageNet）量级远大于此
- **现有 VQ 分词器的缺陷**：
  1. **信息丢失**：将复杂的运动状态（包含关节位置、速度、接地接触等）压缩为单个 1D 嵌入
  2. **码本容量有限**：小码本限制了可生成运动的多样性
- **特征表示问题**：常用的 H3D 格式特征省略了原始旋转信息，恢复需要耗时方法

## 方法详解

### MotionLib 数据集

首个百万级运动数据集，包含 **120 万条运动序列**和 **248 万条文本描述**：

| 数据集 | 序列数 | 文本数 | 小时数 | 文本类型 |
|---|---|---|---|---|
| HumanML3D | 29.2K | 89K | 28.6 | body |
| Motion-X | 81.1K | 142K | 144.2 | body |
| **MotionLib** | **1.21M** | **2.48M** | **1456.4** | **层次化** |

**构建流程：**
1. 从公开数据集和 YouTube 收集 2000 万+视频
2. 使用 WHAM 在世界坐标系中提取 SMPL 参数
3. 生成**层次化文本注释**：部位级描述（如左臂）+ 全身级描述（1-3 句）
4. 使用 RL 策略 $\pi_{\text{refine}}$ 精炼原始运动以遵循物理定律

### MotionBook：高效运动编码

#### 无损运动特征（SMPL-D135）

每帧编码为 $m \in \mathbb{R}^{135}$：
- **根节点（9D）**：6D 旋转 $\mathbf{r}_{rot} \in \mathbb{R}^6$，2D XZ 平面速度，1D 高度
- **身体关节（126D）**：21 个关键关节的 6D 旋转向量 $\mathbf{j}^r \in \mathbb{R}^{21 \times 6}$

相比 H3D 格式（263D），SMPL-D135 更紧凑且保留了完整的旋转信息。

#### 2D 无查找运动量化（2D-LFQ）

**核心创新：**
1. 将运动序列视为单通道图像 $\mathcal{M} \in \mathbb{R}^{T \times D \times 1}$
2. 将特征维度分为 $P$ 个组件，分别编码（如根方向、关节旋转、脚接触等）
3. 编码器输出为 $\mathbb{E}(\mathcal{M}) \in \mathbb{R}^{\lfloor T/\alpha \rfloor \times P \times d}$

**无查找量化**：将码本替换为整数集 $\mathbb{C} = \times_{i=1}^d C_i$，其中 $C_i = \{-1, 1\}$：

$$Q(z_i) = -\mathbb{1}\{z_i \leq 0\} + \mathbb{1}\{z_i > 0\}$$

token 索引计算为 $Index(z) = \sum_{i=1}^d 2^{i-1}\mathbb{1}\{z_i > 0\}$。这使码本大小扩展至少**两个数量级**（从 ~512 到 ~65K+），同时避免码本崩溃。

### Being-M0：大型运动模型

基于预训练 LLM 的自回归运动生成模型：

**两阶段训练：**
1. **运动-文本对齐**：在整个 MotionLib 上预训练，学习基本的运动-文本关联
2. **运动指令微调**：使用 250+ 指令模板和 90 万条指令数据进行微调

**训练损失：**

$$\mathcal{L}(\Theta) = -\sum_{j=1}^{L}\log P_\Theta(y_j | desc, \hat{y}_{1:j-1})$$

## 实验

### Scaling Law 实验

| 解码器 | 指令数 | 参数量 | MotionLib-eval FID ↓ |
|---|---|---|---|
| GPT-2 | 0.02M | 355M | 30.612 |
| GPT-2 | 1.2M | 355M | 6.936 |
| LLaMA-3 | 0.02M | 8B | 29.257 |
| LLaMA-3 | 0.08M | 8B | 21.295 |
| LLaMA-3 | 0.5M | 8B | 8.973 |
| LLaMA-3 | 1.2M | 8B | **6.029** |
| LLaMA-2 | 1.2M | 13B | 6.221 |

**关键发现**：
- 数据量从 0.02M 增加到 1.2M，FID 从 ~30 下降到 ~6，scaling 效果显著
- 更大模型带来一致的改进（LLaMA-3 8B 优于 GPT-2 355M），但**数据规模的影响远大于模型规模**

### 运动分词器对比

2D-LFQ 在保持 MPJPE 误差较低的同时，将码本大小从传统 VQ 的 512 扩展到 65536+，码本利用率接近 100%。

### 泛化能力

现有模型在 MotionLib 上的域外概念测试中挣扎，而 Being-M0 在未见过的运动类别上展现了显著更好的泛化能力。

## 亮点

- **首个百万级运动数据集**：120 万序列、248 万文本注释，比现有数据集大 15 倍以上
- **首次展示运动生成的 scaling law**：数据和模型规模均能有效降低生成误差
- **创新的 2D 无查找量化**：将码本大小扩展两个数量级，从根本上解决了码本容量限制
- **无损特征设计**：SMPL-D135 比 H3D 更紧凑且保留完整旋转信息
- **层次化文本注释**：部位级 + 全身级描述提供了前所未有的文本细节

## 局限性

- 万级视频中提取的运动质量参差不齐，需要额外的 RL 精炼策略
- 部分剧烈运动在 RL 精炼中仍存在滑动问题
- 单人场景为主，多人场景的支持仍有限
- 从视频中提取的 SMPL 参数精度有限（相比 MoCap 数据）
- 大型 LLM 骨干网络的推理成本较高

## 评分

⭐⭐⭐⭐⭐ (5/5)

这是运动生成领域的里程碑式工作。百万级数据集的构建、首次展示的 scaling law 以及创新的 2D-LFQ 分词器都是重要贡献。论文工程量巨大，实验系统性强，为后续研究奠定了坚实基础。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] LLaVA-ReID: Selective Multi-Image Questioner for Interactive Person Re-Identification](llava-reid_selective_multi-image_questioner_for_interactive_person_re-identifica.md)
- [\[ICCV 2025\] SemTalk: Holistic Co-speech Motion Generation with Frame-level Semantic Emphasis](../../ICCV2025/human_understanding/semtalk_holistic_co-speech_motion_generation_with_frame-level_semantic_emphasis.md)
- [\[CVPR 2026\] LLaMo: Scaling Pretrained Language Models for Unified Motion Understanding and Generation with Continuous Autoregressive Tokens](../../CVPR2026/human_understanding/llamo_scaling_pretrained_language_models_for_unified_motion_understanding_and_ge.md)
- [\[CVPR 2026\] Unlocking Motion from Large Vision Models with a Semantic and Kinematic Duality for Gait Recognition](../../CVPR2026/human_understanding/unlocking_motion_from_large_vision_models_with_a_semantic_and_kinematic_duality_.md)
- [\[ICCV 2025\] LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition](../../ICCV2025/human_understanding/lvface_progressive_cluster_optimization_for_large_vision_models_in_face_recognit.md)

</div>

<!-- RELATED:END -->
