---
title: >-
  [论文解读] NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model
description: >-
  [ECCV 2024][图像生成] 提出 NL2Contact，首次利用自然语言描述来可控地建模 3D 手-物体接触图，通过分阶段扩散模型从文本生成手势姿态和接触区域，并构建了首个带有细粒度语言描述的手-物体接触数据集 ContactDescribe。 手-物体接触建模对于动画、VR/AR 和机器人抓取等应用至关重要…
tags:
  - "ECCV 2024"
  - "图像生成"
---

# NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model

**会议**: ECCV 2024  
**arXiv**: [2407.12727](https://arxiv.org/abs/2407.12727)  
**领域**: 图像生成

## 一句话总结

提出 NL2Contact，首次利用自然语言描述来可控地建模 3D 手-物体接触图，通过分阶段扩散模型从文本生成手势姿态和接触区域，并构建了首个带有细粒度语言描述的手-物体接触数据集 ContactDescribe。

## 研究背景与动机

手-物体接触建模对于动画、VR/AR 和机器人抓取等应用至关重要。现有方法（如 ContactOpt、S2Contact）依赖几何约束从点云推断接触，但存在两个核心问题：

**不可控**：无法指定或控制接触模式，生成结果往往是所有手指都接触物体的"全握"模式，与人类实际使用习惯不符（如使用剪刀时只需两个手指）

**缺乏语义**：现有意图标签（动词、物体可供性）过于粗糙，无法精确描述接触模式

本文首次提出用自然语言引导 3D 手-物体接触建模的新任务，利用语言的丰富表达能力实现更精确的接触控制。

## 方法详解

### 整体框架

NL2Contact 包含三个核心模块：

1. **Text-to-Hand-Object Fusion Module**：融合文本、手姿态和物体点云的跨模态特征
2. **Staged Latent Diffusion Module**：两阶段扩散模型，先生成手姿态，再生成接触图
3. **Contact Optimization**：利用生成的接触图迭代优化手姿态参数

输入为初始手姿态 $\widetilde{\mathcal{H}}$、物体点云 $\mathcal{O} \in \mathbb{R}^{2048 \times 3}$ 和文本描述 $\mathbb{T}$，输出手和物体上的接触概率图。

### 关键设计

**ContactDescribe 数据集构建**：

- 基于 ContactPose 数据集（2,300 个抓握实例，25 种日常物体，50 名参与者）
- 设计了多层次（粗到细）的语言描述：高层描述抓握动作、中层描述抓握类型和手指状态、底层指定接触关节位置
- 利用 ChatGPT 生成多样化的自然语言描述，每个抓握实例对应 5 句不同描述，共 11,500 条

**Text-to-Hand Fusion**：
- 使用 VPoser 编码手姿态特征 $f_\theta^g \in \mathbb{R}^{64}$，BERT 提取文本 token 嵌入 $f_{text} \in \mathbb{R}^{768 \times n}$
- 通过两个级联多头注意力模块融合文本、物体和手姿态特征

**Staged Diffusion**：
- **阶段一（Hand Pose Diffusion）**：以 Text-to-Hand 嵌入为条件，在 VPoser 潜空间中去噪生成手姿态
- **阶段二（Contact Diffusion）**：冻结阶段一参数，以生成的手姿态和文本-物体融合特征为条件，使用 PointNet 编码器将接触图编码到 $x_c^0 \in \mathbb{R}^{32 \times 32}$ 潜空间，再通过 U-Net 去噪生成接触图

### 损失函数

手姿态扩散损失：$\mathcal{L}_{\text{pose}} = \mathbb{E}[\|\epsilon - G_\theta^1(\mathbf{x}_h^t | t, f_h)\|_2^2]$

接触扩散损失：$\mathcal{L}_{\text{contact}} = \mathbb{E}[\|\epsilon - G_\theta^2(\mathbf{x}_c^t | t, f_c)\|_2^2]$

接触优化损失：$\mathcal{L}_{opt} = \|\mathcal{C}'_H - \hat{\mathcal{C}}_H\|_2^2 + \lambda_o \|\mathcal{C}'_O - \hat{\mathcal{C}}_O\|_2^2 + \lambda_{pen}\mathcal{L}_{pen}$

其中 $\lambda_O=5$, $\lambda_{pen}=3$，穿透损失 $\mathcal{L}_{pen}$ 惩罚手-物体穿透。

## 实验关键数据

### 主实验

**抓握姿态优化实验 (ContactDescribe + HO3D)**：

| 方法 | MPJPE↓ | Inter.↓ | Cover.↑ | Pr.↑ | Re.↑ |
|------|--------|---------|---------|------|------|
| Perturbed Pose | 79.9 | 8.4 | 2.3% | 9.9% | 11.5% |
| ContactNet | 45.2 | 15.6 | 18.4% | 31.6% | 47.6% |
| ContactOpt | 25.1 | 12.8 | 19.7% | 38.7% | 54.8% |
| S2Contact | 29.4 | 12.2 | 22.2% | 42.5% | 56.1% |
| **NL2Contact** | **21.7** | **7.1** | **30.5%** | **49.2%** | **59.9%** |

**抓握生成实验**：

| 方法 | Inter.↓ | Cover.↑ | Diversity↑ | SD↓ |
|------|---------|---------|------------|-----|
| GrabNet | 15.50 | 99% | 2.06 | 2.34 |
| GraspTTA | 7.37 | 76% | 1.43 | 5.34 |
| ContactGen | 9.96 | 97% | 5.04 | 2.70 |
| **NL2Contact** | **5.89** | **99%** | **5.91** | **2.31** |

### 消融实验

在 HO3D 数据集上的泛化实验表明，NL2Contact 在未见物体上仍能取得接近最优的穿透体积（4.39 vs S2Contact 的 3.52），同时 MPJPE 最低（8.4mm），证明手中心化的接触描述具有良好的泛化性。

### 关键发现

- NL2Contact 相比 ContactOpt 降低 MPJPE 4.4mm，同时**减少**穿透体积（其他方法反而增加穿透）
- 语言引导使接触生成具有可控性，能生成与描述一致的特定手指接触模式
- 在抓握生成中同时取得最低穿透、最高覆盖率和最高多样性

## 亮点与洞察

1. **首创任务定义**：自然语言引导 3D 手-物体接触建模，语言描述比动词/可供性标签提供更精确的控制
2. **LLM 辅助标注**：巧妙利用 ChatGPT 从结构化 prompt 生成多样化的自然语言描述，避免手工标注的高成本和质量不一致
3. **分阶段设计合理**：先生成手姿态再生成接触图，解耦了文本到 3D 交互的跨模态复杂性
4. **实用性强**：每个接触图生成仅需约 3 秒，训练在单 V100 上约 11 小时

## 局限性

- 依赖初始手姿态输入，对严重错误的初始姿态可能优化困难
- ContactDescribe 数据集规模有限（25 种物体），泛化到更多物体类别需要进一步验证
- 自然语言描述的多义性可能导致接触生成的不确定性

## 评分

⭐⭐⭐⭐ (4/5)

- 新颖性：★★★★★ — 首创语言引导接触建模任务 + 首个细粒度接触语言数据集
- 技术：★★★★ — 分阶段扩散设计合理，跨模态融合有效
- 实验：★★★★ — 多任务验证（优化+生成），多数据集评估
- 写作：★★★★ — 结构清晰，图示直观

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation](neusdfusion_a_spatial-aware_generative_model_for_3d_shape_completion_reconstruct.md)
- [\[ECCV 2024\] MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion](macdiff_unified_skeleton_modeling_with_masked_conditional_diffusion.md)
- [\[ECCV 2024\] TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)
- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)

</div>

<!-- RELATED:END -->
