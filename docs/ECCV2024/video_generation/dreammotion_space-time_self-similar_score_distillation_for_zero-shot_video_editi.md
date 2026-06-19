---
title: >-
  [论文解读] DreamMotion: Space-Time Self-Similar Score Distillation for Zero-Shot Video Editing
description: >-
  [ECCV 2024][视频生成][Score Distillation] 提出基于分数蒸馏（Score Distillation）的零样本视频编辑框架DreamMotion，通过时空自相似性正则化在注入目标外观的同时保持原始视频的结构和运动完整性，适用于级联和非级联视频扩散模型。 现有的基于扩散模型的视频编辑方法面临一个核…
tags:
  - "ECCV 2024"
  - "视频生成"
  - "Score Distillation"
  - "视频编辑"
  - "自相似性"
  - "扩散模型"
  - "零样本"
---

# DreamMotion: Space-Time Self-Similar Score Distillation for Zero-Shot Video Editing

**会议**: ECCV 2024  
**arXiv**: [2403.12002](https://arxiv.org/abs/2403.12002)  
**代码**: 有 (项目页面: [https://hyeonho99.github.io/dreammotion](https://hyeonho99.github.io/dreammotion))  
**领域**: 扩散模型 / 视频编辑  
**关键词**: Score Distillation, 视频编辑, 自相似性, 扩散模型, 零样本

## 一句话总结

提出基于分数蒸馏（Score Distillation）的零样本视频编辑框架DreamMotion，通过时空自相似性正则化在注入目标外观的同时保持原始视频的结构和运动完整性，适用于级联和非级联视频扩散模型。

## 研究背景与动机

现有的基于扩散模型的视频编辑方法面临一个核心挑战：如何在编辑过程中建立并保持真实世界的运动。主要问题包括：

**基于T2I的方法局限性**：现有利用T2I扩散模型的方法通常通过膨胀注意力层（inflated attention）来处理多帧，但这种隐式的运动保持策略远不足以实现平滑、完整的运动。常见的补充方案（如注入self-attention map、使用ControlNet等结构引导）增加了系统复杂度。

**T2V模型的不足**：即使使用预训练的T2V模型，公开可用的模型缺乏足够丰富的时间先验来准确描绘真实世界运动（如Fig. 2所示）。因此很多方法需要在输入视频上微调模型权重（one-shot方式），增加了额外开销。

**标准逆扩散过程的固有困难**：无论使用T2I还是T2V，从标准高斯噪声或反转潜在表示开始的传统逆扩散过程，天然难以重新编程复杂的真实世界运动。

**核心洞察**：与其从噪声开始尝试重建运动（逆扩散），不如从已经具有自然运动的视频出发，逐步修改外观——这正是分数蒸馏（SDS）的优势所在。但直接使用视频分数蒸馏会累积结构误差导致运动偏离，因此需要自相似性正则化来约束。

## 方法详解

### 整体框架

DreamMotion的优化策略由三个部分组成：
- 将目标视频变量 $\boldsymbol{x}_0^{1:N}(\theta)$ 初始化为原始视频 $\hat{\boldsymbol{x}}^{1:N}$
- 通过 $\mathcal{L}_{\text{V-DDS}}$ 注入目标文本描述的外观
- 通过 $\mathcal{L}_{\text{S-SSM}}$ 保持空间结构一致性
- 通过 $\mathcal{L}_{\text{T-SSM}}$ 保证时间平滑性

三个损失共享同一噪声 $\epsilon$ 和时间步 $t$，只需一次扩散前向和反向步骤即可高效完成。

### 关键设计

1. **视频Delta去噪分数（V-DDS）**：将图像级别的DDS机制扩展到视频域，注入目标外观。

   DDS通过引入参考文本-图像对消除SDS中的噪声方向：

    $\mathcal{L}_{\text{V-DDS}}(\theta;y) = \|\boldsymbol{\epsilon}_\phi^w(\boldsymbol{x}_t^{1:N}(\theta), t, y) - \boldsymbol{\epsilon}_\phi^w(\hat{\boldsymbol{x}}_t^{1:N}, t, \hat{y})\|_2^2$

   同时使用二值掩码 $m^{1:N}$ 过滤梯度（$\nabla_\theta \mathcal{L}_{\text{V-DDS}} \odot m^{1:N}$），确保非编辑区域不受影响，避免模糊和过饱和。掩码由现成目标检测模型生成的边界框得到。

   **动机**：V-DDS能有效注入目标外观，但会累积结构误差导致运动偏离，因此需要下面两个自相似性正则化。

2. **空间自相似性匹配（S-SSM）**：保持每一帧的结构完整性。

   核心思路：自相似性描述符关注的是物体外观与周围环境的**相对关系**而非绝对外观，因此在外观变化后仍能保持结构信息。

   具体做法：向原始和编辑视频添加相同噪声，通过视频扩散U-Net提取attention key特征 $K(\boldsymbol{x}_t^{1:N}), K(\hat{\boldsymbol{x}}_t^{1:N}) \in \mathbb{R}^{N \times (H \times W) \times C}$，计算每帧的空间自相似图：

    $SS_{i,j}^n(\boldsymbol{x}_t^{1:N}) = \cos(K_i^n(x_t^{1:N}), K_j^n(x_t^{1:N}))$

    $\mathcal{L}_{\text{S-SSM}} = \frac{1}{N}\sum_{n=1}^{N}\|SS^n(\boldsymbol{x}_t^{1:N}) - SS^n(\hat{\boldsymbol{x}}_t^{1:N})\|_2^2$

   **设计动机**：自相似性对局部纹理模式具有鲁棒性，同时能保留全局布局和物体形状。这是首次将深层扩散特征的自相似性应用于视频编辑中的结构保持。

3. **时间自相似性匹配（T-SSM）**：消除帧间伪影，确保时间平滑。

   S-SSM是逐帧独立优化，不考虑帧间相关性，可能导致局部扭曲和闪烁。T-SSM通过空间边际均值将key特征压缩为全局描述符：

    $M[K(\boldsymbol{x}_t^{1:N})] = \frac{1}{H \cdot W}\sum_{i=1}^{H \cdot W}K_i(\boldsymbol{x}_t^{1:N}) \in \mathbb{R}^{N \times C}$

   然后在时间维度计算自相似性：

    $TS_{i,j}(\boldsymbol{x}_t^{1:N}) = \cos(M_i[K(\boldsymbol{x}_t^{1:N})], M_j[K(\boldsymbol{x}_t^{1:N})])$

    $\mathcal{L}_{\text{T-SSM}} = \|TS(\boldsymbol{x}_t^{1:N}) - TS(\hat{\boldsymbol{x}}_t^{1:N})\|_2^2$

   **设计动机**：空间边际均值作为一阶统计量，能在压缩空间信息的同时保留关键空间细节，已被证明是有效的全局描述符。

### 损失函数 / 训练策略

- **总损失**：$\mathcal{L} = \mathcal{L}_{\text{V-DDS}} + \lambda_s \mathcal{L}_{\text{S-SSM}} + \lambda_t \mathcal{L}_{\text{T-SSM}}$
- **优化器**：SGD，学习率0.4，优化200步
- **效率**：8帧视频约2分钟，16帧约4分钟（单A100 GPU）
- **级联扩展**：对于级联视频扩散模型（如Show-1），仅在关键帧生成阶段应用优化，随后通过时间插值和空间超分辨率模块扩展
- **模型无关**：适用于像素空间（Show-1）和潜在空间（ZeroScope）的视频扩散模型

## 实验关键数据

### 主实验（非级联框架 - ZeroScope）

| 方法 | Text-Align↑ | Frame-Con↑ | Motion-Fidelity↑ | Frame-LPIPS↓ | Edit-Acc↑ | SM-Preserve↑ |
|------|------------|------------|-------------------|--------------|-----------|---------------|
| Tune-A-Video | 0.8177 | 0.9218 | 0.6947 | 0.4172 | 3.52 | 2.89 |
| ControlVideo | 0.7850 | 0.9678 | - | 0.3763 | 2.74 | 2.03 |
| Gen-1 | 0.8192 | 0.9704 | - | - | 3.31 | 2.95 |
| Tokenflow | 0.7813 | 0.9576 | 0.9184 | 0.3427 | 3.63 | 3.92 |
| **DreamMotion** | **0.8209** | **0.9726** | **0.9259** | **0.3042** | **4.14** | **4.33** |

### 消融实验

| 配置 | Text-Align | Frame-Con | Motion-Fidelity | Frame-LPIPS | 说明 |
|------|-----------|-----------|-----------------|-------------|------|
| 去除 S-SSM + T-SSM | 0.8202 | 0.9648 | 0.8426 | 0.3247 | 运动保真度大幅下降 |
| 去除 T-SSM | 0.8114 | 0.9567 | 0.9011 | 0.3186 | 帧间出现局部扭曲 |
| 去除 masks | 0.8180 | 0.9695 | 0.8653 | 0.3416 | 非编辑区域受干扰 |
| **完整模型** | **0.8209** | **0.9726** | **0.9259** | **0.3042** | 所有指标最优 |

### 关键发现

- 用户研究（36人）在编辑准确度、帧一致性、结构与运动保持三方面均超过所有基线
- 级联框架（Show-1）上，结构和运动保持（SM-Preserve）得分4.30 vs VMC的3.35
- 空间自相似性解决结构偏离，时间自相似性消除由空间约束带来的伪影，两者协同互补

## 亮点与洞察

1. **范式创新**：避开"从噪声生成运动"这一根本困难，转而"从已有运动出发修改外观"，是对视频编辑问题的优雅重新定义
2. **自相似性的巧妙应用**：利用扩散特征的自相似性同时解决结构保持和时间平滑两个问题
3. **模型无关**：适用于级联和非级联两种框架，无需微调模型权重，真正的零样本方法
4. **高效计算**：三个损失共享一次扩散前向/反向计算，计算开销可控

## 局限与展望

- 框架设计保持原始结构，**不适用于需要显著结构变化的编辑任务**（如改变物体姿态）
- 依赖现成目标检测器提供编辑区域掩码
- 自相似性的计算复杂度为 $O((H \times W)^2)$，对高分辨率视频可能成为瓶颈
- 可探索将方法扩展到更多帧（当前最多16帧）

## 相关工作与启发

- **DDS**（Delta Denoising Score）：通过参考文本-图像对消除SDS噪声方向，是本文的核心基础
- **Tokenflow**：通过扩散内部特征一致性实现时间一致编辑，是最强基线
- **自相似性**：从传统视觉模式匹配到DINO ViT特征，本文首次将其引入扩散特征空间
- 启发：分数蒸馏 + 结构约束的范式可能适用于其他需要保持结构的生成编辑任务（如3D编辑、音频编辑）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将SDS引入视频编辑并提出时空自相似性正则化，范式创新有价值
- **实验充分度**: ⭐⭐⭐⭐ — 自动指标+用户研究，两种框架验证，消融充分
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机到方法逻辑连贯
- **价值**: ⭐⭐⭐⭐ — 零样本+模型无关的视频编辑框架，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Scaling Zero-Shot Reference-to-Video Generation](../../CVPR2026/video_generation/scaling_zero-shot_reference-to-video_generation.md)
- [\[CVPR 2026\] StoryTailor: A Zero-Shot Pipeline for Action-Rich Multi-Subject Visual Narratives](../../CVPR2026/video_generation/storytailora_zero-shot_pipeline_for_action-rich_multi-subject_visual_narratives.md)
- [\[ICML 2026\] SGMD: Score Gradient Matching Distillation for Few-Step Video Diffusion](../../ICML2026/video_generation/sgmd_score_gradient_matching_distillation_for_few-step_video_diffusion_distillat.md)
- [\[ICML 2026\] WIND: Weather Inverse Diffusion for Zero-Shot Atmospheric Modeling](../../ICML2026/video_generation/wind_weather_inverse_diffusion_for_zero-shot_atmospheric_modeling.md)
- [\[CVPR 2025\] Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion](../../CVPR2025/video_generation/zero-1-to-a_zero-shot_one_image_to_animatable_head_avatars_using_video_diffusion.md)

</div>

<!-- RELATED:END -->
