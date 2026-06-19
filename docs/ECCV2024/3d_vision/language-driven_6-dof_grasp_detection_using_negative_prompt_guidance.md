---
title: >-
  [论文解读] Language-Driven 6-DoF Grasp Detection Using Negative Prompt Guidance
description: >-
  [ECCV 2024][3D视觉][6-DoF抓取检测] 提出大规模语言驱动6-DoF抓取数据集Grasp-Anything-6D（1M场景、200M抓取姿态），以及基于扩散模型的LGrasp6D方法，核心创新是**负提示引导（Negative Prompt Guidance）**策略，在推理时引导抓取姿态远离非目标物体。
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "6-DoF抓取检测"
  - "语言引导"
  - "扩散模型"
  - "负提示引导"
  - "机器人操控"
---

# Language-Driven 6-DoF Grasp Detection Using Negative Prompt Guidance

**会议**: ECCV 2024  
**arXiv**: [2407.13842](https://arxiv.org/abs/2407.13842)  
**代码**: [https://airvlab.github.io/grasp-anything](https://airvlab.github.io/grasp-anything)  
**领域**: 3D视觉  
**关键词**: 6-DoF抓取检测, 语言引导, 扩散模型, 负提示引导, 机器人操控

## 一句话总结

提出大规模语言驱动6-DoF抓取数据集Grasp-Anything-6D（1M场景、200M抓取姿态），以及基于扩散模型的LGrasp6D方法，核心创新是**负提示引导（Negative Prompt Guidance）**策略，在推理时引导抓取姿态远离非目标物体。

## 研究背景与动机

**领域现状**: 6-DoF抓取检测是机器人视觉的基础任务，现有方法主要关注抓取稳定性，但忽略了人类通过自然语言表达的抓取意图。

**现有痛点**: 已有语言驱动抓取方法存在严重局限：部分仅支持单物体场景，部分仅检测2D矩形抓取，无法处理3D杂乱场景中的精细语言指令。

**核心矛盾**: 语言驱动抓取本质上是细粒度任务（"抓蓝色杯子" vs "抓黑色杯子"），但现有方法缺乏区分不同物体的精确引导机制。

**本文目标**: 在杂乱3D点云场景中，根据自然语言指令检测目标物体的6-DoF抓取姿态。

**切入角度**: 借鉴图像生成中负提示的思想，在扩散模型训练中学习场景内非目标物体的嵌入表示，推理时将抓取姿态从非目标物体方向"推开"。

**核心 idea**: 通过学习负提示嵌入来编码"不想抓的东西"，在扩散采样中组合正向引导和反向排斥实现精准抓取。

## 方法详解

### 整体框架

LGrasp6D是一个端到端的扩散框架：

1. 输入：3D点云场景 $\mathbf{S}$ + 文本提示 $\mathbf{t}$（如"Grasp the red mug"）
2. 前向过程：对目标抓取姿态 $\mathbf{g}_0$ 使用 $\mathfrak{se}(3)$ 李代数表示，逐步加噪
3. 去噪网络：同时预测噪声和学习负提示嵌入
4. 反向过程：使用带负提示引导的去噪步骤生成抓取姿态

### 关键设计

1. **Grasp-Anything-6D数据集**: 基于2D的Grasp-Anything数据集，使用ZoeDepth估计深度图，将100万2D场景投影为3D点云。通过双线性插值映射抓取位置，用矩形抓取角度推导6-DoF旋转，并手动检查去除碰撞/不稳定抓取。最终规模为**1M点云场景、200M以上6-DoF抓取姿态**，每个抓取关联语言描述。设计动机是弥补语言驱动6-DoF抓取训练数据的空白。

2. **去噪网络架构**: 输入抓取姿态 $\mathbf{g}_t$ 经MLP编码，点云 $\mathbf{S}$ 用PointNet++编码为 $n_s$ 个场景token，文本 $\mathbf{t}$ 用冻结的CLIP ViT-B/32编码。将时间嵌入与抓取嵌入、文本嵌入的融合特征 $\mathbf{f}_{\text{uni}}$ 作为query，场景token作为key/value进行多头交叉注意力，最终MLP输出噪声预测 $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{g}_t, \mathbf{S}, \mathbf{t}, t)$。噪声预测损失：

$$\mathcal{L}_{\text{noise}} = \mathbb{E}_{\boldsymbol{\epsilon}, \mathbf{g}_0, \mathbf{S}, \mathbf{t}, t}\left[\|\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{g}_t, \mathbf{S}, \mathbf{t}, t) - \boldsymbol{\epsilon}\|^2\right]$$

3. **负提示引导学习（Negative Prompt Guidance）**: 网络同时输出负提示嵌入 $\tilde{\mathbf{t}}$：从场景token中减去正向文本嵌入 $\mathbf{t}$，求均值后经MLP得到。该嵌入需逼近场景中**其他物体**的文本嵌入（即"不想抓的东西"的表示）：

$$\mathcal{L}_{\text{negative}} = \min_{i=1}^{m} \|\tilde{\mathbf{t}} - \bar{\mathbf{t}}_i\|_2^2$$

其中 $\{\bar{\mathbf{t}}_i\}_{i=1}^m$ 是同一场景中非目标物体的文本嵌入集合。推理时基于Proposition 1的条件分布分解：

$$p(\mathbf{g}|\mathbf{S}, \mathbf{t}, \neg\tilde{\mathbf{t}}) \propto p(\mathbf{g}|\mathbf{S}) \frac{p(\mathbf{g}|\mathbf{t}, \mathbf{S})}{p(\mathbf{g}|\tilde{\mathbf{t}}, \mathbf{S})}$$

组合去噪步为：

$$\tilde{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}} = \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{g}_t, \mathbf{S}, \varnothing, t) + w\left(\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{g}_t, \mathbf{S}, \mathbf{t}, t) - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{g}_t, \mathbf{S}, \tilde{\mathbf{t}}, t)\right)$$

其中 $w=0.2$ 为负引导强度，$\varnothing$ 表示以概率 $p_{\text{mask}}=0.1$ 随机mask文本条件训练的无条件预测。

### 损失函数 / 训练策略

$$\mathcal{L} = 0.9\mathcal{L}_{\text{noise}} + 0.1\mathcal{L}_{\text{negative}}$$

- 扩散步数 $T=200$，方差从 $\beta_1=10^{-4}$ 线性增至 $\beta_T=0.02$
- 8×A100 GPU，batch size 128，200 epochs
- Adam优化器，lr=$10^{-3}$，weight decay=$10^{-4}$
- 推理可用DDIM加速至50步甚至10步

## 实验关键数据

### 主实验 - Grasp-Anything-6D 数据集

| 方法 | CR↑ | EMD↓ | CFR↑ | IT(s)↓ |
|------|-----|------|------|--------|
| 6-DoF GraspNet | 0.3802 | 0.8035 | 0.6900 | 0.4216 |
| SE(3)-DF | 0.4290 | 0.7565 | 0.7325 | 1.7233 |
| 3DAPNet | 0.4777 | 0.7381 | 0.7213 | 3.4274 |
| Ours w/o NPG | 0.5459 | 0.6262 | 0.7336 | 1.4328 |
| **LGrasp6D (Ours)** | **0.6694** | **0.4013** | **0.7706** | 1.4832 |

### 真实机器人实验

| 方法 | 输入 | 单物体成功率 | 杂乱场景成功率 |
|------|------|-------------|---------------|
| GG-CNN + CLIP | RGB-D | 0.10 | 0.07 |
| CLIPORT | RGB-D | 0.27 | 0.30 |
| CLIP-Fusion | RGB-D | 0.40 | 0.40 |
| LGD | RGB-D | 0.43 | 0.42 |
| 6-DoF GraspNet | 点云 | 0.31 | 0.27 |
| Ours w/o NPG | 点云 | 0.38 | 0.36 |
| **LGrasp6D (Ours)** | **点云** | **0.43** | **0.42** |

### 关键发现

- 负提示引导带来巨大提升：CR从0.5459→0.6694（+22.6%），EMD从0.6262→0.4013（-35.9%）
- t-SNE可视化显示：有NPG时不同物体的抓取姿态聚类明显分离，无NPG时严重混淆
- 跨数据集泛化（Contact-GraspNet上）趋势一致，性能下降轻微
- DDIM 50步加速后推理仅0.40s，快于6-DoF GraspNet且所有指标仍优于其他baseline
- 在合成数据上训练但能泛化到真实桌面/厨房/浴室场景

## 亮点与洞察

- 负提示引导的概念迁移很巧妙：将图像生成中"避免生成某些内容"的负提示思想移植到机器人抓取
- 数学推导完整，Proposition 1给出了条件分布的严格分解，组合去噪公式层次清晰
- 数据集规模巨大（1M场景/200M抓取），为该细分方向提供了重要的训练资源
- 端到端设计：从自然语言到6-DoF抓取，中间无需额外的分割或目标检测模块

## 局限与展望

- 仍存在抓错物体、碰撞和瞄偏的失败案例（论文Figure 7展示）
- 数据集基于ZoeDepth估计的深度图，存在不精确性
- 仅支持物体级别的抓取指令，未覆盖零件级（"抓刀柄"）和任务级（"抓刀来切菜"）
- 仅适用于Robotiq 2F夹爪，泛化到多指灵巧手需额外工作
- 推理速度（1.48s）对实时交互仍偏慢，虽然DDIM可加速但性能有折损

## 相关工作与启发

- **Classifier-free Guidance**: 负提示思想源于此，将无条件预测与条件预测线性组合
- **3DAPNet**: 先前将扩散模型用于6-DoF抓取的代表，但不支持语言
- **Grasp-Anything**: 本文数据集的2D前身，提供2D语言-抓取对应关系
- **启发**: 负提示机制可推广到其他条件生成任务（如：语言驱动的操控规划中避开障碍物）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 负提示引导在抓取检测中首次且有效应用
- **实验充分度**: ⭐⭐⭐⭐⭐ — 基准对比 + 跨数据集 + DDIM加速 + t-SNE分析 + 真实机器人，非常全面
- **写作质量**: ⭐⭐⭐⭐ — 理论推导+实验+实际机器人验证，结构清晰
- **价值**: ⭐⭐⭐⭐ — 大规模数据集+有效方法，对语言驱动操控方向有实质推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] DreamView: Injecting View-specific Text Guidance into Text-to-3D Generation](dreamview_injecting_view-specific_text_guidance_into_text-to-3d_generation.md)
- [\[ECCV 2024\] FALIP: Visual Prompt as Foveal Attention Boosts CLIP Zero-Shot Performance](falip_visual_prompt_as_foveal_attention_boosts_clip_zero-shot_performance.md)
- [\[ECCV 2024\] GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing](gaussctrl_multi-view_consistent_text-driven_3d_gaussian_splatting_editing.md)
- [\[ICCV 2025\] Egocentric Action-aware Inertial Localization in Point Clouds with Vision-Language Guidance](../../ICCV2025/3d_vision/egocentric_action-aware_inertial_localization_in_point_clouds_with_vision-langua.md)
- [\[CVPR 2026\] TextFM: Robust Semi-dense Feature Matching with Language Guidance](../../CVPR2026/3d_vision/textfm_robust_semi-dense_feature_matching_with_language_guidance.md)

</div>

<!-- RELATED:END -->
