---
title: >-
  [论文解读] LEIA: Latent View-Invariant Embeddings for Implicit 3D Articulation
description: >-
  [ECCV 2024][3D视觉][铰接物体] 提出LEIA方法，通过学习视角不变的潜在嵌入来表征铰接物体的不同状态，利用超网络(HyperNetwork)调制NeRF权重，实现在未见过的铰接配置之间进行平滑插值，无需任何运动先验或3D监督。 领域现状： NeRF在静态场景重建上取得了巨大成功，但将其扩展到动态物体或物体铰接…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "铰接物体"
  - "神经辐射场"
  - "超网络"
  - "隐式表示"
  - "状态插值"
---

# LEIA: Latent View-Invariant Embeddings for Implicit 3D Articulation

**会议**: ECCV 2024  
**arXiv**: [2409.06703](https://arxiv.org/abs/2409.06703)  
**代码**: [项目页面](https://archana1998.github.io/leia/)  
**领域**: 3D视觉  
**关键词**: 铰接物体, 神经辐射场, 超网络, 隐式表示, 状态插值

## 一句话总结

提出LEIA方法，通过学习视角不变的潜在嵌入来表征铰接物体的不同状态，利用超网络(HyperNetwork)调制NeRF权重，实现在未见过的铰接配置之间进行平滑插值，无需任何运动先验或3D监督。

## 研究背景与动机

**领域现状**: NeRF在静态场景重建上取得了巨大成功，但将其扩展到动态物体或物体铰接仍然是一个挑战性问题。

**现有痛点**: 
   - 现有方法依赖关于移动部件数量或物体类别的启发式假设，限制了实际应用
   - PARIS等方法需要将物体解耦为静态和运动部分，在多部件铰接场景下失败
   - 基于视频的动态NeRF方法无法很好处理日常物体的大幅铰接

**核心矛盾**: 需要在不依赖运动先验、部件解耦的前提下，灵活建模任意类型和数量的铰接运动

**本文目标**: 仅从多视角图像出发，学习铰接物体不同状态的3D表示，并能生成训练中未见过的中间状态

**切入角度**: 将每个铰接状态编码为一个可学习的潜在嵌入，通过超网络将其映射到NeRF的权重参数化

**核心 idea**: 用超网络将视角不变的状态潜在嵌入映射为NeRF权重，通过潜在空间插值生成新铰接状态

## 方法详解

### 整体框架

系统包含三个核心组件：(1) 可学习的潜在字典 $Z$，为每个铰接状态分配一个嵌入向量；(2) 超网络 $h_l$，将潜在嵌入转换为NeRF的权重调制矩阵；(3) 基于Instant-NGP架构的基础NeRF网络 $f_\theta$。训练时，每个batch采样一个状态并用其对应的多视角图像进行监督；推理时，在潜在空间中进行线性插值生成未见过的中间状态。

### 关键设计

1. **状态条件化的超网络调制**: 超网络不直接预测完整的NeRF权重 $\theta$（代价太高），而是预测低秩矩阵 $P^l \in \mathbb{R}^{K \times r}$ 和 $Q^l \in \mathbb{R}^{r \times K}$（$r \ll K$），通过元素级调制修改基础网络权重：

$$\theta_t^l = \eta(P^l \times Q^l) \circ \theta^l$$

其中 $\eta$ 为激活函数，$\circ$ 为逐元素乘法。这种低秩调制类似子网络选择，rank $r$ 控制压缩-性能的平衡。

2. **可学习潜在字典**: 潜在字典 $Z = \{t: z_t \mid t \in [0,1,...,T]\}$ 使用 `nn.Embedding` 作为查找表，每个状态ID对应一个可学习嵌入 $z_t \in \mathbb{R}^D$。所有超网络共享同一个潜在嵌入作为输入，使其成为连接状态语义与NeRF参数化的桥梁。

3. **潜在空间线性插值**: 给定两个状态 $t_1, t_2$ 及其嵌入 $z_1, z_2$，通过加权线性插值生成中间状态：

$$z_{\text{inter}} = (1 - \beta_i) \cdot z_t + \beta_i \cdot z_{t-1}$$

其中 $\beta_i \in [\frac{1}{\alpha}, \frac{2}{\alpha}, ..., \frac{\alpha-1}{\alpha}]$，可生成 $\alpha - 1$ 个中间状态。插值后的潜在向量通过超网络生成NeRF权重，实现新状态的新视角渲染。

### 损失函数 / 训练策略

总损失包含以下组成部分：

- **Smooth L1重建损失**: $L_{\text{SmoothL1-NeRF}} = \sum_{r \in R} \text{SmoothL1Loss}(\hat{C}(r) - C(r))$，比L2更鲁棒
- **前景掩码损失**: $L_{\text{mask}}$，预测不透明度与GT前景mask之间的BCE损失
- **潜在流形损失**: $\mathcal{L}_{\text{manifold}}(l_i) = \frac{1}{K}\sum_{k=1}^{K} \|l_i - n_k\|_2^2$，鼓励潜在嵌入的K近邻之间保持局部一致性，促进平滑连续的流形结构
- **遮挡正则化**: $L_{\text{occ}} = \frac{1}{K}\sum_{k=1}^{K} \sigma_k \cdot m_k$，减少相机前方的密度累积
- **深度平滑正则化**: $L_{\text{DS}}$，强制相邻像素深度值平滑过渡
- **位置编码**: 对潜在向量添加正弦/余弦位置编码，注入状态顺序信息

使用AdamW优化器训练，每个batch采样一个状态。

## 实验关键数据

### 主实验

使用PartNet-Mobility数据集12个物体（8类），对比PARIS方法进行插值状态重建：

| 指标 | PARIS | VanillaInt | **LEIA** |
|------|-------|------------|----------|
| PSNR↑ | 27.81 | 27.81 | **29.55** |
| SSIM↑ | 0.96 | 0.94 | **0.96** |
| LPIPS↓ | 0.06 | 0.07 | **0.06** |
| CD↓ | 0.45 | 0.37 | **0.36** |

LEIA在多部件铰接物体上的优势尤其显著（Storage2-4, Sunglasses, Box），因为PARIS的运动参数估计在多部件场景下失败。

### 消融实验

| 组件 | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| 有流形损失 | **29.40** | **0.95** | **0.05** |
| 无流形损失 | 28.54 | 0.94 | 0.06 |
| 有深度正则化 | **29.63** | **0.96** | **0.05** |
| 无深度正则化 | 26.93 | 0.93 | 0.07 |
| 有遮挡正则化 | **29.64** | **0.95** | **0.05** |
| 无遮挡正则化 | 28.64 | 0.95 | 0.06 |
| 4个状态 | **29.69** | **0.96** | **0.05** |
| 2个状态 | 28.04 | 0.95 | 0.06 |

### 关键发现

- 流形损失防止潜在空间过拟合到极端状态，是实现有意义插值的关键
- 4个状态比2个状态显著提升，额外状态帮助潜在空间建立更好的结构
- t-SNE可视化显示不同关节的嵌入具有清晰的可分离性，沿各自运动方向形成平滑轨迹
- 位置编码对细小部件（如太阳镜）有帮助，但对大物体可能引入噪声

## 亮点与洞察

- **运动先验无关**: 不需要指定运动类型（旋转/平移），也不需要部件解耦，一个统一模型处理所有情况
- **可扩展到多部件**: 通过t-SNE验证潜在空间可自动分离不同关节的运动，是PARIS等方法无法做到的
- **低秩调制策略**: 借鉴LoRA思想，通过低秩矩阵调制NeRF权重，大幅降低超网络的参数量
- **真实世界验证**: 在手机拍摄的抽屉柜真实场景上也能生成可信的中间状态

## 局限与展望

- 中间状态不保证物理一致性（如门的铰链约束），是放弃运动先验的代价
- 严重遮挡时失败（如微波炉从关闭到打开，形状变化太大）
- 仅在PartNet-Mobility合成数据上做了系统评估，真实场景验证有限
- 可考虑引入轻量级物理约束来提升插值状态的合理性

## 相关工作与启发

- **PARIS** [ICCV 2023]: 将物体解耦为静/动部分，分别估计运动参数→限制了多部件泛化
- **A-SDF** [CVPR 2020]: 需要铰接编码和3D监督→实用性受限
- **CLA-NeRF** [ICRA 2022]: 需要铰接姿态输入→依赖先验
- **超网络用于INR压缩** [NeurIPS 2023]: 低秩调制思路的来源

## 评分

- **新颖性**: ⭐⭐⭐⭐ 用超网络+潜在字典统一建模铰接状态，思路清晰优雅
- **实验充分度**: ⭐⭐⭐ 消融充分，但数据集规模有限（仅12个物体），缺乏更多真实数据
- **写作质量**: ⭐⭐⭐⭐ 方法阐述清晰，公式推导完整
- **价值**: ⭐⭐⭐⭐ 为铰接物体建模提供了一个简洁且可扩展的方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Particulate: Feed-Forward 3D Object Articulation](../../CVPR2026/3d_vision/particulate_feed-forward_3d_object_articulation.md)
- [\[ECCV 2024\] Compress3D: a Compressed Latent Space for 3D Generation from a Single Image](compress3d_a_compressed_latent_space_for_3d_generation_from_a_single_image.md)
- [\[ECCV 2024\] RISurConv: Rotation Invariant Surface Attention-Augmented Convolutions for 3D Point Cloud Classification and Segmentation](risurconv_rotation_invariant_surface_attention-augmented_convolutions_for_3d_poi.md)
- [\[ECCV 2024\] LN3Diff: Scalable Latent Neural Fields Diffusion for Speedy 3D Generation](ln3diff_scalable_latent_neural_fields_diffusion_for_speedy_3d_generation.md)
- [\[ECCV 2024\] Implicit Filtering for Learning Neural Signed Distance Functions from 3D Point Clouds](implicit_filtering_for_learning_neural_signed_distance_functions_from_3d_point_c.md)

</div>

<!-- RELATED:END -->
