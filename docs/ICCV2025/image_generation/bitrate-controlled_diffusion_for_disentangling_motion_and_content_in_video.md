---
title: >-
  [论文解读] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video
description: >-
  [ICCV 2025][图像生成][视频解耦] 提出BCD（Bitrate-Controlled Diffusion），一种通用的自监督视频解耦框架，通过低码率矢量量化作为信息瓶颈来分离视频中的逐帧运动特征和全局内容特征，并以条件扩散模型重建视频，在说话人头部视频和像素风格卡通数据集上展示了高质量的运动迁移和自回归视频生成能力。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "视频解耦"
  - "运动与内容分离"
  - "信息瓶颈"
  - "矢量量化"
  - "扩散模型"
---

# Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video

**会议**: ICCV 2025  
**arXiv**: [2509.08376](https://arxiv.org/abs/2509.08376)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 视频解耦, 运动与内容分离, 信息瓶颈, 矢量量化, 扩散模型

## 一句话总结

提出BCD（Bitrate-Controlled Diffusion），一种通用的自监督视频解耦框架，通过低码率矢量量化作为信息瓶颈来分离视频中的逐帧运动特征和全局内容特征，并以条件扩散模型重建视频，在说话人头部视频和像素风格卡通数据集上展示了高质量的运动迁移和自回归视频生成能力。

## 研究背景与动机

将视频分解为静态内容（content）和动态运动（motion）是视频理解的核心问题，但解耦学习本身是一个不确定性问题（同一视频有多种合理分解方式），再加上视频数据的高维性使问题更加复杂。现有方法的局限：
- **VAE框架方法**引入过多假设：要求低维运动特征、显式假定运动和内容独立/依赖、或假设单帧即可完全建模内容
- **说话人头部专用方法**依赖特定先验：光流（FOMM）、关键点（LivePortrait）、3D参数化人脸模型（HyperReenact）、线性运动基（LIA），限制了适用范围
- 上述方法要么假设过于理想限制了特征表达力，要么绑定特定任务限制了泛化性

BCD的核心洞察是：**低码率约束本身就是一种通用的解耦先验**。信息瓶颈理论指出，最优压缩必然保留最相关信息并丢弃无关细节——这与解耦表征的目标（紧凑且信息丰富）完全一致。

## 方法详解

### 整体框架

输入视频先由预训练图像VAE编码为逐帧潜码序列 $z = \{z_t | t \in [1,T]\}$，经Transformer编码器提取为逐帧运动特征 $m$ 和全局内容特征 $c$，运动特征通过低码率VQ瓶颈量化后，与内容特征一起作为条件输入到DiT扩散解码器重建原始潜码，最终由图像VAE解码器还原为视频帧。

### 关键设计

1. **内容与运动提取（Transformer编码器）**:

    - 使用T5 Transformer（12层，hidden=512，8头注意力，相对位置编码）
    - 在帧序列前添加 $K$ 个可学习查询token作为前缀，经过Transformer后前缀输出为内容特征 $c \in \mathbb{R}^{K \times C_c}$，其余输出为运动特征 $m \in \mathbb{R}^{T \times C_m}$
    - 可学习查询在全训练集上优化，能从多帧鲁棒聚合内容信息（优于单帧参考或简单pooling）
    - 支持灵活的输入帧数，对大变化视频（不同视角、特定帧才有的细节）更友好

2. **低码率矢量量化信息瓶颈**:

    - 采用Group VQ将运动特征分为 $N=64$ 组，每组独立由 $K=32$ 个码字的码本量化
    - 使用距离-Gumbel-Softmax进行可微量化采样：
    $\mu_t^i = \text{GumbelSoftmax}(-\alpha \cdot d_t^i)$
    - **码率控制**：通过Shannon信源编码定理，用平均采样直方图估计量化运动特征的熵 $\mathcal{H}_{model}$，与目标码率 $\mathcal{H}_{target}$ 之间施加MSE约束
    - 目标码率设为4kbps（25fps下每帧160bits），略低于现有视频编码方法报告的说话人头部视频平均码率5kbps
    - 低码率防止内容泄漏到运动（否则运动码率会超标），非零码率防止运动信息不足（避免信息偏好问题）

3. **条件扩散解码器**:

    - 基于DiT-B/4架构，在空间block之间插入时间注意力层保证时间平滑
    - 运动条件通过加到扩散timestep embedding注入
    - 内容条件通过与噪声输入拼接注入
    - 采用EDM框架进行扩散训练和采样

### 损失函数 / 训练策略

率失真优化目标：
$$\mathcal{L} = \mathcal{L}_d + \lambda \mathcal{L}_{VQ} = \text{MSE}(z, \tilde{z}) + 0.04 \cdot \text{MSE}(\mathcal{H}, \hat{\mathcal{H}})$$

**交叉驱动策略**：训练时将每个视频片段沿时间轴均匀分为两段（语义相似但运动不同），用第一段的内容特征和第二段的运动特征重建第二段，避免退化为trivial的纠缠表示。

训练细节：batch=32，每段50帧，先训30 epoch（无时间层），再训15 epoch（加时间层）。8×A100 GPU训练4-5天。

## 实验关键数据

### 主实验（LRS3说话人头部运动迁移）

| 方法 | FID↓ | CSIM↑ | 身份误差↓ | 运动误差↓ | 交叉误差↓ |
|------|------|-------|----------|----------|----------|
| FOMM | 98.5 | 0.76 | 0.75 | 24.3×10⁻² | 24.1×10⁻² |
| MCNET | 98.6 | 0.76 | 0.85 | 23.9×10⁻² | 23.6×10⁻² |
| HyperReenact | 106.8 | 0.58 | 0.57 | 3.94×10⁻² | 4.68×10⁻² |
| LIA | 104.4 | 0.71 | 0.57 | 36.1×10⁻² | 34.1×10⁻² |
| LivePortrait | 100.3 | 0.69 | 0.66 | 24.6×10⁻² | 23.7×10⁻² |
| **BCD (Ours)** | **86.0** | 0.69 | **0.41** | **3.13×10⁻²** | **3.67×10⁻²** |

### 消融实验（目标码率的影响）

| 目标码率(kbps) | FID↓ | CSIM↑ | 身份误差↓ | 运动误差↓ | 交叉误差↓ |
|---------------|------|-------|----------|----------|----------|
| 2.0 | 88.5 | 0.71 | 0.34 | 5.26×10⁻² | 5.68×10⁻² |
| 6.0 | 87.6 | 0.68 | 0.56 | 3.23×10⁻² | 4.13×10⁻² |
| 8.0 | 89.3 | 0.66 | 0.49 | 3.04×10⁻² | 3.74×10⁻² |
| **4.0** | **86.0** | 0.69 | 0.41 | 3.13×10⁻² | **3.67×10⁻²** |
| 4.0 (单参考帧) | 87.9 | 0.69 | 0.47 | 3.13×10⁻² | 3.81×10⁻² |
| 4.0 (无交叉驱动) | 120.1 | 0.64 | 0.58 | 41.5×10⁻² | 40.4×10⁻² |

### 用户研究

| 方法 | 身份保持↑ | 运动一致性↑ | 视觉质量↑ |
|------|----------|-----------|----------|
| FOMM | 3.34 | 2.63 | 2.84 |
| HyperReenact | 2.97 | 4.01 | 3.72 |
| LIA | 3.66 | 2.53 | 3.53 |
| **BCD (Ours)** | **4.10** | **4.30** | **4.00** |

### 关键发现

- BCD不使用任何人脸相关先验，但在身份误差、运动误差、FID上均最优
- 4kbps是最优码率——过低（2kbps）运动信息不足导致误差增大，过高（8kbps）瓶颈不够紧导致解耦不充分
- 无交叉驱动策略时FID从86飙升至120，运动误差增大13倍，表明其对防止解耦退化至关重要
- 基于关键点的方法（FOMM、LivePortrait）身份保持好但运动误差大（受光流伪影和首帧姿态假设限制）
- LIA运动空间受限（线性基组合），运动误差最大
- 方法可直接扩展到原来没见过的数据类型（Sprites卡通数据集）

## 亮点与洞察

- 将率失真理论引入解耦学习非常自然——最优压缩即最好的因子分离
- 隐式特征（无归纳偏置）+ 信息瓶颈的组合策略兼顾灵活性和约束性
- 学习到的运动空间具有结构性，直接支持GPT-2自回归运动生成，证明运动码本捕获了完整的运动分布
- Group VQ + Gumbel-Softmax的可微量化方案可直接端到端训练

## 局限与展望

- 需要大量训练数据来实现无先验解耦
- 最优码率可能因数据集而异，需要手动调整
- 即使经过时间微调仍有轻微视频闪烁
- 训练数据中动态变化通常多于静态变化，面对完全分布外的视频输入时静态建模能力可能下降

## 相关工作与启发

- 低码率VQ从音频编解码（将语音内容与说话人身份分离）迁移到视频解耦，跨域启发
- 扩散模型作为解码器（而非传统VAE解码器）能在信息瓶颈下保持更高重建保真度
- 未来可向通用视频（不限于说话人头部和卡通）扩展

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 码率控制作为通用解耦先验的idea非常原创且有理论支撑
- 实验充分度: ⭐⭐⭐⭐ 定量+用户研究+两个数据集+码率消融+生成验证，比较完整
- 写作质量: ⭐⭐⭐⭐ 动机和理论基础阐述清晰，与编码理论的关联讲解到位
- 价值: ⭐⭐⭐⭐ 提供了一种通用解耦范式，对视频编辑和生成有广泛启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Video Motion Graphs](video_motion_graphs.md)
- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [\[CVPR 2025\] DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning](../../CVPR2025/image_generation/dreamvideo-omni_omni-motion_controlled_multi-subject_video_customization_with_la.md)
- [\[ICCV 2025\] SummDiff: Generative Modeling of Video Summarization with Diffusion](summdiff_generative_modeling_of_video_summarization_with_diffusion.md)
- [\[ICCV 2025\] SMGDiff: Soccer Motion Generation using Diffusion Probabilistic Models](smgdiff_soccer_motion_generation_using_diffusion_probabilistic_models.md)

</div>

<!-- RELATED:END -->
