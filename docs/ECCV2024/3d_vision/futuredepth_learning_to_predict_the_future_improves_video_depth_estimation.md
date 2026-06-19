---
title: >-
  [论文解读] FutureDepth: Learning to Predict the Future Improves Video Depth Estimation
description: >-
  [ECCV 2024][3D视觉][视频深度估计] 提出FutureDepth，通过未来预测网络(F-Net)学习运动线索和重建网络(R-Net)学习多帧对应关系，将隐式的运动和场景特征注入深度解码器，在NYUDv2、KITTI、DDAD、Sintel四个数据集上达到SOTA精度和时序一致性，且推理效率显著优于现有视频深度方法。
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "视频深度估计"
  - "未来预测"
  - "掩码自编码"
  - "时序一致性"
  - "多帧特征聚合"
---

# FutureDepth: Learning to Predict the Future Improves Video Depth Estimation

**会议**: ECCV 2024  
**arXiv**: [2403.12953](https://arxiv.org/abs/2403.12953)  
**代码**: 无（Qualcomm AI Research）  
**领域**: 3D视觉  
**关键词**: 视频深度估计, 未来预测, 掩码自编码, 时序一致性, 多帧特征聚合

## 一句话总结

提出FutureDepth，通过未来预测网络(F-Net)学习运动线索和重建网络(R-Net)学习多帧对应关系，将隐式的运动和场景特征注入深度解码器，在NYUDv2、KITTI、DDAD、Sintel四个数据集上达到SOTA精度和时序一致性，且推理效率显著优于现有视频深度方法。

## 研究背景与动机

**领域现状**: 单目深度估计研究已非常成熟，但忽略了实际应用（自动驾驶、AR/VR）中几乎总是可用的视频连续帧信息。视频深度估计方法通过利用多帧信息提升精度和时序一致性。

**现有痛点**: 基于代价体(cost volume)的方法计算量和内存开销巨大；基于自回归的方法（如MAMo需要光流估计+注意力+在线梯度计算，NVDS需要对每个源帧做逐对交叉注意力）虽更高效但仍开销不小，且难以学习底层动态运动/轨迹信息。

**核心矛盾**: 现有视频深度方法要么精度高但计算量大，要么效率高但精度受限。同时，它们不考虑连续帧的联合深度预测，无法有效学习底层的运动轨迹和空间对应，导致时序一致性次优。

**本文目标**: 如何在保持计算效率的前提下，让模型隐式学习并利用多帧运动和对应线索来显著提升视频深度估计的精度和时序一致性。

**切入角度**: 用"预测未来"作为代理任务迫使网络学习运动规律，用自适应掩码重建迫使网络学习跨帧对应关系。

**核心 idea**: 通过训练网络预测未来帧特征来隐式捕捉运动信息，是一种优雅的表征学习方法，推理时无需显式的光流估计或代价体构建。

## 方法详解

### 整体框架

FutureDepth由四个组件构成：(1) 编码器提取连续$T$帧的特征体 $V_{1,T} \in \mathbb{R}^{H' \times W' \times (T \cdot C)}$；(2) 未来预测网络F-Net学习运动线索；(3) 重建网络R-Net学习多帧对应关系；(4) 深度解码器+细化网络利用F-Net和R-Net的查询特征生成最终深度图。F-Net和R-Net分别产生运动查询 $Q_{\text{motion}}$ 和场景查询 $Q_{\text{scene}}$，通过交叉注意力融合为 $Q_{\text{all}}$ 注入解码过程。

### 关键设计

1. **未来预测网络 (F-Net)**: F-Net以自回归方式迭代预测未来一步的多帧特征。输入当前特征体 $V_{1,T}$ 预测 $\tilde{V}_{2,T+1}$，再用预测结果继续预测 $\tilde{V}_{3,T+2}$，直至 $L$ 步。关键点是**不使用teacher forcing**——预测过程中不使用真实未来帧的特征，而是用自身的预测值迭代，这迫使网络最小化跨时间步的误差累积，从而提取更鲁棒的运动和对应线索。损失函数为 $\mathcal{L}_F = \sum_{i=1}^{L} \| \tilde{V}_{1+i,T+i} - V_{1+i,T+i} \|_2$。最终取所有预测步骤最后一层的特征并平均得到 $Q_{\text{motion}}$。设计动机：预测未来特征需要理解像素级场景和物体如何随时间移动，F-Net因此隐式学会了运动和多帧对应。

2. **重建网络 (R-Net)**: R-Net通过自适应掩码自编码学习多帧对应关系。使用可学习的掩码生成器产生自适应掩码 $M_{1,T}$，对特征体进行掩码 $M_{1,T} \odot V_{1,T}$ 后输入R-Net重建原始特征。训练损失为 $\mathcal{L}_R = \| (1 - M_{1,T}) \odot (\hat{V}_{1,T} - V_{1,T}) \|_2 + \mathcal{L}_D(D_{1,T}, D_{1,T}^{gt})$，即掩码区域的L2重建损失加深度SILog损失。掩码生成器通过深度SILog损失 $\mathcal{L}_A$ 单独训练，学会在不同帧中掩码同一物体的不同部分（如白色卡车跨帧的不同区域），迫使R-Net跨帧搜索信息以完成重建。设计动机：不同于标准视频MAE做预训练，R-Net是额外的辅助网络，推理时与主网络一起工作，生成包含关键场景信息的 $Q_{\text{scene}}$。

3. **特征融合与细化网络**: $Q_{\text{motion}}$ 和 $Q_{\text{scene}}$ 先做交叉注意力生成 $Q_{\text{all}}$，通道维扩展$T$倍。在解码器的每层Transformer中，$Q_{\text{all}}$ 与前一层输出的查询结合参与解码。细化网络由自注意力+交叉注意力层组成，以深度图为输入，用交叉注意力利用 $Q_{\text{all}}$ 迭代精修（重复$N=3$次），显著改善深度图细节（如栏杆、交通灯）。

### 损失函数 / 训练策略

训练分阶段进行：
- **预训练阶段**: 先训练编码器-解码器（5 epochs），再用L2损失预训练R-Net进行随机掩码重建（3 epochs）
- **主训练阶段**: 冻结编码器-解码器，同时训练自适应掩码生成器、F-Net、R-Net（对应 $\mathcal{L}_A$、$\mathcal{L}_F$、$\mathcal{L}_R$）；然后冻结F-Net/R-Net/掩码生成器，训练编码器、解码器和细化网络
- **最终损失**: $\mathcal{L}_{D,\text{final}} = \frac{1}{NT} \sum_{i=0}^{N} \sum_{t=1}^{T} \mathcal{L}_D(D_t^i, D_t^{gt})$，其中 $\mathcal{L}_D$ 为SILog损失
- **关键超参**: $T=4$帧，$L=T$步预测，$N=3$次细化，掩码率$r \in [0.6, 0.9]$，帧间隔从{1,2,3,4}中采样
- 总训练约2天，2块A100 GPU

## 实验关键数据

### 主实验：NYUDv2 & Sintel

| 方法 | NYUDv2 $\delta<1.25$↑ | NYUDv2 Abs Rel↓ | NYUDv2 OPW↓ | Sintel $\delta<1.25$↑ | Sintel Abs Rel↓ | Sintel OPW↓ |
|---|---|---|---|---|---|---|
| NVDS | 0.950 | 0.072 | 0.364 | 0.591 | 0.335 | 0.424 |
| MAMo | 0.942 | 0.074 | 0.388 | 0.579 | 0.358 | 0.493 |
| Baseline (ours) | 0.917 | 0.093 | 0.480 | 0.477 | 0.504 | 0.611 |
| **FutureDepth** | **0.981** | **0.063** | **0.303** | **0.623** | **0.296** | **0.392** |

### 主实验：KITTI

| 方法 | 编码器 | Abs Rel↓ | RMSE↓ | $\delta<1.25$↑ |
|---|---|---|---|---|
| iDisc (单帧) | Swin-L | 0.050 | 2.067 | 0.977 |
| GEDepth (单帧) | - | 0.048 | 2.050 | 0.976 |
| MAMo (视频) | Swin-L | 0.049 | 1.989 | 0.977 |
| NVDS (视频) | DPT-L | 0.052 | 2.101 | 0.976 |
| **FutureDepth** | Swin-L | **0.044** | **1.920** | **0.983** |
| **FutureDepth** | DINOv2 (ViT-L) | **0.041** | **1.856** | **0.984** |

### 时序一致性与效率（KITTI, Swin-L编码器）

| 方法 | rTC↑ | aTC↓ | OPW↓ | Runtime(ms)↓ |
|---|---|---|---|---|
| NVDS | 0.951 | 0.096 | 0.356 | 930 |
| MAMo | 0.963 | 0.088 | 0.328 | 122 |
| **FutureDepth** | **0.988** | **0.076** | **0.281** | **49** |

### 消融实验（KITTI, Swin-L）

| 模型 | R-Net | AM | F-Net | Refine | Sq Rel↓ | RMSE↓ | $\delta<1.25$↑ | OPW↓ |
|---|---|---|---|---|---|---|---|---|
| SF Baseline | | | | | 0.156 | 2.098 | 0.974 | 0.544 |
| MF Baseline | | | | | 0.154 | 2.094 | 0.975 | 0.540 |
| + F-Net | | | ✓ | | 0.129 | 1.978 | 0.981 | 0.311 |
| + R-Net (RM) | ✓ | | | | 0.148 | 2.040 | 0.976 | 0.478 |
| + R-Net (AM) | ✓ | ✓ | | | 0.136 | 1.999 | 0.980 | 0.416 |
| + F-Net + R-Net(AM) | ✓ | ✓ | ✓ | | 0.122 | 1.931 | 0.983 | 0.284 |
| **FutureDepth (full)** | ✓ | ✓ | ✓ | ✓ | **0.119** | **1.920** | **0.983** | **0.281** |

### 关键发现

- 朴素多帧扩展几乎无增益（MF vs SF仅RMSE降0.004），说明简单拼接特征不足以利用时序信息
- F-Net是最大贡献者：Sq Rel降16%，OPW降42%，验证未来预测能有效学习运动线索
- 自适应掩码(AM)优于随机掩码(RM)：OPW从0.478降至0.416，因为AM学会跨帧掩码同一物体的不同部分
- FutureDepth推理仅49ms，比NVDS(930ms)快19倍，比MAMo(122ms)快2.5倍
- DDAD上Abs Rel从MAMo的0.150降至0.114（降24%），展示强泛化能力

## 亮点与洞察

- **代理任务驱动的表征学习**: 不直接估计光流或建代价体，而是用"预测未来"和"掩码重建"两个辅助任务让额外网络自主学会运动和对应信息——思路优雅且高效
- **无teacher forcing的自回归预测**: 借鉴强化学习中的"预测依赖"展开策略，避免了分布偏移问题，提取更鲁棒的运动表征
- **模块化设计**: F-Net和R-Net是即插即用的模块，可与任意基础深度网络（ResNet、Swin、DINOv2等）组合
- **效率与精度的罕见兼得**: 通常视频方法精度高但慢，FutureDepth同时在精度和速度上超越SOTA

## 局限与展望

- 未专门处理遮挡-重现场景（物体被遮挡后重新出现），正确处理此类情况可进一步提升运动理解
- F-Net和R-Net共享架构但目标不同，能否统一为单一多任务网络值得探索
- 推理时仍需掩码不参与（提升了效率），但随之也可能丢失了R-Net在有掩码时学到的部分信息
- 训练流程分阶段，较为复杂，端到端训练可能简化流程

## 相关工作与启发

- **MAMo**: 利用光流+在线梯度更新，精度高但计算昂贵；FutureDepth用未来预测替代光流，更优雅
- **NVDS**: 逐对源帧做交叉注意力，开销随帧数线性增长；FutureDepth用F-Net/R-Net将多帧信息压缩为紧凑查询
- **视频MAE (VideoMAE)**: 用于预训练主编码器-解码器；FutureDepth的R-Net则是独立的辅助网络，推理时一起使用
- **启发**: "预测未来"作为自监督信号在深度估计中效果显著，该范式可推广到其他时序密集预测任务（如光流估计、视频分割）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 用未来预测学运动线索的思路新颖且效果突出，无teacher forcing的自回归训练在视频深度领域首创
- 实验充分度: ⭐⭐⭐⭐⭐ 4个数据集（室内+驾驶+开放域）、精度+时序一致性+效率三维比较、详细消融
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，训练流程偏复杂但有算法伪代码补充
- 价值: ⭐⭐⭐⭐⭐ 精度、时序一致性、效率三方面均SOTA，具有很强的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Video Depth Without Video Models](../../CVPR2025/3d_vision/video_depth_without_video_models.md)
- [\[CVPR 2025\] Video Depth Anything: Consistent Depth Estimation for Super-Long Videos](../../CVPR2025/3d_vision/video_depth_anything_consistent_depth_estimation_for_super-long_videos.md)
- [\[ECCV 2024\] DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)
- [\[ECCV 2024\] Diffusion Models for Monocular Depth Estimation: Overcoming Challenging Conditions](diffusion_models_for_monocular_depth_estimation_overcoming_challenging_condition.md)
- [\[ICCV 2025\] FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution](../../ICCV2025/3d_vision/flashdepth_real-time_streaming_video_depth_estimation_at_2k_resolution.md)

</div>

<!-- RELATED:END -->
