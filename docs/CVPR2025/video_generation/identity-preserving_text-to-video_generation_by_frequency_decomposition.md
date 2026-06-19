---
title: >-
  [论文解读] Identity-Preserving Text-to-Video Generation by Frequency Decomposition
description: >-
  [CVPR 2025][视频生成][身份保持视频生成] ConsisID 提出基于频率分解的 DiT 控制方案，将人脸特征解耦为低频全局信息和高频内在身份信息，分别注入 DiT 的不同位置，实现免微调的身份保持文本到视频生成，在身份保持、文本相关性和视觉质量上全面超越现有方法。 领域现状：身份保持的文本到视频生成（IPT2V…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "身份保持视频生成"
  - "频率分解"
  - "DiT"
  - "人脸一致性"
  - "免微调"
---

# Identity-Preserving Text-to-Video Generation by Frequency Decomposition

**会议**: CVPR 2025  
**arXiv**: [2411.17440](https://arxiv.org/abs/2411.17440)  
**代码**: [https://github.com/PKU-YuanGroup/ConsisID](https://github.com/PKU-YuanGroup/ConsisID)  
**领域**: 扩散模型 / 视频生成  
**关键词**: 身份保持视频生成, 频率分解, DiT, 人脸一致性, 免微调

## 一句话总结

ConsisID 提出基于频率分解的 DiT 控制方案，将人脸特征解耦为低频全局信息和高频内在身份信息，分别注入 DiT 的不同位置，实现免微调的身份保持文本到视频生成，在身份保持、文本相关性和视觉质量上全面超越现有方法。

## 研究背景与动机

**领域现状**：身份保持的文本到视频生成（IPT2V）是视频生成中的重要任务。现有方法主要基于 U-Net 架构，且大多需要针对每个新身份进行逐案微调（如 DreamBooth、LoRA），效率低下。开源社区中仅 ID-Animator 支持免微调 IPT2V，但只能生成类似说话头的视频，身份保持效果差。

**现有痛点**：新兴的 DiT 架构在视频生成中展现了巨大潜力，但将身份控制信号迁移到 DiT 上面临两个核心问题：(1) DiT 缺少 U-Net 的长跳跃连接，低层特征难以聚合，训练收敛困难；(2) Transformer 对高频信息感知能力弱，而高频信息对保持人脸细节至关重要。

**核心矛盾**：U-Net 通过编码器-解码器架构天然具备多尺度特征和高频感知能力，而 DiT 缺乏这些结构优势。直接将 U-Net 上的控制方案套用到 DiT 上行不通。

**本文目标** (1) 如何在 DiT 架构上实现免微调的 IPT2V？ (2) 如何设计频率感知的控制方案来弥补 DiT 的结构缺陷？

**切入角度**：作者从视觉/扩散 Transformer 的频率分析研究中获得启发，发现浅层特征对应低频信息有助于训练收敛，而 Transformer 对高频信息感知不足。人脸特征恰好可以分解为低频（轮廓、比例）和高频（身份标记），这与 DiT 的缺陷形成互补。

**核心 idea**：将人脸身份特征按频率分解为高低频两部分，分别注入 DiT 的浅层输入和注意力块内部，实现频率感知的身份保持视频生成。

## 方法详解

### 整体框架

ConsisID 基于预训练的 CogVideoX-5B（DiT 架构）。给定参考人脸图像，系统通过两个互补的特征提取器分别获取低频和高频人脸信息：全局人脸提取器将参考图和面部关键点拼接到噪声 latent 作为低频信号输入；局部人脸提取器利用 ArcFace 和 CLIP 编码器融合后的高频特征，通过交叉注意力注入每个 Transformer 块。配合层次化训练策略，生成身份一致的视频。

### 关键设计

1. **全局人脸提取器（低频信号注入）**:

    - 功能：提供低频全局人脸信息（轮廓、比例），促进模型收敛
    - 核心思路：从参考图中提取人脸关键点并转化为 RGB 图像，与参考图一起经 VAE 编码后拼接到噪声 latent 上。关键点图像过滤了光照、阴影等无关噪声，让模型聚焦于低频面部结构信息。目标函数变为 $\mathcal{L}_b = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_0, t, \tau_\theta(y), \psi_\theta(f))\|^2]$
    - 设计动机：DiT 缺少 U-Net 的长跳跃连接，直接训练很难收敛。浅层低频信号的注入模拟了 U-Net 跳跃连接的作用，是模型能够训练的前提条件

2. **局部人脸提取器（高频信号注入）**:

    - 功能：补充高频人脸身份细节（眼睛纹理、嘴唇细节等内在身份标记）
    - 核心思路：使用双塔特征提取——ArcFace 提取与表情/姿态无关的内在身份特征，CLIP 编码器提取语义丰富的可编辑特征，通过 Q-Former 融合二者。融合后的特征通过每个注意力块中的交叉注意力与视觉 token 交互：$Z_i' = Z_i + \text{Attention}(Q_i^v, K_i^f, V_i^f)$。同时应用 Dropout 减轻 CLIP 无关特征的影响
    - 设计动机：Transformer 对高频信息感知弱，仅靠低频全局特征无法保持精细身份细节。通过在注意力块内部注入高频信号，引导注意力机制关注面部内在特征

3. **一致性训练策略（Coarse-to-Fine + Dynamic Loss）**:

    - 功能：分阶段训练 + 动态损失设计，提升训练效率和泛化能力
    - 核心思路：(a) 粗到细训练：先用全局提取器学低频特征，再引入局部提取器学高频细节；(b) 动态掩码损失：以概率 $\alpha$ 仅计算人脸区域损失 $\mathcal{L}_d = M \odot \mathcal{L}_c$，避免背景噪声干扰；(c) 动态交叉人脸损失：以概率 $\beta$ 使用训练帧之外的人脸作为参考图，并加入高斯噪声，防止模型学到"复制粘贴"的捷径
    - 设计动机：视频生成需要同时维持时空一致性，直接端到端训练太复杂。分阶段策略降低了学习难度，动态损失则分别解决了背景干扰和过拟合问题

### 损失函数 / 训练策略

最终损失函数 $\mathcal{L}_f$ 综合了动态掩码损失和动态交叉人脸损失。训练设置：分辨率 480×720，49 帧，batch size 80，学习率 $3 \times 10^{-6}$，总步数 1.8k，$\alpha = \beta = 0.5$。推理时使用 DPM sampler，50 步，CFG=6.0。

## 实验关键数据

### 主实验

| 方法 | FaceSim-Arc ↑ | FaceSim-Cur ↑ | CLIPScore ↑ | FID ↓ |
|------|---------------|---------------|-------------|-------|
| ID-Animator | 0.32 | 0.33 | 24.97 | 117.46 |
| **ConsisID** | **0.58** | **0.60** | **27.93** | 151.82 |

ConsisID 在身份保持指标上大幅领先 ID-Animator（FaceSim-Arc +81%），同时在文本相关性上也优于对手。用户研究中 103 份有效问卷显示 ConsisID 在所有维度均被偏好。

### 消融实验

| 配置 | FaceSim-Arc ↑ | FaceSim-Cur ↑ | CLIPScore ↑ | FID ↓ |
|------|---------------|---------------|-------------|-------|
| Full model (plan c) | 0.73 | 0.75 | 36.77 | 127.42 |
| w/o GFE (plan b) | 0.05 | 0.05 | 34.86 | 269.88 |
| w/o LFE (plan a) | 0.66 | 0.68 | 34.48 | 104.34 |
| w/o CFT | 0.54 | 0.58 | 34.47 | 144.62 |
| w/o DML | 0.62 | 0.67 | 34.23 | 187.78 |
| w/o DCL | 0.65 | 0.69 | 32.21 | 117.80 |

### 关键发现

- 去掉全局人脸提取器（GFE）后模型几乎无法收敛，FaceSim-Arc 从 0.73 骤降至 0.05，证明低频信号注入是训练的必要条件
- 高频信号注入位置至关重要：注入注意力块内部（plan c）远优于注入块输出（plan e）或块输入（plan f/g，导致梯度爆炸）
- 傅里叶谱分析直观验证了频率分解的有效性：注入高/低频信号确实增强了对应频段的信息

## 亮点与洞察

- **频率分解控制的思路非常巧妙**：将 DiT 的结构缺陷转化为设计优势——人脸的高低频分解恰好对应 DiT 需要补充的浅层和高频信息，形成了自然的互补关系
- **免微调的实用价值高**：基于预训练 CogVideoX-5B，仅需 1.8k 步训练即可获得免微调 IPT2V 能力，极大降低了使用门槛
- **频率分析的方法论可迁移**：这种从频率域分析模型缺陷再设计针对性补偿的思路，可以推广到其他 DiT 可控生成任务（如姿态控制、风格迁移）

## 局限与展望

- FID 指标反而不如 ID-Animator（151.82 vs 117.46），说明生成的视觉质量/多样性还有提升空间
- 仅在单人场景验证，未处理多人身份保持的情况
- 基于 CogVideoX-5B 固定架构，对其他 DiT 架构（如 HunyuanVideo）的适配性未验证
- 训练数据为内部人体数据集，数据规模和多样性可能限制泛化能力

## 相关工作与启发

- **vs ID-Animator**: ID-Animator 使用类似图像模型的方法做 IPT2V，只能生成人脸区域，无法控制动作/背景。ConsisID 通过频率分解专门为 DiT 设计控制方案，支持全身生成和丰富编辑
- **vs InstantID**: InstantID 在图像域使用 ArcFace + 姿态网络做身份保持生成。ConsisID 将其扩展到视频域，并引入频率分解来解决时序一致性问题
- 这篇论文对理解 DiT 内部的频率特性提供了有价值的实验证据，对后续 DiT 可控生成研究有参考意义

## 评分

- 新颖性: ⭐⭐⭐⭐ 频率分解控制的思路新颖且有理论支撑，但核心组件（ArcFace+CLIP+Q-Former）借鉴较多
- 实验充分度: ⭐⭐⭐⭐ 消融实验详尽，包含频率域可视化分析，但仅与 ID-Animator 一个开源方法对比
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，从 Finding 到设计的逻辑链完整
- 价值: ⭐⭐⭐⭐ 首个基于 DiT 的开源免微调 IPT2V 模型，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] EvoID: Reinforced Evolution for Identity-Preserving Video Generation](../../CVPR2026/video_generation/evoid_reinforced_evolution_for_identity-preserving_video_generation.md)
- [\[CVPR 2025\] HyperNVD: Accelerating Neural Video Decomposition via Hypernetworks](hypernvd_accelerating_neural_video_decomposition_via_hypernetworks.md)
- [\[CVPR 2026\] AnyID: Ultra-Fidelity Universal Identity-Preserving Video Generation from Any Visual References](../../CVPR2026/video_generation/anyid_ultra-fidelity_universal_identity-preserving_video_generation_from_any_vis.md)
- [\[CVPR 2025\] FADE: Frequency-Aware Diffusion Model Factorization for Video Editing](fade_frequency-aware_diffusion_model_factorization_for_video_editing.md)
- [\[CVPR 2026\] ConsID-Gen: View-Consistent and Identity-Preserving Image-to-Video Generation](../../CVPR2026/video_generation/consid-gen_view-consistent_and_identity-preserving_image-to-video_generation.md)

</div>

<!-- RELATED:END -->
