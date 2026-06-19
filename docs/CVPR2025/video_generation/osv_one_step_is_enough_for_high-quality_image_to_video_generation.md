---
title: >-
  [论文解读] OSV: One Step is Enough for High-Quality Image to Video Generation
description: >-
  [CVPR 2025][视频生成][单步视频生成] 提出两阶段训练框架 OSV，结合 GAN 对抗训练和一致性蒸馏，实现单步高质量图像到视频生成，并设计了无需解码的新型视频判别器。 视频扩散模型虽能生成高质量视频，但迭代去噪导致巨大的计算和时间成本——用 SVD 在 A100 上生成2秒视频需要30秒以上…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "单步视频生成"
  - "一致性蒸馏"
  - "对抗训练"
  - "视频扩散加速"
  - "GAN"
---

# OSV: One Step is Enough for High-Quality Image to Video Generation

**会议**: CVPR 2025  
**arXiv**: [2409.11367](https://arxiv.org/abs/2409.11367)  
**代码**: 无  
**领域**: 视频生成  
**关键词**: 单步视频生成, 一致性蒸馏, 对抗训练, 视频扩散加速, GAN

## 一句话总结

提出两阶段训练框架 OSV，结合 GAN 对抗训练和一致性蒸馏，实现单步高质量图像到视频生成，并设计了无需解码的新型视频判别器。

## 研究背景与动机

视频扩散模型虽能生成高质量视频，但迭代去噪导致巨大的计算和时间成本——用 SVD 在 A100 上生成2秒视频需要30秒以上。现有加速方法各有缺陷：一致性蒸馏 (LCM) 在低步数（1-2步）下效果差、训练收敛慢，且高 CFG 值导致曝光问题；GAN 对抗训练能快速收敛但训练不稳定且后期易模式崩溃，基于像素的判别器需要 VAE 解码增加内存和计算开销。更根本的问题是，现有视频加速方法大多简单移植自图像扩散模型，未充分考虑视频特性。本文通过两阶段解耦策略，在早期利用 GAN 快速提升质量，后期引入一致性蒸馏稳定训练并提升上限。

## 方法详解

### 整体框架

OSV 包含三个网络组件：学生模型 $\theta$、EMA 目标模型 $\theta^-$、冻结的教师模型 $\phi$，以及判别器 $\psi$。第一阶段使用 LoRA + GAN 预训练（真实数据作为 GAN 正样本），快速提升低步数生成质量。第二阶段引入一致性蒸馏损失（教师生成数据作为 GAN 正样本），稳定训练并提升性能上限。推理支持一步生成并可多步精化。

### 关键设计

1. **潜空间 GAN 判别器**：创新性地用简单上采样算子替代 VAE 解码器，直接将上采样的视频潜变量送入 DINOv2 预训练判别器（冻结骨干 + 可训练时空判别头）。相比 ADD 需 VAE 解码到像素空间、SF-V 需 UNet 编码器作为骨干，OSV 在 H800 上将每迭代时间从4.29秒降至2.61秒，GPU内存从73.5GB降至35.8GB，同时还避免了半精度训练中的浮点溢出。

2. **两阶段解耦训练**：第一阶段仅用 GAN 对抗损失 + Huber 损失，真实数据 $\mathbf{x}_0$ 为正样本，LoRA 训练确保保持教师知识同时快速收敛。第二阶段加入一致性蒸馏损失，EMA 模型输出为正样本，联合对抗损失精化。关键数学洞察：LGP 阶段对抗损失始终非零（即使模型达到一致性），会干扰后续一致性学习；ACD 阶段对抗损失随训练收敛到零，不会破坏一致性。

3. **多步 ODE 求解器 + Time Travel Sampler**：取消 CFG（观察到 CFG 对蒸馏模型有负面影响），用多步 ODE 求解代替单步求解，在相同训练时间内获得更高蒸馏精度。推理时设计 Time Travel Sampler (TTS)，利用低时间步的预测结果回退到高时间步再次求解，实现高阶求解的效果。

### 损失函数 / 训练策略

- 第一阶段: $\mathcal{L}_{\text{OSV}}^{g_1} = \lambda^{LGP} \cdot \text{ReLU}(1 - D_\psi(f_\theta(\mathbf{x}_{t_n}))) + d(\mathbf{x}_0, f_\theta(\mathbf{x}_{t_n}))$
- 第二阶段: $\mathcal{L}_{\text{OSV}}^{g_2} = \lambda^{ACD} \cdot \text{ReLU}(1 - D_\psi(f_\theta(\mathbf{x}_{t_{n+m}}))) + \lambda(t_n) d(f_{\theta^-}(\mathbf{x}_{t_n}^\phi), f_\theta(\mathbf{x}_{t_{n+m}}))$
- Huber 距离: $d(x,y) = \sqrt{\|x-y\|_2^2 + c^2} - c$
- 第一阶段 LoRA、第二阶段解冻部分层微调

## 实验关键数据

### 主实验

| 方法 | 步数 | FVD↓ | 备注 |
|------|------|------|------|
| AnimateLCM | 8 | 184.79 | 一致性蒸馏 |
| SVD | 25 | 156.94 | 原始扩散 |
| SF-V | 1 | 较高 | GAN 加速 |
| **OSV** | **1** | **171.15** | **单步即超过 AnimateLCM 8步** |
| **OSV** | **2+** | **更低** | **多步精化进一步提升** |

### 消融实验

| 设计选择 | FVD↓ |
|---------|------|
| 仅 LCM (无 GAN) | 差 |
| 仅 GAN (无 LCM) | 训练后期不稳定 |
| 像素空间判别 (ADD风格) | 较高 + 内存大 |
| 潜空间判别 (本文) | 更低 + 内存小 |
| 单步 ODE 求解 | 收敛慢 |
| 多步 ODE 求解 (m=5) | 收敛快且效果好 |

### 关键发现

- 单步 FVD 171.15 超过 AnimateLCM 8步的 184.79，逼近 SVD 25步的 156.94
- 潜空间判别器不仅省资源，性能还更好——DINOv2 特征足够判别质量
- GAN 在早期训练高效但后期不稳定，一致性蒸馏提供稳定的性能上限
- CFG 对蒸馏模型有害——移除 CFG 后反而更好

## 亮点与洞察

- 理论分析 LGP vs ACD 对抗损失的本质区别（非零 vs 收敛到零）指导了两阶段设计
- 潜空间直接做 GAN 判别是一个简单但有效的洞察，大幅降低训练成本
- 将视频加速的问题拆解为"快速初始化"和"精细调优"两个子问题是清晰的思路
- LoRA 在第一阶段防止模式崩溃（阻止生成静态图像）是实用的技巧

## 局限与展望

- 基于 SVD 架构，视频长度和分辨率受 SVD 限制
- 单步生成在动作丰富度上仍可能弱于25步的完整扩散
- Time Travel Sampler 增加了推理步数时的复杂度
- 可推广到文本到视频等其他视频生成任务

## 相关工作与启发

- **vs AnimateLCM**: 纯一致性蒸馏收敛慢，低步数效果差；OSV 通过 GAN 预热快速启动
- **vs SF-V**: 纯 GAN 方法训练不稳定且使用 UNet 编码器做判别器开销大；OSV 两阶段解耦+潜空间判别更稳定高效
- **vs ADD/LADD**: 像素空间对抗蒸馏需 VAE 解码，内存开销大；OSV 证明潜空间判别同样有效

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 两阶段解耦策略有理论支撑，潜空间判别器简单有效
- **实验充分度**: ⭐⭐⭐⭐ — 消融全面，OpenVid-1M 基准定量评估
- **写作质量**: ⭐⭐⭐⭐ — 动机分析深入，问题总结清晰
- **实用价值**: ⭐⭐⭐⭐⭐ — 单步视频生成有巨大实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step](videoscene_distilling_video_diffusion_model_to_generate_3d_scenes_in_one_step.md)
- [\[ICCV 2025\] Dual-Expert Consistency Model for Efficient and High-Quality Video Generation](../../ICCV2025/video_generation/dual-expert_consistency_model_for_efficient_and_high-quality_video_generation.md)
- [\[ICML 2025\] Diffusion Adversarial Post-Training for One-Step Video Generation](../../ICML2025/video_generation/diffusion_adversarial_post-training_for_one-step_video_generation.md)
- [\[CVPR 2025\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)
- [\[CVPR 2025\] Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion](zero-1-to-a_zero-shot_one_image_to_animatable_head_avatars_using_video_diffusion.md)

</div>

<!-- RELATED:END -->
