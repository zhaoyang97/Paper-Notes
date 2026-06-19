---
title: >-
  [论文解读] Unrolled Decomposed Unpaired Learning for Controllable Low-Light Video Enhancement
description: >-
  [ECCV 2024][图像恢复][低光视频增强] 提出 UDU-Net，将低光视频增强建模为 MAP 优化问题并展开为深度网络，通过 Intra/Inter 子网分别处理空间（光照）和时序（一致性）退化，支持无配对训练和人类感知反馈的可控增强。 低光视频增强面临三重挑战： 配对数据获取困难：相比静态图像…
tags:
  - "ECCV 2024"
  - "图像恢复"
  - "低光视频增强"
  - "无配对学习"
  - "深度展开"
  - "时序一致性"
  - "人类感知反馈"
---

# Unrolled Decomposed Unpaired Learning for Controllable Low-Light Video Enhancement

**会议**: ECCV 2024  
**arXiv**: [2408.12316](https://arxiv.org/abs/2408.12316)  
**代码**: [有](https://github.com/lingyzhu0101/UDU.git)  
**领域**: 图像恢复  
**关键词**: 低光视频增强, 无配对学习, 深度展开, 时序一致性, 人类感知反馈

## 一句话总结

提出 UDU-Net，将低光视频增强建模为 MAP 优化问题并展开为深度网络，通过 Intra/Inter 子网分别处理空间（光照）和时序（一致性）退化，支持无配对训练和人类感知反馈的可控增强。

## 研究背景与动机

低光视频增强面临三重挑战：

**配对数据获取困难**：相比静态图像，获取运动场景下的低光/正常光配对视频更加困难，使得无配对学习成为必要的技术路线

**时空退化交织**：噪声、曝光不足和对比度不均在空间域中与时序一致性需求交织在一起

**过曝/欠曝问题**：仅学习分布对齐的方法缺少像素级约束和人类感知反馈，容易导致曝光异常

现有方法的局限：
- **图像增强方法**直接应用于视频会忽略帧间时序上下文，导致增强结果不一致
- **深度展开方法**在低光增强中已有探索（如 Retinex-inspired），但存在模块分离训练收敛局部最优、未能正确建模反射率-光照关系等问题
- 尚无方法在深度展开架构中同时解决低光视频的空间和时序退化

## 方法详解

### 整体框架

UDU-Net 基于 MAP 估计框架，将低光视频增强建模为：

$$\hat{x} = \arg\min_x \frac{1}{2}\|y - Ax\|^2 + \lambda_s J_s(x) + \lambda_t J_t(x)$$

通过 ADMM 分解为三个子问题，展开为级联的 Intra 子网（单帧增强器）和 Inter 子网（多帧增强器），以阶段式方式从空间和时序两个视角逐步优化。

| 阶段 | 子网 | 功能 | 关键技术 |
|------|------|------|----------|
| Stage k | Intra (a) | 粗糙光照估计 | 从无配对专家修图数据学习光照分布 |
| Stage k | Inter (b) | 时序平滑学习 | 3D 卷积 + 光流对齐 + 掩码机制 |
| Stage k+1 | Intra (c) | 精细光照优化 | 人类感知反馈引导的可控增强 |
| Stage k+1 | Inter (d) | 时序细节补偿 | 同 Inter (b)，进一步时序精化 |

### 关键设计

**1. Intra 子网 — 空间先验: 无配对修图光照**

- Stage k：使用 MIT-Adobe FiveK 数据集（Expert C 修图版本）作为无配对数据源，通过对抗学习让增强结果匹配专家修图的光照分布
- 损失包含：语义自监督损失（VGG 特征匹配）、内容自监督损失（多尺度 L1）、Relativistic Average HingeGAN 判别器

**2. 人类感知反馈机制**

Stage k+1 引入可控光照调整：
- 通过 gamma 校正和线性缩放生成目标帧：$m_i^{k+1} = \beta \times (\alpha \times \tilde{x}_i^{k+1})^\gamma$
- 参数从 U(1.00, 1.10) 均匀分布中采样
- 使用 BRISQUE 无参考质量评估模型模拟人类视觉系统反馈
- 自动选择 BRISQUE 分数最低（质量最好）的版本作为优化目标

**3. Inter 子网 — 时序先验: 时序线索探索**

- 使用预训练 RAFT 光流模型估计帧间运动，训练时对其微调
- 第一阶段：3D 卷积聚合 5 个对齐的邻近帧，估计无噪声结构信号 s_t
- 第二阶段：基于结构信号 s_t 和邻近帧，估计补偿细节残差
- 引入软掩码 M_t 平衡噪声抑制和纹理保留

**4. 三维噪声抑制策略**

- **空间域对抗学习**：通过无配对数据让模型减少生成帧中的噪声
- **时序一致性学习**：对齐并融合邻近帧以进一步降噪
- **掩码机制**：基于结构信号差异的指数函数掩码，参数 ω 控制纹理-噪声平衡

### 损失函数 / 训练策略

**Intra 子网损失（Stage k）**：
- 语义自监督损失 L_semantic-G：VGG 特征的 L2 距离
- 内容自监督损失 L_content-G：3 尺度 L1 损失
- 对抗损失：Relativistic Average HingeGAN

**Intra 子网损失（Stage k+1）**：
- 内容自监督损失：与人类感知反馈目标的 L1 距离

**Inter 子网损失**：
- 光流损失 L_flow-R：对齐帧与当前帧的 L1 距离
- 时序内容损失 L_content-T：结构信号约束 + 掩码加权的帧间一致性 L1 损失

## 实验关键数据

### 主实验（表格）

SDSD 数据集上的定量结果（无参考方法对比）：

| 方法 | Outdoor PSNR↑ | Outdoor SSIM↑ | Outdoor warp↓ | Indoor PSNR↑ | Indoor SSIM↑ |
|------|---------------|---------------|---------------|--------------|--------------|
| EnlightenGAN | 18.63 | 0.5399 | 4.49 | 19.59 | 0.5874 |
| CLIP-LIT | 20.88 | 0.5872 | 3.36 | 19.08 | 0.4582 |
| SRIE | 21.89 | 0.6288 | 2.74 | 15.78 | 0.6294 |
| **UDU-Net (Ours)** | **23.94** | **0.7446** | **0.24** | **22.41** | **0.7368** |
| SDSDNet (监督)* | 24.30 | 0.7445 | 0.95 | 27.03 | 0.7788 |

### 消融实验（表格）

| 配置 | PSNR | SSIM | warp |
|------|------|------|------|
| Ours-v1 (仅 Intra-a) | 基础 | 基础 | 基础 |
| Ours-v2 (+Inter-b) | 提升 | 提升 | 显著降低 |
| Ours-v3 (+Intra-c) | 进一步提升 | 进一步提升 | 降低 |
| Default (+Inter-d +人类感知) | 最优 | 最优 | 最优 |
| Default w/o H (无人类感知) | 略低 | 略低 | 略低 |

### 关键发现

- UDU-Net 在 SDSD 室外场景上 PSNR 达到 23.94 dB，比第二好的无参考方法 SRIE 提升 2.05 dB
- 时序质量指标 warp error 仅 0.24，远低于第二好方法（SRIE 的 2.74），展现了卓越的时序一致性
- 在室外场景上达到了与监督方法 SDSDNet 可比甚至略优的 SSIM（0.7446 vs 0.7445）
- 人类感知反馈机制有效抑制了过曝/欠曝，BRISQUE 分数引导使增强结果更符合人类视觉偏好
- 每个组件（Intra-a, Inter-b, Intra-c, Inter-d）都对最终性能有正向贡献

## 亮点与洞察

1. **展开方法的创新应用**：首次将 MAP 优化的展开架构用于低光视频增强，同时建模空间和时序约束
2. **无配对 + 可控**：不依赖成对数据的同时，通过人类感知反馈实现可控增强，这是实用性极强的设计
3. **噪声抑制的系统化**：从空间域（对抗学习）、时序域（帧融合）和掩码机制三个维度协同降噪
4. **BRISQUE 作为代理人类视觉**：使用无参考质量评估模型模拟人类感知，自动调整目标光照水平
5. **端到端训练**：避免了模块分离训练导致的局部最优问题

## 局限与展望

- 对相机运动剧烈或遮挡严重的场景，光流估计可能不准确
- 人类感知反馈依赖 BRISQUE 模型，该模型本身有局限
- MIT-Adobe FiveK 的 expert-retouched 风格可能不适用于所有应用场景
- gamma 校正参数范围 U(1.00, 1.10) 是经验设定，可能需要针对不同场景调整
- 未与最新的扩散模型方法对比

## 相关工作与启发

- **深度展开方法**：PnP、ADMM 框架启发了将优化问题展开为可训练深度网络的思路
- **EnlightenGAN**：无配对增强的先驱，使用双判别器，但不考虑时序
- **StableLLVE**：使用光流模拟动态场景以增强时序稳定性
- **Zero-DCE**：将增强形式化为曲线估计问题
- **Retinex 分解**在展开框架中的局限性为 UDU-Net 设计提供了动机

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 新颖性 | 4 |
| 理论深度 | 4 |
| 实验充分度 | 4 |
| 实用性 | 4 |
| 写作质量 | 3.5 |
| 总体 | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Towards Real-world Event-guided Low-light Video Enhancement and Deblurring](towards_real-world_event-guided_low-light_video_enhancement_and_deblurring.md)
- [\[ECCV 2024\] Learning to Robustly Reconstruct Dynamic Scenes from Low-Light Spike Streams](learning_to_robustly_reconstruct_dynamic_scenes_from_low-light_spike_streams.md)
- [\[ICML 2026\] AnyMod-LLVE: Low-Light Video Enhancement with Modality-Agnostic Inference](../../ICML2026/image_restoration/anymod-llve_low-light_video_enhancement_with_modality-agnostic_inference.md)
- [\[CVPR 2025\] Classic Video Denoising in a Machine Learning World: Robust, Fast, and Controllable](../../CVPR2025/image_restoration/classic_video_denoising_in_a_machine_learning_world_robust_fast_and_controllable.md)
- [\[CVPR 2026\] BiEvLight: Bi-level Learning of Task-Aware Event Refinement for Low-Light Image Enhancement](../../CVPR2026/image_restoration/bievlight_bi-level_learning_of_task-aware_event_refinement_for_low-light_image_e.md)

</div>

<!-- RELATED:END -->
