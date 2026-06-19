---
title: >-
  [论文解读] Pixel-level and Semantic-level Adjustable Super-resolution: A Dual-LoRA Approach
description: >-
  [CVPR 2025][图像恢复][图像超分辨率] 提出PiSA-SR，通过双LoRA模块将像素级回归和语义级增强解耦到两个独立权重空间，实现单步扩散高质量超分辨率，并支持推理时通过两个引导尺度灵活调节保真度和感知质量。 扩散先验的超分辨率面临核心矛盾： - 像素保真 vs 感知质量：$\ell_2$ 损失保证保真度但产生过…
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "图像超分辨率"
  - "双LoRA"
  - "可调节超分"
  - "扩散模型"
  - "保真度-感知平衡"
---

# Pixel-level and Semantic-level Adjustable Super-resolution: A Dual-LoRA Approach

**会议**: CVPR 2025  
**arXiv**: [2412.03017](https://arxiv.org/abs/2412.03017)  
**代码**: [GitHub](https://github.com/csslc/PiSA-SR)  
**领域**: Image Restoration  
**关键词**: 图像超分辨率, 双LoRA, 可调节超分, 扩散模型, 保真度-感知平衡

## 一句话总结

提出PiSA-SR，通过双LoRA模块将像素级回归和语义级增强解耦到两个独立权重空间，实现单步扩散高质量超分辨率，并支持推理时通过两个引导尺度灵活调节保真度和感知质量。

## 研究背景与动机

扩散先验的超分辨率面临核心矛盾：
- **像素保真 vs 感知质量**：$\ell_2$ 损失保证保真度但产生过度平滑，GAN/感知损失增强细节但引入伪影
- **目标纠缠**：现有SD-based方法将两个目标混合在一个扩散过程中，难以优化
- **用户偏好多样**：有人需要内容保真，有人需要丰富语义细节，但模型只能输出固定风格

核心贡献：将SR问题分解为两个可独立调节的LoRA模块，像CFG控制生成强度一样控制SR风格。

## 方法详解

### 整体框架

将SD-based SR公式化为残差学习 $z_H = z_L - \lambda \epsilon_\theta(z_L)$，在SD 2.1上训练两个LoRA：
1. Pixel-LoRA（$\Delta\theta_{pix}$）：用 $\ell_2$ 损失去除退化
2. Semantic-LoRA（$\Delta\theta_{sem}$）：用LPIPS+CSD损失增强语义细节
推理时通过 $\lambda_{pix}$ 和 $\lambda_{sem}$ 独立控制两个LoRA的输出强度。

### 关键设计1：残差学习公式化

- **功能**：将多步扩散SR简化为单步残差预测，支持推理时的输出缩放
- **核心思路**：一步扩散直接从LQ潜码 $z_L$ 得到HQ潜码 $z_H = z_L - \epsilon_\theta(z_L)$，U-Net预测的"噪声"实际是LQ到HQ的残差。训练时固定 $\lambda=1$，推理时调节 $\lambda$ 控制增强强度
- **设计动机**：残差学习让模型聚焦于高频信息，避免从LQ潜码中提取不相关信息，加速收敛。更重要的是，引入缩放因子 $\lambda$ 使得推理时可调节成为可能

### 关键设计2：解耦双LoRA训练

- **功能**：将像素保真和语义增强彻底分离到两个参数空间
- **核心思路**：先训练pixel-LoRA（$\ell_2$ loss，4K步），固定后再在PiSA-LoRA组合中仅训练semantic-LoRA（LPIPS + CSD loss，8.5K步）。推理时按CFG思路分解输出：$\epsilon_\theta = \lambda_{pix}\epsilon_{\theta_{pix}} + \lambda_{sem}(\epsilon_{\theta_{PiSA}} - \epsilon_{\theta_{pix}})$
- **设计动机**：先去退化再增细节的顺序保证语义增强不受噪声/模糊干扰。两个LoRA的差值 $\epsilon_{\theta_{PiSA}} - \epsilon_{\theta_{pix}}$ 精确分离出纯语义增强分量，实现正交控制

### 关键设计3：CSD损失替代VSD

- **功能**：高效利用预训练SD的语义先验进行语义增强
- **核心思路**：CSD（Classifier Score Distillation）损失通过SD的条件/无条件噪声预测差异提取语义梯度，无需VSD的双层优化。梯度 $\nabla\ell_{CSD} = \mathbb{E}[w_t(f(z_t, \epsilon_{real}) - f(z_t, \epsilon_{real}^{\lambda_{cfg}}))]$
- **设计动机**：VSD虽然有效但其 $\lambda_{cfg}=0$ 分量反而削弱语义细节；CSD的CFG分量才是语义增强的核心贡献。CSD避免了VSD的双层优化，显著降低显存占用和训练不稳定性

### 损失函数

- Pixel-LoRA：$\mathcal{L}_{pix} = \|z_H^{pix} - z_{GT}\|_2^2$
- Semantic-LoRA：$\mathcal{L}_{sem} = \mathcal{L}_{LPIPS} + \mathcal{L}_{CSD}$

## 实验关键数据

### 主实验：RealSR引导尺度效果

| $\lambda_{pix}$ | $\lambda_{sem}$ | PSNR↑ | LPIPS↓ | CLIPIQA↑ | MUSIQ↑ |
|---------|---------|-------|--------|----------|--------|
| 0.0 | 1.0 | 25.96 | 0.3426 | 0.4129 | 46.45 |
| 0.5 | 1.0 | 26.75 | 0.2646 | 0.5705 | 63.82 |
| 1.0 | 1.0 | 25.50 | 0.2672 | 0.6702 | 70.15 |
| 1.0 | 0.0 | ~28+ | ~0.35 | ~0.35 | ~40 |

$\lambda_{pix}$ 增大消除退化提升PSNR但过大过平滑；$\lambda_{sem}$ 增大丰富细节提升感知指标但过大引入伪影。

### 与SOTA方法对比（合成数据）

PiSA-SR在默认设置下PSNR/LPIPS/CLIPIQA/MUSIQ等指标上全面领先或持平于StableSR、DiffBIR、SeeSR、OSEDiff等方法，且仅需单步扩散。

### 关键发现

- 双LoRA差值确实纯粹分离出语义细节（可视化清晰展示）
- CSD优于VSD：更稳定、更少显存、语义增强更强
- 单步推理比多步方法更稳定且更快
- 用户研究验证了可调节SR的实用价值

## 亮点与洞察

1. **CFG思想迁移到SR**：将生成模型中CFG的条件/无条件分离思路应用于SR的像素/语义分离，创新自然
2. **可调节≠重训练**：两个引导尺度在推理时即可调节，无需为不同偏好分别训练模型
3. **LoRA解耦的通用性**：双LoRA解耦不同优化目标的策略可推广到其他恢复任务

## 局限与展望

- 推理时需运行两次U-Net（pixel-LoRA和PiSA-LoRA各一次）才能实现可调节
- $\lambda_{pix}$ 和 $\lambda_{sem}$ 的最优值因图像而异，不存在通用最优设置
- 仅支持 ×4 放大，对其他尺度需重新训练
- 基于SD 2.1，未探索SDXL或更新基座模型

## 相关工作与启发

- **OSEDiff**：单步扩散SR的先驱，PiSA-SR在其基础上引入双LoRA解耦
- **SeeSR**：语义tag引导的扩散SR，PiSA-SR用CSD损失实现更直接的语义增强
- **LDL**：GAN-based SR中局部统计约束的方法，与PiSA-SR的像素级LoRA目标相关

## 评分

⭐⭐⭐⭐ — 像素/语义解耦的思路简洁而有效，可调节SR的实用性强，单步推理效率高。双次U-Net推理的额外开销和参数敏感性是小缺陷。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Accelerating Image Super-Resolution Networks with Pixel-Level Classification](../../ECCV2024/image_restoration/accelerating_image_super-resolution_networks_with_pixel-level_classification.md)
- [\[CVPR 2025\] AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution](adversarial_diffusion_compression_for_real-world_image_super-resolution.md)
- [\[NeurIPS 2025\] Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start](../../NeurIPS2025/image_restoration/real-world_adverse_weather_image_restoration_via_dual-level_reinforcement_learni.md)
- [\[CVPR 2025\] Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging](proximal_algorithm_unrolling_flexible_and_efficient_reconstruction_networks_for_.md)
- [\[CVPR 2025\] Augmenting Perceptual Super-Resolution via Image Quality Predictors](augmenting_perceptual_super-resolution_via_image_quality_predictors.md)

</div>

<!-- RELATED:END -->
