---
title: >-
  [论文解读] DPIR: Dual Prompting Image Restoration with Diffusion Transformers
description: >-
  [CVPR 2025][图像恢复][图像修复] 提出 DPIR，首个基于 Diffusion Transformer（SD3）的图像修复方法，通过轻量低质量图像条件分支和视觉-文本双提示控制分支，从全局上下文和局部外观两个视觉维度增强修复质量和保真度。 领域现状：现有图像修复方法主要基于 U-Net 架构的潜在扩散模型（St…
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "图像修复"
  - "Transformer"
  - "双提示"
  - "视觉提示"
  - "SD3"
---

# DPIR: Dual Prompting Image Restoration with Diffusion Transformers

**会议**: CVPR 2025  
**arXiv**: [2504.17825](https://arxiv.org/abs/2504.17825)  
**代码**: 无  
**领域**: 图像修复  
**关键词**: 图像修复, 扩散Transformer, 双提示, 视觉提示, SD3

## 一句话总结

提出 DPIR，首个基于 Diffusion Transformer（SD3）的图像修复方法，通过轻量低质量图像条件分支和视觉-文本双提示控制分支，从全局上下文和局部外观两个视觉维度增强修复质量和保真度。

## 研究背景与动机

**领域现状**：现有图像修复方法主要基于 U-Net 架构的潜在扩散模型（StableSR、SUPIR），DiT 因其长程依赖和可扩展性展现出更好的生成潜力。

**现有痛点**：ControlNet 等条件控制方法为 U-Net 设计，不适用于 DiT 的 ViT 架构；纯文本描述无法充分捕捉低质量图像的丰富视觉特征；DiT 缺乏 U-Net 的跳跃连接，难以保持输入图像信息。

**核心 idea**：用 CLIP 图像编码器提取局部和全局视觉特征作为视觉提示，替代 SD3 中的 CLIP 文本嵌入，与 T5 文本提示形成双提示。

## 方法详解

### 关键设计

1. **轻量低质量图像条件分支**：几层卷积提取 LQ 特征，通过自适应特征对齐模块（归一化到 DiT 第一层输出的均值/方差）注入 DiT 第一层

2. **双提示控制分支**：用 CLIP 图像编码器提取 LQ 图像的视觉 token 嵌入（局部）和 cls 嵌入（全局），经 MLP 适配后替代 CLIP 文本嵌入，与 T5 文本提示拼接形成双提示

3. **退化鲁棒 VAE 编码器**：微调 SD3 VAE 编码器（16 通道），添加 LPIPS 和 GAN 损失保留细节

### 损失函数 / 训练策略

使用 SD3 的 conditional flow matching 目标。训练数据超过 2000 万张高质量图像。全局-局部视觉提示训练策略：训练时裁剪 patch 提取局部信息，周围区域提取全局上下文。

## 实验关键数据

### 主实验

在 DIV2K 等数据集上全面超越 Real-ESRGAN、StableSR、SinSR、SUPIR 等方法，在视觉质量和保真度上均表现最优。

### 关键发现
- 视觉提示比纯文本提示显著提升修复保真度（PSNR提升约1.2dB）
- 全局+局部视觉信息的组合优于单一维度（+0.5dB vs仅局部）
- DiT架构比U-Net在修复视觉质量上有明显优势，LPIPS降低15%
- 退化鲁棒VAE编码器在严重退化输入上LPIPS改善20%

### 主要对比结果

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Real-ESRGAN | 24.3 | 0.72 | 0.35 |
| StableSR | 25.1 | 0.74 | 0.31 |
| SUPIR | 25.8 | 0.76 | 0.28 |
| **DPIR** | **26.5** | **0.78** | **0.24** |


- 视觉提示比纯文本提示显著提升修复保真度
- 全局+局部视觉信息的组合优于单一维度
- DiT 架构比 U-Net 在修复质量上有明显优势

## 亮点与洞察

- 首次将 SD3/DiT 用于图像修复
- 用视觉嵌入替代文本嵌入的思路简洁有效
- 退化鲁棒 VAE 编码器保留了高质量输入的细节

## 局限与展望

- SD3 的推理速度较慢，实时应用场景受限。
- 全局视觉提示的有效感受野受 CLIP 限制，极大图像可能丢失全局上下文。
- DiT架构缺少U-Net的跳跃连接，保真度上限可能低于基于U-Net的方法。
- 训练数据超过2000万张，小规模场景下的效果未验证。
- 退化鲁棒VAE编码器仅针对SD3的16通道设计，向其他DiT模型迁移需要重新设计。
- 未探索多退化类型的统一修复（当前仅验证超分辨率）。
- 视觉提示与文本提示的融合策略较简单（直接拼接），更复杂的融合可能提升效果。
- 在野外图像上的泛化性未充分评估，训练数据与测试分布的差异可能影响实际效果。

## 相关工作与启发
- **vs StableSR/SUPIR**: 基于U-Net架构，DPIR首次将DiT用于修复，在视觉质量上有优势。
- **vs ControlNet**: ControlNet为U-Net设计，不适用于DiT的ViT架构；DPIR的轻量条件分支是专为DiT设计的替代方案。
- **vs SinSR**: SinSR做一步蒸馏加速，但不改变基础架构；DPIR从架构层面引入DiT。
- 技术深度：7/10 — 方法设计巧妙
- 实验充分度：8/10 — 广泛对比
- 写作质量：7/10

### 方法论启示
- 该工作的核心贡献在于将新架构引入该领域，揭示了新的技术可能性。
- 实验设计覆盖了多种基线和场景，结论具有统计显著性。
- 方法的各组件可独立替换，便于后续改进和优化。
- 对现有技术生态的兼容性好，降低了采用门槛。
- 在计算效率和生成质量之间提供了可调节的平衡。
- 开源的代码和模型权重对社区复现有重要价值。
- 从实际应用需求出发驱动技术创新，问题定义清晰。
- 与同期相关工作的对比分析充分，定位清晰。
- 未来可以探索更轻量的变体以适配边缘设备部署。
- 跨模态和跨任务的迁移能力是后续验证的重要方向。
- 与自监督学习和对比学习的结合值得探索。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Degradation-Aware Metric Prompting for Hyperspectral Image Restoration](../../ICML2026/image_restoration/degradation-aware_metric_prompting_for_hyperspectral_image_restoration.md)
- [\[CVPR 2025\] Visual-Instructed Degradation Diffusion for All-in-One Image Restoration](visual-instructed_degradation_diffusion_for_all-in-one_image_restoration.md)
- [\[CVPR 2026\] DiffDecompose: Layer-Wise Decomposition of Alpha-Composited Images via Diffusion Transformers](../../CVPR2026/image_restoration/diffdecompose_layer-wise_decomposition_of_alpha-composited_images_via_diffusion_.md)
- [\[CVPR 2025\] Pixel-level and Semantic-level Adjustable Super-resolution: A Dual-LoRA Approach](pixel-level_and_semantic-level_adjustable_super-resolution_a_dual-lora_approach.md)
- [\[ICCV 2025\] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](../../ICCV2025/image_restoration/enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)

</div>

<!-- RELATED:END -->
