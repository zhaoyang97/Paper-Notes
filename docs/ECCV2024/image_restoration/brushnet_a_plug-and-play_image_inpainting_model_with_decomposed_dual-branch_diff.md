---
title: >-
  [论文解读] BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion
description: >-
  [ECCV2024][图像恢复][图像修复] 提出 BrushNet，一种即插即用的双分支扩散模型图像修复架构，通过将遮罩图像特征提取与图像生成解耦到独立分支，实现逐层像素级特征注入，在图像质量、遮罩区域保持和文本对齐三方面全面超越已有方法。 图像修复（image inpainting）旨在恢复图像中缺失区域…
tags:
  - "ECCV2024"
  - "图像恢复"
  - "图像修复"
  - "扩散模型"
  - "dual-branch"
  - "plug-and-play"
  - "masked image feature"
---

# BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion

**会议**: ECCV2024  
**arXiv**: [2403.06976](https://arxiv.org/abs/2403.06976)  
**代码**: [TencentARC/BrushNet](https://github.com/TencentARC/BrushNet)  
**领域**: 图像生成  
**关键词**: image inpainting, diffusion models, dual-branch, plug-and-play, masked image feature

## 一句话总结

提出 BrushNet，一种即插即用的双分支扩散模型图像修复架构，通过将遮罩图像特征提取与图像生成解耦到独立分支，实现逐层像素级特征注入，在图像质量、遮罩区域保持和文本对齐三方面全面超越已有方法。

## 背景与动机

图像修复（image inpainting）旨在恢复图像中缺失区域，同时保持整体一致性。扩散模型的兴起为该任务带来显著进步，但现有方法存在两大流派的固有缺陷：

1. **采样策略修改方法**（如 Blended Latent Diffusion）：在每一步去噪时将遮罩区域从预训练模型采样、未遮罩区域直接复制粘贴。该方法可适配任意扩散骨干，但由于缺乏对遮罩边界和未遮罩区域上下文的感知，产生的结果语义不连贯。
2. **专用修复模型**（如 SD Inpainting, PowerPaint）：通过扩展 UNet 输入通道维度来融合遮罩图像和 mask 信息。虽然生成质量更好，但存在三个关键问题：
    - 在 UNet 首层就将 noisy latent、masked image latent、mask 和 text 混合，导致后续层无法获取纯净的遮罩图像特征
    - 单分支同时处理条件和生成，增加 UNet 学习负担
    - 需要针对不同扩散骨干分别微调，迁移性差

作者发现 ControlNet 虽然引入了额外分支，但其设计面向稀疏结构控制（如骨架），不适用于需要像素级密集约束的修复任务，直接用于 inpainting 效果不佳。

## 核心问题

如何设计一种即插即用的图像修复架构，使其能够高效提取遮罩图像的像素级特征并注入任意预训练扩散模型，同时保持未遮罩区域的一致性和生成区域的语义连贯性？

## 方法详解

### 整体架构

BrushNet 采用双分支策略：一个冻结的预训练 UNet 负责图像生成，一个可训练的 BrushNet 分支负责遮罩图像特征提取。两个分支的特征逐层融合。

### 三大核心设计

**1. VAE 编码器处理遮罩图像**

与 ControlNet 使用随机初始化卷积层不同，BrushNet 使用预训练 VAE 编码器将遮罩图像映射到隐空间，保证输入特征与预训练 UNet 的数据分布对齐。BrushNet 的输入为三者拼接：noisy latent $z_t$、masked image latent $z_0^{masked}$、下采样后的 mask $m^{resized}$。

**2. 逐层全量特征插入**

BrushNet 将额外分支的特征**逐层**加入预训练 UNet 的**每一层**（而非像 ControlNet 只在部分层添加残差），实现密集的像素级控制。特征插入公式：

$$\epsilon_\theta(z_t, t, C)_i = \epsilon_\theta(z_t, t, C)_i + w \cdot \mathcal{Z}(\epsilon_\theta^{BrushNet}([z_t, z_0^{masked}, m^{resized}], t)_i)$$

其中 $\mathcal{Z}$ 为 zero convolution，$w$ 为控制尺度，$i$ 为层索引。

**3. 移除 cross-attention 层**

BrushNet 分支基于预训练 UNet 的克隆，但移除了所有 text cross-attention 层，确保该分支仅处理纯图像信息，避免文本嵌入对遮罩图像特征的干扰。

### 模糊混合策略（Blurred Blending）

由于 VAE 编解码和 mask 下采样存在固有误差，直接在 latent space 做混合会导致未遮罩区域失真。BrushNet 提出在像素空间先对 mask 做高斯模糊，再用模糊后的 mask 进行 copy-and-paste。虽然 mask 边界会有微小精度损失，但肉眼几乎不可察觉，换来的是显著更好的边界连贯性。

### 灵活控制能力

- **即插即用**：不修改预训练模型权重，可直接适配社区微调的各种扩散模型
- **保留尺度调节**：通过 $w$ 参数控制 BrushNet 对预训练模型的影响程度
- **混合操作可选**：通过调整模糊尺度和是否开启 blending，进一步定制保留程度

## 实验关键数据

### 评测设定

- 提出 **BrushBench**（600 张图，含人、动物、室内、室外，均衡自然/艺术图像）用于分割 mask 修复评测
- 提出 **BrushData**（基于 LAION-Aesthetic 的分割 mask 标注）用于训练
- 使用 **EditBench**（240 张图）评测随机 mask 修复
- 7 项指标覆盖三方面：图像质量（IR, HPS, AS）、遮罩保持（PSNR, LPIPS, MSE）、文本对齐（CLIP Sim）

### 主要结果（BrushBench inside-inpainting）

| 方法 | IR↑ | HPS↑ | AS↑ | PSNR↑ | CLIP Sim↑ |
|------|-----|------|-----|-------|-----------|
| BLD | 9.78 | 25.87 | 6.17 | 21.33 | 26.15 |
| SD Inpainting | 11.72 | 27.06 | 6.50 | 21.52 | 26.17 |
| HD-Painter | 11.68 | 26.90 | 6.42 | 22.61 | 26.37 |
| PowerPaint | 11.46 | 27.35 | 6.24 | 21.43 | 26.48 |
| ControlNet-Inp | 11.21 | 26.92 | 6.39 | 22.73 | 26.22 |
| **BrushNet** | **12.36** | **27.40** | **6.53** | 21.65 | **26.48** |
| **BrushNet*** | **12.64** | **27.78** | 6.51 | **31.94** | 26.39 |

*带 blending 操作

### EditBench 结果

BrushNet 在 EditBench 上同样全面领先：IR 达 4.40（次优 SDI 仅 1.86），HPS 25.10，CLIP Sim 28.67，均为最佳。加 blending 后 PSNR 达 33.66，远超其他方法（~23）。

### 消融实验要点

- 双分支 vs 单分支：双分支设计在所有指标上均优于单分支 SD Inpainting
- VAE vs Conv 编码器：VAE 在 PSNR（14.89→17.96）和 LPIPS 上大幅领先
- 全量特征注入 vs ControlNet 式注入（full > half > CN）：full 在 PSNR 19.86 vs CN 18.49
- 移除 cross-attention 效果更优：保留 cross-attention 反而降低图像质量指标
- 模糊混合 vs 直接粘贴：blur blending 在 PSNR（29.88）和连贯性上最优

## 亮点

1. **架构创新清晰有力**：将遮罩图像特征和生成过程解耦到双分支，动机分析到位，设计合理
2. **即插即用**：冻结预训练模型权重，可无缝适配社区各种微调模型（DreamShaper、MeinaMix 等）
3. **消融实验充分**：逐一验证了 VAE 编码、full 特征注入、移除 cross-attention、模糊混合等每个设计选择
4. **全面的评测体系**：提出 BrushBench 和 BrushData，区分 inside/outside inpainting，7 项指标覆盖三个维度
5. **背景感知能力**：定性结果中，BrushNet 能识别遮罩图像中已有的物体（如金鱼），避免重复生成

## 局限与展望

1. **生成质量依赖基础模型**：BrushNet 作为插件，其输出质量与选择的预训练模型强相关，若基础模型域不匹配（如用动漫模型处理自然图像），结果会不连贯
2. **异形/不规则 mask 处理欠佳**：对形状异常的 mask 仍可能产生较差的生成结果
3. **文本与遮罩图像不匹配时效果下降**：当文本提示与 masked image 内容冲突时，生成质量受影响
4. **仅基于 SD 1.5 验证**：未在 SDXL 或更新架构上验证泛化性
5. **模糊混合的 mask 边界精度**：虽然肉眼难以察觉，但在精细编辑场景可能积累误差

## 与相关工作的对比

| 特性 | BLD | SD Inpainting | PowerPaint | ControlNet-Inp | BrushNet |
|------|-----|---------------|------------|----------------|----------|
| 即插即用 | ✓ | ✗ | ✗ | ✓ | ✓ |
| 灵活尺度 | ✗ | ✗ | ✗ | ✗ | ✓ |
| 内容感知 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 形状感知 | ✗ | ✓ | ✓ | ✓ | ✓ |

BrushNet 是唯一同时具备即插即用、灵活尺度、内容感知和形状感知四项能力的方法。与 ControlNet-Inpainting 相比，BrushNet 使用 VAE 编码（而非随机初始化 Conv）、全量逐层特征注入（而非仅 decoder 残差）、移除 cross-attention，三项改进使其在修复任务上显著优于 ControlNet 的迁移方案。

## 启发与关联

1. **双分支解耦思想**具有通用性，可迁移到其他条件生成任务（如 image-to-image translation、虚拟试穿）中，凡是需要像素级条件注入的场景都可借鉴
2. **移除 cross-attention 保持纯图像特征**的设计启示：在多模态条件注入时，应根据目标信息类型选择性保留或去除注意力机制
3. **VAE 编码对齐分布**的策略比随机初始化编码器更有效，这对 ControlNet 类架构的改进具有参考价值
4. 全量逐层注入 vs 稀疏注入的对比结果表明，密集控制任务（如修复）需要更强的特征耦合，与稀疏控制任务（如姿态引导）有本质区别

## 评分
- 新颖性: ⭐⭐⭐⭐ — 双分支解耦+三项针对性设计，架构创新明确
- 实验充分度: ⭐⭐⭐⭐⭐ — 新 benchmark、7 指标、丰富消融、多域定性结果
- 写作质量: ⭐⭐⭐⭐ — 动机分析清晰，与 ControlNet 的差异阐述到位
- 价值: ⭐⭐⭐⭐ — 即插即用设计实用性强，已开源且社区采用度高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Adaptive Moments are Surprisingly Effective for Plug-and-Play Diffusion Sampling](../../ICLR2026/image_restoration/adaptive_moments_are_surprisingly_effective_for_plug-and-play_diffusion_sampling.md)
- [\[CVPR 2026\] PnP-CM: Consistency Models as Plug-and-Play Priors for Inverse Problems](../../CVPR2026/image_restoration/pnp-cm_consistency_models_as_plug-and-play_priors_for_inverse_problems.md)
- [\[ECCV 2024\] MambaIR: A Simple Baseline for Image Restoration with State-Space Model](mambair_a_simple_baseline_for_image_restoration_with_state-space_model.md)
- [\[ECCV 2024\] Unrolled Decomposed Unpaired Learning for Controllable Low-Light Video Enhancement](unrolled_decomposed_unpaired_learning_for_controllable_low-light_video_enhanceme.md)
- [\[CVPR 2025\] DPIR: Dual Prompting Image Restoration with Diffusion Transformers](../../CVPR2025/image_restoration/dpir_dual_prompting_restoration_dit.md)

</div>

<!-- RELATED:END -->
