---
title: >-
  [论文解读] Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models
description: >-
  [ECCV 2024][图像生成][视频超分辨率] 提出 StableVSR，首次将扩散模型应用于视频超分辨率任务，通过时序条件模块（TCM）和帧级双向采样策略，在显著提升感知质量的同时确保帧间时序一致性。 视频超分辨率（VSR）旨在提升视频的空间分辨率。现有方法存在一个核心矛盾： 感知质量低：当前 SOTA 的 VSR 方…
tags:
  - "ECCV 2024"
  - "图像生成"
  - "视频超分辨率"
  - "扩散模型"
  - "时间一致性"
  - "感知质量"
  - "纹理引导"
---

# Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models

**会议**: ECCV 2024  
**arXiv**: [2311.15908](https://arxiv.org/abs/2311.15908)  
**代码**: [GitHub](https://github.com/claudiom4sir/StableVSR)  
**领域**: 图像生成  
**关键词**: 视频超分辨率, 扩散模型, 时间一致性, 感知质量, 纹理引导

## 一句话总结

提出 StableVSR，首次将扩散模型应用于视频超分辨率任务，通过时序条件模块（TCM）和帧级双向采样策略，在显著提升感知质量的同时确保帧间时序一致性。

## 研究背景与动机

视频超分辨率（VSR）旨在提升视频的空间分辨率。现有方法存在一个核心矛盾：

**感知质量低**：当前 SOTA 的 VSR 方法（如 BasicVSR++、RVRT）专注于重建质量（PSNR/SSIM），生成的结果虽然像素距离近但视觉效果模糊、缺乏细节

**感知-失真权衡**：根据感知-失真权衡定理，在有限模型容量下，提升重建质量不可避免地导致感知质量下降

扩散模型（DM）在单图像超分辨率中已展现出合成逼真纹理和细节的能力，但直接将单图超分应用于视频存在两个问题：

**时序不一致**：每帧独立处理导致相邻帧生成的细节不一致，产生闪烁

**信息利用不足**：未利用视频帧间的时空冗余信息

StableVSR 正是为了在基于扩散模型的视频超分中同时实现**高感知质量**和**时序一致性**。

## 方法详解

### 整体框架

StableVSR 基于预训练的单图像超分 Latent Diffusion Model（Stable Diffusion ×4 Upscaler），通过添加**时序条件模块（TCM）**将其扩展为 VSR 方法。核心组件包括：

1. **Temporal Conditioning Module (TCM)**：注入相邻帧的时序信息到去噪 UNet 中
2. **Temporal Texture Guidance**：提供空间对齐的、细节丰富的纹理引导
3. **Frame-wise Bidirectional Sampling**：帧级双向采样策略，平衡信息传播

给定低分辨率帧序列 $\{LR\}_{i=1}^N$，目标是生成高分辨率序列 $\{\overline{HR}\}_{i=1}^N$。

### 关键设计

#### 时序条件模块 (TCM)

TCM 通过 ControlNet 的架构注入到去噪 UNet 的 decoder 中，目标是：
1. 利用多帧的时空信息改善单帧质量
2. 强制帧间的时序一致性

TCM 接收的输入是来自相邻帧的**时序纹理引导**（Temporal Texture Guidance），这是本文最核心的创新。

#### 时序纹理引导 (Temporal Texture Guidance)

这是方法的核心创新点。关键思路是：在采样步 $t$ 时，利用相邻帧的预测来引导当前帧的生成。

**基于 $\tilde{x}_0$ 而非 $x_t$ 的引导**：直接使用含噪的 $x_t$ 作为引导信息不好——它在大多数采样步中被噪声严重污染。解决方案是通过投影到初始状态获得无噪声的近似：

$$\tilde{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t))$$

$\tilde{x}_0$ 的优势：
- 几乎无噪声，无论 $t$ 多大
- 包含丰富的纹理细节信息
- 随 $t$ 减小逐渐精细化

**空间对齐**：由于视频运动，前一帧的纹理需要与当前帧空间对齐。具体流程：

1. 先将 $\tilde{x}_0^{i-1}$ 通过 VAE decoder $\mathcal{D}$ 解码到像素域
2. 在低分辨率帧 $LR^{i-1}$ 和 $LR^i$ 之间用 RAFT 估计光流
3. 对解码后的帧执行运动补偿

为什么不在潜在空间直接运动补偿？因为在潜在空间做运动补偿会引入伪影（如图4所示）。

完整公式：
$$\widetilde{HR}^{i-1 \rightarrow i} = MC(ME(LR^{i-1}, LR^i), \mathcal{D}(\tilde{x}_0^{i-1}))$$

#### 帧级双向采样策略

现有方案通常以自回归方式逐帧完成所有采样步，存在两个问题：
- **误差累积**：前一帧的错误会传播到后续所有帧
- **单向信息传播**：只从过去到未来传播，未利用未来帧的信息

StableVSR 的双向采样策略：

```
for t = T to 1:
    for i = 1 to N:  // 在所有帧上执行采样步 t
        计算时序纹理引导
        执行一步去噪
    end
    反转帧序列顺序  // 交替前向/后向传播
end
```

关键特征：
1. **帧级而非序列级**：在所有帧上先执行一个采样步，再推进到下一步
2. **双向交替**：每个采样步后反转帧顺序，交替进行前向和后向传播
3. **平衡信息流**：当前帧在前向传播中受过去帧引导，在后向传播中受未来帧引导

### 损失函数 / 训练策略

- **训练目标**：标准的扩散模型去噪损失 $\mathbb{E}[\|\epsilon - \epsilon_\theta(x_t^i, t, LR^i, \widetilde{HR}^{i-1 \rightarrow i})\|_2]$
- **仅训练 TCM**：预训练的 SISR LDM 权重冻结，只训练 ControlNet 部分
- **训练输入**：连续两帧对 $(LR^{i-1}, HR^{i-1})$ 和 $(LR^i, HR^i)$
- **训练细节**：Adam 优化器，lr=$1e-5$，batch=32，20000 步
- **数据增强**：$256 \times 256$ 随机裁剪 + 水平翻转
- **光流估计**：RAFT
- **采样**：训练时 DDPM T=1000，推理时 T=50
- **硬件**：4× NVIDIA Quadro RTX 6000

## 实验关键数据

### 主实验

**Vimeo-90K-T 数据集 (×4 超分)**

| 方法 | LPIPS↓ | DISTS↓ | tLP↓ | tOF↓ | PSNR↑ |
|------|--------|--------|------|------|-------|
| BasicVSR++ | 0.092 | 0.105 | 4.35 | 1.75 | **35.69** |
| RVRT | 0.088 | 0.101 | 4.28 | 1.42 | 36.30 |
| **StableVSR** | **0.070** | **0.087** | **3.89** | **1.37** | 31.97 |

**REDS4 数据集 (×4 超分)**

| 方法 | LPIPS↓ | DISTS↓ | tLP↓ | tOF↓ | PSNR↑ |
|------|--------|--------|------|------|-------|
| BasicVSR++ | 0.131 | 0.068 | 9.02 | 2.75 | **32.38** |
| RVRT | 0.128 | 0.067 | 8.97 | 2.72 | 32.74 |
| RealBasicVSR | 0.134 | 0.060 | 6.44 | 4.74 | 27.07 |
| **StableVSR** | **0.097** | **0.045** | **5.57** | **2.68** | 27.97 |

### 消融实验

论文在方法论层面进行了系统的设计验证：

1. **$\tilde{x}_0$ vs $x_t$ 引导**：$x_t$ 在大 $t$ 时被噪声污染，纹理信息不可用；$\tilde{x}_0$ 在所有步都提供清晰的细节
2. **像素域 vs 潜在空间运动补偿**：在潜在空间做运动补偿会产生严重伪影
3. **双向 vs 单向传播**：双向采样显著改善了时序一致性和感知质量

### 关键发现

1. **感知质量大幅领先**：StableVSR 在所有感知指标（LPIPS、DISTS）上大幅超越现有方法，LPIPS 在 REDS4 上比 RVRT 低 24%
2. **时序一致性更好**：tLP 和 tOF 两个时序一致性指标均最优，temporal profile 更规整
3. **感知-失真权衡**：PSNR/SSIM 有所下降，但仍优于双三次插值和 RealBasicVSR，符合理论预期
4. **生成式能力**：能合成低分辨率帧中不存在的语义一致的细节（如文字纹理、面部细节）

## 亮点与洞察

1. **首次将 DM 用于 VSR**：开创性地在生成范式下处理视频超分，跳出了传统重建方法的框架限制
2. **$\tilde{x}_0$ 引导的技巧**：通过投影到初始状态获得无噪声的引导信号，这一设计巧妙且通用
3. **帧级双向采样**：简洁的策略有效解决了误差累积和信息传播不平衡的问题
4. **在像素域做运动补偿**的发现：潜在空间的空间结构不适合直接做运动补偿，这一实验发现对后续工作有重要参考价值

## 局限与展望

1. **推理速度慢**：每帧需要多次 VAE 解码和光流估计，计算开销显著
2. **采样步数仍然较多**：推理用 T=50，进一步加速（如使用一致性模型）是重要方向
3. **光流误差敏感**：运动补偿依赖光流质量，遮挡区域可能引入错误引导
4. **仅处理 ×4 超分**：未验证在其他放大倍数下的效果
5. **长视频处理**：帧级采样在长视频上的内存和时间开销会更大

## 相关工作与启发

- **StableSR**：利用预训练 text-to-image DM 做单图超分，本文在此基础上扩展到视频
- **BasicVSR/BasicVSR++**：VSR 的基线方法，提出了双向传播框架
- **ControlNet**：本文的 TCM 采用 ControlNet 架构注入时序条件
- 启发：$\tilde{x}_0$ 引导策略可推广到视频编辑、视频生成等其他需要时序一致性的扩散模型应用

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4.5 |
| 理论深度 | 3.5 |
| 实验充分性 | 4 |
| 实用价值 | 4 |
| 写作质量 | 4.5 |
| 总体评分 | 4.1 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] STCDiT: Spatio-Temporally Consistent Diffusion Transformer for High-Quality Video Super-Resolution](../../CVPR2026/image_generation/stcdit_spatio-temporally_consistent_diffusion_transformer_for_high-quality_video.md)
- [\[ECCV 2024\] XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](xpsr_crossmodal_priors_for_diffusionbased_image_superresolut.md)
- [\[ECCV 2024\] Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)
- [\[ECCV 2024\] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [\[ECCV 2024\] DCDM: Diffusion-Conditioned-Diffusion Model for Scene Text Image Super-Resolution](dcdm_diffusion-conditioned-diffusion_model_for_scene_text_image_super-resolution.md)

</div>

<!-- RELATED:END -->
