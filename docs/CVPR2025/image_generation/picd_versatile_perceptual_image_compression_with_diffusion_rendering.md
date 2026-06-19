---
title: >-
  [论文解读] PICD: Versatile Perceptual Image Compression with Diffusion Rendering
description: >-
  [CVPR 2025][图像生成][感知图像压缩] PICD 提出了一种通用的感知图像压缩框架，通过将文本信息无损编码并与压缩图像一起用扩散模型"渲染"融合，在三个层次（领域级、适配器级、实例级）改进条件扩散模型，同时实现屏幕内容和自然图像的高视觉质量与高文本精确度。 领域现状：感知图像压缩（Perceptual Image…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "感知图像压缩"
  - "扩散模型渲染"
  - "屏幕内容压缩"
  - "文本精确性"
  - "图像编解码"
---

# PICD: Versatile Perceptual Image Compression with Diffusion Rendering

**会议**: CVPR 2025  
**arXiv**: [2505.05853](https://arxiv.org/abs/2505.05853)  
**代码**: 无  
**领域**: 扩散模型 / 图像压缩  
**关键词**: 感知图像压缩, 扩散模型渲染, 屏幕内容压缩, 文本精确性, 图像编解码

## 一句话总结

PICD 提出了一种通用的感知图像压缩框架，通过将文本信息无损编码并与压缩图像一起用扩散模型"渲染"融合，在三个层次（领域级、适配器级、实例级）改进条件扩散模型，同时实现屏幕内容和自然图像的高视觉质量与高文本精确度。

## 研究背景与动机

**领域现状**：感知图像压缩（Perceptual Image Compression）利用生成模型（GAN 或扩散模型）作为解码器，在低比特率下维持高视觉质量，代表方法包括 MS-ILLM、PerCo、CDC 等。这些方法通过让重建图像的边际分布匹配原始图像分布来实现感知无损。

**现有痛点**：现有感知编解码器对自然图像效果好，但对屏幕内容（截图、网页等）效果差。核心问题在于：感知编解码器只保证边际分布匹配，不关心具体文字内容是否正确——比如原图中的字母"a"被重建成"c"也被认为是感知无损的，但这对屏幕内容显然不可接受。反过来，现有的屏幕内容编解码器（如 VTM-SCC）优先保证文本准确但感知质量差，低比特率下模糊严重。

**核心矛盾**：文本精确性和感知质量之间存在矛盾——前者要求精确重建特定像素，后者允许"合理的替代"。在 rate-distortion-perception 三方 trade-off 下，同时满足两者非常困难。

**本文目标**：设计一个对屏幕内容和自然图像都有效的通用感知编解码器，同时实现高文本精确性和高视觉质量。

**切入角度**：作者观察到文本信息本身熵很低（几 KB 就能无损编码），而且由 $H(Y|Z) + H(Z) = H(Y)$ 可知，先编码文本再条件编码图像在理论上不增加总比特率。因此可以将文本和图像分开编码，再用扩散模型将两者"渲染"成一张完整重建图像。

**核心 idea**：将文本信息用 OCR 提取后无损编码，将压缩图像和文本信息作为条件输入扩散模型，通过三层条件化改进实现高质量的"扩散渲染"。

## 方法详解

### 整体框架

PICD 的编码端：(1) 用 OCR（Tesseract）从屏幕图像中提取文本内容和位置信息 $Z$，无损压缩（cmix + 指数哥伦布编码）；(2) 以 $Z$ 为条件，用 MLIC 编解码器有损压缩图像 $X$，得到码流 $Y$ 和重建图像 $\bar{X}$。解码端：(1) 先解码文本 $Z$，生成文本字形图（glyph image）$\bar{Z}$；(2) 将 $\bar{X}$ 和 $\bar{Z}$ 送入条件扩散模型进行"扩散渲染"，生成最终重建 $\hat{X} \sim p_\theta(X|\bar{X}, Z)$。对自然图像，简化为不使用文本条件，使用 BLIP 生成的 caption 作为文本输入。

### 关键设计

1. **领域级条件化 (Domain Level)**:

    - 功能：使预训练的 Stable Diffusion 适应屏幕内容的图像域
    - 核心思路：使用 WebUI 数据集（40 万张网页截图），以 OCR 提取的文本内容拼接为提示词（格式："a screenshot with text: ..."），用 LoRA（rank=256）微调 Stable Diffusion。微调前模型无法生成类似截图的图像，微调后能正确生成带有典型屏幕布局的图像
    - 设计动机：Stable Diffusion 从未见过屏幕内容，不微调就无法理解截图的分布特征，是最基础也最关键的一步改进

2. **适配器级条件化 (Adaptor Level)**:

    - 功能：将压缩图像 $\bar{X}$ 和文本字形图 $\bar{Z}$ 的信息高效注入扩散模型
    - 核心思路：提出混合适配器策略——对字形图 $\bar{Z}$，仅使用 ControlNet 的特征编码器（避免 SD-VAE 编码器损坏文字信息）；对压缩图像 $\bar{X}$，同时使用 ControlNet 特征编码器和 SD-VAE 编码器（提供互补信息），并额外加入 pixel shuffle 变换保留完整像素信息。三路特征拼接后通过 SPADE 层注入 UNet
    - 设计动机：vanilla ControlNet 不适合低层次视觉任务（编码器和残差层控制力不够），StableSR 的 VAE 编码器对字形图效果差。混合方案取两者之长——VAN 编码器提供好的图像表示，ControlNet 编码器保护文字信息

3. **实例级条件化 (Instance Level)**:

    - 功能：在采样过程中进一步增强对条件信息的遵循
    - 核心思路：在每步 DDPM 采样后添加梯度引导，包含两项损失：(a) OCR 损失——确保中间去噪结果 $\mathbb{E}[X_0|X_t]$ 的 OCR 输出与 $\bar{Z}$ 一致（用 OCR logits 的 MSE）；(b) 重压缩损失——确保中间去噪结果经过 MLIC 编解码后与 $\bar{X}$ 近似（减少颜色漂移）。引导强度由超参数 $\zeta_1, \zeta_2$ 控制
    - 设计动机：训练好的条件扩散模型在实际采样时可能不完美地遵循条件。通过实例级引导可以在推理时"强迫"模型既保持文字正确又减少颜色偏差，类似 classifier guidance 的思路但用于压缩任务

### 损失函数 / 训练策略

各组件分开训练：(1) 领域级用 LoRA 微调 Stable Diffusion；(2) MLIC 编码器的 ControlNet 分支进行微调（冻结预训练 MLIC、训练复制编码器和零卷积层）；(3) 适配器级的 SPADE 层和 ControlNet 编码器联合训练。实例级引导不需要训练，是纯推理时的优化。

## 实验关键数据

### 主实验（屏幕图像 SCI1K）

| 方法 | BD-TEXT↑ | BD-PSNR↑ | BD-FID↓ | BD-CLIP↑ | BD-DISTS↓ |
|------|---------|---------|--------|---------|----------|
| MLIC (Baseline) | 0.000 | 0.00 | 0.00 | 0.000 | 0.000 |
| VTM-SCC | -0.168 | -1.99 | 31.84 | -0.062 | 0.047 |
| PerCo | -0.057 | -5.01 | -19.90 | -0.023 | -0.035 |
| MS-ILLM | 0.025 | -2.59 | -2.03 | -0.121 | -0.034 |
| **PICD** | **0.107** | -2.97 | **-20.68** | **0.030** | **-0.050** |

### 消融实验

| 配置 | Text Acc↑ | PSNR↑ | FID↓ | CLIP↑ | LPIPS↓ |
|------|----------|-------|------|-------|--------|
| 无字形图 (a) | 0.3468 | 19.10 | 45.83 | 0.8209 | 0.1694 |
| + ControlNet (b) | 0.4404 | 18.84 | 45.35 | 0.8617 | 0.1646 |
| + 提出的混合适配器 (d) | 0.4081 | 19.88 | 37.90 | 0.8922 | 0.1376 |
| + 实例级引导 (f) | 0.4445 | 23.70 | 35.54 | 0.9059 | 0.1172 |
| + 领域级微调 (g, full) | **0.4568** | 23.67 | **34.77** | **0.9082** | **0.1168** |

### 关键发现

- PICD 是唯一同时在文本精确性（BD-TEXT）和感知质量（BD-FID、BD-DISTS）上都优于 baseline 的方法
- 实例级引导带来最大的 PSNR 提升（从 19.88 到 23.70），说明推理时优化非常有效
- 领域级微调虽然 PSNR 变化不大，但 FID 和 CLIP 都有提升，说明它主要改善了生成图像的分布质量
- 在自然图像（Kodak、CLIC）上，PICD 在 FID 指标上也取得了最佳或次佳结果，验证了通用性

## 亮点与洞察

- **文本-图像分离编码的理论保证**非常优雅——通过信息论证明了先编码文本再条件编码图像是最优的，$H(Y|Z) + H(Z) = H(Y)$，理论上不增加比特率但保证文本无损
- **三层条件化的设计哲学**可推广——领域级调整分布、适配器级注入条件、实例级微调输出，这种从粗到细的条件化层次结构适用于任何需要精确控制扩散模型输出的场景
- **把压缩问题转化为条件生成问题**是一个有趣的视角转变——解码器不再是"重建"信号而是"生成"符合条件的信号

## 局限与展望

- 实例级引导需要 OCR 模型的前向+反向传播，显著增加解码时间
- 文本提取依赖 Tesseract OCR 的准确性，对于艺术字体、手写文字等可能不可靠
- 扩散模型解码本身就比传统解码慢几个数量级，实际部署需要考虑蒸馏或少步采样
- 当前只处理了英文文本，对其他语言（如中文、阿拉伯文）的泛化需要额外验证

## 相关工作与启发

- **vs PerCo**: 同为扩散模型做解码器的感知编解码器，但 PerCo 不处理文本精确性问题，在屏幕内容上文本错误严重
- **vs CDC**: GAN-based 感知编解码器，在屏幕内容上感知质量好但文本精确性差
- **vs VTM-SCC**: 传统屏幕内容编解码器，文本清晰但低比特率下整体画面模糊
- **vs Text-Sketch**: 同样关注文本精确性但用 sketch-based 方法，PICD 的扩散渲染方案在感知质量上明显更好

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次在感知压缩中系统性地解决文本精确性问题，三层条件化框架设计优雅
- 实验充分度: ⭐⭐⭐⭐ 屏幕和自然图像均有评估，消融全面，但缺少解码时间对比
- 写作质量: ⭐⭐⭐⭐ 理论分析清晰，但方法部分涉及多个组件导致结构稍显复杂
- 价值: ⭐⭐⭐⭐ 解决了实际痛点（截图压缩文字变糊），但扩散模型解码速度限制了应用场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Uni-Renderer: Unifying Rendering and Inverse Rendering via Dual Stream Diffusion](uni-renderer_unifying_rendering_and_inverse_rendering_via_dual_stream_diffusion.md)
- [\[CVPR 2025\] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)
- [\[CVPR 2025\] Random Conditioning for Diffusion Model Compression with Distillation](random_conditioning_for_diffusion_model_compression_with_distillation.md)
- [\[CVPR 2025\] Unveil Inversion and Invariance in Flow Transformer for Versatile Image Editing](unveil_inversion_and_invariance_in_flow_transformer_for_versatile_image_editing.md)
- [\[CVPR 2025\] Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes](channel-wise_noise_scheduled_diffusion_for_inverse_rendering_in_indoor_scenes.md)

</div>

<!-- RELATED:END -->
