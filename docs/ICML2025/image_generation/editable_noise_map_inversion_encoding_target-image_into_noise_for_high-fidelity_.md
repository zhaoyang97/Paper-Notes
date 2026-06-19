---
title: >-
  [论文解读] Editable Noise Map Inversion: Encoding Target-image into Noise For High-Fidelity Image Manipulation
description: >-
  [ICML 2025][图像生成][扩散模型反演] 提出 Editable Noise Map Inversion (ENM Inversion)，通过在反演过程中同时优化重建误差和编辑对齐误差，使 noise map 同时"铭刻"源图像与目标图像信息，在内容保持和编辑忠实度之间取得最优平衡。 基于扩散模型的图像编辑流程通常…
tags:
  - "ICML 2025"
  - "图像生成"
  - "扩散模型反演"
  - "图像编辑"
  - "Noise Map"
  - "DDIM Inversion"
  - "注意力控制"
---

# Editable Noise Map Inversion: Encoding Target-image into Noise For High-Fidelity Image Manipulation

**会议**: ICML 2025  
**arXiv**: [2509.25776](https://arxiv.org/abs/2509.25776)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 扩散模型反演, 图像编辑, Noise Map, DDIM Inversion, 注意力控制

## 一句话总结

提出 Editable Noise Map Inversion (ENM Inversion)，通过在反演过程中同时优化重建误差和编辑对齐误差，使 noise map 同时"铭刻"源图像与目标图像信息，在内容保持和编辑忠实度之间取得最优平衡。

## 研究背景与动机

基于扩散模型的图像编辑流程通常分为两步：(1) 反演（Inversion）——将输入图像编码为 noise map 序列；(2) 编辑——利用注意力控制（如 Prompt-to-Prompt、MasaCtrl）根据目标 prompt 修改图像。

**核心矛盾**：现有反演方法（DDIM Inversion、Null-Text Inversion、PNP Inversion、Fixed-Point Iteration 等）都以忠实重建源图像为目标，反演得到的 noise map 虽能精确重建原图，但严重限制了编辑灵活性。换言之，目标图像的信息并未被"铭刻"到 noise map 中，导致编辑效果差、产生伪影（如无法将蝴蝶变为鹦鹉）。

**关键观察**：作者在 AFHQ 数据集上分析发现，使用源 prompt 重建得到的 noise map 与使用目标 prompt 编辑得到的 noise map 之间的差异越小，编辑效果越好（LPIPS 更低、CLIP Score 更高）。这一规律在不同反演步骤上均成立。

## 方法详解

### 整体框架

ENM Inversion 在标准 DDIM Inversion 基础上增加了**可编辑噪声精炼（Editable Noise Refinement）**步骤。整体流程：

1. **DDIM 正向反演**：将源图像 $z_0$ 逐步反演为 noise map 序列 $\{z_1, z_2, \dots, z_T\}$
2. **可编辑噪声精炼**：在每一步反演后，迭代优化 $z_t$，使其同时满足重建和编辑需求
3. **注意力控制编辑**：将优化后的 noise map 送入编辑流程（P2P/MasaCtrl/PnP），利用注意力映射从重建路径迁移到编辑路径

### 关键设计

**核心思想**：在每一个反演步骤中，不仅要求 noise map 能精确重建源图像，还要求重建路径和编辑路径产生的 noise map 尽可能接近。

#### 1. 编辑对齐损失 $L_{edit}$

衡量在 $z_t$ 处使用源 prompt $C_{src}$ 和目标 prompt $C_{tgt}$ 分别去噪后的差异：

$$L_{edit} = \|f(z_t, t, C_{src}) - f(z_t, t, C_{tgt})\|^2$$

其中 $f(z_t, t, C)$ 是 DDIM 采样函数（一步去噪）。最小化这一项使得 noise map 对源和目标 prompt 产生相似的去噪结果，从而将目标图像信息"铭刻"到噪声中。

#### 2. 重建保持损失 $L_{prev}$

保证 noise map 仍能精确重建源图像：

$$L_{prev} = \|z_{t-1} - f(z_t, t, C_{src})\|^2$$

这一项防止优化后的 noise map 偏离源图像太远。

#### 3. 联合优化目标

$$\arg\min_{z_t} L = L_{prev} + \lambda \cdot L_{edit}$$

其中 $\lambda$ 为超参数权重，控制编辑对齐的强度。

#### 4. 阈值截断与迭代精炼

为了提高效率，引入预定义阈值 $\tau$：当 $L$ 低于 $\tau$ 时提前停止迭代。每步最多执行 $\mathcal{K}$ 次精炼迭代。

### 算法流程（Algorithm 1）

```
输入: 源图像 z_0, 反演步数 T, 源/目标 prompt, 精炼步数 K, 阈值 τ
输出: 优化后的 noise map {z_T, ..., z_1}

for t = 1 to T:
    z_t ← f_inv(z_{t-1}, t-1, C_src)         # 标准 DDIM 反演
    z_{t-1}^e ← f(z_t, t, C_tgt)              # 使用目标 prompt 去噪
    for k = 1 to K:
        计算 L = L_prev + λ·L_edit
        if L < τ: break
        z_t ← z_t - lr · ∇_{z_t} L           # 梯度下降更新
```

### 与现有编辑方法的兼容性

ENM Inversion 作为即插即用的反演模块，可直接替换以下编辑方法中的反演步骤：
- **Prompt-to-Prompt (P2P)**：替换交叉注意力映射
- **MasaCtrl**：互注意力控制，实现非刚性编辑
- **Plug-and-Play (PnP)**：修改空间特征和自注意力映射

### 视频编辑扩展

将 ENM Inversion 集成到 Video-P2P 框架中：先微调图像扩散模型使其适应视频建模，然后对每一帧执行 ENM Inversion，最后通过跨帧注意力控制确保时间一致性。

### 损失函数 / 训练策略

本方法**无需训练**，完全在推理时优化 noise map。关键超参数：
- $\lambda = 10$（编辑权重，默认值）
- $T = 50$（DDIM 采样步数）
- CFG = 7.5
- 基础模型：Stable Diffusion V1-4（P2P/MasaCtrl）或 V1-5（PnP/Video-P2P）

## 实验关键数据

### 主实验

在 PIE-Bench 数据集（700 张图像 × 9 种编辑任务）上评估，与 Prompt-to-Prompt 结合：

| 方法 | Structure Dist.↓ | PSNR↑ | LPIPS↓ | MSE↓ | SSIM↑ | CLIP-Whole↑ | CLIP-Edited↑ |
|---|---|---|---|---|---|---|---|
| DDIM+P2P | 69.43 | 17.87 | 208.80 | 219.88 | 71.14 | 25.01 | 22.44 |
| NTI+P2P | 13.44 | 27.03 | 60.67 | 35.86 | 84.11 | 24.75 | 21.86 |
| PNPInv+P2P | 11.65 | 27.22 | 54.55 | 32.86 | 84.76 | 25.02 | 22.10 |
| **Ours+P2P** | **10.13** | **28.19** | **45.26** | **27.02** | **86.29** | **25.30** | **22.12** |

与 MasaCtrl 和 Plug-and-Play 结合同样优于所有基线。与 Fixed-Point 方法对比：

| 方法 | Structure Dist.↓ | PSNR↑ | LPIPS↓ | SSIM↑ | CLIP-Whole↑ |
|---|---|---|---|---|---|
| AIDI+P2P | 12.19 | 26.96 | 57.92 | 84.17 | 24.96 |
| FPI+P2P | 14.71 | 26.61 | 61.97 | 83.52 | 23.93 |
| ReNoise | 22.60 | 25.19 | 85.29 | 82.30 | 23.78 |
| **Ours+P2P** | **10.13** | **28.19** | **45.26** | **86.29** | **25.30** |

推理时间对比（与 P2P 结合，单张 RTX 3090）：

| 方法 | 时间 (s) |
|---|---|
| DDIM | 18.22 |
| NTI | 148.48 |
| StyleD | 382.98 |
| PNPInv | 28.17 |
| **Ours** | **38.87** |

### 消融实验

**超参数 $\lambda$（编辑权重）的影响**：

| $\lambda$ | Structure Dist.↓ | PSNR↑ | SSIM↑ | CLIP-Whole↑ | 说明 |
|---|---|---|---|---|---|
| 5 | 10.10 | 28.22 | 86.23 | 25.17 | 编辑能力略弱 |
| **10（默认）** | **10.13** | **28.19** | **86.29** | **25.30** | **最优平衡** |
| 15 | 10.38 | 28.13 | 86.15 | 25.32 | 编辑增强但保持下降 |
| 20 | 12.41 | 28.01 | 86.12 | 25.34 | 过度编辑，结构损失 |

**DDIM 步数 $T$ 的影响**：步数少（20步）利于保持，步数多（75/100步）利于编辑，50 步为最优平衡。

**源 prompt 鲁棒性**：将源 prompt 替换为 null text，重建质量无显著差异，说明方法对源 prompt 不敏感。

### 关键发现

1. **noise map 差异与编辑质量的强相关性**：重建与编辑 noise map 间差异越小，编辑效果越好，这一观察是方法的理论基础
2. **交叉注意力对齐分析**：ENM Inversion 在整个去噪过程中维持较高且稳定的交叉注意力对齐分数，而 DDIM 在后期对齐下降，DDPM 虽有改善但不稳定
3. **可扩展至 Flow-based 模型**：与 RF Inversion 结合后（Ours+RFInv），在 Flux 模型上也显著提升重建质量（LPIPS: 232.88→185.86）

## 亮点与洞察

1. **切入点精妙**：不是更精确地重建源图像，而是"在 noise map 中编码目标图像"，从根本上解决了"重建越好、编辑越差"的矛盾
2. **即插即用**：作为反演模块兼容 P2P、MasaCtrl、PnP 等主流编辑方法，无需修改编辑流程
3. **training-free**：不需要额外训练，纯推理时优化，部署门槛低
4. **实验全面**：覆盖图像编辑（PIE-Bench, 9 种任务类型）和视频编辑（DAVIS），与 7+ 种基线全面对比
5. **损失设计简洁有效**：两项损失一加权，解决两个核心需求，优雅且高效

## 局限与展望

1. **依赖基础模型能力**：若目标图像超出 Stable Diffusion 生成域，方法会失败
2. **每次编辑需重新反演**：不同于传统方法可复用 noise map 用于多个目标 prompt，ENM 需要针对每个源-目标对单独反演，增加计算开销
3. **推理速度**：38.87s 虽远快于 NTI（148s）和 StyleD（383s），但仍比 DDIM（18s）慢约 2 倍
4. **仅验证在 SD v1.4/v1.5 上**：未在 SDXL 或更新的大模型上验证
5. **优化不稳定性**：梯度优化 noise map 可能在极端编辑场景下不收敛

## 相关工作与启发

- **Null-Text Inversion (NTI)**：通过优化 null-text embedding 提升重建质量，但牺牲编辑性
- **PNP Inversion**：通过分离源和目标分支提高效率，但编辑能力仍受限
- **Fixed-Point Iteration (AIDI/FPI/ReNoise)**：迭代求解隐式方程减少近似误差，但更面向重建
- **DDPM Inversion (Edit-Friendly)**：引入随机性提升编辑性，但重建不稳定
- **Pix2pix-Zero**：利用注意力映射变化计算编辑方向，启发了本文分析 noise map 差异的思路

**启发**：在生成模型编辑中，"重建质量"和"编辑灵活性"是一对内在矛盾，ENM 的联合优化思路为这类 trade-off 问题提供了一个通用范式——同时优化两个目标函数而非只关注其中一个。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 核心观察（noise map 差异与编辑质量相关）有洞察力，但优化框架本身相对直接
- 实验充分度: ⭐⭐⭐⭐⭐ — 多种编辑方法 × 多种基线 × 图像/视频 × 消融 × 超参分析，非常全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机阐述到位，图表丰富
- 综合价值: ⭐⭐⭐⭐ — 实用性强，即插即用，但受限于 SD v1 生态

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Exploring Position Encoding in Diffusion U-Net for Training-free High-resolution Image Generation](exploring_position_encoding_in_diffusion_u-net_for_training-free_high-resolution.md)
- [\[CVPR 2025\] Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis](../../CVPR2025/image_generation/noise_diffusion_for_enhancing_semantic_faithfulness_in_text-to-image_synthesis.md)
- [\[ICML 2025\] Taming Rectified Flow for Inversion and Editing](taming_rectified_flow_for_inversion_and_editing.md)
- [\[ECCV 2024\] Diffusion-based Image-to-Image Translation by Noise Correction via Prompt Interpolation](../../ECCV2024/image_generation/diffusion-based_image-to-image_translation_by_noise_correction_via_prompt_interp.md)
- [\[ICCV 2025\] Improved Noise Schedule for Diffusion Training](../../ICCV2025/image_generation/improved_noise_schedule_for_diffusion_training.md)

</div>

<!-- RELATED:END -->
