---
title: >-
  [论文解读] Steering One-Step Diffusion Model with Fidelity-Rich Decoder for Fast Image Compression
description: >-
  [AAAI 2026][图像生成][图像压缩] 提出 SODEC，一种基于单步扩散的图像压缩模型，通过保真度引导模块(FGM)将高保真VAE解码器的先验注入扩散生成过程，结合速率退火训练策略实现极低码率下的高质量压缩，解码速度比多步扩散方法快20×以上，同时在率-失真-感知权衡上达到SOTA。 领域现状 传统编解码器：（JP…
tags:
  - "AAAI 2026"
  - "图像生成"
  - "图像压缩"
  - "扩散模型"
  - "单步解码"
  - "保真度引导"
  - "速率退火"
---

# Steering One-Step Diffusion Model with Fidelity-Rich Decoder for Fast Image Compression

**会议**: AAAI 2026  
**arXiv**: [2508.04979](https://arxiv.org/abs/2508.04979)  
**代码**: [https://github.com/zhengchen1999/SODEC](https://github.com/zhengchen1999/SODEC)  
**领域**: 图像生成  
**关键词**: 图像压缩, 扩散模型, 单步解码, 保真度引导, 速率退火

## 一句话总结

提出 SODEC，一种基于单步扩散的图像压缩模型，通过保真度引导模块(FGM)将高保真VAE解码器的先验注入扩散生成过程，结合速率退火训练策略实现极低码率下的高质量压缩，解码速度比多步扩散方法快20×以上，同时在率-失真-感知权衡上达到SOTA。

## 研究背景与动机

### 领域现状

**传统编解码器**（JPEG2000、VVC）在中高码率下表现可靠，但码率降至极低（<0.1 bpp）时产生严重块效应、模糊和结构失真。

**VAE-based压缩**（如HiFiC、MS-ILLM）通过变分自编码器依靠超先验等概率建模创新超越了传统方法在率-失真指标上的表现（PSNR/MS-SSIM），并进一步引入感知损失优化率-失真-感知框架。但在极低码率下仍难以重建细节，虽"技术正确"但缺乏真实感。

**扩散-based压缩**（如CDC、PerCo、DiffEIC）利用扩散模型强大的生成先验在率-感知权衡上展现出色表现，能在极端压缩下合成高度逼真的纹理和细节。

### 核心矛盾

然而，扩散-based压缩有两个致命缺陷：

**高延迟**：多步去噪过程导致巨大的解码延时和计算成本（PerCo耗时6.2秒，DiffEIC耗时7.8秒），不适合实时或资源受限场景

**低保真度**：扩散模型严重依赖预训练先验而非输入本身，重建偏离原始内容

### 本文切入角度

关键论点：**在图像压缩中，信息量足够丰富的latent使多步细化变得不必要**。利用预训练VAE-based模型产生信息丰富的latent，用单步解码替代迭代去噪过程。同时引入保真度引导机制弥补生成性损失。

### 核心 Idea

三个核心设计：

**单步解码**：信息丰富的latent + 单步扩散 = 高质量重建（无需多步）

**保真度引导模块(FGM)**：用VAE解码器生成高保真初步重建，作为视觉引导条件

**速率退火训练策略**：先高码率预训练学习丰富表征，再逐渐退火到目标低码率

## 方法详解

### 整体框架

SODEC的完整流程：

**编码侧**：
1. 输入图像 $x \in \mathcal{R}^{H \times W \times 3}$ 通过VAE编码器 $\mathcal{E}$ 下采样16倍到latent $y \in \mathcal{R}^{H/16 \times W/16 \times C}$（C通常为220）
2. 超编码器 $\mathcal{H}_a$ 提取超先验 $z = \mathcal{H}_a(y)$
3. 量化 $\hat{y} = \mathcal{Q}(y)$, $\hat{z} = \mathcal{Q}(z)$ 后经概率模型 $\mathcal{P}$ 进行熵编码

**解码侧**：
1. 恢复 $\hat{y}$ 和 $\hat{z}$
2. 变换模块 $\mathcal{T}_s$ 将其转换为适合扩散处理的内容变量 $\hat{y}_t \in \mathcal{R}^{64 \times 64 \times 4}$
3. **单步扩散**模型进行一次去噪得到 $\hat{y}_0$
4. 同时，**FGM** 用VAE解码器生成高保真初步重建 $\hat{x}_f$，经ViT提取特征后作为引导条件 $c_g$ 注入扩散过程

### 关键设计

#### 1. **单步扩散模型**

先用超合成网络 $\mathcal{H}_s$ 从 $\hat{z}$ 提取全局信息 $w$，然后与 $\hat{y}$ 合并转换为内容变量：

$$\hat{y}_t = \sqrt{\bar{\alpha}_t}\hat{y}_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$$

基于Stable Diffusion 2.1的UNet架构 $\epsilon_\theta$ 进行单步预测：

$$\hat{y}_0 = \frac{\hat{y}_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(\hat{y}_t, t, c_g)}{\sqrt{\bar{\alpha}_t}}$$

时步 $t$ 固定为999（最大噪声），通过LoRA微调扩散模型适配压缩任务。

设计动机：与标准T2I不同，压缩任务的latent携带了原始图像的丰富信息——VAE编码的结构性信息远多于随机噪声。因此单步即足以生成高质量重建，无需多步迭代。

#### 2. **保真度引导模块(FGM)**

FGM的设计弥补了扩散模型生成性偏差：

$$\hat{x}_f = \mathcal{D}_a(\hat{y})$$

其中 $\mathcal{D}_a$ 是从HiFiC预训练的VAE解码器初始化并fine-tune的保真度解码器。$\hat{x}_f$ 高度忠实于原始图像但可能缺乏感知丰富性。

然后用预训练ViT提取深层视觉特征，通过投影网络映射到扩散模型的条件空间：

$$c_g = \mathcal{F}_p(\mathcal{F}(\hat{x}_f)) \in \mathcal{R}^{L \times D}$$

其中 $L=77$, $D=1024$。$c_g$ 通过交叉注意力注入UNet。

设计动机：VAE解码器擅长保真度但缺乏感知质量；扩散模型擅长生成逼真纹理但缺乏对源图像的知识。FGM将两者的优势互补——用高保真重建作为强条件引导，指导扩散生成器产出既逼真又忠实的重建。

消融实验证明FGM优于文本提示引导(PerCo)和超先验引导(DiffEIC)：
- 无引导：MS-SSIM=0.8212
- 文本引导：MS-SSIM=0.8185
- 超先验引导：MS-SSIM=0.8258
- **FGM**: **MS-SSIM=0.8481** (+2.7%)

#### 3. **速率退火训练策略**

三阶段训练设计：

**Stage 1: 高码率VAE预训练**
- 用小 $\lambda$（低码率惩罚）训练HiFiC，鼓励学习丰富全面的latent表征
- 高码率 → 信息丰富的编码器-解码器对

**Stage 2: 扩散路径热身**
- 冻结整个VAE编码模块，仅训练扩散生成路径
- ViT和扩散解码器 $\mathcal{D}_m$ 冻结，UNet用LoRA微调
- 全参数训练：$\mathcal{H}_s$, $\mathcal{T}_s$, $\mathcal{D}_a$, $\mathcal{F}_p$
- 仅失真损失，不加码率惩罚和对齐损失
- 目的：教单步扩散生成器有效映射固定latent到高质量重建

**Stage 3: 联合训练 + 速率退火**
- 端到端优化，增加码率惩罚 $\lambda$（速率退火）
- 引入对齐损失确保 $\mathcal{D}_a$ 在latent变化时仍保持高保真：

$$\mathcal{L}_{align} = \mathbb{E}[\|x - \hat{x}_f\|_2^2]$$

总体损失：

$$\mathcal{L}_{overall} = d(x, \hat{x}) + \lambda \cdot r(\hat{y}, \hat{z}) + \alpha \cdot \mathcal{L}_{align}$$

最后加入GAN损失进一步增强细节合成：

$$\mathcal{L}_{finetune} = d(x, \hat{x}) + \lambda \cdot r(\hat{y}, \hat{z}) + \alpha \cdot \mathcal{L}_{align} + \beta \cdot \mathcal{L}_g$$

设计动机：核心直觉是**从丰富表征中选择性丢弃比从贫乏表征中凭空创造要容易**。先在宽松约束下学到全面特征，再逐步收紧码率约束，模型可以智能地保留最重要的信息。

### 损失函数 / 训练策略

失真损失 $d(x, \hat{x}_f) = k_M \cdot \text{MSE} + k_P \cdot \text{LPIPS}$

三阶段各有不同的损失配置，从仅失真到率-失真-对齐-GAN逐步复杂化。

## 实验关键数据

### 主实验

推理效率比较（DIV2K-Val，512×512，A6000）：

| 方法 | 总时间(ms) | 编码(ms) | 解码(ms)↓ | bpp↓ |
|------|-----------|---------|----------|------|
| HiFiC | 9.3 | 5.4 | 3.9 | 0.0310 |
| MS-ILLM | 93.7 | 54.5 | 84.4 | 0.0395 |
| PerCo | 6,242 | 1,540 | **4,702** | 0.0313 |
| DiffEIC | 7,828 | 266 | **7,561** | 0.0391 |
| **SODEC** | **233** | 5.0 | **228** | **0.0314** |

SODEC比PerCo快 **26×**，比DiffEIC快 **33×**！

感知-码率-延迟综合比较（DIV2K-Val）：

| 方法 | bpp↓ | LPIPS↓ | 解码速度 |
|------|------|--------|---------|
| HiFiC | 0.0420 | 较高 | 极快(3.9ms) |
| PerCo | 0.0535 | 较低 | 极慢(4702ms) |
| DiffEIC | 0.0457 | 较低 | 极慢(7561ms) |
| **SODEC** | **0.0322** | **最低** | 快(228ms) |

### 消融实验

保真度引导策略消融：

| 引导策略 | MS-SSIM↑ | LPIPS↓ | bpp↓ |
|---------|---------|--------|------|
| 无引导 | 0.8212 | 0.3625 | 0.0424 |
| 文本提示引导(PerCo) | 0.8185 | 0.3631 | 0.0412 |
| 超先验引导(DiffEIC) | 0.8258 | 0.3527 | 0.0385 |
| **辅助保真度引导(ours)** | **0.8481** | **0.3351** | **0.0368** |

训练策略消融：

| 训练策略 | MS-SSIM↑ | LPIPS↓ | bpp↓ |
|---------|---------|--------|------|
| 冻结VAE模块 | 0.8512 | 0.3761 | 0.0695 |
| 联合训练(匹配bpp) | 0.8621 | 0.3750 | 0.0678 |
| 低到高bpp课程学习 | 0.8643 | 0.3451 | 0.0593 |
| **速率退火(ours)** | **0.8951** | **0.3113** | **0.0604** |

对齐损失消融：

| 对齐损失配置 | MS-SSIM↑ | LPIPS↓ | bpp↓ |
|------------|---------|--------|------|
| 无对齐损失 | 0.7490 | 0.4210 | 0.0203 |
| MSE + LPIPS | 0.7481 | 0.3961 | 0.0199 |
| 合并到主损失 | 0.7984 | 0.4023 | 0.0232 |
| **仅MSE对齐(ours)** | **0.7948** | **0.3827** | **0.0227** |

### 关键发现

1. **单步扩散在压缩任务中足够**：信息丰富的VAE latent使多步细化变得多余
2. **速率退火优于所有其他训练策略**：在给定质量下节省30%码率，或在等码率下显著提升质量
3. **FGM效果远超文本/超先验引导**：MS-SSIM提升2.7%，同时LPIPS也改善
4. **仅MSE对齐损失最优**：加入LPIPS或合并到主损失反而不如纯MSE有效
5. **解码速度量级提升**：228ms vs 4-7.5秒，使扩散压缩首次具备实时应用潜力

## 亮点与洞察

1. **"信息丰富的latent使多步不必要"的核心论点**很有说服力，从根本上颠覆了扩散压缩必须多步的假设
2. **FGM的互补性设计**思路优雅——VAE保真度 + 扩散感知质量的显式融合
3. **速率退火**的"从丰富表征选择性丢弃"思想启发性强，与知识蒸馏有异曲同工之处
4. **三阶段训练**的渐进式设计（先VAE → 再扩散路径 → 最后联合）避免了联合训练初期的不稳定
5. **20-33×解码加速**使扩散压缩从学术概念迈向实用化的关键一步

## 局限与展望

1. **编码器仍是传统VAE**：16倍下采样+220通道的latent可能限制了极端压缩潜力
2. **基于SD 2.1**：未在更新的SD3/FLUX架构上验证
3. **512×512分辨率**：高分辨率图像的表现未知
4. **固定时步t=999**：不同码率下最优时步可能不同
5. **仅图像压缩**：视频压缩的时间维度更适合单步范式，值得探索

## 相关工作与启发

- **PerCo** (Careil et al., 2023)：文本+视觉特征双引导的多步扩散压缩
- **DiffEIC** (Li et al., 2024)：超先验引导的扩散图像压缩
- **HiFiC** (Mentzer et al., 2020)：生成对抗压缩，SODEC的VAE backbone初始化来源
- **One-step diffusion**（如DMD、LCM、SDXL-Turbo）：单步生成的相关工作
- **Key Takeaway**：图像压缩与图像生成的关键区别在于——压缩有源图像信息作为条件，生成则无。这个信息优势使得单步足够，开辟了扩散模型在低延迟视觉任务中的新范式

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 单步扩散压缩+保真度引导+速率退火的组合方案设计完整
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三数据集验证，全面的率失真感知曲线，详细消融
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机论述有力，图表丰富
- **价值**: ⭐⭐⭐⭐⭐ — 20×解码加速+SOTA质量，使扩散压缩首次具备实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal](../../ICCV2025/image_generation/compression-aware_one-step_diffusion_model_for_jpeg_artifact_removal.md)
- [\[CVPR 2026\] Bridging Fidelity-Reality with Controllable One-Step Diffusion for Image Super-Resolution](../../CVPR2026/image_generation/bridging_fidelity-reality_with_controllable_one-step_diffusion_for_image_super-r.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](../../CVPR2026/image_generation/pixelrush_ultrafast_trainingfree_highresolution_im.md)
- [\[AAAI 2026\] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution](mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)

</div>

<!-- RELATED:END -->
