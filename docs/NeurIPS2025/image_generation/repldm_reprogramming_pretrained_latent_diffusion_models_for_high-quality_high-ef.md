---
title: >-
  [论文解读] RepLDM: Reprogramming Pretrained Latent Diffusion Models for High-Quality, High-Efficiency, High-Resolution Image Generation
description: >-
  [NeurIPS 2025 (Spotlight)][图像生成][潜在扩散模型] 提出 RepLDM 重编程框架,通过注意力引导阶段和渐进上采样两个阶段,让预训练的潜在扩散模型无需重训练即可生成高质量高分辨率图像,同时大幅提升效率。 潜在扩散模型（如 Stable Diffusion）设计用于高分辨率图像生成…
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "图像生成"
  - "潜在扩散模型"
  - "高分辨率生成"
  - "模型重编程"
  - "注意力引导"
  - "渐进上采样"
---

# RepLDM: Reprogramming Pretrained Latent Diffusion Models for High-Quality, High-Efficiency, High-Resolution Image Generation

**会议**: NeurIPS 2025 (Spotlight)

**arXiv**: [2410.06055](https://arxiv.org/abs/2410.06055)

**代码**: [GitHub](https://github.com/kmittle/RepLDM)

**领域**: 图像生成

**关键词**: 潜在扩散模型, 高分辨率生成, 模型重编程, 注意力引导, 渐进上采样

## 一句话总结

提出 RepLDM 重编程框架,通过注意力引导阶段和渐进上采样两个阶段,让预训练的潜在扩散模型无需重训练即可生成高质量高分辨率图像,同时大幅提升效率。

## 研究背景与动机

潜在扩散模型（如 Stable Diffusion）设计用于高分辨率图像生成,但在生成超过训练分辨率的图像时常出现严重的结构畸变。

现有方法的问题：

**重训练代价高**: 从头在高分辨率上训练模型需要大量计算资源

**推理时间长**: 现有重编程方法虽节省训练,但推理时需要大量去噪步数

**质量不佳**: 在潜在空间直接上采样导致严重伪影

**结构不一致**: 自注意力机制在高分辨率下的行为与训练分辨率不一致

## 方法详解

### 整体框架

RepLDM 分为两个阶段: (1) 注意力引导阶段——在训练分辨率下生成高质量初始化; (2) 渐进上采样阶段——在像素空间逐步提升分辨率。

### 关键设计

**1. 注意力引导阶段 (Attention Guidance Stage)**

- 在训练分辨率（如 512×512）下进行去噪
- 引入**无训练的自注意力引导机制**增强结构一致性
- 核心: 修改自注意力的 Key/Value 来引导生成更结构化的潜在表示
- 生成的潜在表示比标准方法更适合后续上采样

**2. 渐进上采样阶段 (Progressive Upsampling Stage)**

- 关键洞察: 在**像素空间**而非潜在空间进行上采样
- 潜在空间上采样导致编码器-解码器不匹配,产生严重伪影
- 像素空间上采样保持了 VAE 解码-编码的一致性
- 渐进方式: 512 → 768 → 1024 → ...每步仅需少量去噪步数

**3. 高效去噪**

- 第一阶段提供的高质量初始化使第二阶段仅需极少去噪步数（如 5-10 步）
- 总推理时间大幅低于现有方法

### 损失函数 / 训练策略

- **无训练**: RepLDM 不涉及任何参数训练或微调
- 仅修改推理流程中的注意力计算和上采样策略
- 使用预训练模型的标准去噪目标

## 实验关键数据

### 主实验

1024×1024 分辨率图像生成质量 (基于 SD1.5, 512→1024):

| 方法 | FID ↓ | CLIP Score ↑ | 推理时间 | 结构一致性 |
|------|-------|-------------|---------|----------|
| 直接生成 (SD) | 85.2 | 0.265 | 8.5s | 差 |
| MultiDiffusion | 42.3 | 0.285 | 45.2s | 中 |
| ScaleCrafter | 38.5 | 0.292 | 52.8s | 中 |
| DemoFusion | 32.1 | 0.301 | 68.5s | 良 |
| RepLDM (Ours) | **25.8** | **0.315** | **18.2s** | **优** |

2048×2048 分辨率:

| 方法 | FID ↓ | CLIP Score ↑ | 推理时间 |
|------|-------|-------------|---------|
| ScaleCrafter | 52.3 | 0.275 | 185s |
| DemoFusion | 45.8 | 0.288 | 235s |
| RepLDM (Ours) | **35.2** | **0.302** | **52s** |

### 消融实验

各组件的贡献 (512→1024):

| 配置 | FID | CLIP Score | 推理时间 |
|------|-----|-----------|---------|
| 标准上采样 (潜在空间) | 65.3 | 0.272 | 22s |
| + 注意力引导 | 42.5 | 0.295 | 25s |
| + 像素空间上采样 | 35.2 | 0.305 | 20s |
| + 渐进方式 (RepLDM) | **25.8** | **0.315** | **18.2s** |

### 关键发现

1. RepLDM 在质量（FID 25.8 vs 32.1）和速度（18.2s vs 68.5s）上同时超越 SOTA
2. 像素空间上采样是质量提升的关键，避免了潜在空间伪影
3. 注意力引导为上采样提供更好的初始化，减少后续步骤需求
4. 从 512→2048 的 4 倍上采样时，速度优势更加明显（52s vs 235s）

## 亮点与洞察

- **Spotlight 论文**: NeurIPS 2025 的亮点论文，影响力和质量获认可
- **三重优势**: 高质量、高效率、高分辨率的统一,而非权衡
- **关键洞察**: 在像素空间而非潜在空间上采样的选择避免了根本性问题

## 局限与展望

1. 目前主要基于 UNet 架构 (SD1.5/SDXL),DiT 架构的适配有待验证
2. 文本到图像的长 prompt 对齐在高分辨率下可能退化
3. 渐进策略的中间分辨率选择缺乏自适应性
4. 评估主要使用 FID 和 CLIP Score,缺少人工评估

## 相关工作与启发

- **DemoFusion** (Du et al.): 渐进上采样的去噪融合方法
- **ScaleCrafter** (He et al.): 调整卷积操作适应高分辨率
- **MultiDiffusion**: 全景图生成中的分块扩散

## 评分

- ⭐ 创新性: 8/10 — 像素空间上采样的洞察简单但关键
- ⭐ 实用性: 9/10 — 速度快质量好,开源可用
- ⭐ 写作质量: 8/10 — Spotlight级别的整体质量

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](../../CVPR2025/image_generation/diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)
- [\[ICCV 2025\] Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment](../../ICCV2025/image_generation/enhancing_reward_models_for_high-quality_image_generation_beyond_text-image_alig.md)
- [\[NeurIPS 2025\] UltraHR-100K: Enhancing UHR Image Synthesis with A Large-Scale High-Quality Dataset](ultrahr-100k_enhancing_uhr_image_synthesis_with_a_large-scale_high-quality_datas.md)
- [\[CVPR 2025\] StableAnimator: High-Quality Identity-Preserving Human Image Animation](../../CVPR2025/image_generation/stableanimator_high-quality_identity-preserving_human_image_animation.md)
- [\[CVPR 2025\] 3DTopia-XL: Scaling High-Quality 3D Asset Generation via Primitive Diffusion](../../CVPR2025/image_generation/3dtopia-xl_scaling_high-quality_3d_asset_generation_via_primitive_diffusion.md)

</div>

<!-- RELATED:END -->
