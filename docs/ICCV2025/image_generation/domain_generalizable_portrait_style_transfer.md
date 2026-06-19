---
title: >-
  [论文解读] Domain Generalizable Portrait Style Transfer
description: >-
  [ICCV 2025][图像生成][人像风格迁移] DGPST 提出了一个基于扩散模型的人像风格迁移框架，通过 semantic adapter 建立跨域稠密语义对应来扭曲参考图像，配合 AdaIN-Wavelet Transform 进行潜空间初始化以平衡风格化与内容保持，结合 ControlNet（高频结构引导）和 style adapter（风格引导）的双条件扩散模型生成最终结果，仅在 30K 真实肖像照片上训练即可泛化到照片、卡通、素描、动漫等多种域。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "人像风格迁移"
  - "语义对应"
  - "小波变换"
  - "扩散模型"
  - "跨域泛化"
---

# Domain Generalizable Portrait Style Transfer

**会议**: ICCV 2025  
**arXiv**: [2507.04243](https://arxiv.org/abs/2507.04243)  
**代码**: [https://github.com/wangxb29/DGPST](https://github.com/wangxb29/DGPST)  
**领域**: 图像风格迁移 / 扩散模型  
**关键词**: 人像风格迁移, 语义对应, 小波变换, 扩散模型, 跨域泛化

## 一句话总结
DGPST 提出了一个基于扩散模型的人像风格迁移框架，通过 semantic adapter 建立跨域稠密语义对应来扭曲参考图像，配合 AdaIN-Wavelet Transform 进行潜空间初始化以平衡风格化与内容保持，结合 ControlNet（高频结构引导）和 style adapter（风格引导）的双条件扩散模型生成最终结果，仅在 30K 真实肖像照片上训练即可泛化到照片、卡通、素描、动漫等多种域。

## 研究背景与动机

**领域现状**：人像风格迁移需要对面部各语义区域（皮肤、嘴唇、眼睛、头发、背景）进行精确的局部色调调整，同时保持人物身份和面部结构。现有方法包括传统手工方法（Shih et al.）、GAN-based（StyleGAN 系列）和扩散模型方法。

**现有痛点**：
   - **传统方法**（Shih、Chen 等）依赖显式语义区域对齐，仅在输入和参考结构差异较小时有效，无法处理跨域（照片→卡通）场景
   - **GAN-based 方法**（StyleGAN）不可避免地改变人物身份
   - **已有扩散方法**（StyleID、IP-Adapter+ControlNet、InstantStyle+）主要面向艺术风格迁移，不考虑语义对应，在人像风格迁移中语义区域对齐质量差
   - 通用风格迁移方法在人像这种需要精细语义对齐的任务上表现不佳

**核心矛盾**：人像风格迁移同时需要① 精确的跨域语义对应（眼睛对眼睛、嘴唇对嘴唇）和② 高质量的风格转移（色调、纹理），但现有方法在其中一个或两个方面不足。

**本文目标** 构建一个仅在真实照片上训练就能泛化到任意域（卡通、素描、动漫、旧照片）的人像风格迁移框架。

**切入角度**：利用预训练扩散模型（Stable Diffusion）的特征空间天然具有跨域语义理解能力来建立稠密对应；用小波变换分离高低频来平衡内容和风格。

**核心 idea**：基于扩散特征的语义对应 → 参考图像扭曲 → AdaIN-Wavelet 初始化潜空间 → 双条件扩散模型生成。

## 方法详解

### 整体框架
输入为内容图像 $z_0^c$ 和风格参考图像 $z_0^s$，输出为保持内容身份+应用参考风格的人像。流程分四步：① 利用 SD 特征 + semantic adapter 建立语义对应并扭曲参考图；② 用 ControlNet 提取内容图高频信息作为结构引导；③ 用 style adapter 从扭曲后的参考图提取风格引导；④ 用 AdaIN-Wavelet 初始化潜空间后执行条件去噪生成。

### 关键设计

1. **Semantic-Aware Style Alignment（语义感知对齐）**

    - 功能：建立内容与参考人像之间的稠密语义对应，生成扭曲后的参考图 $z_0^{s\_w}$
    - 核心思路：
        - 用 CLIP 图像编码器提取图像特征，通过 projection network 送入 SD U-Net 做 decoupled cross-attention
        - 将两张图像送入 SD U-Net（注入 semantic adapter 特征），从第三个上采样块提取特征 $F_0^c, F_0^s \in \mathbb{R}^{HW \times C}$
        - 计算 normalized correlation matrix $\mathcal{M}(i,j)$，然后对参考图做 softmax-weighted warping：$z_0^{s\_w}(i) = \sum_j \text{softmax}(\mathcal{M}(i,j)/\tau) \cdot z_0^s(j)$
    - 训练损失：mask warping loss $\mathcal{L}_{mask} = \|M^c - M^{s\_w}\|_1$（语义 mask 对齐）+ cyclic warping consistency loss $\mathcal{L}_{cwc} = \mathcal{L}_{LPIPS}(z_0^s, z^{s'\_w})$（循环一致性）
    - 设计动机：直接用 SD 特征做对应可能语义区域不完整，semantic adapter + 两个损失函数约束对应的准确性

2. **Dual-Conditional Diffusion Model（双条件扩散模型）**

    - 功能：同时利用结构引导和风格引导生成高质量人像
    - **结构引导（ControlNet）**：对内容图像 $z_0^c$ 做 Haar 离散小波变换（DWT），取三个高频子带（LH、HL、HH）作为 ControlNet 输入。只用高频信息（边缘/纹理）而不用原图本身，可以提供与风格无关的结构引导
    - **风格引导（Style Adapter）**：使用 IP-Adapter 架构，从扭曲后的参考图提取 CLIP 图像特征，通过 projection 后在 decoupled cross-attention 层注入：$Z^{new} = \text{softmax}(\frac{QK^t}{\sqrt{d}})V^t + \lambda \cdot \text{softmax}(\frac{QK^i}{\sqrt{d}})V^i$
    - 设计动机：ControlNet 用高频而非原图可以避免将内容的颜色风格带入输出；扭曲后的参考图已经语义对齐，比直接用参考图提供更精准的风格引导

3. **AdaIN-Wavelet Transform（潜空间初始化）**

    - 功能：构建既保持内容结构细节又增强风格色调迁移的初始潜空间
    - 核心思路：
        - 对扭曲参考图做 DDIM inversion 得到 $z_T^{s\_w}$（直接用它初始化会增强色彩迁移但丢失内容细节导致模糊）
        - 先做 AdaIN：$z_T^{cs'} = \sigma(z_T^{s\_w}) \cdot \frac{z_T^c - \mu(z_T^c)}{\sigma(z_T^c)} + \mu(z_T^{s\_w})$（逐通道均值/方差对齐，让内容潜空间的统计量靠近风格）
        - 再做 Wavelet 融合：取 $z_T^{s\_w}$ 的低频 + $z_T^{cs'}$ 的高频，通过 IDWT 合成最终初始潜空间 $z_T^{cs}$
    - 引入 $\gamma$ 参数控制风格化强度，通过插值 $z_T^{cs} = \gamma \cdot z_T^{cs} + (1-\gamma) \cdot z_T^c$ 实现连续风格强度控制
    - 设计动机：从内容潜空间出发保持原色调（风格迁移不足），从参考潜空间出发丢失细节（过度模糊）。AdaIN 对齐统计量+Wavelet 融合高/低频取长补短

### 损失函数 / 训练策略
两阶段训练：
- 第一阶段（500K iterations）：训练 semantic adapter，损失 = $\mathcal{L}_{sem}$（噪声预测）+ $\mathcal{L}_{cwc}$ + $10 \times \mathcal{L}_{mask}$
- 第二阶段（300K iterations）：训练 ControlNet + style adapter，损失 = $\mathcal{L}_{rec}$（条件噪声预测），条件包含高频 ControlNet 和风格 adapter。使用相同图像作为内容和风格（自重建）

## 实验关键数据

### 主实验 - CelebAMask-HQ

| 方法 | Gram loss ↓ | LPIPS ↓ | ID ↓ |
|------|------------|---------|------|
| Shih et al. | 0.376 | 0.187 | 0.093 |
| Wang et al. | 0.208 | 0.181 | 0.106 |
| IP-A + C.N. | 2.835 | 0.245 | 0.774 |
| StyleID | 0.505 | 0.198 | 0.222 |
| InstantStyle+ | 0.557 | 0.294 | 0.272 |
| **Ours** | **0.274** | **0.116** | **0.057** |

### 消融实验

| 配置 | Gram loss ↓ | LPIPS ↓ | ID ↓ |
|------|------------|---------|------|
| Full model | 0.274 | **0.116** | **0.057** |
| w/o ControlNet | **0.236** | 0.333 | 0.450 |
| w/o style adapter | 0.548 | 0.145 | 0.086 |
| w/ Init AdaIN (无 Wavelet) | 1.196 | 0.151 | 0.062 |

### 跨域混合数据集结果

| 方法 | Gram loss ↓ | LPIPS ↓ | ID ↓ |
|------|------------|---------|------|
| Wang et al. | 1.488 | 0.119 | 0.096 |
| InstantStyle+ | 0.723 | 0.192 | 0.203 |
| **Ours** | **0.657** | **0.083** | **0.087** |

### 关键发现
- **ControlNet 对身份保持至关重要**：去掉后 ID 从 0.057 激增到 0.450，说明高频结构引导是保持面部身份的关键
- **Style adapter 对风格迁移必不可少**：去掉后 Gram loss 从 0.274 翻倍到 0.548
- **Wavelet 融合比纯 AdaIN 效果好**：Gram loss 从 1.196 降到 0.274，同时 LPIPS 也改善，证明高低频分离融合的有效性
- 推理速度仅 6.97 秒/张（512×512），比 Deng et al.(24.18s) 和 InstantStyle+(67.4s) 快很多
- 仅在 CelebAMask-HQ（30K 真实照片）训练就能泛化到卡通、素描、动漫、旧照片等多种域

## 亮点与洞察
- **利用 SD 特征空间做跨域语义对应**是核心创新：预训练扩散模型的中间特征天然具有跨域语义理解能力（照片中的眼睛和卡通中的眼睛在特征空间中距离近），加上 semantic adapter 微调，可以建立高质量的稠密对应
- **高频 ControlNet 输入**（DWT 的 LH/HL/HH 子带）巧妙地实现了风格无关的结构引导，避免了用原图或 Canny 边缘导致的色彩/风格泄露
- **$\gamma$ 参数提供连续风格强度控制**，同时作用于潜空间初始化和 style adapter 特征混合，实现直观的风格插值

## 局限与展望
- 基于 SD 1.5 实现，升级到 SDXL 或 SD3 可能进一步提升质量
- 语义对应仍可能在极端姿态差异下失败
- 训练数据仅限人像，其他需要语义对齐的风格迁移场景（如建筑、动物）未探索
- two-stage 训练较复杂（800K iterations），是否可用联合训练简化有待验证
- 区域控制目前需要手动提供 mask，自动化语义区域选择可以改进

## 相关工作与启发
- **vs Wang et al.**：传统方法在同域内效果好（CelebAMask-HQ 上 Gram loss 0.208 略优），但跨域能力差（混合数据集上 Gram loss 1.488 远差于本文 0.657）
- **vs IP-Adapter + ControlNet**：通用组合方案不考虑语义对应，ID 损失 0.774 极高，说明人像风格迁移必须做语义对齐
- **vs StyleID**：免训练方法通过 self-attention 特征注入做风格迁移，但缺乏语义对应导致区域对齐差
- **vs InstantStyle+**：虽然引入了一些结构控制，但推理极慢（67.4s）且效果不如本文

## 评分
- 新颖性: ⭐⭐⭐⭐ 将扩散模型特征用于语义对应+小波变换潜空间融合是新颖组合
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集+跨域测试+完善消融+推理效率对比+区域控制+风格插值
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示丰富，消融逐个验证每个模块
- 价值: ⭐⭐⭐⭐ 实用性强，仅在小规模真实数据训练即可跨域泛化，推理速度快

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] HSI: A Holistic Style Injector for Arbitrary Style Transfer](../../CVPR2025/image_generation/hsi_a_holistic_style_injector_for_arbitrary_style_transfer.md)
- [\[ICCV 2025\] SA-LUT: Spatial Adaptive 4D Look-Up Table for Photorealistic Style Transfer](sa-lut_spatial_adaptive_4d_look-up_table_for_photorealistic_style_transfer.md)
- [\[CVPR 2025\] StyleStudio: Text-Driven Style Transfer with Selective Control of Style Elements](../../CVPR2025/image_generation/stylestudio_text-driven_style_transfer_with_selective_control_of_style_elements.md)
- [\[CVPR 2026\] RegionRoute: Regional Style Transfer with Diffusion Model](../../CVPR2026/image_generation/regionroute_regional_style_transfer_with_diffusion_model.md)
- [\[CVPR 2025\] SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer](../../CVPR2025/image_generation/samam_style-aware_state_space_model_for_arbitrary_image_style_transfer.md)

</div>

<!-- RELATED:END -->
