---
title: >-
  [论文解读] Efficient and Robust Semantic Image Communication via Stable Cascade
description: >-
  [ICML 2025][语义分割][语义通信] 基于 Stable Cascade 架构构建语义图像通信框架，利用 EfficientNet-V2 提取极紧凑图像嵌入（仅占原始大小 0.29%）作为 LDM 条件，通过噪声鲁棒微调使系统在低 SNR 信道下仍能忠实重建图像，同时实现 3-16 倍推理加速。
tags:
  - "ICML 2025"
  - "语义分割"
  - "语义通信"
  - "潜在扩散模型"
  - "Stable Cascade"
  - "图像压缩"
  - "信道鲁棒性"
---

# Efficient and Robust Semantic Image Communication via Stable Cascade

**会议**: ICML 2025  
**arXiv**: [2507.17416](https://arxiv.org/abs/2507.17416)  
**代码**: [GitHub](https://github.com/abilalk02/SC-SIC)  
**领域**: Semantic Communication / Generative AI  
**关键词**: 语义通信, 潜在扩散模型, Stable Cascade, 图像压缩, 信道鲁棒性

## 一句话总结

基于 Stable Cascade 架构构建语义图像通信框架，利用 EfficientNet-V2 提取极紧凑图像嵌入（仅占原始大小 0.29%）作为 LDM 条件，通过噪声鲁棒微调使系统在低 SNR 信道下仍能忠实重建图像，同时实现 3-16 倍推理加速。

## 研究背景与动机

**领域现状**：语义通信 (SemCom) 旨在传输信息的"含义"而非原始比特，通过深度学习和生成模型实现极致带宽压缩。扩散模型 (DM) 因其出色的图像合成能力成为语义图像通信 (SIC) 的主流工具。现有 DM-based SIC 系统包括 GESCO（分割图条件）、Img2Img-SC（SD 文本+图像条件）等方案。

**现有痛点**：
1. **推理慢**：GESCO 需 1000 步去噪，一张 512×512 图需 5 分 24 秒
2. **生成随机性**：基于文本条件的方案每次结果不同，重建不可控
3. **压缩率不够极致**：SD 潜在空间 [4,64,64] 压缩率仅 ~48x

**核心矛盾**：现有方案在速度、压缩率、重建保真度三者间无法同时达标。GESCO 保真但极慢，Img2Img-SC 较快但生成随机性大，JPEG2000+LDPC 低 SNR 下完全崩溃。

**本文目标** 设计同时实现**极致压缩**（0.29%）、**快速推理**（<1秒）和**高保真重建**的语义通信系统。

**切入角度**：利用 Stable Cascade 的极小潜在空间（比 SD 小得多）天然适合极致压缩，加噪声感知微调提高信道鲁棒性。

**核心 idea**：Stable Cascade 超压缩潜在空间 + 噪声感知条件微调 = 速度 × 压缩 × 保真度三重优势。

## 方法详解

### 整体框架

系统分三阶段：
- **发送端**：EfficientNet-V2 编码器提取极紧凑嵌入 $Z \in \mathbb{R}^{16 \times 24 \times 24}$
- **信道传输**：$Z$ 通过 AWGN 信道，接收端得到 $\hat{Z} = Z + \epsilon$
- **接收端**：$\hat{Z}$ 作为 LDM 条件 → 生成 VQGAN 潜在表示 → VQGAN 解码到像素空间

### 关键设计

1. **极致压缩的图像嵌入（EfficientNet-V2 编码器）**:
    - 功能：将原始图像压缩到 0.29% 大小
    - 核心思路：利用 Stable Cascade 预训练的 EfficientNet-V2 编码器，将 $X \in \mathbb{R}^{3 \times 1024 \times 1024}$ 编码为 $Z \in \mathbb{R}^{16 \times 24 \times 24}$。压缩比 = $\frac{3 \times 1024 \times 1024}{16 \times 24 \times 24} = 341$。该嵌入保留高层语义特征，远优于文本嵌入（太抽象导致语义偏差）和分割图（丢失纹理颜色信息）
    - 设计动机：在信息保真和压缩率之间找到最佳平衡点

2. **噪声感知 LDM 微调（Stage B）**:
    - 功能：让 LDM 学会从含噪条件嵌入中恢复高质量图像
    - 核心思路：Stable Cascade 的 Stage B 原本假设条件输入无噪声。本文在训练时对条件嵌入添加信道噪声 $\hat{Z} = Z + \epsilon$，$\epsilon \sim \mathcal{N}(0, \sigma^2)$，SNR 在 1-20 dB 间随机采样。训练目标为标准 MSE 去噪损失：
    $$L = \mathbb{E}_{(X_\text{VG,t}, t, \hat{Z}, \epsilon)}[\|\epsilon - \bar{\epsilon}(X_\text{VG,t}, t, \hat{Z})\|_2^2]$$
    微调 15000 步，batch=4，lr=1e-4
    - 设计动机：原始 SC 模型在信道噪声下直接崩溃（消融实验证实），噪声感知训练让生成模型自身学会信道去噪

3. **VQGAN 编解码复用（Stage A）**:
    - 功能：像素空间与潜在空间的转换
    - 核心思路：复用 SC 预训练的 VQGAN（4x 空间压缩），$\hat{X} = f_\Theta^{-1}(\hat{X}_\text{VG})$。不使用 Stage C（文本→嵌入，本场景不需要）
    - 设计动机：Stage A 已充分预训练，无需再微调

### 损失函数 / 训练策略

- 仅微调 Stage B（LDM），Stage A 和编码器冻结
- 标准扩散 MSE 去噪损失
- 训练时 SNR 在 1-20 dB 随机采样，确保全范围鲁棒
- 不使用文本条件（SC 论文指出对 Stage B 无显著影响）
- 单卡 NVIDIA RTX A6000 (48GB) 训练

## 实验关键数据

### 主实验：压缩效率对比

| 方法 | 传输数据维度 | 压缩比 | 占原始% |
|------|-------------|--------|---------|
| 原始图像 | [3,512,512] | - | 100% |
| **本文 (SC-SIC)** | **[16,12,12]** | **341** | **0.29%** |
| Img2Img-SC | [4,64,64] | 48 | 2.08% |
| DIFFSC | [8,32,32] | 96 | 1.04% |
| CASC | [8,32,32] | 96 | 1.04% |

### 推理速度对比

| 方法 | 512×512 时间 | 1024×1024 时间 | 去噪步数 |
|------|-------------|---------------|----------|
| GESCO | 5分24秒 | - | 1000 |
| Img2Img-SC | 2.34秒 | >12秒 | 30 |
| **本文** | **0.78秒** | **<1秒** | **10** |

加速比：512×512 为 **3x**，1024×1024 为 **>16x**。

### 重建质量（Cityscapes，vs Img2Img-SC 平均改善）

| 指标 | 改善幅度 | 含义 |
|------|---------|------|
| FID ↓ | -43% | 分布级生成质量更好 |
| LPIPS ↓ | -55% | 感知相似度更高 |
| SSIM ↑ | +56% | 结构保持更好 |
| PSNR ↑ | +23% | 像素精度更高 |

### 重建可预测性（LPIPS μ±σ，25次传输）

| SNR (dB) | 本文-1024 | 本文-512 | GESCO | Img2Img-SC |
|----------|----------|---------|-------|------------|
| 20 | 0.173±0.003 | 0.205±0.005 | 0.401±0.014 | 0.520±0.011 |
| 10 | 0.229±0.003 | 0.264±0.008 | 0.424±0.017 | 0.522±0.012 |
| 1 | 0.351±0.006 | 0.371±0.013 | 0.613±0.017 | 0.578±0.019 |

### 消融实验

| 消融项 | 效果 |
|--------|------|
| 无微调（原始 SC） | SNR<10dB 时图像严重损坏，无法使用 |
| 嵌入 [16,24,24]→[16,32,32] | LPIPS/FID/SSIM 改善 >10%，但压缩比降至 192 |
| JPEG2000+LDPC 在 SNR<5dB | 完全失败（cliff effect），无法恢复图像 |

### 关键发现

- 即使 SNR=1dB 极端信道下，重建图像仍感知上接近原图
- 0.29% 极端压缩下质量优于传输 7 倍数据量的 Img2Img-SC
- 生成一致性极高（LPIPS σ=0.003），文本条件方案 σ=0.011-0.019
- 在未见 DIV2K 数据集上仍能重建语义正确图像，但颜色偏向 Cityscapes 色调
- 传统 JPEG2000+LDPC 存在 cliff effect，低 SNR 下完全崩溃

## 亮点与洞察

1. **压缩率记录**：0.29% 是 DM-based SIC 最高已知压缩比
2. **噪声鲁棒性训练的优雅**：无需复杂信道编码，训练时加噪让生成模型自身学会信道去噪，将通信鲁棒性转化为生成模型训练问题
3. **推理实用化**：0.78 秒完成 512×512 重建，语义通信首次在实时场景可行
4. **低方差重建**：LPIPS σ=0.003 意味着多次传输结果几乎完全一致
5. **SC 架构天然优势**：多阶段设计（Stage A 空间压缩 + Stage B 语义生成）形成自然的分层语义传输

## 局限与展望

- 仅在 Cityscapes 微调，跨域泛化时颜色偏差（DIV2K 实验揭示）
- 固定压缩率，缺乏根据信道状态自适应调整的机制
- 未与视频编码标准（H.265/H.266）系统对比
- 仅支持图像，视频语义通信（帧间一致性）未涉及
- 训练依赖 AWGN 信道模型，真实无线信道（衰落、多径）效果未验证

## 相关工作与启发

- **DeepJSCC (Bourtsoulatze et al., 2019)**：端到端联合源信道编码，本文用预训练生成模型替代
- **GESCO (Grassucci et al., 2023)**：分割图条件扩散 SIC，保真但极慢
- **Img2Img-SC (Cicchetti et al., 2024)**：SD 图像+文本条件，压缩率和速度均不如本文
- **Stable Cascade (Pernias et al., 2023)**：提供超压缩潜在空间架构的核心基础
- 启发：多阶段生成模型天然适合分层语义传输，可扩展到渐进式传输

## 评分

- 新颖性: ⭐⭐⭐⭐ Stable Cascade 在语义通信的巧妙应用，噪声感知微调虽简单但有效
- 实验充分度: ⭐⭐⭐⭐ 多基线多 SNR 对比、消融完整、跨数据集泛化测试
- 写作质量: ⭐⭐⭐⭐ 系统架构清晰，公式推导完整
- 价值: ⭐⭐⭐⭐⭐ 压缩率和速度突破使语义通信实时应用成为可能

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Efficient and Versatile Robust Fine-Tuning of Zero-shot Models](../../ECCV2024/segmentation/efficient_and_versatile_robust_fine-tuning_of_zero-shot_models.md)
- [\[CVPR 2025\] MatAnyone: Stable Video Matting with Consistent Memory Propagation](../../CVPR2025/segmentation/matanyone_stable_video_matting_with_consistent_memory_propagation.md)
- [\[ICCV 2025\] Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling](../../ICCV2025/segmentation/harnessing_massive_satellite_imagery_with_efficient_masked_image_modeling.md)
- [\[CVPR 2025\] Robust 3D Shape Reconstruction in Zero-Shot from a Single Image in the Wild](../../CVPR2025/segmentation/robust_3d_shape_reconstruction_in_zero-shot_from_a_single_image_in_the_wild.md)
- [\[NeurIPS 2025\] Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective](../../NeurIPS2025/segmentation/towards_robust_pseudo-label_learning_in_semantic_segmentation_an_encoding_perspe.md)

</div>

<!-- RELATED:END -->
