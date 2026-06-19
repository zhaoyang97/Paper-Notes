---
title: >-
  [论文解读] SAFNet: Selective Alignment Fusion Network for Efficient HDR Imaging
description: >-
  [ECCV 2024][视频理解][HDR成像] SAFNet 提出选择性对齐融合策略，通过金字塔解码器联合精炼有价值区域掩码和跨曝光光流，仅在有价值区域进行精确对齐后显式融合 HDR 图像，在 Kalantari 17 和自建 Challenge123 数据集上超越 SOTA 的同时推理速度快一个数量级。
tags:
  - "ECCV 2024"
  - "视频理解"
  - "HDR成像"
  - "选择性对齐"
  - "光流估计"
  - "多曝光融合"
  - "高效网络"
---

# SAFNet: Selective Alignment Fusion Network for Efficient HDR Imaging

**会议**: ECCV 2024  
**arXiv**: [2407.16308](https://arxiv.org/abs/2407.16308)  
**代码**: [https://github.com/ltkong218/SAFNet](https://github.com/ltkong218/SAFNet)  
**领域**: 视频理解  
**关键词**: HDR成像, 选择性对齐, 光流估计, 多曝光融合, 高效网络

## 一句话总结

SAFNet 提出选择性对齐融合策略，通过金字塔解码器联合精炼有价值区域掩码和跨曝光光流，仅在有价值区域进行精确对齐后显式融合 HDR 图像，在 Kalantari 17 和自建 Challenge123 数据集上超越 SOTA 的同时推理速度快一个数量级。

## 研究背景与动机

多曝光 HDR 成像需要将不同曝光的 LDR 图像合成为 HDR 图像，核心挑战在于动态场景中的运动失配和饱和区域的纹理截断。现有深度学习方法分为两大范式：

**对齐后融合 (Alignment + Fusion)**：先估计跨曝光光流进行对齐，再生成 HDR 图像（如 Kalantari et al.）。问题：在严重饱和和遮挡区域，光流估计极易出错。

**注意力隐式融合**：绕过显式对齐，用各种注意力机制实现空间/通道级特征交互（如 AHDRNet、HDR-Transformer、SCTNet）。问题：计算复杂度高，推理延迟大，难以部署到资源受限设备。

两类方法的效果逐步提升，但计算成本也不断增加。本文的核心观察是：**不是所有非参考帧的区域都值得精确对齐**。例如，过曝/欠曝区域或参考帧中已有良好纹理的区域可以直接丢弃；而非参考帧中包含参考帧缺失的有价值纹理的区域，有纹理区域的运动估计反而比饱和区域容易得多。基于此观察，SAFNet 只在有价值区域进行对齐，跳过困难且无用的饱和区域运动估计。

## 方法详解

### 整体框架

SAFNet 包含三个子网络：金字塔编码器 E、由粗到精解码器 D、细节精炼模块 R。流程为：(1) 编码器提取各输入帧的金字塔特征；(2) 解码器联合精炼选择概率掩码 M 和跨曝光光流 F；(3) 使用光流对齐 + 掩码重加权的融合系数显式合成初始 HDR 图像 Hm；(4) 精炼网络基于光流、掩码、Hm 和 LDR 输入生成最终 HDR 图像 Hr。

### 关键设计

1. **选择性光流估计 (Selective Flow Estimation)**：解码器在由粗到精的光流精炼过程中，同时输出选择概率掩码 M（sigmoid 输出，范围 0-1）标识有价值区域。掩码和光流互相促进：M 告诉解码器关注哪些区域的 F 估计，更好的 F 又能聚合有价值特征促进进一步的区域识别和残差光流估计。公式：
    $[F_{2\to1}^{k-1}, F_{2\to3}^{k-1}, M_1^{k-1}, M_3^{k-1}] = \mathcal{D}^k([F_{2\to1}^k, F_{2\to3}^k, M_1^k, M_3^k, \tilde{\phi}_1^k, \phi_2^k, \tilde{\phi}_3^k])$
   设计动机：跳过饱和区域的运动估计（这些区域本就容易出错），将模型学习能力集中在更有意义的事情上。

2. **显式 HDR 融合 (Explicit HDR Fusion)**：使用选择掩码重加权融合系数，然后显式合成 HDR。关键公式：
    $W_1 = \Lambda_1 \odot M_1, \quad W_3 = \Lambda_3 \odot M_3$
    $W_2 = \Lambda_2 + \Lambda_1 \odot (1-M_1) + \Lambda_3 \odot (1-M_3)$
    $H_m = W_1 \odot \tilde{H}_1 + W_2 \odot H_2 + W_3 \odot \tilde{H}_3$
   未被选中区域的融合权重转移给参考帧，确保归一化。设计动机：显式融合比隐式注意力更高效，且掩码自然地抑制了失配区域的鬼影。

3. **轻量精炼模块 + 窗口分区裁剪 (Refine Module + Window Partition Cropping)**：精炼网络 R 是全卷积网络，在原始分辨率上增强高频细节。利用第一阶段的光流、掩码、Hm 作为额外输入（消融证明 Hm 贡献最大）。训练时提出窗口分区裁剪：第一阶段在 512×512 大 patch 上处理长程纹理聚合，第二阶段在 128×128 小 patch 上精炼局部细节，通过 window partition/reverse 操作统一两种裁剪尺寸。

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \mathcal{L}_r + \beta \mathcal{L}_m$（β=0.1）
- 精炼损失：$\mathcal{L}_r = \mathcal{L}_1(T(H_r), T(H_{gt})) + \alpha \mathcal{L}_p(T(H_r), T(H_{gt}))$（μ-law tonemapping + L1 + 感知损失，α=0.01）
- 融合损失：$\mathcal{L}_m = \mathcal{L}_1(T(H_m), T(H_{gt})) + \mathcal{L}_c(T(H_m), T(H_{gt}))$（L1 + census loss，监督第一阶段对齐和融合）
- 解码器使用 group conv（group=3）+ channel shuffle 提高效率
- 光流和掩码在 1/2 分辨率预测后上采样

## 实验关键数据

### 主实验

| 数据集 | 指标 | SAFNet | 之前SOTA | 速度对比 |
|--------|------|------|----------|------|
| Kalantari 17 | PSNR-μ | 44.66 dB | FlexHDR 44.35 dB | SAFNet快10× |
| Kalantari 17 | PSNR-l | 43.18 dB | FlexHDR 42.60 dB | +0.58 dB |
| Kalantari 17 | SSIM-l | 0.9917 | FlexHDR 0.9902 | +0.0015 |
| Kalantari 17 | 推理时间 | 0.151s | SCTNet 3.466s | 快23× |
| Kalantari 17 | 参数量 | 1.12M | SCTNet 0.99M | 相当 |
| Challenge123 (512²) | PSNR-μ | 41.88 dB | AHDRNet 40.61 dB | +1.27 dB |
| Challenge123 (512²) | PSNR-l | 29.73 dB | AHDRNet 28.33 dB | +1.40 dB |

### 消融实验

| 配置 | PSNR-μ | PSNR-l | 说明 |
|------|---------|------|------|
| 光流 F + 无掩码 M | 33.69 | 36.30 | 无选择性，饱和区光流出错严重 |
| 无光流 F + 掩码 M | 40.69 | 37.08 | 无对齐，缺失移动区域纹理 |
| 光流 F + 掩码 M | 41.68 | 39.61 | 联合精炼效果最好 |
| 精炼输入无 Hm | 43.63 | 41.67 | 失去第一阶段融合信息 |
| 精炼输入有 F+M+Hm | 44.59 | 43.15 | 所有信息互补 |
| S1=128, S2=128 | 44.59 | 43.15 | 小 patch 训练 |
| S1=512, S2=128 (WPC) | 44.66 | 43.18 | 窗口分区裁剪最优 |

### 关键发现

1. **掩码的贡献巨大**：去掉掩码后 PSNR-μ 从 41.68 暴降到 33.69（-8.0 dB），证实选择性对齐的核心价值
2. **速度优势压倒性**：比 HDR-Transformer 快 18 倍、比 SCTNet 快 23 倍、比 FlexHDR 快 10 倍，得益于纯卷积架构无复杂注意力
3. **大运动场景优势明显**：在自建 Challenge123 数据集上（平均运动 128.7 像素 vs Kalantari 17 的 20.1 像素），SAFNet 的优势进一步放大
4. **Transformer 方法的 patch 局限**：基于 patch 的 Transformer 方法无法跨 patch 聚合大运动产生的纹理，导致块状伪影
5. 窗口分区裁剪策略：大 patch 促进长程聚合，小 patch 促进细节精炼，两者互补

## 亮点与洞察

- **"不是所有区域都值得对齐"**这一观察非常精准——将计算资源集中在有意义的区域，同时避免在困难且无用区域的错误传播
- 掩码和光流的**联合精炼形成正反馈循环**：掩码指导光流关注有价值区域，更好的光流促进更准确的区域识别
- 自建 Challenge123 数据集填补了大运动 HDR 评估的空白（平均运动 128.7 vs 20.1 像素，饱和比 0.201 vs 0.061）
- 窗口分区裁剪是一个优雅的训练技巧，巧妙地统一了两阶段不同裁剪尺寸的需求

## 局限与展望

1. 在 Tel 23 数据集上 PSNR-μ 和 HDR-VDP2 略逊于 Transformer 方法，说明在以去鬼影为主（而非大运动聚合）的场景中，注意力机制仍有优势
2. 显式光流对齐在极端遮挡和严重变形场景中仍可能失败
3. 精炼模块使用膨胀残差块较为简单，更复杂的精炼策略可能进一步提升
4. 两阶段流水线引入了额外的超参数（如 β、α、窗口尺寸），需要仔细调优
5. Challenge123 数据集虽有挑战性，但样本量较小（96 训练 + 27 测试），可能不够多样

## 相关工作与启发

- **Kalantari et al. (2017)**：开创性的学习型 HDR 方法（光流对齐 + CNN 融合），SAFNet 在此基础上增加选择性机制
- **SCTNet / HDR-Transformer**：代表性的 Transformer HDR 方法，精度高但速度慢一到两个数量级
- **PWC-Net / LiteFlowNet**：金字塔光流估计的成功架构，SAFNet 的编解码器设计从中借鉴
- 启示：**选择性处理 + 显式操作**在效率敏感的任务中可能比全局隐式注意力更优

## 评分

- 新颖性: ⭐⭐⭐⭐ 选择性对齐融合的思路清晰优雅，窗口分区裁剪是巧妙的训练创新
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集 + 一个自建数据集 + 详尽消融 + 效率对比 + 泛化测试
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示直观，数据集贡献有价值
- 价值: ⭐⭐⭐⭐⭐ 速度-精度权衡的新记录，对移动端 HDR 部署具有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] VideoMamba: Spatio-Temporal Selective State Space Model](videomamba_spatio-temporal_selective_state_space_model.md)
- [\[ECCV 2024\] RGNet: A Unified Clip Retrieval and Grounding Network for Long Videos](rgnet_a_unified_clip_retrieval_and_grounding_network_for_long_videos.md)
- [\[ECCV 2024\] PiTe: Pixel-Temporal Alignment for Large Video-Language Model](pite_pixel-temporal_alignment_for_large_video-language_model.md)
- [\[CVPR 2026\] SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion](../../CVPR2026/video_understanding/savax_egotoexo_imitation_error_detection_via_scene.md)
- [\[CVPR 2025\] Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models](../../CVPR2025/video_understanding/video-panda_parameter-efficient_alignment_for_encoder-free_video-language_models.md)

</div>

<!-- RELATED:END -->
