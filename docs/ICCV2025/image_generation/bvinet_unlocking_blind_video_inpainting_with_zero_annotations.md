---
title: >-
  [论文解读] BVINet: Unlocking Blind Video Inpainting with Zero Annotations
description: >-
  [ICCV 2025][图像生成][图像修复] 首次定义并解决"盲视频修复"（blind video inpainting）任务——在无需任何损坏区域标注的情况下，端到端地同时完成"哪里需要修复"和"如何修复"，通过 mask 预测网络与视频补全网络的一致性约束互相增强，在合成数据和真实应用（弹幕去除/划痕修复）中均取得优异效果。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "图像修复"
  - "mask prediction"
  - "Transformer"
  - "video completion"
  - "consistency loss"
---

# BVINet: Unlocking Blind Video Inpainting with Zero Annotations

**会议**: ICCV 2025  
**arXiv**: [2502.01181](https://arxiv.org/abs/2502.01181)  
**代码**: 无  
**领域**: 图像生成 / 视频修复  
**关键词**: blind video inpainting, mask prediction, wavelet sparse transformer, video completion, consistency loss

## 一句话总结

首次定义并解决"盲视频修复"（blind video inpainting）任务——在无需任何损坏区域标注的情况下，端到端地同时完成"哪里需要修复"和"如何修复"，通过 mask 预测网络与视频补全网络的一致性约束互相增强，在合成数据和真实应用（弹幕去除/划痕修复）中均取得优异效果。

## 研究背景与动机

现有视频修复方法本质上都是"非盲"（non-blind）的——即假设损坏区域的 mask 已知，用户需手动标注每帧的损坏区域。这带来两大实际问题：

**标注成本高昂**：损坏与正常区域的边界往往模糊，精确标注困难且耗时，高帧率/高分辨率视频更是如此

**应用范围受限**：许多场景下无法预知或手动标注损坏区域，如视频划痕、水印、弹幕等

作者将视频中的"损坏"分为两类：（1）外部因素引入，破坏视频原始结构（划痕、水印、弹幕等）；（2）视频中原本存在的不需要内容（物体移除）。本文聚焦第一类。

直接逐帧使用盲图像修复方法的朴素方案会忽略帧间运动连续性，导致闪烁伪影。因此需要一个端到端的视频级解决方案。

## 方法详解

### 整体框架

BVINet 由两个互相约束的子网络组成：**Mask 预测网络**（MPNet）负责预测损坏区域，**视频补全网络**（VCNet）负责利用预测的 mask 从有效区域提取信息来修复损坏内容。两者通过一致性损失互相约束，共同优化。

### 关键设计

1. **Mask 预测网络（MPNet）**:

    - 两级结构：短期预测模块 + 长期精炼模块
    - **短期预测模块（STP）**：
        - Encoder-decoder 结构，独立处理每帧
        - 检测帧内语义不连续区域来预测损坏 mask：$m_i^s = STP(x_i)$
        - 使用离散小波变换（DWT）替代传统下采样（max-pooling/strided-conv），增强对噪声的抗干扰能力
    - **长期精炼模块（LTR）**：
        - 利用视频时序一致性先验精炼预测的 mask 序列
        - 核心：sequence-to-sequence transformer
        - 将深层特征映射为 Q/K/V，分 N 组沿通道维度，通过响应窗口在 T 帧范围内计算空间-时序亲和矩阵
        - Soft-attention 综合多组亲和矩阵和聚合特征：$\hat{E} = E + Conv(D) \odot G$

2. **视频补全网络（VCNet）—— Wavelet Sparse Transformer**:

    - 创新：在频域中隔离噪声，使用稀疏注意力聚合最相关特征
    - 频域分解：对 Q/K/V 进行离散小波变换（DWT），将噪声隔离到高频分量 $Q_i^H, K_i^H, V_i^H$，低频分量 $Q_i^L, K_i^L, V_i^L$ 仅包含干净的基础特征
    - **双分支注意力机制**：
        - Dense Self-Attention (DSA)：$DSA = Softmax(\frac{Q^L \cdot (K^L)^T}{\sqrt{d}} + B)$
        - Sparse Self-Attention (SSA)：$SSA = Softmax(ReLU(\frac{Q^L \cdot (K^L)^T}{\sqrt{d}}) + B)$，用 ReLU 移除负相似度
        - 自适应加权融合：$\hat{V}^L = (\omega_1 \odot DSA + \omega_2 \odot SSA) V^L$
    - 对 DSA 和 SSA 的损坏区域值设为 0，确保仅从有效区域借取信息
    - 逆小波变换（IDWT）恢复最终特征

3. **一致性约束（Consistency Loss）**:

    - 核心思想：如果 mask 预测和视频补全都准确，损坏视频帧 $x_i$ 与补全结果 $\hat{y}_i$ 的差异应仅存在于损坏区域
    - 一致性关系：$m_i^l = \mathcal{B}(\hat{y}_i - x_i)$
    - 一致性损失：$\mathcal{L}_c = \|m_i^l - \mathcal{B}(\hat{y}_i - x_i)\|_1 + \|m_i - \mathcal{B}(\hat{y}_i - x_i)\|_1$
    - 使两个网络精确对应，互相约束

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \lambda_m \mathcal{L}_m + \lambda_v \mathcal{L}_v + \lambda_c \mathcal{L}_c$

- $\mathcal{L}_m$：mask 预测损失
- $\mathcal{L}_v$：视频补全损失
- $\mathcal{L}_c$：一致性损失
- 超参数：$\lambda_m = 3$, $\lambda_v = 5$, $\lambda_c = 0.02$（网格搜索确定）

**专用数据集构建**：
- 用自由形式笔划作为损坏区域 mask，填充真实图像内容（非常量值或噪声）
- 迭代高斯模糊扩展损坏边界，消除明显边缘先验，迫使模型从语义上下文推断
- 数据集：2,400 合成视频 + 1,250 真实弹幕去除视频

## 实验关键数据

### 主实验（盲设置 vs 非盲设置）

YouTube-VOS 数据集：

| 方法 | 盲设置 | PSNR↑ | SSIM↑ | $E_{warp}$↓ | LPIPS↓ |
|------|--------|-------|-------|-------------|--------|
| FGT (非盲) | ✗ | 30.811 | 0.9258 | 0.1308 | 0.4565 |
| WaveFormer (非盲) | ✗ | 33.264 | 0.9435 | 0.1184 | 0.2933 |
| **VCNet (非盲)** | ✗ | **34.107** | **0.9521** | **0.1102** | **0.2145** |
| MPNet+FGT (盲) | ✓ | 27.032 | 0.8755 | 0.1609 | 0.8667 |
| MPNet+WaveFormer (盲) | ✓ | 29.185 | 0.8902 | 0.1508 | 0.7153 |
| **BVINet (盲)** | ✓ | **30.528** | **0.9088** | **0.1362** | **0.6556** |

### 消融实验

MPNet 消融：

| 配置 | BCE↓ | IOU↑ |
|------|------|------|
| STP (strided-conv) | 1.1251 | 0.8437 |
| DWT_STP | 1.0785 | 0.8682 |
| DWT_STP + LTR | 0.9176 | 0.8829 |
| **Full MPNet** | **0.8052** | **0.9017** |

稀疏注意力 + 一致性损失消融：

| DSA | SSA | $\mathcal{L}_c$ | PSNR↑ | SSIM↑ | $E_{warp}$↓ | LPIPS↓ |
|-----|-----|-----------------|-------|-------|-------------|--------|
| ✓ | ✗ | ✗ | 29.172 | 0.8897 | 0.1529 | 0.7264 |
| ✓ | ✓ | ✗ | 29.885 | 0.8962 | 0.1454 | 0.6891 |
| ✓ | ✓ | ✓ | **30.528** | **0.9088** | **0.1362** | **0.6556** |

### 效率分析

| 方法 | FLOPs | 推理时间 |
|------|-------|---------|
| STTN | 477.91G | 0.22s |
| FuseFormer | 579.82G | 0.30s |
| E2FGVI | 442.18G | 0.26s |
| FGT | 455.91G | 0.39s |
| **VCNet** | **396.35G** | **0.21s** |

### 关键发现

- 盲设置下的 BVINet 性能可与非盲设置下的次优方法 E2FGVI 相当（PSNR 30.528 vs 30.064），验证了盲修复的可行性
- VCNet 即使在非盲设置下也大幅超越现有方法（PSNR 34.107 vs 次优 33.264）
- DWT 下采样显著提升 mask 预测质量（IOU 0.8437→0.8682），长期精炼进一步提升至 0.9017
- 双分支（DSA+SSA）比单独使用任一种效果更好，一致性损失带来额外 0.6 PSNR 提升
- 模型对多种损坏模式具有鲁棒性：高斯噪声、纯色填充等训练中未见的模式也能处理
- 弹幕去除等真实应用中效果优于 OGNet 和 RAVUNet

## 亮点与洞察

- **开创性任务定义**：首次提出盲视频修复任务，将"在哪修"和"怎么修"统一到一个框架
- **互约束设计精巧**：mask 预测和视频补全通过一致性损失形成闭环——mask 帮助定位，补全结果反过来验证 mask 质量
- **频域稀疏注意力**：DWT 分离噪声到高频、ReLU 过滤负相似度的组合，比传统注意力更适合修复任务
- **数据集设计思路**：用真实图像而非常量值填充损坏区域 + 高斯模糊边缘，避免模型学到分布先验而非语义理解

## 局限与展望

- 主要聚焦外部损坏（划痕/弹幕等），对视频中原有不需要内容的移除（第二类损坏）效果未验证
- 数据集规模较小（2,400+1,250 视频），更大规模数据可能进一步提升泛化性
- 盲设置下与最强非盲方法仍有 3-4 dB PSNR 差距，mask 预测精度是瓶颈
- 未探索扩散模型在盲视频修复中的潜力
- 损坏类型主要是叠加式损坏，对压缩伪影等退化类型的适用性有待验证

## 相关工作与启发

- 盲图像修复到盲视频修复的跨越不是简单的逐帧扩展——时序一致性是核心挑战
- 视频修复中"仅从有效区域借取信息"的约束与普通注意力的全局聚合形成对比，稀疏注意力的选择性更适合此类任务
- 一致性约束的思想可推广到其他需要"检测+修复"联合优化的视频恢复任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 全新任务定义，端到端盲修复框架具有开创性
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据验证，16种基线对比，详细消融和效率分析
- 写作质量: ⭐⭐⭐⭐ 问题形式化清晰，方法分解逻辑性强
- 价值: ⭐⭐⭐⭐ 为视频修复开辟了零标注的新范式，弹幕/水印去除等应用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Unlocking the Potential of Diffusion Priors in Blind Face Restoration](unlocking_the_potential_of_diffusion_priors_in_blind_face_restoration.md)
- [\[ICCV 2025\] AnyPortal: Zero-Shot Consistent Video Background Replacement](anyportal_zero-shot_consistent_video_background_replacement.md)
- [\[ICCV 2025\] Inpaint4Drag: Repurposing Inpainting Models for Drag-Based Image Editing via Bidirectional Warping](inpaint4drag_repurposing_inpainting_models_for_drag-based_image_editing_via_bidi.md)
- [\[ICCV 2025\] Less-to-More Generalization: Unlocking More Controllability by In-Context Generation](less-to-more_generalization_unlocking_more_controllability_by_in-context_generat.md)
- [\[ICCV 2025\] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers](repa-e_unlocking_vae_for_end-to-end_tuning_of_latent_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
