---
title: >-
  [论文解读] Efficient Video Face Enhancement with Enhanced Spatial-Temporal Consistency
description: >-
  [CVPR 2025][人体理解][盲人脸视频修复] 本文提出一种基于 3D-VQGAN 的高效盲人脸视频增强框架，通过设计空间-时间双码本记录高质量肖像特征和运动残差信息，配合边际先验正则化缓解码本崩溃问题，在 BFVR 和去闪烁任务上实现了 SOTA 效果且推理速度提升 2-140 倍。 领域现状：盲人脸修复 (BFR)…
tags:
  - "CVPR 2025"
  - "人体理解"
  - "盲人脸视频修复"
  - "时空码本"
  - "GAN"
  - "去闪烁"
  - "视频增强"
---

# Efficient Video Face Enhancement with Enhanced Spatial-Temporal Consistency

**会议**: CVPR 2025  
**arXiv**: [2411.16468](https://arxiv.org/abs/2411.16468)  
**代码**: [https://github.com/Dixin-Lab/BFVR-STC](https://github.com/Dixin-Lab/BFVR-STC)  
**领域**: 人体理解 / 视频理解  
**关键词**: 盲人脸视频修复, 时空码本, 3D-VQGAN, 去闪烁, 视频增强

## 一句话总结

本文提出一种基于 3D-VQGAN 的高效盲人脸视频增强框架，通过设计空间-时间双码本记录高质量肖像特征和运动残差信息，配合边际先验正则化缓解码本崩溃问题，在 BFVR 和去闪烁任务上实现了 SOTA 效果且推理速度提升 2-140 倍。

## 研究背景与动机

**领域现状**：盲人脸修复 (BFR) 旨在从退化的低质量人脸中重建高质量结果。图像级方法（GFPGAN、CodeFormer、VQFR）已取得显著进展，主要基于几何先验、生成先验（StyleGAN）和码本先验（VQGAN）。视频级方法（PGTFormer、KEEP、StableBFVR）进一步引入时序约束保证帧间一致性。

**现有痛点**：现有 BFVR 方法存在两大问题。第一，推理效率低下——KEEP 需要额外的 RealESRGAN 预处理背景和人脸检测模型，StableBFVR 需要 BasicVSR++ 做初步修复，复杂的处理流程导致推理时间过长。第二，时间一致性不足——PGTFormer 的感受野仅覆盖两个相邻帧，无法保证全局一致性；码本的离散化本质导致帧间特征跳变引起闪烁。此外，视频闪烁问题（包括真实视频的亮度闪烁和 AI 生成视频的像素闪烁）也缺乏高效解决方案。

**核心矛盾**：高质量视频人脸修复需要同时保证帧级修复质量和帧间时序一致性，但码本的离散表示天然与连续的时域变化矛盾；同时，复杂的视频处理流程导致效率低下。

**本文目标** (1) 如何将 VQGAN 范式高效扩展到视频域以支持视频级压缩和量化？(2) 如何设计码本机制来同时记录空间肖像特征和时序运动信息？(3) 如何解决多码本场景下的码本崩溃问题？

**切入角度**：作者将 VQGAN 从 2D 扩展到 3D（视频域），设计了空间和时间两个独立码本分别记录不同类型的信息。空间码本记录高质量人脸特征的静态外观，时间码本记录帧间的运动残差。通过边际先验正则化取代传统的 one-hot 频率统计来解决码本崩溃。

**核心 idea**：用 3D-VQGAN 做视频级压缩配合空间-时间双码本分别记录外观和运动信息，实现高效且时序一致的视频人脸增强。

## 方法详解

### 整体框架

分为两个训练阶段。**Stage I**：用高质量视频训练 3D-VQGAN（3D 编码器 $E_h$ 和解码器 $D_h$）以及空间/时间码本（$\mathcal{C}_S$ 和 $\mathcal{C}_T$），通过重建任务自监督学习 HQ 人脸的离散表示。**Stage II**：固定 Stage I 的码本和解码器，训练低质量视频编码器 $E_l$ 和两个 Transformer 码本查找模块（$\mathcal{T}_S$ 和 $\mathcal{T}_T$），从 LQ 输入预测对应的码本索引序列，然后用 HQ 解码器重建高质量视频。

### 关键设计

1. **3D-VQGAN 与强判别器**:

    - 功能：实现视频级的高效压缩和量化，支持时空维度的联合下采样
    - 核心思路：编码器和解码器为纯卷积 3D 结构（残差块 + 上下采样块 + 卷积自注意力），实现空间 8 倍、时间 2 倍的压缩比。为解决视频级 VQGAN 训练的不稳定性和伪影问题，使用冻结的预训练 DINOv2 特征网络作为判别器骨干，搭配多个可训练的轻量级判别头：$D_\phi(\hat{x}_h(\theta)) = -\mathbb{E}_{x_h}(\sum_k \mathcal{D}_{\phi,k}(\mathcal{F}(\hat{x}_h(\theta))))$
    - 设计动机：纯卷积结构比 Transformer 更高效且支持任意分辨率输入；DINOv2 预训练特征提供了比从头训练更稳定的判别信号

2. **空间-时间双码本 (Spatial-Temporal Codebooks)**:

    - 功能：分别记录高质量肖像的空间外观特征和帧间的运动残差信息
    - 核心思路：给定编码器输出的压缩表示 $\bm{z}_h$，空间潜变量直接取 $\bm{z}_{h,S} = \bm{z}_h$，时间潜变量通过帧间时间注意力和运动残差计算：$\bm{z}_{h,T} = \text{TA}(\bm{z}_h) + \text{Residual}(\bm{z}_h)$。运动残差定义为相隔时间窗口的两帧潜变量之差。两组潜变量分别在各自码本中做最近邻检索得到量化表示，最后通过逐元素加法融合：$\bm{z}_q = \bm{z}_{q,S} \oplus \bm{z}_{q,T}$
    - 设计动机：传统码本只记录空间特征，无法捕获帧间的运动信息，导致修复结果的时序一致性差。将运动信息（残差）显式编码到独立码本中，让解码器能同时利用"这帧长什么样"和"这帧相比前帧变了什么"两种信息

3. **边际先验正则化 (Marginal Prior Regularization)**:

    - 功能：缓解多码本场景下的码本崩溃（只有少数 code 被使用）
    - 核心思路：计算潜变量与码本之间的欧氏距离矩阵，转换为相似度分数后对行归一化。通过对列求和得到边际后验分布 $P_{post}$，用 KL 散度约束其趋近均匀先验 $P_{prior}$：$\mathcal{L}_{KL}^S = \text{KL}(P_{post}, P_{prior})$。关键区别在于——传统方法用检索到的 one-hot 索引统计频率（对未被检索到的码字不公平），本文用连续的相似度分数累加，让所有码字都能获得梯度
    - 设计动机：多码本（空间+时间）加剧了码本崩溃问题。用相似度分数而非 one-hot 编码统计使用频率，让更多码字参与优化，提高码本利用率

### 损失函数 / 训练策略

**Stage I 损失**：$\mathcal{L}_I = \mathcal{L}_1 + \mathcal{L}_{per} + \mathcal{L}_f + (\mathcal{L}_{KL}^S + \mathcal{L}_{KL}^T) + \lambda_{adv} \cdot \mathcal{L}_{adv}$，包含 L1 重建损失、VGG 感知损失、码本-编码器对齐损失、边际先验正则化和 DINOv2 对抗损失。

**Stage II 损失**：$\mathcal{L}_{II} = \mathcal{L}'_f + \lambda_{CE} \cdot (\mathcal{L}_{CE}^S + \mathcal{L}_{CE}^T)$，包含码本对齐损失和空间/时间码本查找的交叉熵预测损失。

训练在 4 张 A100 GPU 上进行，Stage I 训练 250K iterations（256² 分辨率），Stage II 训练 50K iterations（512² 分辨率）。

## 实验关键数据

### 主实验

VFHQ-Test 上盲人脸视频修复对比（24帧/1秒视频）：

| 方法 | 类型 | SSIM↑ | FVD↓ | Flow-Score↓ | Runtime(s)↓ |
|------|------|-------|------|-------------|-------------|
| GFPGAN | BFIR | 0.8207 | 246.9 | 1.316 | 14.44 |
| CodeFormer | BFIR | 0.8102 | 261.8 | 2.672 | 28.18 |
| BasicVSR++ | VSR | 0.8218 | 392.7 | 1.286 | 72.21 |
| PGTFormer | BFVR | 0.8426 | 107.6 | 1.154 | 7.085 |
| KEEP | BFVR | 0.8223 | 264.9 | 1.302 | 19.01 |
| **Ours** | BFVR | **0.8641** | **105.1** | **1.150** | **2.995** |

亮度去闪烁：

| 方法 | 需GT | FVD↓ | Runtime(s)↓ |
|------|------|------|-------------|
| DVP | 是 | 14.53 | 410.2 |
| FastBlend | 是 | 34.58 | 18.44 |
| **Ours** | **否** | **100.7** | **2.934** |

### 消融实验

| 配置 | SSIM↑ | FVD↓ |
|------|-------|------|
| ViT-S + DINOv2 判别器 | 0.9054 | 49.11 |
| ViT-B + DINOv2 判别器 | 0.9050 | 49.08 |
| ViT-B + CLIP 判别器 | 0.8935 | 66.86 |

| 融合方式 | SSIM↑ | FVD↓ |
|---------|-------|------|
| 逐元素加法 | 0.9054 | 49.11 |
| 卷积融合 | 0.8799 | 97.67 |
| 3DFFT | 0.8741 | 118.7 |

### 关键发现
- 推理效率大幅领先：2.995s vs PGTFormer 7.085s vs KEEP 19.01s vs BasicVSR++ 72.21s，最快的 BFVR 方法快了 2.4 倍
- SSIM 达到 0.8641，显著优于所有 baseline；FVD 和 Flow-Score 也达到最优，证明时序一致性最好
- DINOv2 作为判别器特征网络比 CLIP 效果显著更好（FVD 49.11 vs 66.86），且 ViT-S 与 ViT-B 性能几乎一致，说明不需要更大的特征网络
- 空间-时间码本融合中，简单的逐元素加法显著优于卷积融合和 3DFFT，说明两个码本的信息是互补叠加关系

## 亮点与洞察
- **将 VQGAN 从 2D 扩展到 3D 视频域的整体设计**非常完整——从 3D 编解码器、时空码本、边际正则化到两阶段训练，每个组件都有明确的设计动机和消融验证
- **边际先验正则化**的改进虽然简单但巧妙——将 one-hot 索引替换为连续相似度分数来统计码本使用频率，让未被最近邻命中的码字也能获得梯度信号。这个 trick 可以迁移到任何需要码本的 VQ 方法中
- **DINOv2 作为判别器骨干**的思路值得注意——预训练特征提供了更稳定的训练信号，避免了从头训练判别器的模式崩溃问题。这种"冻结特征网络 + 轻量头"的判别器设计模式具有普适性

## 局限与展望
- 去闪烁任务上 FVD 不如 DVP（100.7 vs 14.53），但 DVP 需要无闪烁的参考视频，本文方法不需要。公平对比下优势明确但效果仍有差距
- 训练数据经过严格筛选（从 16000 视频筛到 3200），部署到多样化场景的泛化性未知
- 视频输入被限制在 24 帧（1秒），更长视频的处理需要滑窗或分段，可能引入段间不一致
- 仅关注人脸区域，需要先做人脸检测裁剪。虽然通过训练数据裁剪简化了流程，但面部占比低的视频仍需额外处理

## 相关工作与启发
- **vs CodeFormer**: CodeFormer 是图像级 VQGAN 码本方法的代表，本文将其扩展到视频域并增加了时间码本，在视频修复上全面超越
- **vs PGTFormer**: PGTFormer 是当前 BFVR SOTA，但时间感受野仅两帧。本文通过 3D 卷积和运动残差码本获得更大的时间感受野
- **vs KEEP**: KEEP 基于卡尔曼滤波利用已修复帧指导后续帧，但需要 RealESRGAN 预处理，流程复杂。本文端到端框架更简洁高效

## 评分
- 新颖性: ⭐⭐⭐⭐ 3D-VQGAN + 时空双码本的设计有创新性，边际先验正则化改进简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ BFVR + 两种去闪烁任务 + 详细消融，指标全面（质量、一致性、效率）
- 写作质量: ⭐⭐⭐⭐ 方法动机清晰，框架描述完整，图表丰富
- 价值: ⭐⭐⭐⭐ 在效率和效果上同时取得突破，具有明确的工业落地价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] IDFace: Face Template Protection for Efficient and Secure Identification](../../ICCV2025/human_understanding/idface_face_template_protection_for_efficient_and_secure_identification.md)
- [\[ECCV 2024\] RePOSE: 3D Human Pose Estimation via Spatio-Temporal Depth Relational Consistency](../../ECCV2024/human_understanding/repose_3d_human_pose_estimation_via_spatio-temporal_depth_relational_consistency.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [\[CVPR 2025\] ShowMak3r++: Compositional Entertainment Video Reconstruction](showmak3r_compositional_tv_show_reconstruction.md)
- [\[CVPR 2025\] Two is Better than One: Efficient Ensemble Defense for Robust and Compact Models](two_is_better_than_one_efficient_ensemble_defense_for_robust_and_compact_models.md)

</div>

<!-- RELATED:END -->
