---
title: >-
  [论文解读] Gaussian Eigen Models for Human Heads
description: >-
  [CVPR 2025][3D视觉][Gaussian人头建模] 提出 Gaussian Eigen Models (GEM)，通过 PCA 将高质量 CNN-based Gaussian Avatar 蒸馏为轻量级线性特征基表示，仅需低维系数的线性组合即可生成面部动画，实现高质量、超轻量（7MB起）和超快速（200+ fps）的可动画头像，并支持从单目视频的实时跨人表情驱动。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "Gaussian人头建模"
  - "特征基蒸馏"
  - "线性形变模型"
  - "跨人reenactment"
  - "实时渲染"
---

# Gaussian Eigen Models for Human Heads

**会议**: CVPR 2025  
**arXiv**: [2407.04545](https://arxiv.org/abs/2407.04545)  
**代码**: [https://zielon.github.io/gem/](https://zielon.github.io/gem/) (项目页，代码将公开)  
**领域**: 3D视觉  
**关键词**: Gaussian人头建模, 特征基蒸馏, 线性形变模型, 跨人reenactment, 实时渲染

## 一句话总结
提出 Gaussian Eigen Models (GEM)，通过 PCA 将高质量 CNN-based Gaussian Avatar 蒸馏为轻量级线性特征基表示，仅需低维系数的线性组合即可生成面部动画，实现高质量、超轻量（7MB起）和超快速（200+ fps）的可动画头像，并支持从单目视频的实时跨人表情驱动。

## 研究背景与动机

1. **领域现状**：3D Gaussian Avatar 方法已可生成逼真的可动画头像。高质量方法（如 Animatable Gaussians）依赖复杂 CNN 架构生成表情相关的外观变化（如皱纹、自阴影），但需要大量计算资源。

2. **现有痛点**：(a) CNN-based 方法模型参数量动辄上千万、检查点超过 500MB，不适合消费级设备；(b) 无 CNN 的方法（如 Gaussian Avatars）质量有限，无法捕捉皱纹等动态细节；(c) 现有方法大多依赖 FLAME 3DMM 追踪，不支持直接从图像驱动。

3. **核心矛盾**：高质量需要重型 CNN，轻量级缺乏表达力。能否在推理时完全去掉 CNN 同时保留 CNN 级别的质量？

4. **本文目标** 将 CNN-based 的高质量 Gaussian Avatar 蒸馏为不需要任何神经网络的轻量线性模型，同时保持高质量和可控性。

5. **切入角度**：受 3DMM（3D Morphable Model）的 PCA 线性基表示启发——如果网格顶点可以用线性基表示，那么 3D Gaussian 的属性（位置、旋转、缩放、不透明度）也应该可以。

6. **核心 idea**：对多帧 Gaussian 点云按模态（位置、旋转、缩放、不透明度）分别做 PCA 构建特征基，新表情只需特征基的线性组合即可生成，完全无需神经网络推理。

## 方法详解

### 整体框架
方法分三大步骤。(1) **构建高质量 Gaussian 数据集**：使用改进版 Animatable Gaussians (CNN) 在多视角视频上训练，生成每帧的规范空间 Gaussian 点云 $\{G_0,...,G_{N-1}\}$。(2) **蒸馏为 GEM**：对这些 Gaussian 按模态分别做 PCA，得到位置、旋转、缩放、不透明度四组特征基，并通过光度损失精炼基向量。(3) **图像驱动的动画**：训练一个轻量回归器，从单张 RGB 图像预测 GEM 系数，实现实时驱动。

### 关键设计

1. **按模态分离的 PCA 蒸馏（Per-modality PCA Distillation）**:

    - 功能：将 CNN 生成的高质量 Gaussian 序列压缩为线性特征基
    - 核心思路：对 $N$ 帧 Gaussian 数据 $\{G_0,...,G_{N-1}\}$，分别对位置 $\phi$、旋转 $\theta$、缩放 $\sigma$、不透明度 $\alpha$ 四种属性做 PCA，得到四组特征基 $B_\phi, B_\theta, B_\sigma, B_\alpha$ 和均值。颜色 $c$ 作为全局参数单独优化不做 PCA（保持 Gaussian 语义一致性）。新表情通过线性组合生成：$G=\{\mu_i + B_i k_i \mid i \in \{\theta,\phi,\alpha,\sigma\}, \vec{c}\}$，其中 $k$ 是系数向量。使用 $M=50$ 个主成分即可达到高质量。关键：PCA 后用训练图像对基向量做光度损失精炼 30K 步，每 1K 步 QR 分解保持正交性，PSNR 从 34.75 提升到 36.85。
    - 设计动机：颜色不做 PCA 是因为它的变化会导致 Gaussian 语义变化（同一个 Gaussian 在不同帧可能表示嘴唇或牙齿），破坏 PCA 的前提。按模态分离允许独立控制每个属性的压缩程度。

2. **Gaussian Map 的 CNN 数据生成器**:

    - 功能：生成用于 PCA 蒸馏的高质量逐帧 Gaussian 点云
    - 核心思路：改进 Animatable Gaussians 架构——合并双 StyleUNet 为单一网络，减少卷积层数，在 FLAME UV 空间操作。使用变形梯度处理从规范空间到变形空间的变换。Gaussian 组织为 2D maps（每个像素代表一个 Gaussian），$256^2$ 分辨率产生约 6 万活跃 Gaussian。这种 CNN 能捕捉表情相关的外观变化（皱纹、自阴影），是蒸馏的质量上限。
    - 设计动机：UV 空间的组织方式确保了帧间 Gaussian 的对应关系（同一 UV 位置的 Gaussian 语义相同），这是 PCA 操作的前提。

3. **基于 EMOCA 的图像驱动回归器（Image-based Regressor）**:

    - 功能：从单张 RGB 图像直接预测 GEM 系数，跳过传统 3DMM 追踪
    - 核心思路：利用预训练 EMOCA 网络的中间特征（表情特征 $f_{expr}$ 和形状特征 $f_{shape}$），去掉最后一层得到 $\mathbf{f} \in \mathbb{R}^{2 \times 1024}$。对训练帧的特征做 PCA 降维，取前 50 个分量得到系数 $\kappa$。再通过小型 MLP（3 层，256 神经元）映射为 GEM 系数：$\mathbf{k}=3 \cdot \sigma_k \cdot \tanh(\text{MLP}(\kappa))$，tanh 限制在 $[-3\sigma_k, 3\sigma_k]$ 范围内避免越界。使用相对特征（减去中性表情参考帧），实现跨人表情迁移。
    - 设计动机：GEM 不依赖 FLAME，因此不能用传统 3DMM 追踪来获取驱动参数。直接从图像回归系数跳过了追踪步骤，既快（实时）又避免了 3DMM 误差传播。

### 损失函数 / 训练策略
CNN 模型训练：$\mathcal{L}_{Color}=(1-\omega)\mathcal{L}_1+\omega\mathcal{L}_{D-SSIM}+\zeta\mathcal{L}_{VGG}$。GEM 基向量精炼使用相同损失。特征基精炼 30K 步，每 1K 步做 QR 正交化。回归器训练使用 5 个正面相机的训练帧。

## 实验关键数据

### 主实验

| 设置 | 指标 | GEM | AG (CNN) | GA (无CNN) | INSTA (NeRF) |
|------|------|-----|----------|------------|------------|
| 新视角 PSNR↑ | dB | **33.55** | 32.42 | 31.32 | 27.78 |
| 新视角 LPIPS↓ | | **0.068** | 0.071 | 0.079 | 0.123 |
| 新表情+视角 PSNR↑ | dB | **32.68** | 29.01 | 28.31 | 27.92 |
| 新表情+视角 LPIPS↓ | | **0.068** | 0.081 | 0.082 | 0.115 |
| 跨人 FID↓ | | 0.429 | 0.409 | 0.559 | 0.530 |
| 渲染速度 FPS↑ | | **201.7** | 16.5 | 142.7 | 20.6 |

### 消融实验

| 配置 (#components × texture) | PSNR↑ | 模型大小 | FPS |
|------|-------|---------|-----|
| 10 comp × 128² | 31.81 | **7MB** | 238 |
| 30 comp × 128² | 34.20 | 20MB | 241 |
| 50 comp × 128² | 34.67 | 34MB | 239 |
| 50 comp × 256² | 34.61 | 138MB | **202** |
| 50 comp × 512² | **35.45** | 553MB | 117 |
| Ours CNN (256²) | 34.99 | 109MB | 36 |
| AG (256²) | 34.40 | 529MB | 17 |

### 关键发现
- **GEM 质量超越原始 CNN**：新表情+新视角上 GEM 32.68 vs CNN 29.01，因为 analysis-by-synthesis 直接优化系数更精确，避免了 3DMM 追踪误差
- **极致压缩**：10 个主成分仅 7MB，质量仍有 31.81dB，足以用于低端设备
- **速度碾压**：201 fps vs CNN 的 17-36 fps，因为推理只需一次点积运算
- **颜色固定是成功关键**：固定颜色防止 Gaussian 语义漂移，确保 PCA 成立
- 精炼步骤从 34.75→36.85 PSNR，QR 正交化保证基向量质量
- GEM 30K 步精炼未出现过拟合，测试集上也有提升

## 亮点与洞察
- **PCA 蒸馏的优雅简洁**：用最经典的线性代数工具（PCA + 线性组合）替代复杂 CNN，实现了质量和效率的双赢。体现了"复杂模型训练、简单模型部署"的思想
- **GEM 超越原始 CNN 的反直觉结果**：蒸馏后的学生模型质量居然高于教师模型，因为 analysis-by-synthesis 的精确优化避免了 3DMM 追踪的误差传播。这提示：限流瓶颈可能不在模型表达力而在驱动信号质量
- **可灵活调节质量-大小权衡**：10/30/50 个主成分对应 7/20/34 MB，不同硬件条件下均可部署。这种连续可调性是 CNN 模型不具备的
- **推理时完全无网络**：与 GASP 类似，推理只需线性代数运算，可在 CPU 上以 67fps 运行（i9-13900K），极具实用价值

## 局限与展望
- 依赖高质量多视角视频训练 CNN 教师模型，数据采集门槛高
- 固定拓扑的 UV 空间限制了对大型附属物（如大帽子）的建模
- 颜色作为全局参数不参与动画，光照变化时可能不自然
- PCA 是线性模型，极端表情的表达力可能不足（虽然实验中未观察到明显退化）
- 跨人 reenactment 的 FID 略逊于 CNN 方法（0.429 vs 0.409），说明回归器还有提升空间
- 未探索将方法扩展到全身头像

## 相关工作与启发
- **vs Gaussian Avatars (GA)**: GA 将 Gaussian 绑定到 FLAME mesh，无 CNN 但也无表情依赖的外观变化，缺少皱纹细节；GEM 通过 PCA 基保留了 CNN 学到的动态外观，同时比 GA 还快
- **vs Animatable Gaussians (AG)**: AG 是 GEM 的教师模型，需 CNN 推理（17fps, 529MB）；GEM 蒸馏后 202fps, 34MB，且新表情质量更好
- **vs 3D Gaussian Blendshapes**: 也用线性插值控制 Gaussian，但依赖 3DMM 表情系数；GEM 是独立的特征基，推理时不需要 3DMM
- **与模型压缩方法的互补性**：Compact3D 等压缩技术可进一步应用于 GEM 减小存储

## 评分
- 新颖性: ⭐⭐⭐⭐ PCA 蒸馏 Gaussian Avatar 的思路直觉但执行精巧，按模态分离和颜色固定的设计有insight
- 实验充分度: ⭐⭐⭐⭐⭐ 新视角/新表情/跨人三种评估、完整压缩消融、速度分析、实时demo
- 写作质量: ⭐⭐⭐⭐ 结构清晰，与 3DMM 的类比恰当，易于理解
- 价值: ⭐⭐⭐⭐⭐ 极致轻量化的高质量头像方案，7MB+200fps 对 AR/VR 应用价值极大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads](../../ICCV2025/3d_vision/vit-split_unleashing_the_power_of_vision_foundation_models_via_efficient_splitti.md)
- [\[CVPR 2025\] Feat2GS: Probing Visual Foundation Models with Gaussian Splatting](feat2gs_probing_visual_foundation_models_with_gaussian_splatting.md)
- [\[ECCV 2024\] Sapiens: Foundation for Human Vision Models](../../ECCV2024/3d_vision/sapiens_foundation_for_human_vision_models.md)
- [\[CVPR 2025\] UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting](unipre3d_unified_pre-training_of_3d_point_cloud_models_with_cross-modal_gaussian.md)
- [\[CVPR 2025\] PromptHMR: Promptable Human Mesh Recovery](prompthmr_promptable_human_mesh_recovery.md)

</div>

<!-- RELATED:END -->
