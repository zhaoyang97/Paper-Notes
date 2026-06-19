---
title: >-
  [论文解读] VIGFace: Virtual Identity Generation for Privacy-Free Face Recognition Dataset
description: >-
  [ICCV 2025][图像生成][虚拟身份生成] 提出 VIGFace 框架，通过在人脸识别模型的特征空间中预先分配与真实身份正交的虚拟原型（virtual prototypes），训练扩散模型从虚拟原型生成不存在于真实世界的人脸图像，实现隐私无忧的人脸识别数据集构建和数据增强。 深度学习人脸识别模型的训练依赖于大规模人脸…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "虚拟身份生成"
  - "隐私安全"
  - "合成人脸数据集"
  - "扩散模型"
  - "人脸识别"
---

# VIGFace: Virtual Identity Generation for Privacy-Free Face Recognition Dataset

**会议**: ICCV 2025  
**arXiv**: [2403.08277](https://arxiv.org/abs/2403.08277)  
**代码**: [GitHub](https://github.com/kim1102/VIGFace)  
**领域**: 扩散模型/人脸识别  
**关键词**: 虚拟身份生成, 隐私安全, 合成人脸数据集, 扩散模型, 人脸识别

## 一句话总结

提出 VIGFace 框架，通过在人脸识别模型的特征空间中预先分配与真实身份正交的虚拟原型（virtual prototypes），训练扩散模型从虚拟原型生成不存在于真实世界的人脸图像，实现隐私无忧的人脸识别数据集构建和数据增强。

## 研究背景与动机

深度学习人脸识别模型的训练依赖于大规模人脸数据集，但这些数据集面临严峻的隐私和伦理挑战：

**隐私权问题**：现有数据集（如 CASIA-WebFace、MS-Celeb-1M）通过网络爬取收集，未经当事人同意。包含儿童面部图像的数据集（如 VGGFace2）已因隐私原因被撤回。

**现有合成方法的三个不足**：理想的合成人脸数据需同时满足：(a) 数据分布与真实数据一致，(b) 生成的身份与真实人不重叠，(c) 同一身份内保持一致性。但现有方法均无法同时满足三点——SynFace 仅能生成不到 500 个不同身份；DigiFace 的 3D 渲染风格与真实图像差距大；DCFace 缺乏数据增强能力；IDiffFace 无法保证身份唯一性。

**身份泄露风险**：部分 SOTA 方法（CemiFace、HSFace）存在身份泄露问题——从 WebFace4M 数据集采样身份嵌入，生成的人脸与训练集中的真实人高度相似，从隐私角度看是重大缺陷。

**类内多样性不足**：真实数据集常存在长尾分布，部分身份的图像数量少、变化度低，导致模型泛化能力受限。

本文的核心思路是：如果能在特征空间中预先规划好虚拟身份的位置（与所有真实身份正交），那么从这些位置生成的人脸图像就天然地不会与任何真实人重叠。

## 方法详解

### 整体框架

VIGFace 包含两个阶段：
1. **阶段 1：FR 模型训练 + 虚拟原型分配**：在真实数据上训练 FR 模型，同时学习虚拟身份原型
2. **阶段 2：扩散模型人脸生成**：基于预训练 FR 模型的特征空间，训练条件扩散模型生成人脸

### 关键设计

1. **虚拟原型的正交分配（核心创新）**：在标准 ArcFace 训练中，仅有真实身份原型 $W_R = [w_r^1, ..., w_r^n]$。本方法额外引入 $k$ 个虚拟原型 $W_V = [w_v^1, ..., w_v^k]$，扩展原型矩阵为 $W \in \mathbb{R}^{(n+k) \times D}$。关键问题在于：虚拟身份没有对应的真实图像，原型无法通过 ArcFace 损失自然更新。解决方案是生成虚拟嵌入（模拟真实嵌入的分布）来更新虚拟原型：

    $f'_{FR}(x_j) = w_v^j + \mathcal{N}(0, 1) \cdot \sigma$
    $\sigma^2 = \frac{1}{b} \sum_{i=1}^{b} (f_{FR}(x_i) - w_r^i)^2$

   虚拟嵌入的标准差 $\sigma$ 与真实嵌入的分布匹配（通过 EMA 平滑），然后将虚拟嵌入和真实嵌入一起送入 ArcFace 损失。这样虚拟原型在训练中被推离其他所有原型（真实 + 虚拟），最终在特征空间中形成正交分布。虚拟嵌入的梯度不回传到 backbone，仅更新虚拟原型。

2. **条件扩散人脸生成**：采用 DiT 架构，输入条件包括：FR 原型向量 $w_r$（身份条件）、五点人脸关键点图像 $y$（姿态条件）和时间步 $t$。模型预测速度 $v_t$ 而非噪声。关键约束是最小化生成图像与输入原型的特征距离：

    $\min_\theta \mathbb{E}_{\epsilon,t} \| f_{FR}(\hat{x}_\theta(x_t, t, w_r, y)) - w_r \|_2^2$

   使用 classifier-free guidance（10% 概率将 $w_r$ 置零），推理时引导权重 $g=4.0$ 效果最佳。五点关键点由 RetinaFace 提取，允许控制生成图像的姿态变化。

3. **数据集属性度量体系**：提出三项评估指标量化合成数据集质量：

    - **类一致性** $C_k$：类内图像特征余弦相似度的均值，衡量同身份图像的统一性
    - **类可分离性** $S_k$：类中心与负类中心的平均距离，衡量身份的唯一性
    - **类内多样性** $D_k$：基于 CR-FIQA 评分方差，衡量姿态/遮挡/光照等条件的丰富度

### 损失函数 / 训练策略

- **阶段 1**：ArcFace 损失 $L_{arc}$ 同时作用于真实嵌入和虚拟嵌入
- **阶段 2**：速度预测 MSE 损失 + 特征距离约束损失
- 虚拟嵌入数量 $b_v = (k \times b_r) / n$，确保虚拟和真实原型更新均衡
- EMA 平滑系数 $\alpha = 0.9$
- 训练数据集：CASIA-WebFace（约 0.49M 图像，~10.5K 身份）

## 实验关键数据

### 主实验

**纯合成数据训练的 FR 基准对比（IR-SE50 + AdaFace）**：

| 方法 | 训练数据来源 | 图像数 | LFW | CFP-FP | CPLFW | AgeDB | CALFW | 平均 |
|------|-----------|--------|-----|--------|-------|-------|-------|------|
| CASIA (真实) | - | 0.49M | 99.40 | 96.63 | 90.23 | 94.68 | 93.70 | 94.93 |
| SynFace | FFHQ | 0.5M | 91.93 | 75.03 | 70.43 | 61.63 | 74.73 | 74.75 |
| DCFace | FFHQ+CASIA | 0.5M | 98.55 | 85.33 | 82.62 | 89.70 | 91.60 | 89.56 |
| CemiFace | CASIA+WF4M | 0.5M | 99.03 | 91.06 | 87.62 | 91.33 | 92.42 | 92.30 |
| HSFace300K | WF4M | 15M | 99.30 | 91.54 | 87.70 | 94.45 | 94.58 | 93.52 |
| **VIGFace(S)** | **CASIA** | **0.5M** | **99.02** | **95.09** | **87.72** | 90.95 | 90.00 | **92.56** |
| **VIGFace(L)** | **CASIA** | **6.0M** | 99.33 | **97.31** | **91.12** | 93.82 | 92.95 | **94.91** |

### 消融实验

**数据增强效果（真实+合成数据组合）**：

| 条件 | 真实图像 | 合成(真实ID) | 合成(虚拟ID) | LFW | CFP-FP | 平均 | Δ |
|------|--------|------------|------------|-----|--------|------|---|
| CASIA | ✓ | | | 99.40 | 96.63 | 94.93 | - |
| +虚拟ID增强 | ✓ | | ✓ | 99.45 | 97.23 | 95.19 | +0.26 |
| +真实ID增强 | ✓ | ✓ | | 99.55 | 98.03 | 95.85 | +0.92 |
| +全部增强 | ✓ | ✓ | ✓ | **99.70** | **98.10** | **95.92** | **+0.99** |

**真实ID增强**将长尾类（<50张图像）扩充至 50 张，额外增加约 0.15M 图像。

### 关键发现

- VIGFace(L) 仅用 6M 图像就超越了使用 15M 图像的 HSFace300K（94.91 vs 93.52 平均准确率）
- 在 CFP-FP（跨姿态）和 CPLFW 基准上甚至超越真实数据训练的模型，得益于五点关键点带来的姿态多样性
- VIGFace(B) 的最近真实身份余弦相似度低于 CASIA-WebFace 自身的最近负类相似度，证明无身份泄露
- CemiFace 和 Vec2Face 存在身份泄露：生成的人脸与 WebFace4M 训练集中的真实人高度相似
- 数据增强效果：真实ID增强（+0.92）比虚拟ID增强（+0.26）效果更大，组合使用效果最佳（+0.99）

## 亮点与洞察

- **正交原型设计的数学优雅性**：利用高维空间中向量近似正交的性质，通过 ArcFace 损失自然地将虚拟原型推到远离真实身份的位置。不需要后处理阈值或选择性采样。
- **隐私安全的严格验证**：通过余弦相似度定量证明生成的虚拟人脸确实不存在于真实世界，而非仅凭视觉判断。同时揭示了 CemiFace/Vec2Face 的身份泄露问题。
- **同时具备替代和增强能力**：既能作为真实数据集的完全替代（隐私安全场景），又能与真实数据配合使用提升性能（数据增强场景）。
- **数据集质量评价体系**：提出的一致性/可分离性/多样性三维度评估框架可推广到其他合成数据集的质量评估。

## 局限与展望

- 虚拟身份数量受限于嵌入空间维度，当 $n+k$ 过大时正交性可能下降
- 五点关键点的姿态控制相对粗糙，可以探索更精细的 3DMM 或 landmark 条件
- 目前仅在 CASIA-WebFace 规模上验证，更大规模（如 WebFace4M）的扩展性需进一步探索
- 生成图像的分辨率和质量仍有提升空间
- 可探索将虚拟原型方法推广到其他需要身份保护的生物特征识别场景

## 相关工作与启发

- ArcFace 损失的角度间隔机制是虚拟原型正交化的基础
- DiT 架构在条件生成中的应用趋势
- DCFace 的双条件解耦（身份+风格）启发了本文的身份条件设计
- 本文的身份泄露分析为合成人脸数据集的评估设定了新标准

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 虚拟原型正交化的思路原创性强，同时解决了隐私和性能两个问题
- **实验充分度**: ⭐⭐⭐⭐⭐ 5 个验证基准、多种数据规模对比、数据增强实验、身份泄露分析均完备
- **写作质量**: ⭐⭐⭐⭐ 方法动机清晰，可视化丰富（t-SNE、相似度矩阵）
- **价值**: ⭐⭐⭐⭐⭐ 解决人脸识别领域长期存在的隐私困境，具有重大现实意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] OmniVTON: Training-Free Universal Virtual Try-On](omnivton_training-free_universal_virtual_try-on.md)
- [\[CVPR 2025\] GIF: Generative Inspiration for Face Recognition at Scale](../../CVPR2025/image_generation/gif_generative_inspiration_for_face_recognition_at_scale.md)
- [\[ICML 2025\] Synthetic Face Datasets Generation via Latent Space Exploration from Brownian Identity Diffusion](../../ICML2025/image_generation/synthetic_face_datasets_generation_via_latent_space_exploration_from_brownian_id.md)
- [\[ICCV 2025\] NullSwap: Proactive Identity Cloaking Against Deepfake Face Swapping](nullswap_proactive_identity_cloaking_against_deepfake_face_swapping.md)
- [\[ICCV 2025\] CaO2: Rectifying Inconsistencies in Diffusion-Based Dataset Distillation](cao2_rectifying_inconsistencies_in_diffusion-based_dataset_distillation.md)

</div>

<!-- RELATED:END -->
