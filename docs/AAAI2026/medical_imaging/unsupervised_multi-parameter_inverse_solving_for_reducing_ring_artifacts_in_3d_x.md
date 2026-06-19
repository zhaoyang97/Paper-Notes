---
title: >-
  [论文解读] Unsupervised Multi-Parameter Inverse Solving for Reducing Ring Artifacts in 3D X-Ray CBCT
description: >-
  [AAAI 2026][医学图像][CT环形伪影去除] 提出 Riner，将 CT 环形伪影去除（RAR）建模为基于物理的多参数逆问题，通过隐式神经表示（INR）联合学习无伪影图像和探测器物理参数，实现无监督且优于有监督 SOTA 方法的 3D CBCT 重建。 1. 领域现状：3D 锥束计算机断层扫描（CBCT）广泛应用于…
tags:
  - "AAAI 2026"
  - "医学图像"
  - "CT环形伪影去除"
  - "隐式神经表示"
  - "无监督学习"
  - "多参数逆问题"
  - "CBCT重建"
---

# Unsupervised Multi-Parameter Inverse Solving for Reducing Ring Artifacts in 3D X-Ray CBCT

**会议**: AAAI 2026  
**arXiv**: [2412.05853](https://arxiv.org/abs/2412.05853)  
**代码**: [https://github.com/iwuqing/Riner](https://github.com/iwuqing/Riner)  
**领域**: 医学影像  
**关键词**: CT环形伪影去除, 隐式神经表示, 无监督学习, 多参数逆问题, CBCT重建

## 一句话总结

提出 Riner，将 CT 环形伪影去除（RAR）建模为基于物理的多参数逆问题，通过隐式神经表示（INR）联合学习无伪影图像和探测器物理参数，实现无监督且优于有监督 SOTA 方法的 3D CBCT 重建。

## 研究背景与动机

1. **领域现状**：3D 锥束计算机断层扫描（CBCT）广泛应用于医学诊断、生物研究和材料科学。由于 X 射线探测器的非理想响应，重建图像中常出现严重的环形伪影（ring artifacts），显著降低图像质量和诊断可靠性。

2. **现有痛点**：当前 SOTA 的有监督深度学习方法（如 DeepRAR、Restormer）需要大规模配对数据集训练，但存在两个关键限制：(1) 数据收集昂贵且泛化能力差——模型通常在模拟数据上训练，在真实数据上性能大幅下降；(2) 可扩展性差——2D 模型难以直接扩展到 3D CBCT，逐切片处理会导致 Z 轴不连续。

3. **核心矛盾**：有监督方法将 RAR 视为端到端的后处理去噪任务，缺乏对伪影物理成因的建模，导致域外泛化能力不足。

4. **本文目标**：如何在不依赖外部训练数据的情况下，从原始 CT 测量数据中同时恢复高质量图像和估计探测器物理参数。

5. **切入角度**：从 X 射线 CT 物理出发，理论分析环形伪影的两个物理成因——不一致响应（IR）和无效测量（IM），将 RAR 重新建模为多参数逆问题。

6. **核心 idea**：用可微分的物理正向模型参数化探测器的非理想行为，通过 INR 的频谱偏置先验正则化病态逆问题，实现无需训练数据的 RAR。

## 方法详解

### 整体框架

Riner 采用基于射线的优化流水线：给定原始测量数据 $\widetilde{\rho}(\theta,s)$，MLP 网络 $f_\Phi$ 接收 X 射线路径上的空间坐标 $\mathbf{x}$ 作为输入，预测对应的 CT 强度 $\mu(\mathbf{x})$。预测的 CT 强度连同可学习的响应因子 $\alpha_s$ 和掩码 $\beta_s$，通过可微分物理正向模型生成估计测量值 $\widehat{\rho}(\theta,s)$，然后通过最小化损失函数联合优化所有参数。

### 关键设计

1. **环形伪影的理论建模**：
    - 功能：从 Lambert-Beer 定律出发，理论分析环形伪影的物理成因
    - 核心思路：非理想探测器（$\alpha_s \neq 1$）在测量数据中引入额外非线性项 $-\ln\alpha_s$（IR 效应）；缺陷探测器（$\alpha_s = 0$）产生无效测量（IM 效应）。这两种物理畸变使得从真实测量重建 CT 图像的逆问题本质上是非线性的
    - 设计动机：传统线性算法（如 FDK）假设理想探测器，无法处理这些非线性畸变

2. **可微分物理正向模型**：
    - 功能：将 MLP 预测的 CT 强度和物理参数转换为估计测量值，同时支持梯度反向传播
    - 核心思路：正向模型定义为 $\widehat{\rho}(\theta,s) = [-\ln\alpha_s + \sum_{\mathbf{x}\in L(\theta,s)}\mu(\mathbf{x})\cdot\Delta\mathbf{x}]\cdot\beta_s$，其中 $\alpha_s = \max(\alpha_s^0, \epsilon)$ 确保非负性，$\beta_s = \sigma(\beta_s^0)$ 作为二值掩码屏蔽缺陷探测器的无效信号
    - 设计动机：相比传统线性积分模型，引入额外物理参数 $(\alpha_s, \beta_s)$ 实现更精确的采集建模

3. **基于 INR 的图像参数化**：
    - 功能：用 MLP 网络表示连续的 CT 图像函数 $f_\Phi: \mathbf{x} \to \mu(\mathbf{x})$
    - 核心思路：利用 hash encoding + 2 层 FC 层的轻量 MLP，INR 固有的频谱偏置（先学低频全局结构，再恢复高频细节）作为正则化约束，缓解引入物理参数后逆问题的病态性
    - 设计动机：INR 的学习先验自然约束了解空间，无需外部数据即可实现高质量重建

### 损失函数 / 训练策略

损失函数包含两项：
$$\mathcal{L} = \underbrace{\sum_{L(\theta,s)\in\bar{\Pi}}\|\widehat{\rho}(\theta,s) - \widetilde{\rho}(\theta,s)\cdot\beta_s\|_1}_{\text{Data Consistency}} + \underbrace{\lambda\cdot\sum_{s\in\bar{S}}-\beta_s^2}_{\text{Negative }\ell_2}$$

- **数据一致性项**：L1 距离最小化预测与实际测量的差异
- **负 $\ell_2$ 正则化**：防止所有掩码收敛到零（即避免将所有探测器标记为缺陷），$\lambda=0.01$

训练策略：每步随机采样 80 条 X 射线（来自 2 个探测器、40 个投影视角），使用 Adam 优化器，学习率 $10^{-3}$，4000 次迭代。仅需约 3GB 显存（NVIDIA RTX 4090）。

## 实验关键数据

### 主实验

| 数据集 | 指标 | Riner (本文) | Restormer (之前SOTA) | 提升 |
|--------|------|------|----------|------|
| DeepLesion (2D) | PSNR | 39.02±2.18 | 37.31±1.92 | +1.71 dB |
| LIDC (2D) | PSNR | 39.11±2.17 | 36.69±1.90 | +2.42 dB |
| AAPM (3D) | PSNR | 36.53±1.28 | 33.94±1.50 | +2.59 dB |
| DeepLesion | SSIM | 0.967±0.019 | 0.947±0.028 | +0.020 |
| LIDC | SSIM | 0.962±0.030 | 0.925±0.051 | +0.037 |
| AAPM | SSIM | 0.949±0.020 | 0.907±0.031 | +0.042 |

### 消融实验

| 配置 | PSNR | 说明 |
|------|---------|------|
| w/o β, α (积分模型) | 35.34±2.26 | 无法处理 IR 和 IM 效应 |
| w/o β (仅 α) | 31.05±9.16 | 无法处理 IM 效应，方差极大 |
| w/o α (仅 β) | 35.72±1.93 | 无法处理 IR 效应 |
| 完整模型 | 38.98±1.41 | 两种效应都有效消除 |
| λ=0 (无正则化) | 31.63±1.77 | β 坍缩为 0，优化退化 |
| λ=0.01 (默认) | 38.98±1.41 | 最佳平衡 |
| λ=1 (强正则化) | 31.55±8.97 | β 收敛为 1，无法识别缺陷探测器 |

### 关键发现

- 无监督 Riner 首次在 RAR 任务上全面超越有监督方法，在 3 个模拟数据集上均取得最佳性能
- 在 3D CBCT 数据上优势更为明显（+2.59 dB），因为有监督方法在 3D 场景下性能显著下降
- 物理参数 $(\alpha, \beta)$ 的估计结果与 ground truth 高度一致
- 在真实世界 micro-CT 数据上也展示了优异的伪影去除效果

## 亮点与洞察

- **物理建模的威力**：将领域知识（CT 物理）显式融入无监督框架，消除了对大规模配对数据的依赖
- **内存友好**：ray-based 优化策略使方法天然适用于大规模 3D CBCT，每次迭代仅需前向/反向传播少量采样体素
- **理论-方法-实验一致性**：消融实验结果与理论建模对 IR/IM 效应的预期完全吻合
- INR 的频谱偏置作为"免费"正则化，优雅地解决了多参数逆问题的病态性

## 局限与展望

- 作为无监督方法需要逐案优化：2D 切片约 30 秒，3D 体积约 10 分钟（256×256×100）
- 可考虑用 3D Gaussian Splatting 等先进表示替代 INR 以提升效率
- 未讨论对不同伪影严重程度的鲁棒性
- 真实数据无法定量评估（缺少 ground truth）

## 相关工作与启发

- **vs DeepRAR/Restormer (有监督)**：有监督方法依赖模拟数据训练，域外泛化差；Riner 直接从原始测量推断，无需外部数据
- **vs NeRF/NAF 等 INR 方法**：现有 INR-CT 方法假设理想探测器，不处理非线性物理畸变；Riner 通过引入物理参数扩展了框架
- **vs Super (传统模型方法)**：Super 用手工设计算法处理多种伪影类型，但性能受限于先验不足；Riner 通过端到端优化实现更好重建

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个将 RAR 建模为多参数物理逆问题的无监督方法，理论分析扎实
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集（3模拟+2真实）、10+基线对比、多维度消融
- 写作质量: ⭐⭐⭐⭐⭐ 从理论分析到方法设计到实验验证逻辑清晰，公式推导严谨
- 价值: ⭐⭐⭐⭐⭐ 无监督超越有监督在 CT 领域具有重要意义，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Unsupervised Motion-Compensated Decomposition for Cardiac MRI Reconstruction via Neural Representation](unsupervised_motion-compensated_decomposition_for_cardiac_mri_reconstruction_via.md)
- [\[CVPR 2026\] Solving a Nonlinear Blind Inverse Problem for Tagged MRI with Physics and Deep Generative Priors](../../CVPR2026/medical_imaging/solving_a_nonlinear_blind_inverse_problem_for_tagged_mri_with_physics_and_deep_g.md)
- [\[CVPR 2026\] Efficient Unrolled Networks for Large-Scale 3D Inverse Problems](../../CVPR2026/medical_imaging/efficient_unrolled_networks_for_large-scale_3d_inverse_problems.md)
- [\[AAAI 2026\] Fine-Tuned LLMs Know They Don't Know: A Parameter-Efficient Approach to Recovering Honesty](fine-tuned_llms_know_they_dont_know_a_parameter-efficient_approach_to_recovering.md)
- [\[ECCV 2024\] Unsupervised Multi-modal Medical Image Registration via Invertible Translation](../../ECCV2024/medical_imaging/unsupervised_multi-modal_medical_image_registration_via_invertible_translation.md)

</div>

<!-- RELATED:END -->
