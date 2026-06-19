---
title: >-
  [论文解读] Ordinal Label-Distribution Learning with Constrained Asymmetric Priors for Imbalanced Retinal Grading
description: >-
  [NeurIPS 2025 Workshop (GenAI for Health)][医学图像][diabetic retinopathy] 提出 CAP-WAE（Constrained Asymmetric Prior Wasserstein Autoencoder），通过非对称先验、序数边距正交紧凑损失和方向感知序数损失三重创新，解决糖尿病视网膜病变分级中长尾分布和序数结构的挑战，在多个 DR 基准上达到 SOTA。
tags:
  - "NeurIPS 2025 Workshop (GenAI for Health)"
  - "医学图像"
  - "diabetic retinopathy"
  - "ordinal classification"
  - "Wasserstein autoencoder"
  - "label distribution"
  - "imbalanced learning"
---

# Ordinal Label-Distribution Learning with Constrained Asymmetric Priors for Imbalanced Retinal Grading

**会议**: NeurIPS 2025 Workshop (GenAI for Health)  
**arXiv**: [2509.26146](https://arxiv.org/abs/2509.26146)  
**代码**: 无  
**领域**: 医学影像 / 序数分类  
**关键词**: diabetic retinopathy, ordinal classification, Wasserstein autoencoder, label distribution, imbalanced learning

## 一句话总结
提出 CAP-WAE（Constrained Asymmetric Prior Wasserstein Autoencoder），通过非对称先验、序数边距正交紧凑损失和方向感知序数损失三重创新，解决糖尿病视网膜病变分级中长尾分布和序数结构的挑战，在多个 DR 基准上达到 SOTA。

## 研究背景与动机
**领域现状**：糖尿病视网膜病变（DR）分级是典型的序数分类任务（0-4 级），且数据严重长尾——重度 DR（3/4 级）样本稀少但临床最关键。

**现有痛点**：
   - 传统方法使用各向同性高斯先验，无法建模少数类的重尾/偏斜结构
   - 对称损失函数（如 CE）对"误低估"和"误高估"惩罚相同，不符合临床需求（误低估更危险）
   - 潜空间缺乏级别有序性，相邻级别重叠严重

**核心矛盾**：长尾分布 + 序数结构 + 非对称临床代价，三重困难交织。

**切入角度**：用 WAE 替代 VAE 避免后验塌陷，非对称先验匹配少数类分布特性。

**核心 idea**：非对称先验 WAE + 距边距正交紧凑损失 + 方向感知序数软标签，端到端解决 DR 长尾序数分级。

## 方法详解

### 整体框架
编码器 → 潜空间（受 MAOC 损失约束） → 解码器（WAE 重建） + 序数分级头（方向感知软标签）

### 关键设计

1. **Wasserstein Autoencoder with Asymmetric Prior**

    - 功能：用 WAE 对齐聚合后验 $Q_Z$ 与非对称先验 $P_Z$
    - 非对称先验：使用偏斜分布（如 skew-normal / log-normal）替代标准高斯
    - 优势：保留少数类的重尾结构，避免标准高斯先验将少数类"挤压"
    - WAE 目标：$\min_{E,D} \mathbb{E}[\|x - D(E(x))\|^2] + \lambda \cdot \text{MMD}(Q_Z, P_Z)$

2. **Margin-Aware Orthogonality and Compactness (MAOC) Loss**

    - 功能：在潜空间中强制级别有序且可分
    - 正交性：不同级别的潜向量均值彼此正交 $\langle \mu_i, \mu_j \rangle \to 0$
    - 紧凑性：同级别样本聚集 $\text{Var}(z | y=k) \to \text{small}$
    - 边距感知：相邻级别间保持 $\|\mu_k - \mu_{k+1}\| \geq m$
    - 公式：$\mathcal{L}_{MAOC} = \alpha \sum_{i \neq j} |\langle \mu_i, \mu_j \rangle| + \beta \sum_k \text{tr}(\Sigma_k) + \gamma \sum_k \max(0, m - \|\mu_k - \mu_{k+1}\|)$

3. **Direction-Aware Ordinal Loss**

    - 功能：生成反映临床优先级的软标签
    - 轻量级头预测非对称散度参数 $(\sigma_L^k, \sigma_R^k)$
    - 软标签：$\tilde{y}_j^k = \exp(-\frac{(j-k)^2}{2\sigma_{L/R}^{k,2}})$（左右散度不同）
    - 核心：$\sigma_L < \sigma_R$ 使得"误低估"的惩罚更重
    - KL 散度度量预测分布与软标签分布的差异

### 训练策略
- 自适应多任务加权（Adaptive MTL Weighting）平衡 WAE 重建、MAOC 和序数损失
- 端到端训练，无需单独预训练
- 数据增强：标准翻转/旋转 + Mixup

## 实验关键数据

### 主实验 — DR 分级基准

| 方法 | EyePACS QWK↑ | EyePACS Acc↑ | EyePACS F1↑ | APTOS QWK↑ |
|------|-------------|-------------|-------------|------------|
| ResNet-50 + CE | 0.812 | 78.5 | 52.3 | 0.845 |
| CORN (序数回归) | 0.836 | 80.1 | 55.7 | 0.862 |
| UniOrdinal | 0.849 | 81.3 | 57.2 | 0.871 |
| BalancedMix | 0.841 | 80.8 | 58.9 | 0.868 |
| VAE + Ordinal | 0.853 | 81.7 | 58.4 | 0.875 |
| **CAP-WAE (本文)** | **0.878** | **83.6** | **63.1** | **0.894** |

### 消融实验

| 配置 | EyePACS QWK↑ | Macro-F1↑ |
|------|-------------|-----------|
| Baseline (WAE + CE) | 0.838 | 54.8 |
| + Asymmetric Prior | 0.852 | 57.3 |
| + MAOC Loss | 0.861 | 59.6 |
| + Direction-Aware Ordinal | 0.871 | 61.8 |
| + Adaptive MTL Weighting | **0.878** | **63.1** |

### 各级别 F1（EyePACS）

| 级别 | Baseline F1 | CAP-WAE F1 | 提升 |
|------|-----------|-----------|------|
| Grade 0 (多数) | 89.2 | 90.1 | +0.9 |
| Grade 1 | 61.3 | 67.8 | +6.5 |
| Grade 2 | 52.7 | 61.3 | +8.6 |
| Grade 3 (少数) | 38.1 | 52.4 | +14.3 |
| Grade 4 (少数) | 32.8 | 43.9 | +11.1 |

### 关键发现
- CAP-WAE 在 QWK 上超越 SOTA 2.5-4.2 个百分点
- 少数类（Grade 3/4）F1 提升最显著（+11-14%），验证了非对称设计的有效性
- MAOC 损失对潜空间结构化贡献最大
- 方向感知损失使"误低估"错误减少约 23%
- t-SNE 可视化显示潜空间形成紧凑有序的级别簇

## 亮点与洞察
- **临床导向设计**：非对称惩罚（误低估更重）直接对应临床需求
- **WAE 优于 VAE**：避免后验塌陷，潜空间更可控
- **三重创新互补**：先验（分布层面）+ MAOC（表示层面）+ 序数损失（监督层面）

## 局限与展望
- 仅在 DR 分级上验证，其他序数任务（如年龄估计、疼痛评级）有待扩展
- NeurIPS Workshop 论文，篇幅受限，消融实验可更深入
- 非对称先验的具体分布族选择（skew-normal vs log-normal）缺乏比较
- 多中心泛化性未验证

## 相关工作与启发
- **CORAL (CVPR 2020)**：序数回归经典方法
- **Label Distribution Learning (Geng 2016)**：LDL 基础理论
- **WAE (Tolstikhin et al. 2018)**：Wasserstein 自编码器
- 启发：非对称先验思想可推广到其他长尾序数任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 非对称先验+方向感知序数损失新颖
- 实验充分度: ⭐⭐⭐⭐ 消融和逐级别分析完整
- 写作质量: ⭐⭐⭐⭐ 动机-方法-实验逻辑清晰
- 价值: ⭐⭐⭐⭐ 医学影像长尾序数分类有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Divide, Conquer, and Aggregate: Asymmetric Experts for Class-Imbalanced Semi-Supervised Medical Image Segmentation](../../CVPR2026/medical_imaging/divide_conquer_and_aggregate_asymmetric_experts_for_class-imbalanced_semi-superv.md)
- [\[CVPR 2026\] KLIP: localized distribution shift detection via KL-divergence with diffusion priors in Inverse Problems](../../CVPR2026/medical_imaging/klip_localized_distribution_shift_detection_via_kl-divergence_with_diffusion_pri.md)
- [\[CVPR 2025\] CycleULM: A Unified Label-Free Deep Learning Framework for Ultrasound Localisation Microscopy](../../CVPR2025/medical_imaging/cycleulm_a_unified_label-free_deep_learning_framework_for_ultrasound_localisatio.md)
- [\[CVPR 2025\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](../../CVPR2025/medical_imaging/a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[CVPR 2025\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](../../CVPR2025/medical_imaging/semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)

</div>

<!-- RELATED:END -->
