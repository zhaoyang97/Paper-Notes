---
title: >-
  [论文解读] Vision Transformers for Cosmological Fields: Application to Weak Lensing Mass Maps
description: >-
  [NeurIPS 2025][物理/科学计算][Transformer] 首次将 Vision Transformers（ViT 和 Swin Transformer）应用于弱引力透镜收敛场的宇宙学参数（$\Omega_m$ 和 $S_8$）约束，通过模拟推断框架系统比较了注意力架构与 CNN 的性能。
tags:
  - "NeurIPS 2025"
  - "物理/科学计算"
  - "Transformer"
  - "弱引力透镜"
  - "宇宙学参数"
  - "模拟推断"
---

# Vision Transformers for Cosmological Fields: Application to Weak Lensing Mass Maps

**会议**: NeurIPS 2025  
**arXiv**: [2512.07125](https://arxiv.org/abs/2512.07125)  
**代码**: 无  
**领域**: 宇宙学, 深度学习, 弱引力透镜  
**关键词**: Vision Transformer, 弱引力透镜, 宇宙学参数, Swin Transformer, 模拟推断

## 一句话总结
首次将 Vision Transformers（ViT 和 Swin Transformer）应用于弱引力透镜收敛场的宇宙学参数（$\Omega_m$ 和 $S_8$）约束，通过模拟推断框架系统比较了注意力架构与 CNN 的性能。

## 研究背景与动机

### 领域现状

**领域现状**：弱引力透镜是宇宙结构形成的重要探针，小尺度非线性结构含有丰富的非高斯信息

### 现有痛点

**现有痛点**：传统二点统计仅捕获高斯特征，CNN 已被证明能提取非高斯信息用于宇宙学参数约束

### 核心矛盾

**核心矛盾**：Vision Transformers 在计算机视觉中取得突破，但在弱引力透镜领域尚未被系统评估

### 解决思路

**解决思路**：ViT 通过注意力机制直接捕获全局上下文，无需 CNN 的层级聚合，且具有更好的可解释性

## 方法详解

### 整体框架
- 模拟推断 (SBI) 框架：训练神经密度估计器 (NDE) 近似 $p(d|\theta)$
- 视觉模型作为特征压缩器：将 512×512 收敛场映射为低维数据向量
- NDE 集成：3 个 MAF + 3 个 MDN，筛选偏差超过 5% 的 NDE，合并其余后验采样

### 关键设计
1. **模拟数据**：

    - DarkGridV1 N-body 模拟套件生成收敛场
    - 采用 DES-Y3 红移分布的 4 个层析 bin
    - 三种测试设置：无噪声单通道、LSST-Y1 单通道、LSST-Y1 四通道（完整层析）
    - 共 13680 张 512×512 收敛场map

2. **模型架构覆盖**：

    - CNN 系列：Baseline CNN(500K)、ResNet-18/34/50/101(11M-44M)
    - ViT 系列：ViT-B(86M)、ViT-L(307M)、ViT-H(632M)
    - Swin 系列：Swin-T(29M)、Swin-S(50M)、Swin-B(88M)、Swin-L(197M)

3. **预训练策略**：

    - 使用解析模型快速生成与弱引力透镜场统计特性相似的合成数据
    - 在合成数据上预训练，然后在真实模拟数据上微调
    - 对 Transformer 架构的提升显著，对 CNN 影响较小

### 损失函数 / 训练策略
- L2 损失（RMSE）用于模型训练
- AdamW 优化器
- ReduceLROnPlateau 调度器（patience=10，factor=0.3）
- CNN 学习率 $10^{-3}$，ViT/Swin 学习率 $10^{-5}$
- 80:10:10 训练/验证/测试划分，early stopping patience=30

## 实验关键数据

### 主实验（模型性能概览）

| 模型 | 参数量 | 无噪声 RMSE (S8) | LSST-Y1 RMSE (S8) |
|------|------|-----------------|-------------------|
| Baseline CNN | 500K | 较好 | 较好 |
| ResNet-50 | 24M | 较好 | 较好 |
| ViT-B | 86M | 差 | 差 |
| ViT-L | 307M | 差 | 差 |
| Swin-T | 29M | 中等 | 接近 CNN |
| Swin-L | 197M | **最优** (无噪声) | 接近 CNN |

### 预训练消融

| 设置 | Swin (无预训练) | Swin (有预训练) | CNN (无预训练) |
|------|---------------|---------------|--------------|
| 25% 训练数据 | 显著下降 | 接近满数据性能 | 稳定 |
| 50% 训练数据 | 中等下降 | 接近满数据性能 | 稳定 |
| 100% 训练数据 | 基线 | 最优 | 基线 |

### 关键发现
- **在无噪声场景中**，Swin Transformer 初步显示出优于 CNN 的表现（更大的模型灵活性）
- **加入真实噪声后**，Swin 与 CNN 性能可比，Transformer 的优势消失
- Vanilla ViT 始终表现较差，可能因为在小数据集上训练效率低且难以捕获精细尺度特征
- 预训练对 Transformer 影响显著，对 CNN 影响极小
- 通过 TARP 覆盖率测试验证表明两类模型的后验估计均well-calibrated

## 亮点与洞察
- 首次系统性地将注意力架构应用于弱引力透镜 mass map 的宇宙学参数约束
- Swin Transformer 与 CNN 性能可比（Figure of Merit 相当），但在噪声条件下并无优势
- 预训练策略揭示了 Transformer 对训练数据量的依赖：预训练可有效弥补数据不足
- Transformer 的注意力权重可解释性是未来值得探索的方向

## 局限与展望
- 仅变化 $\Omega_m$ 和 $\sigma_8$ 两个参数，未考虑系统效应（如光度红移不确定性）
- 数据集规模小（13680 张），限制了 Transformer 的潜力发挥
- 未尝试 ViT/Swin 的掩码自编码预训练等更高级的方案
- 可探索 Transformer 注意力权重的物理可解释性（如哪些区域贡献最大信息）

## 相关工作与启发
- 预训练对 Transformer 的重要性在科学数据领域再次得到验证
- Swin 的窗口注意力在科学图像上表现更好，可能因为弱引力透镜信号本身具有局部+全局双重特性
- 对于信噪比有限的科学数据，Transformer 不一定优于 CNN，更强的归纳偏差（如平移不变性）可能更有用
- NDE 集成（MAF+MDN）+ 覆盖率验证（TARP）的推断框架值得借鉴
- 合成数据预训练策略对天文学中数据稀缺问题的实际价值

## 评分
- 新颖性：⭐⭐⭐ （应用层面新颖，方法较标准）
- 技术贡献：⭐⭐⭐ （系统比较有参考价值）
- 实验充分度：⭐⭐⭐⭐ （12 种模型 × 3 种设置 + SBI 验证）
- 写作质量：⭐⭐⭐⭐ （结构清晰，结论客观）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Neural Deprojection of Galaxy Stellar Mass Profiles](neural_deprojection_of_galaxy_stellar_mass_profiles.md)
- [\[NeurIPS 2025\] Quantum Doubly Stochastic Transformers](quantum_doubly_stochastic_transformers.md)
- [\[CVPR 2025\] Accurate Differential Operators for Hybrid Neural Fields](../../CVPR2025/physics/accurate_differential_operators_for_hybrid_neural_fields.md)
- [\[NeurIPS 2025\] AstroCo: Self-Supervised Conformer-Style Transformers for Light-Curve Embeddings](astroco_self-supervised_conformer-style_transformers_for_light-curve_embeddings.md)
- [\[ICML 2026\] Interpretable Equivariant Marks for Contrastive Cosmological Inference](../../ICML2026/physics/interpretable_neural_marked_statistics_for_cosmological_inference.md)

</div>

<!-- RELATED:END -->
