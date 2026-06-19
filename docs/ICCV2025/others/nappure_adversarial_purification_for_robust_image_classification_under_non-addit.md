---
title: >-
  [论文解读] NAPPure: Adversarial Purification for Robust Image Classification under Non-Additive Perturbations
description: >-
  [ICCV 2025][对抗纯化] 提出 NAPPure 框架，通过联合优化底层干净图像和扰动参数（基于似然最大化），将对抗纯化从仅处理加性扰动扩展到模糊、遮挡、几何扭曲等非加性扰动，在GTSRB上实现73.93%的平均鲁棒准确率（传统方法仅43.2%）。 加性 vs 非加性扰动：- 现有对抗攻击和防御主要关注加性扰动：（$…
tags:
  - "ICCV 2025"
  - "对抗纯化"
  - "非加性扰动"
  - "扩散模型"
  - "鲁棒分类"
  - "模糊攻击"
  - "遮挡攻击"
  - "几何扭曲"
---

# NAPPure: Adversarial Purification for Robust Image Classification under Non-Additive Perturbations

**会议**: ICCV 2025  
**代码**: 无  
**领域**: 其他  
**关键词**: 对抗纯化, 非加性扰动, 扩散模型, 鲁棒分类, 模糊攻击, 遮挡攻击, 几何扭曲

## 一句话总结

提出 NAPPure 框架，通过联合优化底层干净图像和扰动参数（基于似然最大化），将对抗纯化从仅处理加性扰动扩展到模糊、遮挡、几何扭曲等非加性扰动，在GTSRB上实现73.93%的平均鲁棒准确率（传统方法仅43.2%）。

## 研究背景与动机

**加性 vs 非加性扰动：**

- 现有对抗攻击和防御主要关注**加性扰动**（$x_{adv} = x + \epsilon$），即在图像上直接加噪声
- 但现实世界中**非加性扰动**同样常见且危险：模糊（贴光学膜）、遮挡（贴贴纸/补丁）、几何扭曲（变形）
- 这些非加性扰动也已被证明能有效欺骗分类器

**现有对抗纯化方法的失败：**

- DiffPure和LM等方法本质上假设扰动是加性的（即干净图像在被扰动图像附近）
- 对于非加性扰动，被扰动图像与原始图像的 $l_2$ 距离可能很大，导致纯化过程中的**语义漂移**——模型可能将图像修复为其他类别的内容
- 例如模糊攻击后，DiffPure无法恢复清晰的边缘和纹理

**核心洞察：** 如果已知扰动的类型（如模糊/遮挡/扭曲），可以将扰动建模为参数化变换 $x_{adv} = f(x, \epsilon)$，然后通过最大化似然来分离底层干净图像和扰动参数。

## 方法详解

### 整体框架

NAPPure 通过联合优化干净图像 $x$ 和扰动参数 $\epsilon$ 来净化对抗图像，优化目标基于贝叶斯分解：

$$\log p(x, \epsilon | x_{adv}) \propto \log p(x) + \log p(\epsilon) + \log p(x_{adv} | x, \epsilon)$$

### 三个优化项

**1. 图像似然项 $\log p(x)$**：
使用EDM扩散模型估计数据分布，通过ELBO近似：

$$-\mathbb{E}_{\sigma, n} [\lambda(\sigma) \| D_\theta(x_\sigma; \sigma) - x \|_2^2]$$

推动图像向高概率密度区域移动，消除潜在扰动。

**2. 扰动先验项 $\log p(\epsilon)$**：
使用能量模型表示，约束扰动幅度不会过大：

$$\log p(\epsilon) = -\phi(\epsilon) - \log Z$$

势函数 $\phi(\epsilon)$ 在变换恒等元处取最小值（如加性扰动的恒等元为 $\epsilon_0 = 0$，卷积的恒等元为单位核）。

**3. 图像重建项 $\log p(x_{adv} | x, \epsilon)$**：
约束解必须在已知变换下有效，避免语义漂移：

$$\log p(x_{adv} | x, \epsilon) = -\frac{1}{2\sigma^2} \| x_{adv} - f(x, \epsilon) \|_2^2 + C_\sigma$$

### 最终优化目标

$$\min_{x, \epsilon} \mathbb{E}_{\sigma, n} \| D_\theta(x_\sigma; \sigma) - x \|_2^2 + \lambda_1 \cdot \phi(\epsilon) + \lambda_2 \cdot \| x_{adv} - f(x, \epsilon) \|_2^2$$

使用Adam优化器交替更新 $x$ 和 $\epsilon$，迭代 $T=500$ 步。

### 针对不同扰动类型的实例化

| 扰动类型 | 变换函数 $f$ | 恒等元 $\epsilon_0$ | 势函数 $\phi$ |
|----------|-------------|---------------------|---------------|
| 加性 | $x + \epsilon$ | $0$ | $\|\epsilon\|_2^2$ |
| 卷积/模糊 | $x * \epsilon$ | 单位核 | $\|\epsilon - \epsilon_0\|_2^2$ |
| 补丁/遮挡 | $x \cdot (1-m) + p \cdot m$ | $(x_{adv}, h/2, w/2, 0)$ | $|s|$ |
| 光流/扭曲 | 光流变换 | $0$ | $\|\epsilon\|_2^2$ |

### 复合变换处理（NAPPure-joint）

对于多种扰动同时存在的情况，引入可学习权重 $w \in [0,1]$，将每种基础变换替换为插值形式：

$$\hat{f}(x, \hat{\epsilon}) = w \cdot f(x, \epsilon) + (1-w) \cdot x$$

复合变换通过 $f = \hat{f}_n \circ \cdots \circ \hat{f}_1$ 构建。

### 退化到传统加性情况的理论证明

当 $f(x, \epsilon) = x + \epsilon$ 且 $p(\epsilon)$ 为均匀分布时，NAPPure退化为 $\max_x \log p(x)$，恰好等同于标准对抗纯化方法LM，证明NAPPure是传统方法的兼容扩展。

## 实验关键数据

### 主实验：GTSRB数据集上的鲁棒准确率

| 防御方法 | 卷积攻击 | 补丁攻击 | 光流攻击 | 加性攻击 | **平均** |
|----------|---------|---------|---------|---------|--------|
| 无防御 | 57.42 | 13.67 | 1.56 | 3.12 | 18.95 |
| AT | 61.72 | 19.92 | 19.72 | 47.85 | 37.30 |
| DiffPure | 61.52 | 46.29 | 21.88 | 60.74 | 47.61 |
| LM | 53.32 | 13.67 | 8.79 | 79.07 | 38.71 |
| **NAPPure** | **86.91** | **74.22** | **51.37** | **83.20** | **73.93** |
| NAPPure-joint | 76.17 | 57.23 | 37.37 | 66.40 | 59.29 |

- NAPPure在所有非加性攻击类型上**大幅超越**所有基线（平均+26.3% vs DiffPure）
- 在加性攻击上也保持有竞争力的性能（83.20%）
- NAPPure-joint（不知道具体攻击类型）仍然明显优于基线方法

### CIFAR-10数据集上的鲁棒准确率

| 防御方法 | 卷积攻击 | 补丁攻击 | 光流攻击 | 加性攻击 | **平均** |
|----------|---------|---------|---------|---------|--------|
| DiffPure | 59.38 | 69.73 | 23.06 | 79.10 | 57.82 |
| LM | 60.16 | 36.13 | 13.09 | 70.12 | 44.88 |
| **NAPPure** | **66.40** | **76.75** | **48.24** | **82.81** | **66.94** |

- CIFAR-10上同样验证了NAPPure的有效性
- 光流攻击的提升最为显著（48.24% vs DiffPure的23.06%）

### 复合攻击实验（GTSRB）

| 防御方法 | 鲁棒准确率 |
|----------|----------|
| 无防御 | 12.70 |
| DiffPure | 30.00 |
| LM | 15.82 |
| NAPPure | 37.10 |
| **NAPPure-joint** | **54.49** |

- 在最困难的4种扰动同时存在的复合攻击下，NAPPure-joint达到54.49%
- NAPPure-joint在复合攻击下反而优于NAPPure，验证了插值技术的有效性

### 关键发现

1. 传统对抗纯化方法在非加性扰动下性能急剧下降（DiffPure在GTSRB光流攻击下仅21.88%）
2. $\lambda_1=0.01, \lambda_2=5$ 是光流攻击下的最优超参数组合
3. $\lambda_1$ 过大会过度约束扰动，过小会引入新扰动；$\lambda_2$ 过大限制纯化灵活性，过小导致语义漂移
4. GTSRB上效果比CIFAR-10更显著，因为交通标志依赖清晰的形状边界

## 亮点与洞察

1. **问题定义有价值**：首次系统地研究对抗纯化在非加性扰动下的应用，填补了重要空白
2. **统一框架**：通过贝叶斯分解自然地统一了不同类型扰动的处理，且理论上兼容传统加性方法
3. **模块化设计**：不同扰动类型的处理模块可即插即用，扩展性强
4. **退化定理**：证明NAPPure在加性情况下退化为LM，增强了理论完整性
5. **复合攻击的处理方案**：NAPPure-joint提供了攻击类型未知时的实用防御

## 局限性

1. **假设已知扰动类型**：NAPPure需要预先知道扰动的类型，实际中可能不可用
2. **计算开销大**：500步迭代优化比前向推理慢很多，实时性受限
3. **超参数敏感**：$\lambda_1$ 和 $\lambda_2$ 需要针对不同攻击类型调整
4. **仅在32×32图像上验证**：是否能扩展到高分辨率图像尚未探索
5. **补丁攻击处理**：需要额外训练辅助模型来近似不可微的变换

## 相关工作与启发

- **DiffPure / LM**：标准对抗纯化方法，仅适用于加性扰动
- **对抗训练**：可自然扩展到非加性扰动，但对未见攻击泛化能力差
- **图像恢复**：去模糊/修复等任务相关但未考虑对抗场景
- **启发**：将扰动参数显式建模的思路可能适用于其他需要逆变换的任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将对抗纯化扩展到非加性扰动的统一框架，贡献清晰
- **实验**: ⭐⭐⭐⭐ — 两个数据集、四种攻击类型、复合攻击，消融充分
- **写作**: ⭐⭐⭐⭐ — 数学推导严谨，理论与实践结合良好
- **价值**: ⭐⭐⭐⭐ — 解决了实际场景中非加性攻击防御的空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Active Generation for Image Classification](../../ECCV2024/others/active_generation_for_image_classification.md)
- [\[AAAI 2026\] Boosting Adversarial Transferability via Ensemble Non-Attention](../../AAAI2026/others/boosting_adversarial_transferability_via_ensemble_non-attention.md)
- [\[ACL 2025\] Battling against Tough Resister: Strategy Planning with Adversarial Game for Non-collaborative Dialogues](../../ACL2025/others/battling_against_tough_resister_strategy_planning_with_adversarial_game_for_non-.md)
- [\[ICCV 2025\] Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponents](adversarial_data_augmentation_for_single_domain_generalization_via_lyapunov_expo.md)
- [\[CVPR 2026\] Prototype-based Causal Intervention for Multi-Label Image Classification](../../CVPR2026/others/prototype-based_causal_intervention_for_multi-label_image_classification.md)

</div>

<!-- RELATED:END -->
