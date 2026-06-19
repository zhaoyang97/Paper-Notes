---
title: >-
  [论文解读] Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems
description: >-
  [CVPR 2025][图像生成][扩散模型] 证明 DDIM 确定性反向链是一个分区迭代函数系统（PIFS），由此推导出三个无需模型评估的可计算几何量（收缩阈值 $L_t^*$、膨胀函数 $f_t(\lambda)$、全局膨胀阈值 $\lambda^{**}$），并据此从理论上解释了四个现有的经验性设计选择（cosine offset、分辨率 logSNR shift、Min-SNR 加权、Align Your Steps）。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "扩散模型"
  - "分形几何"
  - "PIFS"
  - "DDIM"
  - "噪声调度"
  - "理论分析"
---

# Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems

**会议**: CVPR 2025  
**arXiv**: [2603.13069](https://arxiv.org/abs/2603.13069)  
**代码**: 无  
**领域**: 图像生成 / 扩散模型理论  
**关键词**: 扩散模型, 分形几何, PIFS, DDIM, 噪声调度, 理论分析

## 一句话总结

证明 DDIM 确定性反向链是一个分区迭代函数系统（PIFS），由此推导出三个无需模型评估的可计算几何量（收缩阈值 $L_t^*$、膨胀函数 $f_t(\lambda)$、全局膨胀阈值 $\lambda^{**}$），并据此从理论上解释了四个现有的经验性设计选择（cosine offset、分辨率 logSNR shift、Min-SNR 加权、Align Your Steps）。

## 研究背景与动机

**领域现状**：扩散模型的理论基础建立在 SDE/ODE 上，提供了分布收敛的全局保证。但连续视角将 score network 视为黑箱，无法解释两个核心现象：(a) 为什么去噪链在高噪声端组装全局空间上下文、在低噪声端合成局部细节？(b) 为什么 self-attention 如此有效？

**现有痛点**：许多扩散模型的设计选择仍是经验性的——cosine schedule 的 offset=0.008 为什么好？Min-SNR 加权为什么有效？缺乏统一的理论框架来理解和预测这些设计。

**核心矛盾**：理论优雅但缺乏结构性洞察——SDE 理论告诉你"收敛了"但不告诉你"怎么收敛的"。

**本文目标** 提供一个统一的设计语言来理解和优化扩散模型的调度、架构和训练目标。

**切入角度**：1988 年 Barnsley 提出自然图像具有局部自相似性，可用分区迭代函数系统（PIFS）压缩。本文发现 DDIM 反向链恰好也是 PIFS——每步去噪就是一次分区收缩映射。

**核心 idea**：扩散模型的去噪链就是一个 PIFS，其分形几何完全刻画了去噪动态的两阶段结构。

## 方法详解

### 整体框架

将 DDIM 单步算子 $\Phi_t(x) = \frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}} x + b_t \hat\varepsilon_\theta(x, t)$ 视为 PIFS 的一步。核心是分析其 Jacobian 的收缩/膨胀特性——对角块（patch 内部动力学）和跨 patch 块（attention 耦合）的交互。

### 关键设计

1. **收缩结构（Section 3）**：

    - 推导两个收缩条件：(EC) 欧几里得收缩和 (PC) 块最大范数收缩
    - 收缩阈值 $L_t^* = (\sqrt{\bar\alpha_{t-1}/\bar\alpha_t} - 1) / |b_t|$——仅由噪声调度决定，与数据/模型无关
    - Score-matching 训练是 Barnsley 拼贴误差最小化的扩散模型类比
    - L2-W1 桥接：训练损失控制 PIFS 不动点的 Wasserstein 距离

2. **两阶段结构（Section 4）**：

    - **Regime I（高噪声）**：diffuse attention 维持强跨 patch 耦合（$\delta_t^{cross}$ 大），学到的"方向抑制场" $S_{k,t}$ 将对角块保持在膨胀阈值以下 → 全局上下文组装
    - **Regime II（低噪声）**：attention 局部化，抑制按 variance 顺序逐 patch 释放 → 局部细节合成
    - Self-attention 为什么有效：它精确控制了 $\delta_t^{cross}$（通过 softmax 权重的上界），是 PIFS 收缩的自然原语
    - 两阶段转换与 Raya & Ambrogioni (2023) 报告的自发对称性破缺一致

3. **吸引子几何（Section 5）**：

    - PIFS 吸引子的 Kaplan-Yorke 维度由离散 Moran 方程决定：$\prod_t f_t(\lambda^{**}) = 1$
    - 一个 patch 方向对样本多样性有贡献 ⟺ 其 leading variance 超过 $\lambda^{**}$

4. **三个设计准则（Section 6）**：

    - **准则 1**：最大化最弱环节收缩阈值 $\min_t L_t^*$（尽早注入噪声，抬高 $v_1$）
    - **准则 2**：均衡每步 Lyapunov 贡献 = 最小化 $\text{Var}_t(\Delta d_t)$ ≈ 信息常数准则
    - **准则 3**：平衡采样步骤工作负载——将步数集中在 $L_t^*$ 最小的地方

### 四个经验设计的理论解释

| 经验设计 | 对应准则 | PIFS 解释 |
|---------|---------|----------|
| Cosine offset $s_{off}=0.008$ | 准则 1 | 将 $L_1^*$ 从 $7.9 \times 10^{-4}$ 提高到 $3.2 \times 10^{-3}$（4x），增强最弱步的收缩余量 |
| 分辨率 logSNR shift | 准则 1 前提 | 调度必须覆盖细节 patch 转换的 logSNR 范围 |
| Min-SNR 加权 | 准则 2 | 均衡每步信息增益，等价于均衡 KY 维度增长 |
| Align Your Steps | 准则 3 | 将采样步数集中在几何贡献最大的位置 |

## 实验关键数据

### 调度对比

| 调度 | 步数 | 平均 $L_t^*$ | CV($L_t^*$) | 最细步 $L_t^*$ |
|------|------|-------------|-------------|---------------|
| Linear (DDPM) | 1000 | 0.805 | **0.341** | 0.00500 |
| Cosine ($s_{off}=0$) | 1000 | 0.637 | 0.483 | 0.00079 |
| Cosine ($s_{off}=0.008$) | 1000 | 0.641 | 0.474 | **0.00321** |
| 50-step DDIM | 50 | 0.637 | 0.483 | 0.01571 |

### 信息增益均衡

| 调度 | CV(IG_t) | CV(|Δd_t|) | Spearman ρ(IG, Δd) |
|------|---------|-----------|-------------------|
| Linear | 1.107 | 0.836 | 0.9999 |
| Cosine | **0.867** | **0.570** | 0.9998 |

### 关键发现

- **$L_t^*$ 在 $t=1$（最细步）处最小**：$L_t^* \approx \frac{1}{2}\sqrt{v_t}$，细节合成是最受约束的阶段
- **CIFAR-10 所有 8×8 patch 在整个 1000 步链中都是膨胀强迫的**：leading eigenvalue 远超 $\lambda^*(t) \approx 1.002$
- **IG 与 KY 维度增长近乎完美成正比**：Spearman ρ > 0.999，验证了理论 CS 不等式的紧性
- **Linear 调度 $L_t^*$ 均衡好但 IG 均衡差；Cosine 反之**：不存在两项都最优的调度

## 亮点与洞察

- **将 1988 年的分形图像压缩与 2020 年的扩散模型统一**——深层次的数学联系：Barnsley 的自相似结构驱动了扩散模型的成功。Score-matching 就是拼贴误差最小化，不是类比而是数学恒等。
- **三个无需模型评估的几何量构成"设计语言"**：$L_t^*$、$f_t(\lambda)$、$\lambda^{**}$ 完全由调度和数据协方差谱决定。在训练任何模型之前就能预测调度的优劣。
- **两阶段的 PIFS 解释极其清晰**：Regime I 的"抑制场"保持收缩→全局结构组装，Regime II 的抑制释放→细节涌现。这不是事后描述而是数学推导的必然结果。
- **Self-attention 的必要性有了结构证明**：它控制跨 patch 耦合 $\delta_t^{cross}$，是 PIFS 需要的收缩原语。

## 局限与展望

- **只分析 DDIM（确定性采样）**：DDPM（随机采样）的 PIFS 结构仍是开放问题
- **实验主要基于 CIFAR-10 的分析**：未在高分辨率数据（ImageNet 256/512）上验证
- **Gaussian 假设**：attractor 维度分析依赖块对角 Gaussian 假设
- **PIFS 正则化器的实际训练效果未充分验证**：理论上可以拓宽收缩余量，但何时值得额外计算成本未清楚

## 相关工作与启发

- **vs Raya & Ambrogioni (2023)**：他们发现了两阶段行为和对称性破缺，本文给出了精确的几何机制解释。
- **vs Kingma et al. (2021) 信息常数准则**：本文从完全不同的角度（KY维度增长均衡）推导出了等价的结论，提供了更深层的理解。
- **启发**：这种"从已有经验设计反推理论最优解"的研究方式值得学习——不是提出新方法，而是理解为什么已有方法有效。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 扩散模型-分形几何的统一是全新视角，数学贡献深刻
- 实验充分度: ⭐⭐⭐ 理论为主，实证验证集中在 CIFAR-10 分析
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，但对非理论读者门槛高
- 价值: ⭐⭐⭐⭐⭐ 为扩散模型提供了统一的理论设计语言，解释了多个经验设计

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Optimizing for the Shortest Path in Denoising Diffusion Model](optimizing_for_the_shortest_path_in_denoising_diffusion_model.md)
- [\[ACL 2025\] Planning with Diffusion Models for Target-Oriented Dialogue Systems](../../ACL2025/image_generation/difftod_diffusion_dialogue_planning.md)
- [\[CVPR 2025\] AniDoc: Animation Creation Made Easier](anidoc_animation_creation_made_easier.md)
- [\[NeurIPS 2025\] AccuQuant: Simulating Multiple Denoising Steps for Quantizing Diffusion Models](../../NeurIPS2025/image_generation/accuquant_simulating_multiple_denoising_steps_for_quantizing.md)
- [\[CVPR 2026\] Smoothing the Score Function to Enhance Generalization in Diffusion Models](../../CVPR2026/image_generation/smoothing_the_score_function_to_enhance_generalization_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
