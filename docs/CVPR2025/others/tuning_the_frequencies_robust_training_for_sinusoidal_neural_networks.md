---
title: >-
  [论文解读] Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks
description: >-
  [CVPR 2025][隐式神经表示] 提出 TUNER，一种基于 Bessel 函数振幅-相位展开理论的正弦 MLP 训练方案，通过将隐藏神经元展开为输入频率整数组合的傅里叶级数实现鲁棒的频率初始化和训练中带限控制，显著提升隐式神经表示的收敛稳定性和重建质量。 领域现状：正弦 MLP（如 SIREN）因其光滑性和高表示能力…
tags:
  - "CVPR 2025"
  - "隐式神经表示"
  - "正弦网络"
  - "频率控制"
  - "带限"
  - "傅里叶级数"
---

# Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks

**会议**: CVPR 2025  
**arXiv**: [2407.21121](https://arxiv.org/abs/2407.21121)  
**代码**: 无  
**领域**: 信号与通信  
**关键词**: 隐式神经表示, 正弦网络, 频率控制, 带限, 傅里叶级数

## 一句话总结
提出 TUNER，一种基于 Bessel 函数振幅-相位展开理论的正弦 MLP 训练方案，通过将隐藏神经元展开为输入频率整数组合的傅里叶级数实现鲁棒的频率初始化和训练中带限控制，显著提升隐式神经表示的收敛稳定性和重建质量。

## 研究背景与动机

**领域现状**：正弦 MLP（如 SIREN）因其光滑性和高表示能力成为低维信号的主流隐式神经表示（INR）方法，广泛用于图像、音频、SDF、位移场等信号的编码。

**现有痛点**：正弦 MLP 的初始化和训练仍然是经验性的。SIREN 将输入频率随机初始化在一个范围内，可能引入不需要的高频导致过拟合和噪声重建。BACON 用乘法滤波网络（MFN）硬截断频谱来控制带限，但会导致振铃伪影，且 MFN 缺少非线性激活难以高效表示精细细节。

**核心矛盾**：正弦层的组合会产生大量新频率，但现有方法对这一频率生成过程缺乏理论理解——不清楚层组合如何生成频率、输入频率如何决定网络频谱、以及如何在训练中控制频率范围。

**本文目标**：建立正弦 MLP 的频率生成理论，基于此设计鲁棒的初始化方案和训练中带限控制机制。

**切入角度**：利用 Jacobi-Anger 恒等式的推广，将正弦层组合展开为类傅里叶级数，发现隐藏神经元可表示为输入频率整数组合的正弦之和，振幅由 Bessel 函数给出。

**核心 idea**：正弦 MLP 的频谱完全由输入频率 $\omega$ 决定（频率 = 输入频率的整数线性组合），振幅由隐藏权重通过 Bessel 函数控制——因此初始化输入频率等价于频谱采样，约束隐藏权重等价于带限控制。

## 方法详解

### 整体框架
TUNER 针对三层正弦 MLP $f(\mathbf{x}) = \mathbf{C} \circ \mathbf{S} \circ \mathbf{D}(\mathbf{x}) + e$，其中 $\mathbf{D}$ 是输入层（将坐标投影为正弦列表），$\mathbf{S}$ 是隐藏正弦层，$\mathbf{C}$ 是线性输出层。TUNER 包括两部分：(1) 基于傅里叶级数理论的输入频率初始化——在带限内选择合适的整数频率；(2) 基于 Bessel 函数振幅上界的隐藏权重约束——训练中裁剪隐藏权重以控制高频振幅。

### 关键设计

1. **振幅-相位展开（Theorem 1）**:

    - 功能：将隐藏神经元精确展开为输入频率整数组合的正弦级数
    - 核心思路：证明每个隐藏神经元 $h_i(\mathbf{x}) = \sin(\sum_j W_{ij} \sin(\omega_j \mathbf{x} + \varphi_j) + b_i)$ 可展开为 $h_i(\mathbf{x}) = \sum_{\mathbf{k} \in \mathbb{Z}^m} \alpha_\mathbf{k} \sin(\beta_\mathbf{k} \mathbf{x} + \lambda_\mathbf{k})$，其中频率 $\beta_\mathbf{k} = \langle \mathbf{k}, \omega \rangle$（输入频率的整数线性组合），振幅 $\alpha_\mathbf{k} = \prod_j J_{k_j}(W_{ij})$（Bessel 函数之积）
    - 设计动机：此展开首次严格解释了正弦层组合为何能大幅增加表示能力——$m$ 个输入频率可在截断阶 $B$ 下生成 $(2B+1)^m - 1)/2$ 个非零频率

2. **频谱采样初始化**:

    - 功能：初始化输入层频率使网络生成的频率覆盖目标信号的完整频谱
    - 核心思路：将输入频率限制为整数频率 $\omega_j \in \frac{2\pi}{p}\mathbb{Z}^d$（保证周期性），训练中冻结输入频率。采用混合采样策略：低频区域密集采样（因为信号能量主要集中在低频），高频区域稀疏采样（利用层组合生成填充）。这等价于对目标频谱进行有策略的采样
    - 设计动机：随机初始化可能产生过多高频导致过拟合，或遗漏关键频率导致重建失败。整数频率保证了傅里叶级数的正交性

3. **训练中带限控制（Theorem 2）**:

    - 功能：在训练过程中约束网络频谱在指定带限内
    - 核心思路：证明振幅上界 $|\alpha_\mathbf{k}| \leq \prod_j (|W_{ij}|/2)^{|k_j|} / |k_j|!$。当 $|W_{ij}| < 2$ 时，高阶 $k_j$ 对应的振幅指数衰减；$|W_{ij}|$ 越小，衰减越快。因此通过在训练中裁剪隐藏权重 $|W_{ij}| \leq c$（$c < 2$），可以有效抑制高频分量，实现软带限滤波
    - 设计动机：BACON 的硬截断会导致振铃伪影。基于 Bessel 函数的软衰减提供了更平滑的频谱控制，避免了吉布斯现象

### 损失函数 / 训练策略
使用标准 MSE 损失训练。输入频率冻结不训练，仅优化隐藏层权重 $W$、偏置 $b$ 和输出层 $C$。每次梯度更新后对 $W$ 进行裁剪以维持带限。使用 Adam 优化器，典型训练 3000 epochs。

## 实验关键数据

### 主实验

| 方法 | 数据集 | PSNR | 特点 |
|------|--------|------|------|
| SIREN | Kodak | 较低 | 均匀随机初始化，易过拟合出噪声 |
| FFM | Kodak | 中等 | 傅里叶特征映射 |
| BACON | Kodak | 中等 | 硬截断带限，有振铃伪影 |
| TUNER | Kodak | 最优 | 快速收敛，无噪声/伪影 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅奇数频率初始化 | 重建质量差 | 遗漏偶数频率导致周期倍增 |
| 加入 (1,0),(0,1) 频率 | 显著改善 | 基频保证完整频谱覆盖 |
| 均匀随机初始化 | 梯度有噪声 | 高频过拟合 |
| TUNER 初始化 | 梯度干净 | 无需梯度监督即可得到光滑梯度 |

### 关键发现
- 输入频率的选择对网络表示能力至关重要——缺少关键基频会导致无法学到完整频谱
- 隐藏权重裁剪提供的软带限比 BACON 的硬截断更优，避免了振铃伪影
- TUNER 初始化下的网络即使不对梯度做监督，重建的信号梯度也是干净的（无噪声），说明频率控制有效防止了过拟合
- 低频密集+高频稀疏的混合采样策略符合自然信号的能量分布规律

## 亮点与洞察
- **优美的理论框架**：将 Jacobi-Anger 恒等式推广到多变量多层情形，建立了正弦 MLP 与傅里叶级数的严格联系。这一理论贡献独立于实际应用本身就有价值
- **初始化即频谱采样**：将网络初始化问题转化为频域采样问题的视角非常优雅，提供了一个新的理解正弦网络的思维框架
- **Bessel 函数的实用上界**：Theorem 2 提供的振幅上界简洁实用（仅依赖权重绝对值），可直接转化为训练中的权重裁剪规则

## 局限与展望
- 理论分析主要针对三层网络，虽然可推广但深层网络的展开复杂度指数增长
- 输入频率在训练中冻结，限制了网络自适应调整频谱的能力
- 实验主要在 2D 图像重建上验证，未充分探索 3D 场景（NeRF、SDF）等高维应用
- 权重裁剪是启发式的软带限，不是精确的频谱截断，对极端带限要求的场景可能不够严格

## 相关工作与启发
- **vs SIREN**: SIREN 随机初始化输入频率且不控制训练中的频率增长，容易产生高频噪声。TUNER 通过整数频率初始化+权重裁剪从根本上解决了这一问题
- **vs BACON**: BACON 使用 MFN 做硬截断带限，优点是理论上精确截断，缺点是振铃伪影且无非线性激活。TUNER 保留了正弦激活的非线性表达力，用软衰减替代硬截断
- **vs FFM (Fourier Feature Mapping)**: FFM 用随机傅里叶特征做输入映射但不使用正弦激活。TUNER 的理论显示正弦激活的层组合本身就能生成丰富频率，提供了统一视角

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 理论贡献突出，振幅-相位展开和 Bessel 上界是全新的数学工具
- 实验充分度: ⭐⭐⭐ 图像重建实验充分，但缺少 3D/视频等更多应用场景
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，可视化直观，整体组织清晰
- 价值: ⭐⭐⭐⭐ 为正弦 INR 领域提供了首个完整的理论框架，具有深远的理论和实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] EVOS: Efficient Implicit Neural Training via EVOlutionary Selector](evos_efficient_implicit_neural_training_via_evolutionary_selector.md)
- [\[CVPR 2025\] Subnet-Aware Dynamic Supernet Training for Neural Architecture Search](subnet-aware_dynamic_supernet_training_for_neural_architecture_search.md)
- [\[ICLR 2026\] Breaking Gradient Temporal Collinearity for Robust Spiking Neural Networks](../../ICLR2026/others/breaking_gradient_temporal_collinearity_for_robust_spiking_neural_networks.md)
- [\[CVPR 2025\] Improving Transferable Targeted Attacks with Feature Tuning Mixup](improving_transferable_targeted_attacks_with_feature_tuning_mixup.md)
- [\[CVPR 2025\] VKDNW: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights](training-free_neural_architecture_search_through_variance_of_knowledge_of_deep_n.md)

</div>

<!-- RELATED:END -->
