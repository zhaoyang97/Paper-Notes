---
title: >-
  [论文解读] SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs
description: >-
  [AAAI 2026][物理/科学计算][神经算子] 提出 SAOT（Spectral Attention Operator Transformer），通过线性复杂度的小波注意力（WA）捕获高频局部细节，与傅里叶注意力（FA）的全局感受野经门控融合互补，在 6 个算子学习基准上取得 SOTA，Navier-Stokes 相对误差比 Transolver 下降 22.3%。
tags:
  - "AAAI 2026"
  - "物理/科学计算"
  - "神经算子"
  - "小波变换"
  - "傅里叶注意力"
  - "Transformer"
  - "偏微分方程"
---

# SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs

**会议**: AAAI 2026  
**arXiv**: [2511.18777](https://arxiv.org/abs/2511.18777)  
**代码**: [https://github.com/chenhong-zhou/SAOT](https://github.com/chenhong-zhou/SAOT)  
**作者**: Chenhong Zhou, Jie Chen, Zaifeng Yang
**领域**: 科学计算 / 算子学习  
**关键词**: 神经算子, 小波变换, 傅里叶注意力, 谱Transformer, PDE求解

## 一句话总结

提出 SAOT（Spectral Attention Operator Transformer），通过线性复杂度的小波注意力（WA）捕获高频局部细节，与傅里叶注意力（FA）的全局感受野经门控融合互补，在 6 个算子学习基准上取得 SOTA，Navier-Stokes 相对误差比 Transolver 下降 22.3%。

## 研究背景与动机

### 领域现状

PDE 求解是科学计算的基础任务。传统数值方法（有限元、有限差分）精度高但计算开销大，深度学习驱动的神经算子方法近年迅速崛起。FNO（Fourier Neural Operator）开创了频域学习范式：在傅里叶空间参数化积分算子核，等价于空间域的全局卷积，能高效捕获长程依赖。后续涌现出 U-FNO、Geo-FNO、F-FNO 等变体，以及 Transolver 等基于 Transformer 的新架构。

### 核心痛点：傅里叶方法的谱偏差

尽管傅里叶方法擅长全局特征，但存在根本性缺陷——**谱偏差（spectral bias）**。具体表现为：

**过度平滑**：全局卷积操作倾向于生成平滑解，高频成分被抑制。论文通过能量谱分析直观展示了这一点：FA 在低波数段与真实解吻合良好，但在高波数段能量急剧衰减，偏离真实分布。

**局部细节丢失**：傅里叶变换本质是全局操作，基函数在整个定义域上有支撑，缺乏空间局部性。但 PDE 解中常包含边界层、激波、涡结构等需要空间局部+高频信息才能准确捕捉的特征。

**现有补救方案效果有限**：MWT（多小波算子）和 WNO（小波神经算子）尝试用小波变换替代傅里叶变换，但在实际精度上未能达到预期，尤其在 Transformer 架构日益主流的背景下表现力不足。

### 本文切入角度

小波变换天然具有**时频局部化**特性：小波基函数在空间和频率上都是局部化的，既能保留频率信息又能保持空间位置信息。这恰好弥补了傅里叶变换的局部性缺陷。作者提出的核心思路是：**小波注意力捕获局部高频特征 + 傅里叶注意力捕获全局低频特征，二者通过门控机制自适应融合**。这在信号处理中是经典的互补思路，但在算子学习领域是首次被有效实现。

## 方法详解

### 整体架构

SAOT 采用标准的 Encoder-Processor-Decoder Transformer 架构：

- **Encoder**：线性层将输入函数从 $\mathbb{R}^{d_a}$ 提升到高维特征空间 $\mathbb{R}^D$
- **Processor**：$L$ 层 pre-norm Transformer 块堆叠，核心是将标准自注意力替换为提出的 **Spectral Attention (SA)**
- **Decoder**：线性层投影到输出维度 $\mathbb{R}^{d_u}$

每个 Transformer 块遵循：$\hat{X}^l = \text{SA}(\text{LN}(X^{l-1})) + X^{l-1}$，$X^l = \text{MLP}(\text{LN}(\hat{X}^l)) + \hat{X}^l$。

### 小波注意力（Wavelet Attention, WA）—— 核心创新

WA 的设计目标是以线性复杂度 $O(ND^2)$ 在小波域学习局部敏感特征。具体流程如下：

**Step 1 — 通道降维**：输入特征 $X \in \mathbb{R}^{H \times W \times D}$ 先通过卷积层投影为 $\bar{X} \in \mathbb{R}^{H \times W \times D/4}$。降维是为 FWT 展开后通道拼接做准备，确保最终维度不膨胀。

**Step 2 — 快速小波变换（FWT）分解**：使用 Haar 小波将 $\bar{X}$ 分解为 4 个子带。高通滤波器 $f_H = (1/\sqrt{2}, -1/\sqrt{2})$ 和低通滤波器 $f_L = (1/\sqrt{2}, 1/\sqrt{2})$ 先沿行方向再沿列方向分别作用，得到：

- $X_{LL}$：低频分量，捕获粗粒度基础特征
- $X_{LH}, X_{HL}, X_{HH}$：三个高频分量，保留不同方向的精细细节

每个子带尺寸为 $\mathbb{R}^{H/2 \times W/2 \times D/4}$，沿通道维度拼接得到 $\tilde{X} \in \mathbb{R}^{H/2 \times W/2 \times D}$。空间分辨率减半使后续注意力计算量降为原来的 1/4。

**Step 3 — 局部卷积增强**：可选地对 $\tilde{X}$ 施加 $3 \times 3$ 卷积，强化小波子带间的空间局部关联，生成局部上下文化特征 $X^w$。

**Step 4 — 线性化注意力**：将 $X^w$ reshape 为 $Q^w, K^w, V^w \in \mathbb{R}^{n \times D}$（$n = H/2 \times W/2$），使用核特征映射 $\phi(x) = \text{elu}(x) + 1$ 将标准注意力线性化。关键公式：

$$s'_i = \frac{\phi(q_i)^T \left(\sum_j \phi(k_j) \otimes v_j\right)}{\phi(q_i)^T \sum_l \phi(k_l)}$$

由于 $\sum_j \phi(k_j) \otimes v_j$ 和 $\sum_l \phi(k_l)$ 可预计算并跨所有 query 复用，复杂度从 $O(N^2)$ 降为 $O(N)$。

**Step 5 — 小波逆变换与输出合成**：线性注意力输出 reshape 回 $\mathbb{R}^{H/2 \times W/2 \times D}$，经 IFWT 重建为 $X^r \in \mathbb{R}^{H \times W \times D/4}$，再与原始输入 $X$ 拼接后通过线性层得到最终输出 $X^{WA}$。

**设计动机总结**：WA 的核心洞察是利用小波变换将输入分解为多个频率子带，在子带间做注意力计算——这让模型在紧凑的表示空间中同时学习低频全局模式和高频局部细节，而线性注意力确保了计算效率。

### 傅里叶注意力（Fourier Attention, FA）

FA 沿用 AFNO 的设计，在傅里叶空间近似积分算子核：

$$\mathcal{K}(X)(g) = \mathcal{F}^{-1}(R_\psi \cdot \mathcal{F}(X))(g)$$

其中 $R_\psi$ 用分块 MLP 层实现（而非 FNO 的可学习复值权重张量），大幅减少参数和显存。与 AFNO 的区别是**不做稀疏化/截断**，保留所有频率模式以维持完整表达力。复杂度 $O(ND\log N)$，输出加残差连接：$X^{FA} = X + X'$。FA 与 WA 并行运行。

### 门控融合块（Gated Fusion Block）

不同空间区域对全局信息和局部信息的需求不同（例如边界区域更依赖局部高频，平坦区域更依赖全局低频），因此设计自适应门控融合：

$$G = \sigma(\text{Linear}(\text{Concat}(X^{FA}, X^{WA})))$$
$$X^{SA} = G \odot X^{FA} + (1 - G) \odot X^{WA}$$

其中 $\sigma$ 是 sigmoid 函数，$G \in \mathbb{R}^{N \times D}$ 是逐元素门控权重。这种设计让网络根据数据自身特征决定每个位置、每个通道应更多依赖全局还是局部信息。

### 计算复杂度

SA 的复杂度为 $O(\max(ND^2, ND\log N))$，整个 SAOT（$L$ 层）为 $O(L \cdot \max(ND^2, ND\log N))$——对比标准自注意力的 $O(N^2D)$，在大规模网格上显著更高效。

### 训练细节

相对 $L^2$ 误差作为损失和评估指标，Adam 优化器，初始学习率 $10^{-3}$，训练 500 epochs，单卡 V100S-32GB。

## 实验关键数据

### 表1：主实验——6 个基准上的相对 $L^2$ 误差（↓越低越好）

| 模型 | Darcy | NS | Airfoil | Pipe | Plasticity | Elasticity |
|------|-------|------|---------|------|-----------|------------|
| FNO | 0.0108 | 0.1556 | - | - | - | - |
| Geo-FNO | 0.0108 | 0.1556 | 0.0138 | 0.0067 | 0.0074 | 0.0229 |
| MWT | 0.0067 | 0.1553 | 0.0076 | 0.0072 | 0.0027 | 0.0334 |
| WNO | 0.0242 | 0.1613 | 0.0188 | 0.0070 | - | 0.0465 |
| GNOT | 0.0105 | 0.1380 | 0.0076 | - | - | 0.0086 |
| IPOT | 0.0085 | 0.0885 | 0.0088 | - | 0.0033 | 0.0156 |
| Transolver | 0.0058 | 0.0985 | 0.0053 | 0.0043 | 0.0012 | 0.0067 |
| **SAOT** | **0.0049** | **0.0688** | **0.0048** | 0.0063 | **0.0008** | 0.0080 |
| vs Transolver | ↓15.5% | **↓22.3%** | ↓9.4% | ↑46.5% | **↓33.3%** | ↑19.4% |

SAOT 在 Darcy、NS、Airfoil、Plasticity 上均取得最优，NS 和 Plasticity 上改进最为显著。Pipe 和 Elasticity 上不如 Transolver。

### 表2：消融实验——注意力机制对比

| 注意力 | Darcy 参数量(M) | Darcy $L^2$ | Elasticity 参数量(M) | Elasticity $L^2$ |
|--------|----------------|------------|---------------------|-----------------|
| FA only | 0.651 | 0.0058 | 0.576 | 0.0232 |
| WA only | 2.361 | 0.0057 | 1.514 | 0.0129 |
| **SA (FA+WA)** | 2.694 | **0.0049** | 2.040 | **0.0080** |

- WA 单独即大幅优于 FA（Elasticity: 0.0129 vs 0.0232，↓44.4%），验证了高频局部信息的重要性
- SA 在二者基础上进一步提升——门控融合确实实现了互补增益
- WA 参数量约为 FA 的 3.6 倍，但带来了质的性能跃升

## 亮点与洞察

1. **能量谱分析作为诊断工具**：论文通过对比 FA 和 WA 预测的能量谱与真实解的差异，直观揭示了傅里叶方法的谱偏差根源。WA 在高波数段能量谱与真实解对齐更好。这种分析方法可推广用于诊断任何频域方法的频率捕获缺陷。

2. **小波+傅里叶互补的有效实现**：时频互补在信号处理中是经典思路，但本文首次在算子学习中将二者整合进 Transformer 架构并取得一致性提升。门控融合是关键——不是简单加权平均，而是让网络根据空间位置和通道自适应分配权重。

3. **零样本超分辨率泛化能力出色**：在 85² 分辨率训练的模型，在 43²~421² 范围内测试均保持最低误差，展现了强离散化不变性。所有方法的误差曲线呈 U 形（偏离训练分辨率越远误差越大），但 SAOT 的 U 形最浅。

4. **WA 远优于现有小波算子**：WA 的 Darcy 误差（0.0057）远低于 MWT（0.0067）和 WNO（0.0242），说明在 Transformer 框架内引入小波注意力比直接用小波构建算子层更有效。

## 局限性

1. **Pipe 和 Elasticity 上不如 Transolver**：Pipe 上误差高出 46.5%（0.0063 vs 0.0043），Elasticity 上高出 19.4%。Transolver 的物理感知 slice attention 在特定几何结构上可能更有针对性。

2. **仅使用 Haar 小波**：Haar 小波是最简单的小波基，在平滑度和逼近能力上不如 Daubechies、Coiflet 等高阶小波。论文也在结论中提到探索其他小波基变体是未来方向。

3. **参数量增长显著**：SA（2.694M）是 FA-only（0.651M）的 4.1 倍，效率-性能权衡需关注。在参数量受限的部署场景可能不够实用。

4. **仅在 2D PDE 上验证**：所有基准均为 2D 问题，更高维的 PDE（3D 流体、时空问题）上的有效性未经验证。

## 相关工作

- **vs Transolver**：Transolver 用物理启发的 slice attention 将网格点聚合到物理切片上做注意力，关注物理结构；SAOT 从频域互补角度设计注意力。二者在不同 PDE 类型上各有优势，互为补充。
- **vs MWT / WNO**：同属小波方法阵营，但 MWT 用多小波基做算子层，WNO 用小波积分层——都未与 Transformer 架构有效结合。SAOT 的 WA 将小波嵌入注意力机制，实现了质的飞跃。
- **vs AFNO / FNet / GFNet**：三种傅里叶谱 Transformer 仅在频域做 token mixing，缺乏局部性。SAOT 在 Elasticity 上误差仅 0.0080，远低于 GFNet（0.0230）和 AFNO（0.0228）。
- **vs FNO 及其变体**：FNO 系列做全局傅里叶卷积，SAOT 额外引入小波分支捕获局部细节，在所有规则网格基准上全面超越。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 小波注意力+傅里叶注意力互补在算子学习中是新颖且有效的组合，门控融合设计合理
- **实验充分度**: ⭐⭐⭐⭐⭐ — 6 个基准、11+ 个 baseline、详细消融、能量谱分析、超分辨率泛化测试、与谱 Transformer 的额外对比
- **写作质量**: ⭐⭐⭐⭐ — 动机链条清晰（谱偏差→小波局部性→互补融合），能量谱可视化直观有力
- **实用价值**: ⭐⭐⭐⭐ — 为算子学习的频域注意力设计提供了新方向，代码已开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Knowledge-Guided Masked Autoencoder with Linear Spectral Mixing and Spectral-Angle-Aware Reconstruction](knowledge-guided_masked_autoencoder_with_linear_spectral_mixing_and_spectral-ang.md)
- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](../../ICLR2026/physics/drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[AAAI 2026\] FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer](flashkat_understanding_and_addressing_performance_bottlenecks_in_the_kolmogorov-.md)
- [\[ICML 2026\] TINNs: Time-Induced Neural Networks for Solving Time-Dependent PDEs](../../ICML2026/physics/tinns_time-induced_neural_networks_for_solving_time-dependent_pdes.md)
- [\[ICML 2026\] Quiver: Quantum-Informed Views for Enhanced Representations in Large ML Models](../../ICML2026/physics/quiver_quantum-informed_views_for_enhanced_representations_in_large_ml_models.md)

</div>

<!-- RELATED:END -->
