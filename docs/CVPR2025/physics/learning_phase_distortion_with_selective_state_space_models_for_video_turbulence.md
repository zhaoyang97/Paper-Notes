---
title: >-
  [论文解读] Learning Phase Distortion with Selective State Space Models for Video Turbulence Mitigation
description: >-
  [CVPR 2025][物理/科学计算][大气湍流消除] 提出 MambaTM——首个基于 Mamba 的视频大气湍流消除网络，通过 VAE 将传统 Zernike 多项式表示的相位畸变重参数化为潜在相位畸变（LPD），用 LPD 引导 SSM 的状态转移；在保持线性复杂度和全局感受野的同时，实现了 SOTA 恢复质量和接近 2× 的推理加速（55.4 FPS vs 32.7 FPS）。
tags:
  - "CVPR 2025"
  - "物理/科学计算"
  - "大气湍流消除"
  - "Mamba"
  - "状态空间模型"
  - "潜在相位畸变"
  - "退化感知恢复"
---

# Learning Phase Distortion with Selective State Space Models for Video Turbulence Mitigation

**会议**: CVPR 2025  
**arXiv**: [2504.02697](https://arxiv.org/abs/2504.02697)  
**代码**: [https://xg416.github.io/MambaTM](https://xg416.github.io/MambaTM)  
**领域**: 图像/视频恢复 / 大气湍流消除  
**关键词**: 大气湍流消除, Mamba, 状态空间模型, 潜在相位畸变, 退化感知恢复

## 一句话总结
提出 MambaTM——首个基于 Mamba 的视频大气湍流消除网络，通过 VAE 将传统 Zernike 多项式表示的相位畸变重参数化为潜在相位畸变（LPD），用 LPD 引导 SSM 的状态转移；在保持线性复杂度和全局感受野的同时，实现了 SOTA 恢复质量和接近 2× 的推理加速（55.4 FPS vs 32.7 FPS）。

## 研究背景与动机

1. **领域现状**：大气湍流导致远距离成像出现时空变化的像素位移和模糊，严重影响检测识别等下游任务。现有深度学习方法分为单帧和多帧两类，多帧方法因可利用时序"幸运效应"（某些帧受湍流影响较小）通常更优。
2. **现有痛点**：
    - **空间维度**：CNN 的有限感受野无法处理湍流的大范围空间依赖。
    - **时间维度**：Self-attention 理论上可聚合长时序信息，但二次复杂度难以扩展到多帧；循环聚合方法面临并行化和训练不稳定问题。
    - **退化表示**：传统 Zernike 多项式表示相位畸变存在严重不适定性——多组 Zernike 系数可产生相同退化模式，且需要不可微的 PSF 尺寸。
3. **核心矛盾**：需要同时实现大空间+长时序感受野、低复杂度、以及可准确估计的退化表示。
4. **本文目标** (1) 设计高效且全局感受野的湍流消除骨架；(2) 提出可学习的、一一映射的退化表示替代 Zernike。
5. **切入角度**：用 Mamba（选择性状态空间模型）替代 attention/RNN 获得线性复杂度+全局感受野；用 VAE 将 Zernike 表示压缩为潜在空间中的 LPD，消除不适定性。
6. **核心 idea**：LPD 让退化估计更 well-posed + Mamba 让时空建模更高效 = 退化感知的高效视频湍流消除。

## 方法详解

### 整体框架
输入为 $T$ 帧退化视频 $I \in \mathbb{R}^{T \times H \times W \times 3}$。多尺度编码器逐帧提取特征，经 $N_1$ 组 Mamba 模块处理后，LPD 解码器估计 4 通道相位畸变图（2 通道 tilt + 2 通道 blur），LPD 编码器将其压缩为引导特征送入后续 $N_2$ 组引导式 Mamba 模块，最后多尺度解码器输出恢复图像。ReBlurNet（冻结预训练）负责用 LPD 重退化恢复结果，实现联合优化。

### 关键设计

1. **潜在相位畸变（Latent Phase Distortion, LPD）**:

    - 功能：用可学习的低维表示替代传统 Zernike 多项式，实现与退化模式的一一映射。
    - 核心思路：使用条件 VAE 将 Zernike 系数场 $\mathbf{a}$ 编码为潜在空间 $\tilde{\mathbf{a}} \sim \mathcal{N}(\mu, \sigma^2)$，其中 $(\mu, \log\sigma)$ 即为 LPD。VAE 的解码器（ReBlurNet）根据 LPD 条件生成模糊图案，替代原始的大核深度卷积。LPD 比 Zernike 快 50×（0.02s vs 0.16~6.10s），完全可微，且重退化 PSNR 从 25.84/31.17 提升到 34.08 dB。
    - 设计动机：Zernike 到退化的映射是多对一的（相位恢复问题不适定），直接估计 Zernike 系数训练甚至无法收敛。VAE 将无穷多种可能映射到一个唯一的高斯分布，从根本上解决不适定性。

2. **多扫描混合 Mamba 模块**:

    - 功能：以线性复杂度实现视频特征的全局时空建模。
    - 核心思路：每组 Mamba 包含三个双向 Mamba 块，分别使用不同的扫描顺序——空间优先（SFMB，宽-高-时间轴）、时间优先（TFMB，时间-高-宽轴）、局部 Hilbert 扫描（LHMB，保持多维局部性的空间填充曲线）。三种扫描提供不同轴向的连接强度，相互补充。双向扫描确保非因果视觉数据的无方向偏差建模。
    - 设计动机：将 3D 张量展平为 1D 序列时，单一扫描顺序会导致某些轴向的连接被割裂。Hilbert 曲线特别适合保持展平后的局部邻近性，弥补标准光栅扫描的不足。

3. **相位畸变引导的 SSM（Guided SSM, GSSM）**:

    - 功能：用估计的 LPD 信息显式引导状态空间模型的状态转移。
    - 核心思路：标准 Mamba 中，控制参数 $\Delta, B, C$ 仅依赖输入特征 $x$。GSSM 将 LPD 编码 $r$ 与 $x$ 拼接后共同决定这些参数：$\Delta = s_\Delta(x; r), B = s_B(x; r), C = s_C(x; r)$。此外 LPD 也用于调制 Mamba 层输出的门控机制。这使得状态演化和转移依赖于退化信息。
    - 设计动机：ReBlurNet 通过重退化损失隐式注入退化感知，但 GSSM 进一步显式引导每个时空位置的聚合方式——退化严重区域可能需要聚合更多帧信息。

### 损失函数 / 训练策略
两阶段训练：
- **阶段一**（ReBlurNet）：VAE 损失 $\mathcal{L}_{VAE} = \mathcal{L}_{returb} + \alpha_k \mathcal{L}_{KL}$，其中 $\mathcal{L}_{returb}$ 为 L1 重建损失，$\mathcal{L}_{KL}$ 为 KL 散度正则。
- **阶段二**（MambaTM）：冻结 ReBlurNet，总损失 $\mathcal{L} = \mathcal{L}_{restore} + 0.2 \mathcal{L}_{returb}$。$\mathcal{L}_{restore}$ 包含 Charbonnier 损失 + 0.01× 感知损失。$\mathcal{L}_{returb}$ 包含 tilt 重建、blur 重建和 KL 散度。
- 渐进训练：从 batch 16、patch 192、18 帧逐步增大到 batch 4、patch 256、36 帧。

## 实验关键数据

### 主实验

**ATSyn-dynamic 动态场景（PSNR / SSIM / LPIPS）：**

| 方法 | Weak | Medium | Strong | Overall |
|------|------|--------|--------|---------|
| DATUM (prev SOTA) | 30.21 / 0.887 / 0.179 | 29.62 / 0.878 / 0.183 | 28.26 / 0.846 / 0.219 | 29.42 / 0.871 / 0.192 |
| **MambaTM** | **30.87 / 0.899 / 0.143** | **30.08 / 0.890 / 0.143** | **28.61 / 0.860 / 0.172** | **29.92 / 0.884 / 0.152** |

**TMT 动态场景 + 速度：**

| 方法 | PSNR | SSIM | LPIPS | FPS |
|------|------|------|-------|-----|
| DATUM | 28.60 | 0.844 | 0.225 | 32.7 |
| **MambaTM** | **28.90** | **0.856** | **0.200** | **55.4** |

**ATSyn-static + Turb-Text 真实世界：**

| 方法 | PSNR(static) | CRNN/DAN/ASTER 识别率 |
|------|-------------|----------------------|
| DATUM | 26.76 | 93.55 / 97.95 / 97.25 |
| **MambaTM** | **27.01** | **97.80 / 99.35 / 98.15** |

### 消融实验

**LPD vs Zernike 表示：**

| 表示 | 速度(s) | PSNR_returb | 可微性 |
|------|---------|-------------|--------|
| Zernike (严格监督) | 0.16~6.10 | 25.84 | 部分 |
| Zernike (松弛监督) | 0.16~6.10 | 31.17 | 部分 |
| **LPD** | **0.02** | **34.08** | **完全** |

**扫描策略消融（ATSyn-dynamic Overall）：**

| 配置 | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| 仅 SF + TF | 29.13 | 0.872 | 0.167 |
| SF + TF + LH | 29.67 | 0.883 | 0.157 |
| SF + TF + LH + LPD 引导 | **29.75** | 0.881 | **0.153** |

### 关键发现
- **LPD 是最关键设计**：将不适定的 Zernike 估计问题转化为良定的潜在空间估计，重退化 PSNR 提升 3~8 dB。
- **Mamba 的线性复杂度**带来了真正的实时性——55.4 FPS，是 DATUM 的 1.7×，且更稳定（SSM 是线性循环，不像 DATUM 的非线性循环那样存在推理不稳定问题）。
- **三种扫描互补但非必须**：去掉任何一种扫描不会导致性能崩溃，说明 SSM 具有较好的鲁棒性，但混合扫描在各轴提供更均匀的连接。
- **ReBlurNet 设计**：NAFNet + 多尺度调制优于 PlainUNet（46.72 vs 43.26 dB），但过大的 NAFNet 只有边际收益。

## 亮点与洞察
- **LPD 重参数化技巧精妙**：通过 VAE 将不适定的物理参数估计转化为适定的潜在编码估计，同时保留物理意义。这个 trick 可迁移到任何存在"多参数一效果"问题的退化估计场景（如光学像差、运动模糊核估计）。
- **退化引导 Mamba 的设计**：把退化信息注入 SSM 的 $\Delta, B, C$ 参数，使得状态演化和信息聚合具有自适应性。这种"条件化 SSM"的思路可迁移到其他条件生成/恢复任务。
- **渐进训练策略**：从小 patch/多 batch 到大 patch/少 batch 的渐进训练，既保证早期快速收敛又让后期学到长程依赖，是视频恢复任务的实用训练技巧。

## 局限与展望
- **仅在合成数据上训练**：虽然真实世界定性结果不错，但缺少真实湍流数据的定量评估（因为没有 GT）。
- **LPD 的 VAE 需要预训练**：两阶段训练增加了复杂度，能否端到端联合训练 VAE 和恢复网络值得探索。
- **动态场景处理有限**：论文假设动态区域服从刚性运动，对非刚性运动（如行人四肢）可能失效。
- **Hilbert 扫描的局限**：Hilbert 曲线在边界处仍不可避免地会产生跳跃，且对 3D 张量的最优展平策略仍未充分探索。

## 相关工作与启发
- **vs DATUM [Zhang et al.]**: DATUM 使用非线性循环结构，虽有全局时序感受野但训练不稳定且推理慢。MambaTM 用线性 SSM 替代，稳定性和速度均大幅提升。
- **vs TMT [Zhang et al.]**: TMT 用时间-通道 self-attention，二次复杂度限制了帧数。MambaTM 的线性复杂度可处理更多帧。
- **vs DRBNet [Chen et al.]**: DRBNet 用可微模拟器做退化感知但仍基于 Zernike，LPD 从根本上解决了 Zernike 的不适定性。

## 评分
- 新颖性: ⭐⭐⭐⭐ LPD 重参数化和退化引导 Mamba 都是有意义的新设计
- 实验充分度: ⭐⭐⭐⭐⭐ 合成+真实、静态+动态、多个 benchmark、详细消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但物理背景部分对非光学读者可能较难
- 价值: ⭐⭐⭐⭐ 首个 Mamba 用于湍流消除，LPD 设计可迁移到其他退化估计场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Physics-Guided Machine Learning for Uncertainty Quantification in Turbulence Models](../../NeurIPS2025/physics/physics-guided_machine_learning_for_uncertainty_quantification_in_turbulence_mod.md)
- [\[CVPR 2025\] Towards Faithful Multimodal Concept Bottleneck Models](towards_faithful_multimodal_concept_bottleneck_models.md)
- [\[ICML 2026\] MōLe-Λ: Learning the Coupled-Cluster Response State for Energies, Gradients, and Properties](../../ICML2026/physics/mōle-λ_learning_the_coupled-cluster_response_state_for_energies_gradients_and_pr.md)
- [\[CVPR 2025\] KAC: Kolmogorov-Arnold Classifier for Continual Learning](kac_kolmogorov-arnold_classifier_for_continual_learning.md)
- [\[NeurIPS 2025\] GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations](../../NeurIPS2025/physics/gyroswin_5d_surrogates_for_gyrokinetic_plasma_turbulence_simulations.md)

</div>

<!-- RELATED:END -->
