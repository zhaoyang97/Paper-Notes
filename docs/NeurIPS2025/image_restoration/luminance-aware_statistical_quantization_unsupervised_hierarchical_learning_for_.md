---
title: >-
  [论文解读] Luminance-Aware Statistical Quantization: Unsupervised Hierarchical Learning for Illumination Enhancement
description: >-
  [NeurIPS 2025][图像恢复][低光图像增强] 提出 LASQ 框架，将低光图像增强重新定义为基于分层亮度分布的统计采样过程，利用自然亮度转换中固有的幂律分布特性，通过 MCMC 采样生成层次化亮度适配算子，嵌入扩散模型前向过程实现无监督增强，无需正常光照参考即可工作。 领域现状： 低光图像增强(LLIE)方法分为…
tags:
  - "NeurIPS 2025"
  - "图像恢复"
  - "低光图像增强"
  - "扩散模型"
  - "幂律分布"
  - "MCMC采样"
  - "无监督学习"
---

# Luminance-Aware Statistical Quantization: Unsupervised Hierarchical Learning for Illumination Enhancement

**会议**: NeurIPS 2025  
**arXiv**: [2511.01510](https://arxiv.org/abs/2511.01510)  
**代码**: [GitHub](https://github.com/XYLGroup/LASQ)  
**领域**: 低光图像增强 / 图像恢复  
**关键词**: 低光图像增强, 扩散模型, 幂律分布, MCMC采样, 无监督学习

## 一句话总结

提出 LASQ 框架，将低光图像增强重新定义为基于分层亮度分布的统计采样过程，利用自然亮度转换中固有的幂律分布特性，通过 MCMC 采样生成层次化亮度适配算子，嵌入扩散模型前向过程实现无监督增强，无需正常光照参考即可工作。

## 研究背景与动机

**领域现状**: 低光图像增强(LLIE)方法分为监督（需配对数据）和无监督两类，近期扩散模型引入提升了灵活性。

**现有痛点**: 
   - 监督方法过拟合于像素级对应关系，忽略亮度转换的连续物理过程
   - 无监督方法依赖伪参考（如经验性 gamma 校正），继承先验偏差
   - 两类方法都将本质连续、上下文相关的亮度动态过度简化，导致泛化能力有限

**核心矛盾**: 重建保真度 vs 跨场景泛化能力——专注于域内精度则泛化差，追求泛化则域内不足。

**本文目标**: 建立基于自然照明物理规律的 LLIE 统计模型，不依赖配对数据。

**切入角度**: 经验发现自然亮度转换服从幂律密度分布，可用分层幂函数近似。

**核心idea**: 将 LLIE 从确定性像素映射重新定义为在层次化亮度分布上的统计采样过程。

## 方法详解

### 整体框架

三个核心组件：(1) 层次化亮度建模——建立亮度变化坐标系、设计分层亮度适配算子 → (2) MCMC 采样——生成从粗到细的亮度适配算子集合 → (3) 扩散模型——将层次化采样嵌入前向过程，实现无监督学习。

### 关键设计

1. **亮度变化坐标系 (Luminance Variation Coordinate System)**:

    - **功能**: 建立低光-正常光亮度关系的几何框架
    - **为什么**: 需要将亮度转换的物理规律数学化
    - **怎么做**: 对每个像素 $i$，以 $(I_L^{(i)}, I_N^{(i)})$ 为坐标点，发现这些点遵循幂律分布 $y = ax^\kappa$，不同 $\kappa$ 对应不同的亮度适配策略（$\kappa < 0.5$: 暗区恢复，$0.5 < \kappa < 1$: 中间调增强，$\kappa \to 1$: 高光保持）

2. **层次化亮度适配算子 (Hierarchical Luminance Adaptation Operator, LAO)**:

    - **功能**: 构建从全局到局部的多尺度亮度校正算子
    - **怎么做**: 对区域 $\mathcal{P}$ 计算亮度标量 $G_\mathcal{P}$ 及 LAO：
    $\gamma_\mathcal{P} = (\alpha + G_\mathcal{P})^{\beta_\mathcal{P}}, \quad \beta_\mathcal{P} = 2G_\mathcal{P} - 1 + \eta\frac{\sigma_{G_\mathcal{P}}^2}{\sigma_{G_\mathcal{P}}^2 + \delta}$
    - **分布建模**: LAO 服从截断高斯分布 $\gamma \sim \mathcal{N}_{\text{trunc}}(\mu=\gamma_0, \sigma^2; \gamma_{\min}, \gamma_{\max})$
    - **物理含义**: 高概率算子对应物理合理的全局适配，低概率算子代表局部精调

3. **MCMC 层次采样**:

    - **功能**: 从 LAO 分布空间中渐进采样，生成从粗到细的增强图像集合
    - **怎么做**: 第 $n$ 次迭代产生 $2^{n-1}$ 个 LAO 配置：
    $p(\mathcal{I}_H^{(n)}) \approx \sum_{z=1}^{2^{n-1}} p(\mathcal{I}_H^{(n)}|\gamma_{\mathcal{P},z}^{(n)}) p(\gamma_{\mathcal{P},z}^{(n)})$
   转移核为截断高斯：$q(\gamma_z^{(n)}|\gamma_{z-1}^{(n)}) = \mathcal{N}_{\text{trunc}}(\gamma_z^{(n)}|\gamma_{z-1}^{(n)}, \lambda^2)$
    - **网格策略**: 第 $n$ 次将图像分为 $m_n \times w_n$ 非重叠块（$m_n = 2^{\lceil(n-1)/2\rceil}$），实现从粗到细

4. **层次引导扩散模型 (Hierarchically-Guided Diffusion)**:

    - **功能**: 将 MCMC 采样的层次化增强嵌入扩散前向过程
    - **怎么做**: 通过时间映射 $\psi(t) = \lfloor t \cdot N/T \rfloor$ 将 $T$ 步扩散对齐到 $N$ 层层次化特征。前向过程在每个时间区间 $T_n$ 内使用对应的 $\mathcal{F}_H^{(\psi(t))}$ 作为照明归一化参考
    - **训练**: 噪声预测目标 $\mathcal{L}_d$ + 全局标签弱引导 $\mathcal{L}_g$
    - **LASQ++ 扩展**: 可选加入非配对正常光照参考的对抗判别器：
    $\mathcal{L}_{\text{total}} = \lambda_d\mathcal{L}_d + \lambda_g\mathcal{L}_g + \lambda_{\text{GAN}}\mathbb{E}[-\log\mathcal{D}_\phi(G_\theta(\mathcal{I}_L))]$

### 损失函数 / 训练策略

- 噪声预测损失 $\mathcal{L}_d$（权重 0.9）+ 全局引导损失 $\mathcal{L}_g$（权重 0.005）
- 可选 GAN 损失（权重 0.7，LASQ++ 模式）
- Adam 优化器，学习率 $2 \times 10^{-5}$，U-Net 架构，$T=1000$ 步

## 实验关键数据

### 主实验

配对数据集（LOLv1 / LSRW）对比：

| 类型 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|-------|-------|--------|
| SL | PyDiff | 23.275 | 0.859 | 0.108 |
| SL | SMG | 23.814 | 0.809 | 0.144 |
| UL | LightenDiffusion | 20.453 | 0.803 | 0.192 |
| UL | NeRCo | 19.738 | 0.740 | 0.239 |
| **UL** | **LASQ** | **20.375** | **0.814** | **0.191** |
| UL+ | LASQ++ | 20.481 | 0.807 | 0.205 |

无参考数据集（DICM/NPE/VV）——LASQ 的真正优势场景：

| 方法 | DICM NIQE↓ | NPE NIQE↓ | VV NIQE↓ |
|------|-----------|----------|---------|
| LightenDiffusion | 3.724 | 3.618 | 2.941 |
| NeRCo | 4.107 | 3.902 | 3.765 |
| **LASQ** | **3.715** | **3.571** | **2.777** |

LASQ 在无参考数据集上全面超越所有方法（含监督方法），展现出色的跨场景泛化。

### 消融实验

| 方法 | LOLv1 PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------------|-------|--------|
| Fixed Luminance Adj. | 16.741 | 0.715 | 0.273 |
| Limited Hierarchy (2层) | 19.139 | 0.792 | 0.243 |
| **LASQ (完整)** | **20.375** | **0.814** | **0.191** |

### 计算效率

| 方法 | FLOPs (G) | 参数 (M) | 推理时间 (ms) |
|------|----------|---------|-------------|
| SCI | 0.13 | — | 50.14 |
| LightenDiffusion | 367.99 | 27.83 | 257.94 |
| **LASQ** | **219.75** | **24.08** | **213.89** |

LASQ 在保持扩散模型性能优势的同时，推理效率接近非扩散方法。

### 关键发现

- 自适应 MCMC 采样远优于固定亮度调整（PSNR 差 3.6 dB）
- 中间层次级别不可或缺——两层简化版虽优于固定方案但不及完整 LASQ（PSNR 差 1.2 dB）
- LASQ 在无参考场景超越所有监督方法——证实物理驱动的泛化优势
- 加入正常光照参考（LASQ++）可提升域内色彩保真度，但可能轻微降低泛化能力
- 超参数敏感性低：$\alpha$, $\eta$, $\lambda_d$, $\lambda_g$ 变化范围内 PSNR 波动 <0.3dB

## 亮点与洞察

- **物理驱动的范式转换**: 首次将 LLIE 从确定性像素映射重新定义为基于自然亮度物理规律的统计过程
- **无需配对数据**: MCMC 采样嵌入扩散前向过程，完全无监督即可工作，从根本上消除了对配对数据的依赖
- **泛化能力突出**: 在无参考数据集上甚至击败监督方法，说明物理先验比数据驱动的映射更具泛化力
- **双模式兼容**: 无缝支持有/无正常光照参考两种场景
- **幂律分布发现**: 自然亮度转换的幂律分布特性本身是有价值的经验发现

## 局限与展望

- MCMC 采样增加了训练时间（虽然推理时不使用）
- 幂律假设可能在极端场景（如纯黑/纯白区域）不成立
- 当前使用静态幂律参数化，对时变场景（如视频）的适用性待验证
- U-Net 架构可替换为更先进的去噪网络（如 DiT）进一步提升性能
- 硬件-软件协同设计以匹配传感器特定噪声特性的方向未被探索

## 相关工作与启发

- 与 Zero-DCE 的"曲线估计"思路相关但本质不同——LASQ 是统计采样而非单一曲线拟合
- LightenDiffusion 将 Retinex 理论融入扩散步骤，LASQ 则基于幂律分布建立更通用的物理框架
- 启发方向：将层次化 MCMC 采样思想推广到其他图像退化恢复任务（如去雾、去雨）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将 LLIE 重新定义为统计采样问题，理论视角独特且有经验支撑
- 实验充分度: ⭐⭐⭐⭐ 配对/无参考/消融/计算效率/超参数敏感性全面覆盖
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，物理直觉清晰，但数学符号略显密集
- 价值: ⭐⭐⭐⭐⭐ 无监督+物理驱动+泛化强，对实际部署（无需配对数据）有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Outlier-Aware Post-Training Quantization for Image Super-Resolution](../../ICCV2025/image_restoration/outlier-aware_post-training_quantization_for_image_super-resolution.md)
- [\[ICCV 2025\] Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)](../../ICCV2025/image_restoration/low-light_image_enhancement_using_event-based_illumination_estimation.md)
- [\[CVPR 2026\] BiEvLight: Bi-level Learning of Task-Aware Event Refinement for Low-Light Image Enhancement](../../CVPR2026/image_restoration/bievlight_bi-level_learning_of_task-aware_event_refinement_for_low-light_image_e.md)
- [\[AAAI 2026\] ICLR: Inter-Chrominance and Luminance Interaction for Natural Color Restoration in Low-Light Image Enhancement](../../AAAI2026/image_restoration/iclr_inter-chrominance_and_luminance_interaction_for_natural_color_restoration_i.md)
- [\[ICLR 2026\] ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting](../../ICLR2026/image_restoration/protots_learning_hierarchical_prototypes_for_explainable_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
