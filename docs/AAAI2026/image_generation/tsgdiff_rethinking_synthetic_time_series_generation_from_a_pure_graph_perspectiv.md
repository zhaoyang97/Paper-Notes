---
title: >-
  [论文解读] TSGDiff: Rethinking Synthetic Time Series Generation from a Pure Graph Perspective
description: >-
  [AAAI 2026][图像生成][时间序列生成] 提出 TSGDiff，首次从纯图的视角重新审视时间序列生成任务，将时间序列表示为基于傅里叶频谱特征构建的动态图，在图的潜在空间中进行扩散建模，并提出 Topo-FID 指标评估生成时间序列的结构忠实度。 1. 领域现状： 多变量时间序列生成在能源管理、金融预测、医疗监测等领…
tags:
  - "AAAI 2026"
  - "图像生成"
  - "时间序列生成"
  - "图神经网络"
  - "扩散模型"
  - "傅里叶变换"
  - "拓扑结构忠实度"
---

# TSGDiff: Rethinking Synthetic Time Series Generation from a Pure Graph Perspective

**会议**: AAAI 2026  
**arXiv**: [2511.12174](https://arxiv.org/abs/2511.12174)  
**代码**: [TSGDiff](https://github.com/jvaeylee/TSGDiff)  
**领域**: 时间序列  
**关键词**: 时间序列生成, 图神经网络, 扩散模型, 傅里叶变换, 拓扑结构忠实度

## 一句话总结

提出 TSGDiff，首次从纯图的视角重新审视时间序列生成任务，将时间序列表示为基于傅里叶频谱特征构建的动态图，在图的潜在空间中进行扩散建模，并提出 Topo-FID 指标评估生成时间序列的结构忠实度。

## 研究背景与动机

1. **领域现状**: 多变量时间序列生成在能源管理、金融预测、医疗监测等领域有广泛需求。现有方法包括 GAN 系列（TimeGAN）、VAE 系列（TimeVAE）和扩散模型系列（Diffusion-TS、CSDI）。
2. **现有痛点**: (a) 传统生成模型难以有效捕获变量间复杂的时空依赖关系；(b) 它们基于欧几里得空间假设，无法表示时间序列数据固有的拓扑和结构特征；(c) 如 Diffusion-TS 使用 encoder-decoder transformer 分别处理空间和时间信息，限制了对复杂变量间依赖的建模；(d) 图方法在预测任务中表现出色，但在生成任务中几乎未被探索。
3. **核心矛盾**: 时间序列具有丰富的时间依赖结构（短期/中期/长期周期性），但现有生成方法仅在原始数据域操作，未能从结构角度建模这些依赖关系。
4. **本文目标**: 从图的视角统一建模时间序列中的时间依赖和结构关系，实现高保真的合成时间序列生成。
5. **切入角度**: 将时间步视为图的节点，利用傅里叶频谱特征构建边来编码周期性依赖，在图的潜在空间中进行扩散建模。
6. **核心 idea**: 用傅里叶变换发现时间序列的周期性规律来构建图结构，在图的潜在表示空间中进行扩散生成，同时用拓扑结构忠实度指标衡量生成质量。

## 方法详解

### 整体框架

TSGDiff 分为三个阶段：(1) 图构建与编码——对时间序列进行滑动窗口分割和实例归一化后，利用傅里叶变换提取周期信息构建动态图，再用 GNN 编码器得到潜在表示；(2) 潜在扩散建模——在潜在图空间中执行扩散-去噪过程；(3) 图解码——将去噪后的潜在表示解码回合成时间序列。

### 关键设计

1. **Fourier-based Graph Construction（基于傅里叶的图构建）**:
    - 功能：将时间序列转换为能编码周期性依赖的图结构
    - 核心思路：对每个窗口段 $\tilde{X}_i$ 做离散傅里叶变换 $\tilde{X}_i^{(p)} = \sum_{n=0}^{w-1} \tilde{x}_{i+n} \cdot e^{-j\frac{2\pi}{w}pn}$，检测幅值最高的 top-3 频率并转换为对应周期。节点 = 时间步，边 = 相邻时间步 + 周期邻居。邻接矩阵 $\mathbf{A}_{ij}=1$ 表示存在时序或周期连接
    - 设计动机：傅里叶频谱自然揭示了时间序列的周期性结构，用周期来定义边连接能同时编码短期、中期和长期依赖模式

2. **Graph VAE Encoder-Decoder（图变分自编码器）**:
    - 功能：将图结构编码为潜在空间表示，并能从潜在向量解码恢复
    - 核心思路：编码器堆叠多层 GCN（图卷积 $y' = \text{Mish}(\text{BN}((W\mathbf{A}\tilde{X}+b)^T)^T)$ + 残差连接），经均值池化后输出 $\mu$ 和 $\log\sigma$，通过重参数化技巧采样潜在向量 $z = \mu + \epsilon \odot \exp(\log\sigma / 2)$。解码器用全连接层+Tanh 恢复节点特征。KL 散度损失 $\mathcal{L}_{KL}$ 约束潜在分布接近标准正态
    - 设计动机：GCN 能天然利用邻接矩阵定义的图结构进行信息聚合，VAE 框架提供正则化的潜在空间

3. **Latent Diffusion + Topo-FID Metric（潜在扩散 + 拓扑结构忠实度指标）**:
    - 功能：在潜在图空间进行生成建模；评估生成时间序列的结构保真度
    - 核心思路：前向过程逐步加噪 $z_k = \sqrt{\alpha_k} \cdot z + \sqrt{1-\alpha_k} \cdot \epsilon$；反向过程用 MLP 网络预测去噪，输入为 $\text{concat}(z_k, t_{emb})$。Topo-FID 定义为 $\alpha \cdot S_{edit} + (1-\alpha) \cdot S_{entropy}$，其中 $S_{edit}$ 比较邻接矩阵差异，$S_{entropy}$ 比较节点度分布的结构熵差异
    - 设计动机：在潜在空间做扩散比原始数据空间更高效且能保持结构一致性；传统指标无法捕获时间序列的结构特征，Topo-FID 从图的角度补充了这一缺失

### 损失函数 / 训练策略

总损失函数：
$$\mathcal{L} = \mathcal{L}_{recon} + \beta \mathcal{L}_{KL} + \gamma \mathcal{L}_{denoising} + \delta \mathcal{L}_{Fourier}$$

- $\mathcal{L}_{recon}$：输入与解码器输出的 MSE 重建损失
- $\mathcal{L}_{KL}$：KL 散度正则化（$\beta=0.2$）
- $\mathcal{L}_{denoising}$：扩散去噪损失 $\|epsilon - \epsilon_\theta(z_k, k)\|^2$（$\gamma=1$）
- $\mathcal{L}_{Fourier}$：频域一致性损失 $\|\text{FFT}(x_0) - \text{FFT}(\hat{x}_0)\|^2$（$\delta=1$）

训练配置：batch size 128，lr 0.01，500 epochs，滑窗大小 48，步长 1。

## 实验关键数据

### 主实验

| 数据集 | 指标 | TSGDiff | Diffusion-TS | TimeGAN | 提升 |
|--------|------|------|----------|------|------|
| ETTh | Topo-FID↑ | **0.986** | 0.864 | 0.798 | +12.2% |
| Weather | Context-FID↓ | **0.353** | 1.161 | 2.420 | -69.6% |
| EEG | Discriminative↓ | **0.301** | 0.492 | 0.391 | -23.0% |
| ETTh | Predictive↓ | **0.020** | 0.026 | 0.030 | -23.1% |
| Stocks | Correlational↓ | **0.026** | 0.027 | 0.061 | -3.7% |
| Wind | Topo-FID↑ | **0.869** | 0.819 | 0.824 | +5.0% |

在 6 个数据集上平均 Discriminative score 提升 50%。

### 消融实验

| 配置 | Topo-FID↑ | Context-FID↓ | Discriminative↓ | 说明 |
|------|---------|---------|---------|------|
| w/o KL | 0.787 | 44.467 | 0.500 | 潜在空间无约束，崩塌 |
| w/o Denoising | 0.908 | 3.922 | 0.432 | 生成能力受限 |
| w/o Fourier | 0.883 | 0.407 | 0.061 | 频域信息丢失 |
| Full (TSGDiff) | **0.986** | **0.224** | **0.056** | 所有组件协同最优 |

### 关键发现

- KL 散度损失对模型至关重要——去除后 Context-FID 从 0.224 飙升至 44.467，潜在空间完全退化
- Fourier 损失对 Topo-FID 有显著影响（0.986→0.883），说明频域约束对保持结构忠实性很重要
- 核密度估计可视化显示 TSGDiff 生成的分布比 Diffusion-TS 更接近真实数据分布

## 亮点与洞察

- **视角创新**：首次从纯图的视角看待时间序列生成问题，将时间依赖关系建模为图的边，这是一种有意义的范式转换
- **Fourier + Graph 的自然结合**：用傅里叶频谱发现周期性来构建图边，比手动定义或全连接图更有物理意义
- **Topo-FID 新指标**：传统指标（FID、Discriminative score 等）无法评估时间序列的结构特性，Topo-FID 填补了这一空白
- **统一框架**：图构建 + 图 VAE + 潜在扩散的三阶段设计，将结构建模与生成过程统一在同一框架中

## 局限与展望

- 图构建仅考虑了 top-3 频率的周期性，可能遗漏重要的非周期性或非线性依赖关系
- 每个变量独立做傅里叶变换构建边，但不同变量之间的交叉依赖关系未被显式建模（节点 = 时间步而非变量）
- 潜在扩散网络使用简单的 MLP，可能限制了对复杂潜在分布的建模能力
- Topo-FID 仍然依赖与真实数据的成对比较，对于评估多样性可能不够全面
- 滑窗大小固定为 48，不同时间尺度的序列可能需要不同的窗口设置

## 相关工作与启发

- **vs Diffusion-TS**: Diffusion-TS 在原始数据域做扩散+季节趋势分解，空间和时间分开处理；TSGDiff 在图的潜在空间统一建模，能更好捕获结构依赖
- **vs TimeGAN**: TimeGAN 通过联合优化监督和对抗目标捕获时间动态，但面临模式崩塌问题；TSGDiff 的图VAE+扩散避免了这个问题
- **vs FourierGNN (预测)**: FourierGNN 在预测任务中用傅里叶+图方法，但未用于生成任务；TSGDiff 将这一思路成功扩展到生成领域

## 评分

- 新颖性: ⭐⭐⭐⭐ 图视角看时间序列生成是新颖的，Topo-FID 指标也是有价值的贡献
- 实验充分度: ⭐⭐⭐⭐ 6个数据集、5类指标、4个基线、消融实验+可视化完整
- 写作质量: ⭐⭐⭐⭐ 框架图清晰、数学符号规范、但部分公式推导可以更详细
- 价值: ⭐⭐⭐⭐ 为时间序列生成提供了新的建模范式和评估指标，有较好的启发性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] SimDiff: Simpler Yet Better Diffusion Model for Time Series Point Forecasting](simdiff_simpler_yet_better_diffusion_model_for_time_series_point_forecasting.md)
- [\[ICLR 2026\] Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting](../../ICLR2026/image_generation/conditionally_whitened_generative_models_for_probabilistic_time_series_forecasti.md)
- [\[NeurIPS 2025\] A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](../../NeurIPS2025/image_generation/a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)
- [\[CVPR 2026\] Rethinking Prompt Design for Inference-time Scaling in Text-to-Visual Generation](../../CVPR2026/image_generation/rethinking_prompt_design_for_inference-time_scaling_in_text-to-visual_generation.md)
- [\[CVPR 2025\] Everything to the Synthetic: Diffusion-driven Test-time Adaptation via Synthetic-Domain Alignment](../../CVPR2025/image_generation/everything_to_the_synthetic_diffusion-driven_test-time_adaptation_via_synthetic-.md)

</div>

<!-- RELATED:END -->
