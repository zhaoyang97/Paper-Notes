---
title: >-
  [论文解读] Graph-based Neural Space Weather Forecasting
description: >-
  [NEURIPS2025][时间序列][space weather] 提出基于图神经网络的空间天气神经模拟器，在 Vlasiator 混合 Vlasov 模拟数据上训练，实现确定性和概率性自回归预测近地空间状态，速度比原始模拟快 100 倍以上，并通过隐变量生成集合预报来量化预测不确定性。 空间天气描述了太阳风驱动的近地空间…
tags:
  - "NEURIPS2025"
  - "时间序列"
  - "space weather"
  - "图神经网络"
  - "probabilistic forecasting"
  - "hybrid-Vlasov"
  - "ensemble forecasting"
---

# Graph-based Neural Space Weather Forecasting

**会议**: NEURIPS2025  
**arXiv**: [2509.19605](https://arxiv.org/abs/2509.19605)  
**代码**: [fmihpc/spacecast](https://github.com/fmihpc/spacecast)  
**领域**: 图像生成  
**关键词**: space weather, graph neural network, probabilistic forecasting, hybrid-Vlasov, ensemble forecasting

## 一句话总结

提出基于图神经网络的空间天气神经模拟器，在 Vlasiator 混合 Vlasov 模拟数据上训练，实现确定性和概率性自回归预测近地空间状态，速度比原始模拟快 100 倍以上，并通过隐变量生成集合预报来量化预测不确定性。

## 背景与动机

空间天气描述了太阳风驱动的近地空间条件，对现代基础设施构成严重威胁：地磁感应电流可破坏电力网络，卫星运行受到不利影响，电磁通信也可能出现故障。当前业务预报系统依赖全球磁流体力学（MHD）模型（如 BATS-R-US），由 L1 拉格朗日点卫星的实时太阳风数据驱动。但这一范式面临两大挑战：

1. **物理保真度不足**：MHD 模型将等离子体近似为流体，忽略了离子动力学过程。更高保真度的混合 Vlasov 模型（如 Vlasiator）可以捕获这些过程，但计算成本极高，无法用于实时预报。
2. **缺乏不确定性量化**：多数业务预报是确定性的单点预测，缺少关键的不确定性信息。虽然集合预报的需求已被提出，但现有方法要么扰动物理模型的太阳风输入，要么用机器学习对确定性输出做后处理。

本文受大气天气预报中 ML 突破（GraphCast、Pangu-Weather 等）的启发，将基于图的有限区域建模方法迁移到空间天气领域，旨在同时解决上述两个问题。

## 核心问题

如何利用机器学习让计算昂贵的混合 Vlasov 模拟（Vlasiator）变得可用于业务预报？具体而言：
- 如何在保持物理保真度的前提下实现快速确定性预报？
- 如何为空间天气预报系统增加当前缺失的不确定性量化能力？

## 方法详解

### 问题形式化

将空间天气预报建模为：给定两个连续磁层状态 $X^{-1:0} = (X^{-1}, X^{0})$（捕获一阶动力学），预测未来轨迹 $X^{1:T} = (X^1, \dots, X^T)$。每个状态 $X^t \in \mathbb{R}^{N \times d_x}$ 表示 $N$ 个网格位置上 $d_x$ 个物理变量。预测变量包括：磁场 $(B_x, B_y, B_z)$、电场 $(E_x, E_y, E_z)$、速度场 $(v_x, v_y, v_z)$、粒子数密度 $\rho$、等离子体温度 $T$、等离子体压力 $P$，共 12 个物理量。

### 数据来源：Vlasiator 混合 Vlasov 模拟

使用 Vlasiator 进行 2D-3V（两个空间维度、三个速度维度）配置模拟：
- **时间步长**：$\Delta t = 1\,\text{s}$，空间分辨率 600 km
- **模拟域**：日-夜子午面（GSE 坐标系的 $x$-$z$ 平面），$x$ 方向 $[-60R_E, +30R_E]$，$z$ 方向 $\pm 30R_E$
- **太阳风参数**：密度 $\rho = 1/\text{cm}^3$，速度 $\mathbf{v} = (-750, 0, 0)\,\text{km/s}$，温度 $T = 0.5\,\text{MK}$
- **行星际磁场**：南向 $\mathbf{B} = (0, 0, -5)\,\text{nT}$，Alfvén 马赫数 $M_A = 6.9$
- **内边界**：距地心 $3.7\,R_E$ 的完美导体

### 图神经网络架构

采用 encode-process-decode 架构：
1. **编码器**：将网格输入编码到 mesh 表示
2. **处理器**：多层 GNN 处理隐表示
3. **解码器**：将处理后的数据映射回原始网格

关键设计：GNN 预测残差更新而非直接预测下一状态（$\hat{X}^t = X^{t-1} + \tilde{g}(X^{t-2:t-1})$），降低学习难度。边界强制：在日侧开放边界区域（$x = 27R_E$ 到 $30R_E$），通过静态二值掩码在每次预测后用真实边界数据替换。

### 三种 Mesh 架构

| 架构 | 节点数 | 边数 | 特点 |
|------|--------|------|------|
| Simple | 58,592 | 465,584 | 单层粗网格 |
| Multiscale | 58,592 | 522,054 | 多层边连接到同一组节点，单层 GNN 同时处理所有尺度 |
| Hierarchical | 65,825 | 587,156 | 三层独立图，通过层间图传递信息 |

Hierarchical 架构使用 interaction network 保留信息，propagation network 传递新信息，在高低层之间做完整的上下扫描。

### 确定性模型：Graph-FM

自回归映射 $\hat{X}^t = f(X^{t-2:t-1})$，使用加权 MSE 损失训练：

$$\mathcal{L} = \frac{1}{N} \sum_{n=1}^{N} \sum_{i=1}^{d_x} \omega_i \lambda_i (\hat{X}_{n,i} - X_{n,i})^2$$

其中 $\lambda_i$ 为时间差异逆方差（归一化不同物理量的贡献），$\omega_i = 1/d_x$。

### 概率模型：Graph-EFM

引入隐随机变量 $Z^t$ 来建模预测不确定性，结构类似条件 VAE：

$$p(X^t | X^{t-2:t-1}) = \int p(X^t | Z^t, X^{t-2:t-1}) \, p(Z^t | X^{t-2:t-1}) \, dZ^t$$

- **隐映射（Latent Map）**：使用 GNN 将输入映射到各向同性高斯分布的均值，方差固定。隐分布定义在最粗层 $\mathcal{V}_L$ 的节点上，确保在低维空间引入随机性。
- **预测器（Predictor）**：确定性映射，$Z^t$ 在顶层注入并向下传播，产生高分辨率、空间连贯的预报。

训练分三阶段：
1. 自编码器预训练（200 epochs，$\lambda_{\text{KL}} = 0, \lambda_{\text{CRPS}} = 0$）
2. 变分训练（300 epochs，$\lambda_{\text{KL}} = 1$）
3. CRPS 微调（50 epochs，$\lambda_{\text{CRPS}} = 10^{-3}$），改善集合校准

## 训练与推理

### 训练策略

所有模型使用 AdamW 优化器（$\beta_1=0.9$, $\beta_2=0.95$, weight decay $0.01$），effective batch size 为 8。

**Graph-FM（确定性）**：标准两阶段训练
- 阶段一：300 epochs，学习率 $10^{-3}$
- 阶段二：200 epochs，学习率 $10^{-4}$

**Graph-EFM（概率性）**：三阶段渐进训练
1. **自编码器预训练**（200 epochs，lr $10^{-3}$）：$\lambda_{\text{KL}}=0, \lambda_{\text{CRPS}}=0$，让 encoder/decoder 先学会从隐空间重建场量
2. **变分训练**（300 epochs，lr $10^{-3}$）：$\lambda_{\text{KL}}=1$，引入 KL 散度正则化，训练完整变分目标
3. **CRPS 微调**（50 epochs，lr $10^{-4}$）：$\lambda_{\text{CRPS}}=10^{-3}$，使用 CRPS 损失改善集合校准

### 损失函数

- **确定性**：加权 MSE，使用时间差异逆方差 $\lambda_i$ 归一化不同物理量的动态范围
- **概率性**：变分目标（负 ELBO）= KL 散度正则 + 重建损失（负对数似然），微调时叠加 CRPS 损失
- **CRPS** 使用无偏两样本估计器：$\mathcal{L}_{\text{CRPS}} = \sum \frac{1}{2}(|\hat{X}-X| + |\check{X}-X| - |\hat{X}-\check{X}|)$，其中 $\hat{X}, \check{X}$ 为两个独立集合成员

### 训练成本

使用 8 张 AMD MI250X GPU 训练：

| 模型 | 图架构 | 训练时间 (GPU h) | 推理时间 (s/step) |
|------|--------|-----------------|-------------------|
| Graph-FM | Simple | 101 | 0.47 |
| Graph-FM | Multiscale | 102 | 0.48 |
| Graph-FM | Hierarchical | 108 | 0.52 |
| Graph-EFM | Simple | 119 | 3.20 |
| Graph-EFM | Multiscale | 122 | 3.31 |
| Graph-EFM | Hierarchical | 131 | 3.45 |

对比基线：Vlasiator 在 50 个 AMD EPYC 7H12 CPU 上模拟 1 秒需 4-5 分钟。确定性模型加速约 **500 倍**，概率模型（集合大小 5）加速约 **80 倍**。

### 图构建细节

数据网格为 $671 \times 1006\,(z, x)$，排除 5124 个内边界节点。通过递归下采样构建图：每个粗层节点位于 $3 \times 3$ 细层节点正方形的中心。Grid-to-Mesh 图 $\mathcal{G}_{G2M}$ 有 1,411,687 条边，Mesh-to-Grid 图 $\mathcal{G}_{M2G}$ 有 2,679,608 条边。所有 MLP 使用单隐层 + Swish 激活 + Layer Normalization。

## 实验设置

### 数据划分
因 Vlasiator 极高的模拟精度，数据量有限：训练集 10 分钟、验证集 1 分钟、测试集 1 分钟，按因果顺序（时间序列）划分，确保测试是对未来的有意义泛化。

### 评估指标

- **RMSE**：集合均值（或确定性预报）与真实值的均方根误差，归一化为无量纲得分（除以变量标准差）
- **CRPS**（Continuous Ranked Probability Score）：评估概率预报的整体质量，同时衡量准确性和校准度，使用有限样本估计器
- **Spread**：集合成员相对集合均值的均方根偏差，量化预报不确定性。良好校准时 Spread 应等于 RMSE

## 实验结果

### 确定性模型对比
- **Hierarchical 架构误差累积最慢**，Simple 和 Multiscale 架构的误差增长更快
- 可能原因：Hierarchical 的分层信息传播（interaction network 保留 + propagation network 传递）更有效地处理多尺度空间结构

### 概率 vs 确定性
- **概率模型 RMSE 低于确定性模型**，特别是在更长预报时间上
- 原因：多成员平均缓解了轨迹漂移的影响，产生更稳定的估计
- 三种 Mesh 架构的概率模型性能相近，差异不如确定性模型明显

### 校准分析
- 所有概率模型存在**欠散（underdispersion）**：集合 Spread 始终小于 RMSE
- 主要原因：训练数据有限 + 变分目标的固有特性（在有限区域建模中的已知问题）
- CRPS 微调较保守（$\lambda_{\text{CRPS}}=10^{-3}$，仅 50 epochs），加大权重可进一步改善

### 逐变量分析
- 模型成功捕获复杂磁层动力学，集合标准差正确定位在弓激波、磁尾等物理活跃区域
- **伪影问题**：在磁尾北叶区域的 $B_y$ 和 $v_y$ 分量出现训练数据中不存在的虚假结构，归因于有限的训练样本
- 12 个物理变量（$B_{x,y,z}$, $E_{x,y,z}$, $v_{x,y,z}$, $\rho$, $T$, $P$）的 RMSE/CRPS/Spread 均有详细逐变量分析

## 亮点

1. **领域创新**：首次将图神经网络天气预报框架（GraphCast 风格）迁移到空间天气领域，弥合了大气天气 ML 与空间天气预报之间的鸿沟
2. **双模式预报**：同一框架支持确定性和概率性预报，概率版本通过条件 VAE + CRPS 微调实现集合预报
3. **不确定性量化**：为空间天气预报系统增加了当前大部分运行系统所缺失的不确定性量化能力，且不确定性正确定位在物理活跃区域（弓激波、磁尾等）
4. **计算效率**：在保持混合 Vlasov 模拟物理保真度的同时实现百倍以上加速
5. **残差预测 + 边界强制**：简单而有效的工程设计，降低学习难度并处理开放边界

## 局限性

1. **训练数据极其有限**：仅 10 分钟训练数据，是概率预报校准不佳和产生伪影的首要原因
2. **二维模拟**：仅在 2D 子午面（$x$-$z$ 平面）验证，未扩展到三维全球模拟
3. **欠散（Underdispersion）**：集合 Spread 始终小于 RMSE，预报系统未充分校准
4. **伪影**：$B_y$ 和 $v_y$ 在磁尾北叶区域出现虚假结构
5. **误差累积**：自回归预报固有的误差增长，长时间预报退化
6. **单一太阳风条件**：仅训练于固定参数（$\rho=1/\text{cm}^3$, $v=-750\,\text{km/s}$, $B_z=-5\,\text{nT}$），未覆盖多样真实太阳活动
7. **仅预测 VDF 矩**：训练在速度分布函数的矩上，未模拟完整的速度分布函数

### 作者提出的未来方向

- **三维扩展**：配合自适应网格细化（AMR），在关键区域提高分辨率，GNN 天然适合不规则网格
- **更大数据集**：跨多个太阳周期的多样条件，支持更大时间步长以缓解累积误差
- **物理约束**：如磁场散度为零（$\nabla \cdot \mathbf{B} = 0$），提高物理真实性
- **Thermalization 技术**：控制误差增长，延长稳定前推时间
- **基础模型方向**：空间天气数据可纳入 The Well 等多物理模拟数据集，构建跨尺度/跨系统的基础模型
- **完整 VDF 预测**：参考 5D 陀螺动力学代理模型，预测完整的离子速度分布函数

## 与相关工作的对比

| 方法 | 类型 | 特点 |
|------|------|------|
| BATS-R-US (MHD) | 物理模型 | 业务运行标准，但忽略离子动力学 |
| Vlasiator | 物理模型 | 最高保真度，混合 Vlasov，但计算成本禁止实时使用 |
| GraphCast | ML 大气天气 | 本文方法直接借鉴其图结构 |
| Graph-FM/EFM (Oskarsson 2024) | ML 大气天气 | 本文基础架构来源，做了空间天气领域适配 |
| 太阳风扰动集合 | 集合预报 | 扰动物理模型输入，本文用 ML 直接生成集合 |

本文的核心贡献在于：不是简单地将大气天气 ML 方法复制到空间天气，而是针对磁层模拟的特殊结构（不规则网格、内边界排除、开放边界强制）做了适配，并且首次在混合 Vlasov 数据（而非 MHD）上训练 ML 模型。

## My Notes

- **GNN 在科学模拟中的通用性**：encode-process-decode + 残差预测 + 多尺度图层次结构已成为物理模拟 ML 的通用范式（GraphCast → Graph-FM/EFM → 本文的空间天气迁移），架构可复用性极强
- **概率预报的架构选择**：条件 VAE + CRPS 微调是实用的概率预报方案，三阶段渐进训练（AE → VAE → CRPS fine-tune）值得参考，可推广到其他时空预测任务
- **数据受限场景的 ML**：仅 10 分钟训练数据，GNN 仍能学到有意义的动力学。这说明 inductive bias（图结构、残差预测、边界强制）在小数据下至关重要
- **基础模型趋势**：The Well、Multimodal Universe 等大规模物理模拟数据集 + Aurora/Surya 等基础模型，空间天气是天然的下一个目标领域
- **分类疑问**：本文被分在 image_generation 目录下，但实际是科学计算/时空预测任务，与图像生成关系不大。如有重分类需求可移至更合适的类别
- **与大气天气 ML 的差异**：空间天气的特殊性在于不规则内边界（地球）、开放边界的日侧强制、以及 Vlasov 方程的动力学复杂度远高于 Navier-Stokes

## 评分
- 新颖性: ⭐⭐⭐⭐ (领域迁移创新，非方法创新)
- 实验充分度: ⭐⭐⭐ (数据极少，但消融和对比完整)
- 写作质量: ⭐⭐⭐⭐ (清晰规范)
- 价值: ⭐⭐⭐⭐ (为空间天气 ML 开辟新方向)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting](a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)
- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)
- [\[ICLR 2026\] Weight-Space Linear Recurrent Neural Networks](../../ICLR2026/time_series/weight-space_linear_recurrent_neural_networks.md)
- [\[NeurIPS 2025\] OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales](omnicast_a_masked_latent_diffusion_model_for_weather_forecasting_across_time_sca.md)
- [\[NeurIPS 2025\] RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting](rivermamba_a_state_space_model_for_global_river_discharge_and_flood_forecasting.md)

</div>

<!-- RELATED:END -->
