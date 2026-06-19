---
title: >-
  [论文解读] I-INR: Iterative Implicit Neural Representations
description: >-
  [AAAI 2026][自动驾驶][隐式神经表示] 提出 I-INR（Iterative Implicit Neural Representations），一个即插即用的迭代精修框架，通过引入轻量级 FeedbackNet 和 FuseNet 模块（仅增加 0.5-2% 参数），对信号进行渐进式多步重建，有效缓解 INR 的频谱偏差问题，在图像拟合、超分辨率、去噪和 3D 占位预测等任务上均显著超越基线。
tags:
  - "AAAI 2026"
  - "自动驾驶"
  - "隐式神经表示"
  - "迭代精修"
  - "高频重建"
  - "去噪鲁棒性"
  - "即插即用框架"
---

# I-INR: Iterative Implicit Neural Representations

**会议**: AAAI 2026  
**arXiv**: [2504.17364](https://arxiv.org/abs/2504.17364)  
**代码**: [https://github.com/optimizer077/I-INR](https://github.com/optimizer077/I-INR)  
**领域**: 计算机视觉 / 信号表示  
**关键词**: 隐式神经表示, 迭代精修, 高频重建, 去噪鲁棒性, 即插即用框架

## 一句话总结

提出 I-INR（Iterative Implicit Neural Representations），一个即插即用的迭代精修框架，通过引入轻量级 FeedbackNet 和 FuseNet 模块（仅增加 0.5-2% 参数），对信号进行渐进式多步重建，有效缓解 INR 的频谱偏差问题，在图像拟合、超分辨率、去噪和 3D 占位预测等任务上均显著超越基线。

## 研究背景与动机

### INR 的核心挑战

隐式神经表示（INR）用神经网络（通常是 MLP）将空间/时间坐标直接映射到信号属性（如像素强度、颜色、3D 占位），具有分辨率无关、紧凑编码和无缝插值的优势。然而 INR 面临三个持续性挑战：

**频谱偏差（Spectral Bias）**：使用 L1/L2 损失优化时，INR 天然倾向学习低频分量，导致高频细节丢失

**噪声鲁棒性差**：现有方法通常假设输入干净完整，面对噪声和遮挡时表现退化

**泛化能力有限**：单次拟合的结果难以泛化到未见过的分辨率或退化条件

### 现有缓解策略及其不足

- **位置编码方案**（如 Fourier features）：通过正交 Fourier 基注入高频信号，但不适应噪声场景
- **替代激活函数**（如 SIREN 的正弦、WIRE 的 Gabor 小波、Gauss 的高斯）：能更好地捕获高频结构，但高精度与噪声鲁棒性难以兼顾
- **WIRE**：使用 Gabor 小波激活提升噪声鲁棒性，但仍有改进空间

### 迭代方法的启发

扩散模型等迭代方法在图像修复、视频生成等领域取得巨大成功——通过多步逆转退化过程产生高质量样本。然而，**迭代策略在 INR 领域几乎未被探索**。

**核心动机**：借鉴迭代精修的思想，将信号重建从单次预测（one-shot）变为多步渐进精修，从而同时提升高频保持能力和噪声鲁棒性。

## 方法详解

### 整体框架

I-INR 将信号重建建模为从初始状态 $\mathcal{Z}$（t=1）到最终重建 $\mathcal{I}(x)$（t=0）的迭代过程，使用步长 $\delta$ 逐步逼近。整体架构包含三个组件：
- **Backbone**：任意现有 INR 架构（如 SIREN、WIRE、Gauss），仅执行一次前向传播
- **FeedbackNet**：轻量级 MLP（2层，宽度30），整合中间状态和时间条件
- **FuseNet**：轻量级 MLP（2层，宽度100），融合 Backbone 和 FeedbackNet 特征

### 关键设计

#### 1. **迭代重建的数学建模**

前向过程将目标信号与初始状态线性插值：

$$g(x)_t = \mathcal{I}(x)(1-t) + \mathcal{Z}t, \quad t \in [0,1]$$

重建过程通过条件期望迭代更新：

$$\hat{g}(x)_{t-\delta} = \frac{\delta}{t}\mathbb{E}[g(x)_0 | \hat{g}(x)_t] + (1 - \frac{\delta}{t})\hat{g}(x)_t$$

初始状态 $\mathcal{Z}$ 从标准正态分布采样（实验证明优于全零或全一初始化）。

**设计动机**：借鉴 InDI（Inversion by Direct Iteration）的思想——将困难的逆问题分解为一系列更简单的子问题，每步更新仅需估计条件期望。该方法从一个较差的初始猜测逐步过渡到精确重建，自然地实现了粗到细的多尺度学习。

#### 2. **训练目标**

训练隐式神经网络 $f_\theta$ 直接预测干净目标：

$$\min_\theta \mathbb{E}_{x,t,n}\|f_\theta(\tilde{g}(x)_t, x, t) - \mathcal{I}(x)\|_2^2$$

其中中间状态添加小扰动以保证正则性：

$$\tilde{g}(x)_t = (1-t)\mathcal{I}(x) + t\mathcal{Z} + \varepsilon t n$$

$\varepsilon$ 经验设为 0.1，$n \sim \mathcal{N}(0,1)$。

**设计动机**：小扰动项 $\varepsilon tn$ 满足正则性要求，保证推理时重建过程的稳定性。同时，随机时间步 $t$ 的采样使网络学会处理从完全模糊到几乎清晰的各种中间状态。

#### 3. **网络架构设计**

信息流形式化为：

$$f_\theta(\hat{g}(x)_t, x, t) = \text{FuseNet}(\text{concat}(\mathbf{f}, \mathbf{b})) \odot \mathbf{b}$$

其中 $\mathbf{b} = \text{Backbone}(x)$，$\mathbf{f} = \text{FeedbackNet}(\hat{g}(x)_t, x, t)$。

关键效率设计：
- **Backbone 仅执行一次**：提取基础特征后在所有迭代步中复用
- **FeedbackNet 和 FuseNet 极轻量**：每步迭代仅增加 0.43 GFLOPs（Backbone 为 106.8 GFLOPs）
- **乘法融合**：使用逐元素乘法（$\odot$）而非加法融合，实验证明效果更好

**设计动机**：乘法融合让 FuseNet 的输出充当对 Backbone 特征的调制门控——可以选择性地增强或抑制特定特征通道，适合高频细节的精细调控。

### 训练与推理策略

**训练**（Algorithm 1）：
1. 采样初始状态 $\mathcal{Z} \sim \mathcal{N}(0,1)$
2. 随机采样坐标 $x$，时间步 $t \sim \mathcal{U}(0,1)$，噪声 $n \sim \mathcal{N}(0,I)$
3. 构造中间状态并计算重建损失
4. 梯度下降更新参数

**推理**（Algorithm 2）：
1. 从 $\hat{g}(x)_1 = \mathcal{Z}$ 开始
2. 以步长 $\delta$ 从 $t=1$ 迭代到 $t=0$
3. 每步：$\hat{g}(x)_{t-\delta} = \frac{\delta}{t}f_\theta(\hat{g}(x)_t, x, t) + (1 - \frac{\delta}{t})\hat{g}(x)_t$

默认使用 2 步推理（$\delta = 0.5$），平衡质量和效率。

## 实验关键数据

### 主实验

**图像拟合**（Kodak 数据集，3层 MLP，300 神经元/层）：

| 基线 | PSNR | SSIM | I-版本 PSNR | I-版本 SSIM | 提升 |
|------|------|------|------------|------------|------|
| SIREN | 34.57 | 0.931 | **37.53** | **0.961** | +2.96 |
| WIRE | 32.15 | 0.898 | **33.73** | **0.924** | +1.58 |
| Gauss | 31.33 | 0.880 | **31.93** | **0.884** | +0.60 |

**超分辨率**（DIV2K 数据集，仅在 2× 上训练）：

| 尺度 | 方法 | SIREN PSNR/LPIPS | WIRE PSNR/LPIPS | Gauss PSNR/LPIPS |
|------|------|-----------------|-----------------|------------------|
| 2× | Baseline | 26.77/0.414 | 26.14/0.457 | 25.19/0.538 |
| 2× | **I-版本** | **27.64/0.367** | **27.21/0.388** | **26.82/0.363** |
| 4× | Baseline | 25.03/0.597 | 24.57/0.618 | 23.85/0.673 |
| 4× | **I-版本** | **25.53/0.575** | **25.78/0.496** | **25.18/0.620** |

注：模型仅在 2× 上训练，4× 是零样本泛化测试。

**图像去噪**（DIV2K，Poisson 噪声）：

| 基线 | PSNR | LPIPS | I-版本 PSNR | I-版本 LPIPS | PSNR 提升 |
|------|------|-------|------------|-------------|----------|
| SIREN | 23.86 | 0.604 | **25.59** | **0.540** | +1.73 |
| WIRE | 23.32 | 0.746 | **24.76** | **0.490** | +1.44 |
| Gauss | 23.10 | 0.783 | **24.20** | **0.533** | +1.10 |

I-SIREN 在去噪上实现了高达 **+3.25 dB PSNR** 的提升（最佳情况）。

**3D 占位重建**（IoU）：

| 方法 | SIREN | WIRE | Gauss |
|------|-------|------|-------|
| Baseline | 0.9840 | 0.9917 | 0.9855 |
| **I-版本** | **0.9934** | **0.9950** | **0.9967** |

### 消融实验

**FeedbackNet 和 FuseNet 的影响**（I-SIREN，Kodak 图像拟合）：

| FeedbackNet | FuseNet | PSNR |
|-------------|---------|------|
| ✗ | ✗ | 20.27 |
| ✓ (1×) | ✗ | 31.88 |
| ✓ (1×) | ✓ (1×) | **37.53** |
| ✓ (2×) | ✓ (1×) | 37.25 |
| ✓ (1×) | ✓ (2×) | 37.77 |
| ✓ (2×) | ✓ (2×) | 37.56 |

**重建步数的影响**：
- PSNR 在 steps=4 达到峰值，从 1 到 2 步提升最大
- 超过 4 步后 PSNR 略降，但感知质量（LPIPS）持续改善——体现了保真度-感知权衡（Perception-Distortion Tradeoff）
- 对去噪任务，PSNR 在 steps=2 达峰，步数过多会开始重建噪声

**推理计算成本**：

| 组件 | GFLOPs |
|------|--------|
| Backbone（仅执行1次） | 106.8 |
| 每步精修（FeedbackNet+FuseNet） | 0.43 |
| 总计（steps=2） | 107.6 |
| 额外开销 | ~0.8% |

**训练时间对比**（Kodak，2000迭代，RTX 4090）：

| 指标 | SIREN | I-SIREN | 增幅 |
|------|-------|---------|------|
| 训练时间 | 基线 | +6% | 微量 |
| 推理延迟/图 | 基线 | +3ms | 微量 |

### 关键发现

1. 迭代改进在**所有三种激活函数**（SIREN、WIRE、Gauss）上均带来一致提升，证明框架的通用性
2. 高斯初始状态始终优于全零/全一初始状态——噪声提供更好的探索起点
3. I-SIREN 在 3 层深度就超过了 5 层 SIREN，证明迭代比增加深度更高效
4. 乘法融合一致优于自适应加权融合，说明门控机制比线性组合更适合精修任务
5. 统计显著性检验（5种随机种子，Wilcoxon signed-rank test）：所有改进 p < 0.00001

## 亮点与洞察

1. **即插即用是最大卖点**：不改变原有 INR 架构，仅添加两个微型模块即可获得提升，工程友好度极高
2. **计算开销几乎为零**：0.8-2% 的额外 FLOPs 换来高达 +2.96 dB PSNR，投入产出比极高
3. **从扩散模型到 INR 的巧妙迁移**：将 InDI 的迭代思想适配到坐标网络，概念简单但效果显著
4. **跨任务/跨分辨率泛化**：仅在 2× 上训练的模型能直接用于 4× 超分，展现了学到的精修能力的泛化性
5. **感知-保真度权衡的自然涌现**：步数增加时 PSNR 下降但 LPIPS 改善，说明迭代过程自然地从低频到高频逐步精修

## 局限与展望

1. 当推理步数过多时会出现过拟合现象（去噪时重建噪声），需要合理选择步数
2. FeedbackNet 和 FuseNet 的架构（层数、宽度）设计较为简单，可能还有优化空间
3. 未与基于位置编码的方法（如 DINER、FINER）结合测试
4. 三维场景仅测试了小规模模型（3个物体），未在 NeRF 级别的大规模场景上验证
5. 迭代步数的最优值因任务而异（拟合=4，去噪=2），缺乏自适应步数选择机制

## 相关工作与启发

- **与扩散模型的关系**：共享迭代精修理念，但 I-INR 是在坐标空间而非像素空间操作，更紧凑
- **与 InDI 的关系**：直接采用 InDI 的迭代框架，但将其从图像修复任务迁移到 INR 信号表示
- **启发**：Backbone 一次前向 + 轻量精修的设计模式可推广到其他连续表示学习场景，如 NeRF 加速

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 迭代思想在 INR 中的首次系统应用，将 InDI 迁移到坐标网络
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4 个任务、3 种基线、完整消融、统计检验，极其全面
- **写作质量**: ⭐⭐⭐⭐ — 公式推导清晰，可视化丰富，实验设置透明
- **实用价值**: ⭐⭐⭐⭐⭐ — 即插即用、几乎零开销、代码开源，实用性极强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] NeMo-map: Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping](../../ICLR2026/autonomous_driving/nemo-map_neural_implicit_flow_fields_for_spatio-temporal_motion_mapping.md)
- [\[CVPR 2026\] Neural Distribution Prior for LiDAR Out-of-Distribution Detection](../../CVPR2026/autonomous_driving/neural_distribution_prior_for_lidar_ood_detection.md)
- [\[NeurIPS 2025\] Continuous Simplicial Neural Networks](../../NeurIPS2025/autonomous_driving/continuous_simplicial_neural_networks.md)
- [\[ICLR 2026\] Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](../../ICLR2026/autonomous_driving/spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)
- [\[CVPR 2025\] Neural Inverse Rendering from Propagating Light](../../CVPR2025/autonomous_driving/neural_inverse_rendering_from_propagating_light.md)

</div>

<!-- RELATED:END -->
