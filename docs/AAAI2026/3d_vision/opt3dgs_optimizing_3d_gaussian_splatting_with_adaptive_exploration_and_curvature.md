---
title: >-
  [论文解读] Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation
description: >-
  [AAAI 2026][3D视觉][3D高斯泼溅] 提出 Opt3DGS 框架，将 3DGS 训练分为探索和利用两阶段：探索阶段用自适应加权 SGLD 逃离局部最优，利用阶段用局部拟牛顿 Adam 优化器实现精确收敛，在不修改高斯表示的前提下达到 SOTA 渲染质量。 3D 高斯泼溅（3DGS）通过显式高斯基元建模场景…
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "3D高斯泼溅"
  - "非凸优化"
  - "随机梯度朗日万动力学"
  - "拟牛顿法"
  - "新视角合成"
---

# Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation

**会议**: AAAI 2026  
**arXiv**: [2511.13571](https://arxiv.org/abs/2511.13571)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 3D高斯泼溅, 非凸优化, 随机梯度朗日万动力学, 拟牛顿法, 新视角合成

## 一句话总结

提出 Opt3DGS 框架，将 3DGS 训练分为探索和利用两阶段：探索阶段用自适应加权 SGLD 逃离局部最优，利用阶段用局部拟牛顿 Adam 优化器实现精确收敛，在不修改高斯表示的前提下达到 SOTA 渲染质量。

## 研究背景与动机

3D 高斯泼溅（3DGS）通过显式高斯基元建模场景，在新视角合成中取得了卓越表现。然而，优化高斯基元以重建辐射场本质上是一个**高度非凸的优化问题**，面临两个核心挑战：

### 挑战一：局部最优陷阱

原始 3DGS 使用启发式规则（自适应密度控制 ADC）进行高斯的克隆、分裂和剪枝，但这些规则缺乏鲁棒性。后续工作 3DGSMCMC 将优化建模为 SGLD（随机梯度朗日万动力学）过程，引入随机噪声促进探索。但 3DGSMCMC 存在**聚类效应**：

- 新增高斯的位置从基于不透明度的概率分布 $\pi(x)$ 中 i.i.d. 采样
- 早期发现的主导结构变得高不透明度后，后续采样严重偏向这些区域
- 导致已重建好的区域过度积累高斯，而几何复杂/探索不足的区域得不到覆盖
- 从 MCMC 角度看，这种偏差将采样链局限在单一后验模式中

### 挑战二：收敛质量不足

现有 3DGS 方法普遍使用一阶优化器（Adam），缺乏曲率信息，在训练后期难以精确收敛到最优点。虽然有工作尝试使用牛顿法或 LM 算法，但计算量大（需 Hessian 矩阵或其近似）。

**核心思路**：将训练过程分为**探索（Exploration）**和**利用（Exploitation）**两阶段，分别解决上述两个问题。

## 方法详解

### 整体框架

- **探索阶段**（前 29,000 次迭代）：使用自适应加权 SGLD（AW-SGLD）增强全局搜索，逃离局部最优
- **利用阶段**（最后 1,000 次迭代）：使用局部拟牛顿 Adam（LQNAdam）进行精确的曲率感知收敛

总训练 30,000 次迭代，高斯基元增长率 5%。

### 关键设计

1. **自适应加权 SGLD（AW-SGLD）**

   **核心思路**：受"平直方图"（flat histogram）原理启发，通过展平后验分布来降低模式间的能量壁垒，使模型更容易跨越局部最优。

   将高斯基元的配置视为概率分布：
    $P(g) \propto \exp\left(-\frac{\mathcal{L}_{total}(g)}{\tau}\right)$

   将样本空间按能量水平划分为 $m$ 个子区域 $\mathcal{G}_n = \{g: u_{n-1} < \mathcal{L}_{total}(g) < u_n\}$。

   构造展平分布 $\rho(g)$：
    $\rho(g) \propto \frac{P(g)}{\Psi^\zeta(\Theta, \mathcal{L}_{total}(g))}$
   其中 $\zeta > 0$ 控制展平程度，$\Psi$ 是基于能量的分段指数插值加权函数，权重向量 $\Theta$ 通过随机近似在线更新。

   展平分布引入额外的**梯度乘子** $\nu$：
    $\nu = 1 + \zeta\tau \frac{\log\theta(J(g)) - \log(\theta(J(g)-1) \vee 1)}{\Delta u}$

   将梯度乘子融入 SGLD 更新：
    $g_k \leftarrow g_{k-1} - \lambda_{lr} \cdot \nu \cdot \nabla_g \mathbb{E}[\mathcal{L}_{total}(g_{k-1})] + \lambda_{noise} \cdot \epsilon$

   权重向量 $\Theta$ 的更新使用随机近似：
    $\theta_k(i) = \theta_{k-1}(i) + \lambda_\theta \theta_{k-1}^\zeta(J(g_k)) \cdot (1_{i=J(g_k)} - \theta_{k-1}(i))$

   **设计动机**：直接增大噪声 $\lambda_{noise}$ 不鲁棒（场景复杂度不同），自适应加权方法根据能量分布自动调整探索强度，通过展平后验实现更均匀的模式探索。高能量区域（重建差的区域）获得更大的探索促进。

2. **局部拟牛顿 Adam 优化器（LQNAdam）**

   **核心思路**：在利用阶段，对每个高斯基元独立应用 L-BFGS 估计拟牛顿方向，作为 Adam 的伪梯度输入，获得曲率感知的更新方向。

   具体步骤：
    - 对每个高斯基元的位置 $\mu$ 独立执行 L-BFGS（历史长度 $K=5$），估计拟牛顿方向 $\mathbb{D}$
    - 将 $\mathbb{D}$ 作为伪梯度输入 Adam，计算最终更新方向 $\text{Adam}(\mathbb{D})$
    - 在 MCMC 框架下的更新规则：
    $\mu_{t+1} = \mu_t - \lambda_{lr} \cdot \text{Adam}(\mathbb{D}) + \lambda_{noise} \cdot \epsilon_\mu$

   **关键设计选择**：
    - "局部"：每个高斯基元独立处理，可在 CUDA 上并行
    - 无需线搜索：用 Adam 代替传统拟牛顿方法的线搜索，保持鲁棒性
    - L-BFGS 不需要计算 Hessian 矩阵，与各种损失函数兼容
    - 利用阶段将 L1 损失替换为 L2 损失，禁用梯度乘子 $\nu$

   **设计动机**：基于 3DGS² 的观察，位置属性对渲染质量影响最大，且高斯基元间弱耦合，因此适合对位置做独立的拟牛顿优化。

3. **探索→利用的切换策略**

    - 在第 29,000 次迭代切换
    - 探索阶段使用 AW-SGLD，前 2,500 次迭代作为 warm-up 稳定能量估计
    - 利用阶段禁用梯度乘子，切换到 L2 损失和 LQNAdam
    - 展平系数 $\zeta = 0.75$ 在所有数据集上通用

### 损失函数 / 训练策略

与 3DGSMCMC 相同的损失函数：
$$L_{total} = (1-\lambda_{ssim}) \times L_1 + \lambda_{ssim} \times L_{ssim} + \lambda_o \sum_i |o_i|_1 + \lambda_\Sigma \sum_{ij} |\sqrt{\text{eig}_j(\Sigma_i)}|_1$$

后两项分别是不透明度稀疏正则化和协方差矩阵尺度约束。利用阶段将 L1 替换为 L2。

能量区间：大多数场景 [0.0, 0.2]，特殊场景（Train）[0.0, 0.3]，划分为 200 个均匀 bin。

## 实验关键数据

### 主实验

**标准设置（SfM 初始化）**：

| 方法 | MipNeRF360 PSNR/SSIM/LPIPS | T&T PSNR/SSIM/LPIPS | DeepBlending PSNR/SSIM/LPIPS |
|------|---------------------------|---------------------|----------------------------|
| 3DGS | 28.69/0.870/0.182 | 23.14/0.841/0.183 | 29.41/0.903/0.243 |
| 3DGSMCMC | 29.89/0.900/0.190 | 24.29/0.860/0.190 | 29.67/0.900/0.320 |
| SSS | 29.90/0.893/0.145 | 24.87/0.873/0.138 | 30.07/0.907/0.247 |
| **Opt3DGS** | **29.96**/0.897/**0.143** | 24.80/**0.875**/**0.139** | **30.09**/**0.911**/**0.229** |

9 个指标中最优 5 个，次优 4 个。相比 3DGSMCMC，在 T&T 上 LPIPS 改善 26.84%。

**随机初始化（无 SfM）**：

| 方法 | MipNeRF360 PSNR/SSIM/LPIPS | T&T PSNR/SSIM/LPIPS | DeepBlending PSNR/SSIM/LPIPS |
|------|---------------------------|---------------------|----------------------------|
| 3DGS | 27.89/0.840/0.260 | 21.93/0.800/0.270 | 29.55/0.900/0.330 |
| 3DGSMCMC | 29.72/0.890/0.190 | 24.21/0.860/0.190 | 29.71/0.900/0.320 |
| **Opt3DGS** | **29.78/0.893/0.149** | **24.39/0.865/0.151** | **29.90/0.905/0.236** |

在所有 9 个指标上均为最优，证明了即使初始状态很差，优化框架仍能引导模型找到高质量解。

### 消融实验

| 配置 | Train PSNR | Truck PSNR | Train 时间 | Truck 时间 |
|------|-----------|-----------|----------|----------|
| Baseline (3DGSMCMC) | 22.47 | 26.11 | 11min | 22min |
| + AW-SGLD | 22.74 (+0.27) | 26.49 (+0.38) | 12min | 22min |
| + AW-SGLD + LQNAdam | **23.01** (+0.54) | **26.61** (+0.50) | 12min | 23min |

两个组件均有贡献，AW-SGLD 贡献更大，LQNAdam 进一步精细化。额外计算开销 < 1 分钟。

**展平系数 $\zeta$ 的影响**：$\zeta = 0.75-0.8$ 为最佳范围，过小探索不足，过大可能导致训练不稳定。

### 关键发现

- 纯优化改进（不修改高斯表示）可达到甚至超过修改表示的方法（SSS）
- 随机初始化条件下优势更明显，证明增强探索能力在困难条件下尤为重要
- 高分辨率输入下（更复杂的后验地形），Opt3DGS 的优势持续存在
- 有限高斯数量下，Opt3DGS 依然表现优异，说明优化效率提升可弥补表示能力不足
- 额外计算开销极小（约 1 分钟）

## 亮点与洞察

1. **优化视角的纯粹性**：本文完全从优化角度改进 3DGS，不修改高斯表示或引入辅助网络，证明了"优化比表示更重要"的观点
2. **探索-利用框架的可迁移性**：这种两阶段优化框架独立于表示方式，可作为即插即用模块替换其他 3DGS 系统的优化组件
3. **平直方图原理在 3DGS 中的应用**：将统计物理/MCMC 中的高级采样技术（本用于模拟蛋白质折叠等）引入三维重建，跨领域启发
4. **拟牛顿方向 + Adam 的巧妙结合**：LQNAdam 保留了 Adam 的鲁棒性，同时引入了曲率信息，避免了传统二阶方法的线搜索开销
5. **在困难条件（随机初始化、高分辨率、少高斯）下优势更明显**：说明增强的探索能力在解空间复杂时价值最大

## 局限与展望

- 展平系数 $\zeta$ 和能量区间仍需手动设置，不同场景可能需要微调
- 利用阶段仅 1,000 次迭代可能不足以充分利用曲率信息
- L-BFGS 历史长度固定为 5，未探索自适应调整的可能性
- 仅对位置属性使用拟牛顿方向，未扩展到其他高斯参数（颜色、不透明度等）
- 与 SSS 在部分指标上互有胜负，可考虑将 Opt3DGS 的优化策略与更好的表示结合

## 相关工作与启发

- **3DGSMCMC (2024)**：将 3DGS 优化建模为 SGLD/MCMC 的先驱工作，本文的直接基线
- **SSS (2025)**：改进高斯表示（Student's T 分布）+ SGHMC 采样，与本文互补
- **Wang-Landau 算法 (2001)**：平直方图原理的源头，本文的理论灵感
- **L-BFGS**：经典的有限内存拟牛顿优化方法，本文创新性地将其用于每个高斯基元的独立优化
- **启发**：优化策略的改进是正交于表示改进的维度，二者可组合使用；统计力学中的采样方法值得更多关注

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （将平直方图原理和拟牛顿方向引入 3DGS 优化，视角独特）
- 实验充分度: ⭐⭐⭐⭐⭐ （标准/随机初始化/高分辨率/少高斯，条件覆盖全面）
- 写作质量: ⭐⭐⭐⭐⭐ （理论推导严谨，贝叶斯视角分析清晰，图示直观）
- 价值: ⭐⭐⭐⭐⭐ （框架通用性强，可作为 3DGS 优化的标准组件）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction](oceansplat_object-aware_gaussian_splatting_with_trinocular_view_consistency_for_.md)
- [\[CVPR 2025\] DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds](../../CVPR2025/3d_vision/dashgaussian_optimizing_3d_gaussian_splatting_in_200_seconds.md)
- [\[AAAI 2026\] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting](gt2-gs_geometry-aware_texture_transfer_for_gaussian_splatting.md)
- [\[CVPR 2026\] Curvature-Aware Captioning: Leveraging Geodesic Attention for 3D Scene Understanding](../../CVPR2026/3d_vision/curvature-aware_captioning_leveraging_geodesic_attention_for_3d_scene_understand.md)
- [\[CVPR 2026\] BA-GS: Bayesian Adaptive Gaussian Splatting for SFM-Free 3D Reconstruction](../../CVPR2026/3d_vision/ba-gs_bayesian_adaptive_gaussian_splatting_for_sfm-free_3d_reconstruction.md)

</div>

<!-- RELATED:END -->
