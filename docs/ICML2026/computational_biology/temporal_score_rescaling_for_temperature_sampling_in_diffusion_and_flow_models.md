---
title: >-
  [论文解读] Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models
description: >-
  [ICML 2026][计算生物][temperature sampling] 通过对预训练扩散/流模型的 score 输出乘以一个仅依赖时间步、变量 $k$ 与 $\sigma$ 的解析重标缩因子 $r_t$，即可在推理阶段把采样分布"局部"地变得更尖或更平，而无需任何微调，对 DDIM 等确定性采样器也完全兼容。
tags:
  - "ICML 2026"
  - "计算生物"
  - "temperature sampling"
  - "score rescaling"
  - "扩散模型"
  - "flow matching"
  - "training-free"
---

# Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models

**会议**: ICML 2026  
**arXiv**: [2510.01184](https://arxiv.org/abs/2510.01184)  
**代码**: https://temporalscorerescaling.github.io  
**领域**: 计算生物  
**关键词**: temperature sampling, score rescaling, diffusion, flow matching, training-free  

## 一句话总结
通过对预训练扩散/流模型的 score 输出乘以一个仅依赖时间步、变量 $k$ 与 $\sigma$ 的解析重标缩因子 $r_t$，即可在推理阶段把采样分布"局部"地变得更尖或更平，而无需任何微调，对 DDIM 等确定性采样器也完全兼容。

## 研究背景与动机
**领域现状**：扩散模型（DDPM）与流匹配模型（Flow Matching）通过学习噪声分布 $p(\mathbf{x}_t)$ 的 score $\nabla\log p_t(\mathbf{x})$ 进行采样，已经成为图像生成、深度估计、姿态预测、机器人策略、蛋白设计等任务的通用范式。但实际部署常常希望偏离训练分布——例如深度估计想要"更可能"的预测，而图像生成想要"更多样"的样本。

**现有痛点**：传统的温度采样在扩散模型上非常难做。Classifier-Free Guidance（CFG）虽然能换取多样性 vs 似然，但本质并不等价于温度缩放，而且只能用在条件模型；Likelihood-weighted 微调（Shih et al., 2023）需要重新训练，对大模型不现实；Langevin/MCMC 校正（Du et al., 2023）会让推理开销翻 5 倍以上；最常用的 Constant Noise Scaling（CNS）只是把 SDE 里的噪声项乘一个常数 $1/\sqrt{k}$，不仅不能用在 ODE 采样器上，还会在高噪声步过度压制探索、在低噪声步又压制不足，导致 mode collapse。

**核心矛盾**：温度采样要的是"改变模式宽度但不改变模式权重"，而所有现有 training-free 方法在跨时间步上用了一个常数的"探索强度"，自然无法精确实现这种局部缩放。

**本文目标**：设计一种 (a) 无需训练、(b) 兼容 DDIM/Euler 等确定性采样器、(c) 不增加 score 评估次数、(d) 在简单分布上有可证明性的温度采样机制。

**切入角度**：作者注意到，如果原始数据是 isotropic Gaussian 或其混合，那么"温度缩放后数据"在前向加噪后的 score 与原 score 之间存在一个解析的、随时间变化的线性比值。换言之，对一个已经训练好的 score 网络，只要在推理时乘上一个 $r_t$，就能等价地把采样目标分布换成方差被 $1/k$ 缩放的版本。

**核心 idea**：用 $r_t(k,\sigma)=\frac{\eta_t\sigma^2+1}{\eta_t\sigma^2/k+1}$（$\eta_t=\alpha_t^2/\sigma_t^2$ 为信噪比）这一时间相关的标量去重标缩 score，把"全局温度"问题转化为"局部温度"问题，避开 mode 漂移。

## 方法详解

### 整体框架
方法称为 **Temporal Score Rescaling (TSR)**，要解决的是"如何在不重新训练的前提下，让预训练扩散/流模型在推理时变得更尖（偏似然）或更平（偏多样）"。它的做法只在推理时插一步：拿到任何预训练扩散或流匹配模型的输出，先把噪声预测 $\boldsymbol\epsilon_\theta$ 或速度 $\boldsymbol v_\theta$ 等价换算到 score 视角，乘上一个仅依赖时间步的标量重标缩因子 $r_t(k,\sigma)$，再把放大后的 score 喂回原本的采样器（DDPM/DDIM/Euler/Heun 都行）。对用户而言只多出两个超参 $k$（控制变尖/变平的强度）和 $\sigma$（控制从哪一步开始介入），和 CFG 一样"一次调好可重用"。

### 关键设计

**1. 单 Gaussian 上的解析 score 缩放公式：把温度操作压成一个标量**

TSR 的全部威力来自一个极简观察：对数据分布做温度缩放 $\Sigma\to\Sigma/k$ 这件事，可以等价地只作用在 score 上。在 stochastic interpolant 框架 $\mathbf{x}_t=\alpha_t\mathbf{x}_0+\sigma_t\boldsymbol\epsilon$ 下，若 $\mathbf{x}_0\sim\mathcal{N}(\boldsymbol\mu,\sigma^2\mathbf{I})$，则噪声分布的 score 有闭式 $\nabla\log p_t(\mathbf{x})=-(\mathbf{x}-\alpha_t\boldsymbol\mu)/(\alpha_t^2\sigma^2+\sigma_t^2)$。把数据方差换成 $\sigma^2/k$ 之后，这个表达式只有分母变了，于是温度前后两个 score 的比值就是一个解析标量 $r_t(k,\sigma)=\frac{\eta_t\sigma^2+1}{\eta_t\sigma^2/k+1}$（其中 $\eta_t=\alpha_t^2/\sigma_t^2$ 为信噪比），且 $k=1$ 时 $r_t\equiv1$ 恰好恢复原始 score。正因为它是闭式而非额外网络或迭代修正，方法才能对任何采样器"插即用"。

**2. 推广到混合高斯：从"全局温度"退守到"局部温度"**

真正让 TSR 区别于旧方法的，是它放弃了改变模式权重、只改模式宽度。作者证明当数据是"良好分离"的等方差高斯混合时，上面那个为单 Gaussian 推出的 $r_t$ 仍是真实 score 的一个有界近似——小 $t$ 时单一分量主导，误差呈指数衰减；大 $t$ 时分布近似纯噪声，误差呈多项式衰减，两端都趋零。其物理含义是 TSR 只压缩每个 mode 内部的方差、不动 mode 之间的相对权重，因此采样结果会均匀覆盖所有 mode 而不是塌向中心。这正好绕开了 CNS 那种"向中心模式塌缩"和 CFG 那种"模式权重失衡"的副作用。论文在 1D 高斯混合、2D checkerboard、Swiss-roll 上验证了这种"局部"特性：CNS 会丢掉边缘 mode，TSR 则全部保留。真实数据虽然不是显式 GMM，但任何足够平滑的分布在局部都可用 Gaussian 近似，这让公式具备跨任务通用性。

**3. 统一适配 score / $\epsilon$ / velocity 三种参数化：一套公式驱动异构模型**

为了让同一个 $r_t$ 既能驱动 DDPM/DDIM 这类噪声预测模型、又能驱动 Flow Matching 这类速度预测模型，作者把所有参数化都换算回 score。score 与噪声预测有线性关系 $\mathbf{s}_\theta=-\sigma_t^{-1}\boldsymbol\epsilon_\theta$，因此对扩散模型直接做 $\tilde{\boldsymbol\epsilon}_\theta=r_t(k,\sigma)\boldsymbol\epsilon_\theta$；对流匹配模型，速度与 score 满足 $\mathbf{s}_\theta=-(\alpha_t\boldsymbol v_\theta-\dot{\alpha}_t\mathbf{x})/[\sigma_t(\dot{\alpha}_t\sigma_t-\alpha_t\dot{\sigma}_t)]$，反代后得 $\tilde{\boldsymbol v}_\theta=\alpha_t^{-1}(r_t(k,\sigma)(\alpha_t\boldsymbol v_\theta-\dot{\alpha}_t\mathbf{x})+\dot{\alpha}_t\mathbf{x})$，$x_0$-prediction 与 $v$-prediction 也能同样代入。这样就不必为每种模型重搭采样栈，TSR 才能对 Stable Diffusion 2/3、Flux.1 dev、FoldingDiff、Marigold、Pi-0 等异构模型即插即用。

### 损失函数 / 训练策略
无新训练。$k$ 与 $\sigma$ 两个超参的选法：先固定 $\sigma=1.0$ 用二分搜索找最优 $k$，再固定 $k$ 用二分搜索找最优 $\sigma$，必要时迭代一次即可——比网格搜索高效得多，且经验上同一对 $(k,\sigma)$ 可跨同任务不同模型迁移。

## 实验关键数据

### 主实验

| 任务 / 数据集 | 模型 | 指标 | 默认采样 | + TSR | 备注 |
|---------------|------|------|----------|-------|------|
| Text-to-Image / LAION Aesth. 5k | SD3 | FID ↓ / CLIP ↑ | 24.77 / 32.82 | **22.81 / 33.05** | $k=0.93, \sigma=3.0$，越过 CFG 的 Pareto 前沿 |
| Text-to-Image | SD2 | FID / CLIP | 22.81 / 33.66 | **19.75 / 33.75** | 同一对 $(k,\sigma)$ 迁移 |
| Text-to-Image | Flux.1 dev | FID / CLIP | 53.99 / 31.97 | **51.79 / 32.14** | 同一对 $(k,\sigma)$ 迁移 |
| 深度估计 / ETH3D | Marigold(DDIM) | AbsRel ↓ / $\delta_1$ ↑ | 7.1 / 90.4 | **6.68 / 95.7** | 优于 CNS (6.82 / 95.6) |
| 深度估计 / NYUv2 | Marigold | AbsRel / $\delta_1$ | 6.0 / 95.9 | **5.84 / 96.0** | — |
| 姿态预测 / SYMSOL | $SO(3)$ diffusion | mean err (deg) ↓ | 0.444 | **0.356** ($k=7,\sigma=0.5$) | CNS 略优(0.350) 但仅适用 SDE |
| 机器人 / LIBERO-10 | Pi-0 (flow) | 平均成功率 ↑ | 81.7 | **82.8** ($k=1.25,\sigma=0.25$) | 10 任务里 6 升 2 平 |
| 蛋白生成 | FoldingDiff | designability ↑ | 0.22 | 明显改善且 FID 优于 CNS | 同时保多样性 |

### 消融实验

| 配置 | FID ↓ (SD3) | CLIP ↑ | 含义 |
|------|-------------|--------|------|
| 默认 Euler-ODE | 24.77 | 32.82 | 基线 |
| + CFG 调整 | Pareto 曲线 | Pareto 曲线 | 多样性/对齐 trade-off |
| + CNS | flow ODE 下不可用 | — | 仅适用 SDE 采样器 |
| + TSR ($k=0.93,\sigma=3.0$) | **22.81** | **33.05** | 越过 CFG 前沿，可与 CFG 叠加 |
| TSR $k<1$（更平） | 细节更多但接近噪声 | — | 生成任务收益 |
| TSR $k>1$（更尖） | 平滑、缺细节 | — | 预测类任务收益 |

### 关键发现
- "局部温度"概念是核心区别：CNS 用常数压制噪声会向中心 mode 塌缩，TSR 用时间相关因子只压模式内方差，从而保住所有 mode。
- 创作型任务（图像生成）偏好 $k<1$（更平），而预测型任务（深度、姿态、机器人、蛋白设计）偏好 $k>1$（更尖），统一框架两面通吃。
- $(k,\sigma)$ 的最优值在同任务不同模型之间稳定迁移，调参成本接近 CFG。
- 与 CFG 正交，可叠加使用；与 DDIM / Euler-ODE 等确定性采样器原生兼容，零额外 NFE。

## 亮点与洞察
- 用"温度缩放后的数据分布的 score 与原 score 存在闭式比值"这一极简观察，把一个一直被认为需要重新训练或 MCMC 的难题压成"乘一个标量"，是非常漂亮的"分析换算法"案例。
- 把温度从"全局"放宽到"局部"是关键概念跃迁：放弃改变 mode 权重，只改 mode 宽度，正好绕开 mode dropping，这种"自我设限换通用性"的取舍值得在其他生成控制问题里复用。
- 同一套 $r_t$ 公式覆盖 score/$\epsilon$/velocity 三种参数化，体现了 stochastic interpolant 视角的统一价值——这种"找最小公共变量"的设计哲学对其他扩散/流的推理时干预方法都有借鉴意义。

## 局限与展望
- 只能做"局部"温度：无法改变 GMM 中各 mode 的相对权重，对那些希望"压制错误 mode、放大正确 mode"的任务（例如条件采样里抑制虚假对象）效果有限。
- 理论保证仅限于良好分离的等方差高斯混合，对真实复杂分布只能依赖经验有效性，给出的误差界对小 $\sigma$ 数据较松。
- 需要为每个任务/模型分别调 $(k,\sigma)$，虽然成本类似 CFG，但仍无法零样本自适应；缺少基于数据驱动估计 $\sigma$ 的方法。
- 在 $k$ 不当（如机器人 LIBERO Task 2/8 上）反而掉点的现象提示：对基础模型本身已经表现差的任务，sharpening 可能放大错误，需要任务级动态控制。

## 相关工作与启发
- **vs CFG (Ho & Salimans 2022)**：CFG 仅对条件模型有效，需训练时 condition dropout；TSR 对无条件/条件模型均可，且与 CFG 正交可叠加。
- **vs CNS (Yim et al., 2023; Geffner et al., 2025)**：CNS 用常数 $1/\sqrt{k}$ 缩噪声项，只兼容 SDE，且会丢 mode；TSR 用时间相关 $r_t$ 缩 score，兼容 ODE 且保 mode。
- **vs Likelihood-weighted Finetuning (Shih et al., 2023)**：要重新训练并需访问训练数据；TSR 完全 training-free。
- **vs Langevin / MCMC Corrector (Du et al., 2023)**：推理开销翻数倍；TSR 零额外 NFE。
- **vs Variance-reduced sampling for proteins (Geffner et al., 2025)**：本质是 CNS 的特例，TSR 在 FoldingDiff 上同时拿到更高 designability 与更低 FID。

## 评分
- 新颖性: ⭐⭐⭐⭐ 思路精炼优雅，但"重标缩 score"的大方向并非首创，亮点在于解析公式与"局部温度"概念。
- 实验充分度: ⭐⭐⭐⭐⭐ 跨 5 个完全不同任务、覆盖 ODE 与 SDE 采样器，并系统对比 CFG/CNS。
- 写作质量: ⭐⭐⭐⭐ 推导清晰、图例充分；对超参直觉解释友好。
- 价值: ⭐⭐⭐⭐⭐ 零训练成本、零推理开销、几乎所有扩散/流模型立刻可用，落地价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](../../NeurIPS2025/computational_biology/consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[ICML 2026\] Flow Sampling: Learning to Sample from Unnormalized Densities via Denoising Conditional Processes](flow_sampling_learning_to_sample_from_unnormalized_densities_via_denoising_condi.md)
- [\[ICML 2026\] Stein Diffusion Guidance: Training-Free Posterior Correction for Sampling Beyond High-Density Regions](stein_diffusion_guidance_training-free_posterior_correction_for_sampling_beyond_.md)
- [\[ICLR 2026\] Scalable Spatio-Temporal SE(3) Diffusion for Long-Horizon Protein Dynamics](../../ICLR2026/computational_biology/scalable_spatio-temporal_se3_diffusion_for_long-horizon_protein_dynamics.md)
- [\[ICML 2026\] Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models](scalable_single-cell_gene_expression_generation_with_latent_diffusion_models.md)

</div>

<!-- RELATED:END -->
