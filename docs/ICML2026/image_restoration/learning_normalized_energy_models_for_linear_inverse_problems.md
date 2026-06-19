---
title: >-
  [论文解读] Learning Normalized Energy Models for Linear Inverse Problems
description: >-
  [ICML 2026][图像恢复][各向异性去噪] 作者把"线性逆问题"重写为"各向异性去噪"，并提出 Anisotropic Covariance Score Matching (A-CSM) 训出一个**归一化**的能量模型 $U_\theta(\mathbf{y},\boldsymbol{\Sigma})\approx -\log p(\mathbf{y}|\boldsymbol{\Sigma})$，单个模型即可处理 inpainting、deblurring、super-resolution，并解锁能量引导自适应调度、MALA 无偏校正和盲逆问题三大新能力。
tags:
  - "ICML 2026"
  - "图像恢复"
  - "各向异性去噪"
  - "协方差分数匹配"
  - "归一化能量模型"
  - "后验采样"
  - "盲逆问题"
---

# Learning Normalized Energy Models for Linear Inverse Problems

**会议**: ICML 2026  
**arXiv**: [2605.15487](https://arxiv.org/abs/2605.15487)  
**代码**: https://github.com/nzilberstein/Anisotropic-energy-Model (有)  
**领域**: 图像恢复 / 能量模型 / 扩散模型 / 线性逆问题  
**关键词**: 各向异性去噪, 协方差分数匹配, 归一化能量模型, 后验采样, 盲逆问题

## 一句话总结
作者把"线性逆问题"重写为"各向异性去噪"，并提出 Anisotropic Covariance Score Matching (A-CSM) 训出一个**归一化**的能量模型 $U_\theta(\mathbf{y},\boldsymbol{\Sigma})\approx -\log p(\mathbf{y}|\boldsymbol{\Sigma})$，单个模型即可处理 inpainting、deblurring、super-resolution，并解锁能量引导自适应调度、MALA 无偏校正和盲逆问题三大新能力。

## 研究背景与动机

**领域现状**：扩散模型已经是图像逆问题（去模糊、补全、超分）的主流先验，做法基本分两派——Bayes 派把预训练无条件扩散模型当 $p(\mathbf{x})$，在采样时用 Bayes 公式拼出 $\nabla\log p(\mathbf{x}_t|\mathbf{y})$；Regression 派则把 $(\mathbf{x},\mathbf{y})$ 配对直接学条件得分 $\nabla\log p(\mathbf{x}_t|\mathbf{y})$。

**现有痛点**：Bayes 派的 likelihood 项 $p(\mathbf{y}|\mathbf{x}_t)=\int p(\mathbf{y}|\mathbf{x})p(\mathbf{x}|\mathbf{x}_t)\mathrm{d}\mathbf{x}$ 是高维积分，只能用 DPS 之类的近似，引入采样偏差；Regression 派虽然规避了近似，但每换一种退化算子 $\mathbf{H}$ 都要重训一个模型，丢掉了先验/似然解耦的灵活性。更根本的问题是：两派都是 score-based，只学梯度不学**密度本身**，所以无法做归一化对数概率比较，也就无法做 MCMC 接受概率、能量引导调度、或 $\arg\max_{\boldsymbol{\Sigma}} p(\mathbf{y}|\boldsymbol{\Sigma})$ 这种盲估计任务。

**核心矛盾**：要同时拿到 (i) 跨退化共享的先验灵活性、(ii) 不依赖 likelihood 近似的无偏采样、(iii) 显式归一化密度——而既有 EBM-with-diffusion 工作（Du 2023, Thornton 2025）只支持 isotropic noise，覆盖不了线性逆问题对应的**各向异性**协方差。

**切入角度**：作者注意到 $\mathbf{y}=\mathbf{H}\mathbf{x}+\sigma\mathbf{v}$ 经 $\mathbf{H}^{-1}\mathbf{y}$ 重写后等价于 $\mathbf{y}=\mathbf{x}+\boldsymbol{\Sigma}^{1/2}\mathbf{v}'$，其中 $\boldsymbol{\Sigma}=\sigma^2\mathbf{H}^{-1}(\mathbf{H}^{-1})^\top$。于是"解一族线性逆问题"等价于"在一族协方差 $\boldsymbol{\Sigma}$ 上做去噪"，只要学一个 $\boldsymbol{\Sigma}$ 为条件的密度，就能一统所有线性退化。

**核心 idea**：将 Guth 2025 的 dual score matching 从 isotropic 推广到 anisotropic，新增一个**协方差分数**项 $\nabla_{\boldsymbol{\Sigma}}U_\theta$，由 Fokker-Planck 等式约束跨 $\boldsymbol{\Sigma}$ 的质量守恒，从而把"未归一化能量"训成"归一化能量"。

## 方法详解

### 整体框架
方法要解决的是"用一个模型统一处理一族线性逆问题"，关键转换是把退化观测 $\mathbf{y}=\mathbf{H}\mathbf{x}+\sigma\mathbf{v}$ 重写成各向异性去噪问题 $\mathbf{y}=\mathbf{x}+\boldsymbol{\Sigma}^{1/2}\mathbf{v}'$，于是只需学一个以噪声协方差 $\boldsymbol{\Sigma}$ 为条件的归一化能量 $U_\theta(\mathbf{y},\boldsymbol{\Sigma})\approx -\log p(\mathbf{y}|\boldsymbol{\Sigma})$。对 $\mathbf{y}$ 求梯度得 score $\nabla_\mathbf{y}U_\theta$，配合各向异性 Tweedie 公式 $\mathbb{E}[\mathbf{x}|\mathbf{y},\boldsymbol{\Sigma}]=\mathbf{y}-\boldsymbol{\Sigma}\nabla_\mathbf{y}U_\theta$ 做去噪/重建；对 $\boldsymbol{\Sigma}$ 求梯度得协方差分数，用于自适应调度和盲估计。架构上能量写成 $U_\theta(\mathbf{y},\boldsymbol{\Sigma})=\tfrac{1}{2}\langle\mathbf{y},\mathbf{s}_\theta(\mathbf{y},\boldsymbol{\Sigma})\rangle$，骨干是 EDM (Karras 2022) UNet。

### 关键设计

**1. 各向异性去噪分数匹配 A-DSM：把能量模型训成协方差感知去噪器**

标准 DSM 假设 $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}$，碰到 inpainting 这种沿不同方向噪声幅度差好几个数量级的协方差时，梯度尺度会爆炸或塌缩。A-DSM 用各向异性 Tweedie 公式把能量逼近成去噪器，损失取 $\ell_{\text{A-DSM}}=\mathbb{E}[\|\boldsymbol{\Sigma}^{1/2}\nabla_\mathbf{y}U_\theta(\mathbf{y},\boldsymbol{\Sigma})-\boldsymbol{\Sigma}^{-1/2}(\mathbf{y}-\mathbf{x})\|^2]$，两侧用 $\boldsymbol{\Sigma}^{1/2}$ 重加权使损失尺度无关，相当于 maximum-likelihood weighting 的各向异性推广。这种 scale-invariant 重加权让 $\nabla_\mathbf{y}U_\theta$ 能在 $[10^{-9},10^3]$ 的极宽噪声方差范围内稳定逼近 $\nabla_\mathbf{y}\log p(\mathbf{y}|\boldsymbol{\Sigma})$，是后面 any-order 生成和盲估计能跑通的前提。

**2. 各向异性协方差分数匹配 A-CSM：用 Fokker-Planck 约束把能量训成归一化的**

A-DSM 只学到 $\mathbf{y}$ 方向的梯度，跨不同 $\boldsymbol{\Sigma}$ 的能量值之间还差一个未知常数，无法做对数概率比较——这正是 isotropic 版本（Guth 2025, Yu 2025）的天花板。A-CSM 额外监督能量关于 $\boldsymbol{\Sigma}$ 的梯度：先证明协方差版 Tweedie 恒等式 $\nabla_{\boldsymbol{\Sigma}}U(\mathbf{y},\boldsymbol{\Sigma})=\mathbb{E}[\tfrac{1}{2}\boldsymbol{\Sigma}^{-1}-\tfrac{1}{2}\boldsymbol{\Sigma}^{-1}(\mathbf{y}-\mathbf{x})(\mathbf{y}-\mathbf{x})^\top\boldsymbol{\Sigma}^{-1}]$，再配一个 Frobenius 范数下的 scale-invariant 损失 $\ell_{\text{A-CSM}}$，整体目标 $\tfrac{1}{d}\ell_{\text{A-DSM}}+\tfrac{1}{d^2}\ell_{\text{A-CSM}}$。本质上是用 Fokker-Planck 连续性方程 $\nabla_{\boldsymbol{\Sigma}}p(\mathbf{y}|\boldsymbol{\Sigma})=\tfrac{1}{2}\nabla_\mathbf{y}^2 p(\mathbf{y}|\boldsymbol{\Sigma})$ 强约束所有边际密度一致，使归一化常数项不依赖 $\boldsymbol{\Sigma}$。训练后用 $U_\theta(\mathbf{y},\boldsymbol{\Sigma})-\mathbb{E}_\mathbf{y}[U_\theta|\boldsymbol{\Sigma}]+\tfrac{1}{2}\log\det(2\pi e\boldsymbol{\Sigma})$ 重归一化（大 $\boldsymbol{\Sigma}$ 时 $\mathbf{y}\sim\mathcal{N}(0,\boldsymbol{\Sigma})$ 充当锚点）。正是这步让模型能算 $p(\mathbf{y}|\boldsymbol{\Sigma}_T,\mathbf{y})/p(\mathbf{y}|\boldsymbol{\Sigma}_t,\mathbf{y})$ 这种概率比、做 MALA 接受概率和 $\arg\max_{\boldsymbol{\Sigma}}\log p(\mathbf{y}|\boldsymbol{\Sigma})$，这是它区别于一切纯 score 模型的根本所在。

**3. 双域协方差嵌入架构：一个 UNet 同时吃像素域和频域两类协方差**

完全通用的 $\boldsymbol{\Sigma}$ 有 $d(d-1)/2$ 个自由度，在 $d=64^2=4096$ 时内存直接爆炸，但 spatial-diagonal 已能覆盖 inpainting、frequency-diagonal 已能覆盖 deblur/SR，因此把条件限制在这两族、压到 $d$ 维向量并不损失实际表达力。空间协方差表示成空间变化噪声图 $\mathbf{e}_\ell\in\mathbb{R}^{c_\ell\times d_\ell}$，频域协方差表示成只调制通道的 $\mathbf{e}_\ell\in\mathbb{R}^{c_\ell}$，两个分支并行算出 embedding 后按 $\mathbf{x}_\ell\leftarrow\mathrm{SiLU}(\mathbf{x}_\ell\odot(1+\mathbf{e}_\ell))$ 注入每层。这一调制形式与 EDM 原生的 isotropic 通道 gain modulation 完全兼容，几乎不增加额外计算量，也让模型能直接继承现有 score architecture 在去噪上的 inductive bias。

### 损失函数 / 训练策略
总损失 $\mathcal{L}=\tfrac{1}{d}\ell_{\text{A-DSM}}+\tfrac{1}{d^2}\ell_{\text{A-CSM}}$。训练协方差按 0.5/0.5 在 spatial 类（中心/横向 box，box size 1~64）和 spectral 类（Gaussian deblur kernel size 8×8, $\sigma_g=0.8$；4× SR）之间采样。骨干 EDM UNet。所有方法（包括 baseline）统一用同一架构，区别仅在 $p(\boldsymbol{\Sigma})$ 和输入：Bayesian baseline 取 $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}$，Palette 把 measurement 与噪图 stack 作输入。采样最多 1000 NFE（CelebA inpainting 1200）。

## 实验关键数据

### 主实验
CelebA 64×64 和 ImageNet 64×64 上 inpainting (中心 45×45 box, $\sigma=10^{-4}$) 与 Gaussian deblurring (8×8 kernel, $\sigma=10^{-2}$)，对比 DPS、RED-Diff、DAPS、Palette。

| 数据集 / 任务 | 指标 | 本文 | DPS | RED-Diff | DAPS |
|---------------|------|------|-----|----------|------|
| CelebA Inpainting | LPIPS↓ | **0.093** | 0.110 | 0.100 | 0.098 |
| CelebA Inpainting | FID↓ | **34.57** | 36.76 | 47.82 | 45.76 |
| CelebA Deblurring | LPIPS↓ | **0.002** | 0.004 | 0.006 | 0.005 |
| CelebA Deblurring | DISTS↓ | **0.04** | 0.08 | 0.08 | 0.10 |
| ImageNet Inpainting | FID↓ | **47.54** | 55.61 | 58.50 | 54.07 |
| ImageNet Deblurring | FID↓ | **44.82** | 59.09 | 63.10 | 79.43 |
| ImageNet Deblurring | DISTS↓ | **0.07** | 0.10 | 0.11 | 0.15 |

PSNR 上 RED-Diff 在 CelebA inpainting 略高（17.96 vs 17.70），符合其 MAP 行为偏 over-smooth 的特点；但在感知质量 (LPIPS/FID/DISTS) 上本文几乎全胜，特别是 deblurring 任务 DISTS 砍掉一半。

### 消融实验
CelebA inpainting 任务上 ULA vs MALA 校正步数（LPIPS↓）：

| 校正器 | 1 步 | 5 步 | 8 步 |
|--------|------|------|------|
| ULA | 0.093 | 0.093 | 0.093 |
| MALA | 0.093 | 0.091 | **0.089** |

A-CSM 消融见 §4.3 blind 任务：去掉 A-CSM 的纯 A-DSM 模型完全无法定位 box size 和 $\sigma_1$，验证归一化常数 $\boldsymbol{\Sigma}$ 无关性才是 blind estimation 可行的根因。

### 关键发现
- **能量与样本质量校准**：在固定观测下用同一 $U_\theta$ 评估 DPS / RED-Diff / 本文的样本——DPS 样本 prior 概率明显低于 GT（被 likelihood 近似带偏到 OOD 区域），RED-Diff 样本 prior 偏高但 posterior 低（过 smooth），只有本文样本在 prior 和 posterior 上都贴近 GT，首次让"先验/后验对数概率"成为衡量逆问题采样器的可计算指标。
- **能量引导自适应调度在小测量量下显著获益**：MNIST 上随机 $k$ 像素重建，能量引导调度 $\boldsymbol{\delta}\boldsymbol{\Sigma}_t\propto\boldsymbol{\Sigma}_t\nabla_{\boldsymbol{\Sigma}}U_\theta\boldsymbol{\Sigma}_t$（Bregman 几何下的 steepest descent）在 $k$ 较小时分类错误率持续低于几何调度，约 300 测量时两者收敛到 clean baseline；但在 CelebA inpainting 上还输给固定调度，作者归因于 diagonal-covariance 自由度不足。
- **MALA 无偏校正只有归一化能量模型能用**：MALA 接受概率要算 $p(\mathbf{x}'|\boldsymbol{\Sigma}_T,\mathbf{y})/p(\mathbf{x}|\boldsymbol{\Sigma}_T,\mathbf{y})$，纯 score model 永远做不了；本文模型用 8 步 MALA 把 LPIPS 从 0.093 砍到 0.089，且增加 ULA 步数完全无效，说明这是真正的"不可替代能力"而非超参调好的边际收益。
- **盲估计直接 $\arg\max_{\boldsymbol{\Sigma}}\log p_\theta(\mathbf{y}|\boldsymbol{\Sigma})$**：在 box-size 和 $\sigma_1$ 都未知的 inpainting 任务上，对数概率曲面在真值处取得明确单峰，可直接读出退化参数；这是 Bayesian 派完全做不到的（它们没有 $\log p(\mathbf{y}|\boldsymbol{\Sigma})$），也是 regression 派完全做不到的（需要为每个 $\boldsymbol{\Sigma}$ 重训）。

## 亮点与洞察
- **"线性退化 ≡ 各向异性噪声"的视角转换**：$\mathbf{H}^{-1}\mathbf{y}$ 这一步重写把整族逆问题统一进同一个去噪框架，本身没有新数学但把问题降维到只需建模一族 $\boldsymbol{\Sigma}$，是整篇论文成立的关键支点。
- **A-CSM 作为 Fokker-Planck 的隐式 enforcement**：把"密度归一化常数跨 $\boldsymbol{\Sigma}$ 不变"这一硬性物理约束变成可微的训练损失，类似 PINN 但避开了显式二阶导，这种"用守恒方程做正则化"的思路可直接迁移到任何条件密度建模（如条件 flow / 时间序列扩散）。
- **能量值反过来成为评测指标**：以前比较逆问题采样器只能看 PSNR/LPIPS，现在可以画 $\log p(\hat{\mathbf{x}})$ 直方图诊断哪种采样器在 prior/posterior 分布上偏哪边——这一可视化在 §4.1 直接揭示 DPS 落 OOD、RED-Diff over-smooth，是相当犀利的"显微镜"工具。
- **Diagonal 受限 ≠ 表达力差**：选 spatial-diagonal + spectral-diagonal 两族就覆盖了 90%+ 的常用线性逆问题，把 $O(d^2)$ 自由度压到 $O(d)$，工程上极度务实。

## 局限与展望
- 作者承认 dual score matching 训练比纯 score model 贵（额外 backprop 计算 $\nabla_{\boldsymbol{\Sigma}}U_\theta$），未来可用 sliced score matching + forward-mode JVP 加速。
- 协方差只支持 spatial / spectral 对角，无法表达任意 $\boldsymbol{\Sigma}$（例如旋转模糊、空间变化模糊），CelebA inpainting 上能量引导调度跑输固定调度也归因于此。
- 实验分辨率最高仅 192×192 (AFHQ-Cat)，到 256+ 或 1024 是否仍 scale 没有证据；EBM 训练数值稳定性在更高维下也存在风险。
- 整体目标里的 $1/d$ 和 $1/d^2$ 权重缺乏 ablation，A-DSM/A-CSM 比重对最终性能的敏感度未知。
- 加新退化算子（不在训练 $p(\boldsymbol{\Sigma})$ 支持集内）仍需额外训练以扩展 $\log p(\mathbf{y}|\boldsymbol{\Sigma})$ 的 range，相比 Bayes 派"训一次先验适配所有逆问题"的承诺仍打折扣。

## 相关工作与启发
- **vs DPS / DAPS (Bayes 派)**：他们靠 Gaussian 近似 $p(\mathbf{y}|\mathbf{x}_t)$ 然后做 guidance，本文直接学归一化 $p(\mathbf{y}|\boldsymbol{\Sigma})$，绕开 likelihood 近似导致的 OOD 偏差；代价是必须事先知道协方差族 $p(\boldsymbol{\Sigma})$。
- **vs Palette / InDI (Regression 派)**：他们每种退化训一个条件模型，本文单个模型覆盖一族 $\boldsymbol{\Sigma}$ 且支持盲估计；架构开销几乎相同因为协方差 embedding 复用 EDM 的 gain modulation。
- **vs Guth 2025 / Yu 2025 / Plainer 2025 (isotropic EBM)**：他们的 dual / time score matching 只能在 $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}$ 一维流形上做归一化，本文推广到任意（受限族）协方差矩阵，是真正的"isotropic → anisotropic"扩展，也是把 EBM 从纯生成任务拓展到逆问题的关键一跃。
- **vs Du 2023 / Thornton 2025 (compositional EBM-diffusion)**：他们用未归一化能量做组合生成，本文强调"归一化"打开了 MALA / blind / 概率比较等新闸门，这一系列能力是组合派完全没有的。
- **启发**：把"逆问题"重新表述成"条件密度族建模"加上"Fokker-Planck 守恒约束"这一 recipe，原则上可移植到非线性逆问题（用更一般的 SDE 替代 $\boldsymbol{\Sigma}^{1/2}\mathbf{v}$）、跨模态条件生成、甚至带物理约束的科学计算反演问题。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Solving Inverse Problems with Flow-based Models via Model Predictive Control](solving_inverse_problems_with_flow-based_models_via_model_predictive_control.md)
- [\[CVPR 2026\] PnP-CM: Consistency Models as Plug-and-Play Priors for Inverse Problems](../../CVPR2026/image_restoration/pnp-cm_consistency_models_as_plug-and-play_priors_for_inverse_problems.md)
- [\[ICML 2026\] Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules](triadic_dynamics_aware_diffusion_posterior_sampling_for_inverse_problems_optimiz.md)
- [\[CVPR 2026\] Dual Ascent Diffusion for Inverse Problems](../../CVPR2026/image_restoration/dual_ascent_diffusion_for_inverse_problems.md)
- [\[CVPR 2026\] Outlier-Robust Diffusion Solvers for Inverse Problems](../../CVPR2026/image_restoration/outlier-robust_diffusion_solvers_for_inverse_problems.md)

</div>

<!-- RELATED:END -->
