---
title: >-
  [论文解读] One-shot Conditional Sampling: MMD meets Nearest Neighbors
description: >-
  [ICML 2026][图像恢复][条件采样] CGMMD 用 k 近邻图把"期望条件 MMD（ECMMD）"估计成一个可直接最小化的非对抗目标，训出一个能在单次前向传播内从 $P_{Y\mid X}$ 采样的条件生成器，并给出了非渐近误差界与分布收敛性证明。 领域现状：条件分布建模是统计与机器学习的基础问题——回归只给出条…
tags:
  - "ICML 2026"
  - "图像恢复"
  - "条件采样"
  - "MMD"
  - "最近邻估计"
  - "一次性生成"
  - "核均值嵌入"
---

# One-shot Conditional Sampling: MMD meets Nearest Neighbors

**会议**: ICML 2026  
**arXiv**: [2509.25507](https://arxiv.org/abs/2509.25507)  
**代码**: https://github.com/anirbanc96/cgmmd (有)  
**领域**: 科学计算 / 条件生成 / 核方法  
**关键词**: 条件采样, MMD, 最近邻估计, 一次性生成, 核均值嵌入

## 一句话总结
CGMMD 用 k 近邻图把"期望条件 MMD（ECMMD）"估计成一个可直接最小化的非对抗目标，训出一个能在单次前向传播内从 $P_{Y\mid X}$ 采样的条件生成器，并给出了非渐近误差界与分布收敛性证明。

## 研究背景与动机
**领域现状**：条件分布建模是统计与机器学习的基础问题——回归只给出条件均值/分位数，而很多下游任务（不确定性量化、模拟推断、图模型、维度约简）需要整条 $P_{Y\mid X}$。现代主流做法是条件 GAN、CVAE、条件扩散模型，把"密度估计"重新表述为"用噪声 $\eta$ 加输入 $x$ 生成样本"。

**现有痛点**：三类方法各有短板。条件 GAN 走 min-max 优化，依赖 JS/KL 散度，当生成器与目标分布支撑在低维流形上时几乎不交、梯度消失，且训练不稳、易模式塌缩。Wasserstein/IPM 类损失（如 W-GAN、MMD-GAN）在无条件设定下缓解了不稳定，但**条件**场景下既缺乏有限样本理论，也没有简洁的 k 近邻类估计。条件扩散虽稳定，但采样需要几十到上千步迭代去噪，测试时复杂度高。

**核心矛盾**：训练目标的稳定性、统计意义上的相合性、采样时间这三者之间存在 trade-off——对抗损失牺牲稳定换灵活性，扩散牺牲采样速度换样本质量，IPM 类目标缺统计保证。

**本文目标**：构造一个条件采样框架，同时满足：(i) **非对抗、可直接最小化**；(ii) **单次前向**就能采样；(iii) 有非渐近误差界并能证明收敛到真分布。

**切入角度**：Chatterjee et al. (2024) 已经把 MMD 推广到 Expected Conditional MMD（ECMMD），证明它是一个严格 scoring rule（ECMMD$^2 = 0$ 当且仅当条件分布相等）。但要把 ECMMD 当成训练 loss 用，需要一个能从有限样本一致估计的形式——k 近邻在条件均值估计里早已是经典工具，把它嫁接到 ECMMD 的 U-统计核函数上，就能得到一个免对抗、免迭代采样、可端到端反传的目标。

**核心 idea**：用 k-NN 图近似"在 $X=X_i$ 条件下"取期望，把生成器输出和真实样本喂进核函数 $\mathsf{H}$，直接最小化得到的 ECMMD 估计量；训练完后给任意 $x$ 抽一个 $\eta$ 单次前向即得 $\hat g(\eta, x) \sim P_{Y\mid X=x}$。

## 方法详解

### 整体框架
CGMMD 要解决的是"给定 $X=x$ 时如何一次性抽出 $P_{Y\mid X=x}$ 的样本"。它把这个生成问题转成一个纯最小化目标：用 $X$ 上的 $k$ 近邻图把"期望条件 MMD（ECMMD）"估计成可反传的经验损失，直接训一个 ReLU 生成器 $\hat g(\eta, x)$，测试时给新 $x$ 采一个噪声 $\eta$ 做单次前向即得条件样本。

具体地，输入训练对 $\{(Y_i, X_i)\}_{i=1}^n$、参考噪声 $P_\eta=\mathcal{N}(0, I_m)$、核函数 $\mathsf{K}$ 与生成器类 $\mathcal{G}$；每轮先对每个样本采辅助噪声 $\eta_i$ 前向得到伪样本 $g(\eta_i, X_i)$，再在小批量 $X$ 上现建有向 $k$ 近邻图 $G(\mathcal{X}_n)$，沿图上的近邻对求和得到经验损失 $\hat{\mathcal{L}}(g)$（即 ECMMD$^2$ 的一致估计），反传更新参数；训练完得到 $\hat g$，采样阶段 $\eta\sim P_\eta \to \hat g(\eta, x)$ 一步出样本。

### 关键设计

**1. ECMMD 的 k-NN 估计量：把"条件期望"做成可微的近邻求和**

直接最小化 ECMMD$^2$ 的障碍在于它含一个"在 $X$ 条件下取期望"的内层算子，而我们手里只有有限样本。论文先用核技巧把 ECMMD$^2$ 写成 $\mathbb{E}[\mathsf{H}(W, W')]$（$W=(Y,Z)$，核 $\mathsf{H}$ 由四个核值组合而成），再用 tower 性质把外层关于 $X$ 的期望与内层关于 $Y, Z\mid X$ 的条件期望分离。关键一步是不做核回归来估内层条件期望，而是在 $X$ 上建 $k$-NN 有向图 $G(\mathcal{X}_n)$，把邻居集 $N_G(i)$ 里的样本视作"近似同条件下"的伪重复样本，于是估计量写成 $\widehat{\mathrm{ECMMD}}^2 = \frac{1}{n k_n}\sum_i \sum_{j\in N_G(i)} \mathsf{H}(W_i, W_j)$。这样做的好处是 k-NN 免去核回归的带宽选择，能自适应到 $X$ 的内蕴维度 $\bar d$，且图只依赖 $X_i$、求和里只有 $g$，梯度直通而无须额外的 reparameterization 技巧。

**2. 非对抗的直接最小化目标：去掉判别器，只留一个生成器**

把估计量当 loss 后，训练就退化成对生成器参数 $\theta$ 的纯最小化 $\hat g \in \arg\min_{g\in\mathcal{G}} \hat{\mathcal{L}}(g)$，其中 $\hat{\mathcal{L}}(g) = \frac{1}{n k_n}\sum_i \sum_{j\in N_G(i)} \mathsf{H}\big((Y_i, g(\eta_i, X_i)), (Y_j, g(\eta_j, X_j))\big)$；算法 1 的循环就是每个 mini-batch 重建 $k_B$-NN 图、前向算 $\hat{\mathcal{L}}$、再 $\theta \leftarrow \theta - \alpha\nabla_\theta \hat{\mathcal{L}}$。MMD-GAN 已证明这类核损失能避免 JS/KL 散度在不相交支撑下的梯度消失；本文进一步把它推广到条件设定并彻底去掉判别器，既绕开了条件 GAN 常见的模式崩塌与 min-max 不稳定，工程上也只需维护一个生成器网络。

**3. 一次前向采样 + ReLU 网络函数类：把分布信息压进权重**

测试时的单步采样依据 noise outsourcing 引理——对联合分布 $(Y, X)$ 存在 Borel 可测的 $\bar g$ 与独立噪声 $\eta$ 使 $(Y, X)\overset{d}{=}(\bar g(\eta, X), X)$，因此只要在 ReLU 网络类 $\mathcal{G}_{\mathcal{H},\mathcal{W},\mathcal{S},\mathcal{B}}$（深度 $\mathcal{H}$、宽度 $\mathcal{W}$、参数量 $\mathcal{S}$、$\ell_\infty$ 界 $\mathcal{B}$）里学到逼近 $\bar g$ 的 $\hat g$，采样就是 $\eta\sim\mathcal{N}(0, I_m)\to\hat g(\eta, x)$ 一步完成。扩散模型的采样瓶颈来自把分布建模摊到几十上千步去噪上；CGMMD 反过来把分布信息整合进单个网络的权重里，由 ECMMD 损失保证生成分布与真分布一致，因此单次前向就够，相对扩散快两到三个量级——这是它最直接的实用优势。

### 损失函数 / 训练策略
核心损失为 $\hat{\mathcal{L}}(g) = \frac{1}{n k_n}\sum_i \sum_{j\in N_G(i)} \mathsf{H}(W_{i,g}, W_{j,g})$，其中 $\mathsf{H}(W_i, W_j) = \mathsf{K}(Y_i, Y_j) - \mathsf{K}(Y_i, g_j) - \mathsf{K}(g_i, Y_j) + \mathsf{K}(g_i, g_j)$；实验取高斯核，batch size 200，每个 batch 在 $X$ 上现建 $k_B$-NN 图，理论上要求 $k_n = o(\sqrt n)$、网络规模满足 $\mathcal{B}^2\mathcal{H}\mathcal{S}\log\mathcal{S}\log n / n \to 0$。配套的非渐近理论（Theorem 4.4）在 Assumption 2.1（核有界、特征核）、4.1（网络规模条件）、4.2（$X$ 次高斯、$\bar g$ 一致连续、条件均值嵌入的 Lipschitz 灵敏度）下给出，以概率至少 $1-\delta$ 有 $\mathcal{L}(\hat g) \lesssim \frac{\mathrm{polylog}\, n}{n^{1/(2d)}} + \sqrt{\frac{\mathcal{B}^2\mathcal{H}\mathcal{S}\log\mathcal{S}\log n}{n}} + \omega_{\bar g}\!\big(\frac{2\sqrt{\log n}}{(\mathcal{H}\mathcal{W})^{1/(d+m)}}\big) + \sqrt{\frac{\log(1/\delta)}{n}}$，三项分别对应 k-NN 估计的随机误差、网络泛化误差与网络逼近误差，且当 $X$ 集中在低维流形上时维度 $d$ 可换成内蕴维度 $\bar d$；Corollary 4.5 进一步证明 $\hat g$ 诱导的条件分布在 MMD 与特征函数意义下都收敛到真条件分布。

## 实验关键数据

### 主实验

| 任务 / 数据集 | 设定 | 关键观察 |
|---|---|---|
| Bivariate Helix 合成 | $\sigma \in \{0.2, 0.4, 0.6\}$ | 低噪 $\sigma=0.2$ 三种方法都能恢复螺旋结构；噪声升高后 CGMMD 仍保住螺旋"眼"，GCDS 与 WGAN 明显退化 |
| MNIST 4× 超分 | $7\times 7 \to 28\times 28$ | 数字 $\{0\dots4\}$ 重建清晰 |
| STL-10 4× 超分 | $3\times 24\times 24 \to 3\times 96\times 96$ | 重建均值图清晰，像素级标准差图显示生成结果有显著多样性 |
| MNIST 去噪 | $\sigma=0.5$，数字 $\{5\dots9\}$ | CGMMD 恢复干净字形 |
| CelebHQ 去噪 | $3\times 64\times 64$, $\sigma=0.25$ | 重建人脸保留面部结构 |

### 与扩散模型对比（MNIST 去噪，$\sigma=0.9$）

| 模型 | PSNR | SSIM | Time/batch (s) | Time/img (s) |
|---|---|---|---|---|
| Diffusion (CFG) | 13.326 | 0.861 | 6.94 | $5.42\times 10^{-2}$ |
| Distilled Diffusion | 10.658 | 0.508 | $1.18\times 10^{-1}$ | $9.2\times 10^{-4}$ |
| **CGMMD** | 8.922 | 0.718 | $7.21\times 10^{-2}$ | $\mathbf{5.6\times 10^{-4}}$ |

### 关键发现
- 在高噪声合成任务上 CGMMD 相比 GCDS/WGAN 稳定性优势显著——WGAN 没有 $\ell_1$ 正则常常训不动，这一点作者明确指出。
- 与扩散模型的对比展示了清晰的速度-质量 trade-off：CGMMD 单图采样比 CFG 扩散快约 100 倍，PSNR 落后但 SSIM 不至于太差；与蒸馏扩散在速度上同级，SSIM 反而更高。
- ECMMD + k-NN 框架对 $X$ 的内蕴维度有适应性（附录 C.2 的合成实验），说明理论分析中 $d \to \bar d$ 的论断在实践中可观察。

## 亮点与洞察
- **把 k-NN 当成"条件期望近似器"嵌入到 MMD 估计里**是个简洁却强力的设计——既继承了无条件 MMD-GAN 的稳定性，又自然引入了条件依赖。它绕开了核回归的带宽选择，并把"近邻"的概念变成可微目标里的求和指标。
- 一次前向 + 非对抗训练这两点叠加，让 CGMMD 在"轻量条件采样器"这条赛道上极有吸引力——很多模拟推断、posterior sampling 任务对单样本时延敏感，扩散模型并不合适。
- 论文证明的"k-NN 类非线性泛函的一致集中"是独立有趣的工具，可以迁移到其他依赖条件均值估计的统计学习问题（如条件独立性检验、条件期望回归）。
- 主结果对内蕴维度 $\bar d$ 的自适应给出了一个温和的"高维但流形"假设下的可用界，与现实世界中数据流形分布的直觉一致。

## 局限与展望
- 作者承认：当前理论要求网络规模随样本量增长，无法直接覆盖固定架构网络；图像任务的 PSNR 暂时比不过专业的扩散/超分模型。
- 训练时每个 mini-batch 都要现建 $k_B$-NN 图，batch 较大或维度较高时成图开销不可忽视，论文未讨论近似最近邻或缓存策略。
- 实验局限于 MNIST、FashionMNIST、CelebHQ、STL-10 这类相对小的图像数据集，没有触碰高分辨率自然图像或文本-图像条件生成；高维 $Y$ 下核选择（高斯核带宽）对实际效果的影响也没详细分析。
- 改进方向：把损失推广到 flow-matching / OT-flow 类目标，把 k-NN 替换成更可扩展的近邻结构（如可微 ANN），以及把理论延伸到 fixed-architecture 网络的有限逼近误差设定。

## 相关工作与启发
- **vs GCDS (Zhou et al., 2023)**：GCDS 用 GAN 形式做条件采样，min-max 优化、易模式塌缩；CGMMD 改用 ECMMD 直接最小化，去掉判别器，理论上还多了一致性证明。
- **vs Wasserstein-GAN 条件版 (Song et al., 2025)**：W-GAN 用 Wasserstein 距离作条件 IPM，但训练对 $\ell_1$ 正则敏感；CGMMD 用核 MMD，损失光滑、训练更稳。
- **vs 条件扩散 (Ho & Salimans, 2021)**：扩散迭代采样质量更高但每图 ~50 ms 级别；CGMMD 单步前向 ~0.56 ms，差两个数量级，适合需要海量采样的科学计算 / 后验近似场景。
- **vs MMD-GAN 无条件版 (Li et al., 2015; Bińkowski et al., 2018)**：本文是把它推广到条件设定的统计学化版本——既给出 k-NN 估计器，又补齐了非渐近界。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把 ECMMD 与 k-NN 结合用作条件生成的训练目标，并配合非渐近理论，路线清晰且此前未见。
- 实验充分度: ⭐⭐⭐ 合成 + 三个图像任务覆盖了概念验证，但缺乏更大规模 benchmark 与对扩散 SOTA 的硬碰硬。
- 写作质量: ⭐⭐⭐⭐ 推导严谨、记号统一、定理与算法穿插紧凑；附录里独立的 k-NN 一致集中结果是亮点。
- 价值: ⭐⭐⭐⭐ 对"需要快速条件采样且关心理论保证"的科学计算 / 模拟推断社区有直接实用价值，框架易扩展到 flow-based 方法。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] AceTone: Bridging Words and Colors for Conditional Image Grading](../../CVPR2026/image_restoration/acetone_bridging_words_and_colors_for_conditional_image_grading.md)
- [\[ICML 2026\] Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules](triadic_dynamics_aware_diffusion_posterior_sampling_for_inverse_problems_optimiz.md)
- [\[ICLR 2026\] A Statistical Benchmark for Diffusion-Posterior-Sampling Algorithms](../../ICLR2026/image_restoration/a_statistical_benchmark_for_diffusion-posterior-sampling_algorithms.md)
- [\[CVPR 2026\] Zero-Shot Image Denoising via Hybrid Prior-Guided Pseudo Sample Generation](../../CVPR2026/image_restoration/zero-shot_image_denoising_via_hybrid_prior-guided_pseudo_sample_generation.md)
- [\[ICLR 2026\] Adaptive Moments are Surprisingly Effective for Plug-and-Play Diffusion Sampling](../../ICLR2026/image_restoration/adaptive_moments_are_surprisingly_effective_for_plug-and-play_diffusion_sampling.md)

</div>

<!-- RELATED:END -->
