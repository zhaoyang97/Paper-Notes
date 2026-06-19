---
title: >-
  [论文解读] Fractional Diffusion Bridge Models
description: >-
  [NeurIPS 2025][计算生物][扩散桥模型] 提出分数扩散桥模型（FDBM），将分数布朗运动（fBM）引入生成扩散桥框架，通过 Hurst 指数 $H$ 控制轨迹的粗糙度和长程依赖性，在蛋白质构象预测和图像翻译任务上超越布朗运动基线。 领域现状：扩散桥模型通过条件化随机过程在两个分布间构建随机插值路径…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "扩散桥模型"
  - "分数布朗运动"
  - "蛋白质构象预测"
  - "图像翻译"
  - "Schrödinger桥"
---

# Fractional Diffusion Bridge Models

**会议**: NeurIPS 2025  
**arXiv**: [2511.01795](https://arxiv.org/abs/2511.01795)  
**代码**: [GitHub-paired](https://github.com/GabrielNobis/FDBM_paired) / [GitHub-unpaired](https://github.com/mspringe/FDBM_unpaired) / [SBFlow](https://github.com/mspringe/Schroedinger-Bridge-Flow)  
**领域**: 计算生物  
**关键词**: 扩散桥模型, 分数布朗运动, 蛋白质构象预测, 图像翻译, Schrödinger桥

## 一句话总结

提出分数扩散桥模型（FDBM），将分数布朗运动（fBM）引入生成扩散桥框架，通过 Hurst 指数 $H$ 控制轨迹的粗糙度和长程依赖性，在蛋白质构象预测和图像翻译任务上超越布朗运动基线。

## 研究背景与动机

**领域现状**：扩散桥模型通过条件化随机过程在两个分布间构建随机插值路径，广泛用于配对/非配对数据翻译（如蛋白质构象预测、图像翻译）。现有模型统一使用标准布朗运动（BM）作为驱动噪声。

**现有痛点**：标准 BM 是马尔可夫过程，增量独立，无法捕捉真实数据中的**记忆效应、长程依赖、粗糙性和反常扩散**现象。对于蛋白质等复杂系统，时间上的独立增量假设可能导致建模不充分。

**核心矛盾**：现有选择 BM 的动机是数学简便性而非物理忠实性——选择更富表达力的驱动噪声应能更好地匹配真实数据动力学。

**本文目标**：将分数布朗运动（非马尔可夫、具有长程相关性）引入扩散桥，同时保持可训练性和高效推理。

**切入角度**：利用 fBM 的马尔可夫近似（MA-fBM），通过 $K$ 个 Ornstein-Uhlenbeck (OU) 过程的线性叠加来逼近 fBM，使得增广系统是马尔可夫的。

**核心 idea**：用 fBM 的马尔可夫近似替代 BM 驱动扩散桥，通过 Hurst 指数 $H$ 灵活控制生成轨迹特性，在配对和非配对设置下均改善性能。

## 方法详解

### 整体框架

FDBM 的核心是将标准扩散桥中的 BM 替换为缩放的 MA-fBM $X = \sqrt{\varepsilon}\hat{B}^H$，其中 $\hat{B}^H_t = \sum_{k=1}^K \omega_k Y_t^k$ 是 $K$ 个 OU 过程的加权和。参考过程 $X$ 本身是非马尔可夫的，但增广过程 $Z = (X, Y^1, \ldots, Y^K)$ 是马尔可夫的。

### 关键设计

#### 1. MA-fBM 作为驱动噪声

**Type II fBM 定义**：$B_t^H = \frac{1}{\Gamma(H+1/2)} \int_0^t (t-s)^{H-1/2} dB_s$

- $H > 0.5$：正相关增量（超扩散），轨迹更光滑
- $H < 0.5$：负相关增量（亚扩散），轨迹更粗糙
- $H = 0.5$：退化为标准 BM

**MA-fBM**：选择 $K$ 个 OU 过程，速率 $\gamma_k = r^{k-n}$，最优权重 $\omega$ 由闭式线性系统 $A\omega = b$ 给出，最小化 $L^2(\mathbb{P})$ 误差。实验中固定 $K=5$。

#### 2. MA-fBB（马尔可夫近似分数布朗桥）

对增广过程 $Z$ 进行端点条件化，得到部分钉住的过程 $Z_{|x_0,x_1}$，其 SDE 漂移包含可解析计算的引导项 $u(t,z)$。关键在于条件均值 $\mu_{1|t}(z)$ 和条件方差 $\sigma^2_{1|t}$ 均可闭式计算。

#### 3. 配对数据翻译

**Proposition 5（耦合保持性）**：存在一个保持训练数据耦合 $\Pi_{0,1}$ 的随机过程 $Z^\star$，满足 $(X_0^\star, X_1^\star) \sim \Pi_{0,1}$。

**训练目标**：
$$\mathcal{L}_{\text{FDBM}}^{\text{paired}}(\theta) = \int_0^1 \mathbb{E}_{\mathbb{P}^\star}\left[\left\|\frac{X_1^\star - \mu_{1|t}(Z_t^\star)}{\sigma_{1|t}^2} - \tilde{u}^\theta(t, X_0, \mu_{1|t}(Z_t^\star))\right\|^2\right] dt$$

**关键**：神经网络 $\tilde{u}^\theta$ 的输出维度与数据维度 $d$ 相同（不是增广维度），通过缩放变换映射到增广空间。因此 FDBM 可复用 ABM 的网络架构，额外计算开销极小。

#### 4. 非配对数据翻译（Schrödinger桥）

将 SB 问题中的参考过程替换为 MA-fBM，定义增广空间上的 SB 问题，引入增广互反类和增广马尔可夫投影。采用 $\alpha$-IMF 训练方案（预训练 + 微调）。

### 损失函数/训练策略

- **配对**：最小化神经网络预测与分数桥漂移目标之间的 $L^2$ 距离
- **非配对**：类似公式但不依赖起始值 $X_0$，采用 forward-forward 训练策略
- **注意**：非配对设置中 MA-fBM 参考过程在 $H$ 远离 0.5 时微调阶段不稳定

## 实验关键数据

### 配对数据：蛋白质构象预测（D3PM 数据集）

| 方法 | Median RMSD(Å)↓ | Mean RMSD(Å)↓ | RMSD<2Å(%)↑ | RMSD<5Å(%)↑ | Δ RMSD Mean↑ |
|------|-----------------|---------------|-------------|-------------|-------------|
| SBALIGN | 3.67 | 4.82 | 0% | 71% | 1.92 |
| Sesame | 2.87 | 3.65 | 38% | 82% | 3.11 |
| ABM (baseline) | 2.40 | 3.49 | 43% | 84% | 3.35 |
| **FDBM (H=0.2)** | **2.12** | **3.34** | **48%** | **86%** | 3.39 |
| FDBM (H=0.3) | 2.33 | 3.42 | 43% | 85% | **3.49** |
| FDBM (H=0.1) | 2.20 | 3.44 | 46% | 83% | 3.45 |

### 非配对数据：AFHQ 图像翻译

在 AFHQ-256 和 AFHQ-512 上评估 cat↔wild 翻译：

| 方法/设置 | FID (wild→cat)↓ | FID (cat→wild)↓ |
|-----------|----------------|----------------|
| SBFlow (AFHQ-256) | 基线 | 基线 |
| FDBM (H=0.6, AFHQ-256) | 19.42 | 11.62 |
| FDBM (H=0.4, AFHQ-512) | 30.11 | 14.27 |

### 合成数据：Moons & T-Shape

| 数据集 | ABM WSD | FDBM 最优 WSD |
|--------|---------|--------------|
| Moons | 0.015±0.019 | **0.012±0.002** (H=0.7) |
| T-Shape | 0.082±0.028 | **0.048±0.039** (H=0.2) |

### 消融实验

- **H 的选择依赖任务**：Moons 适合平滑轨迹（H=0.6-0.7），T-Shape 和蛋白质适合粗糙轨迹（H=0.1-0.3）
- **K=5 个 OU 过程**已足够，增加 K 改善有限
- **扩散系数 $\sqrt{\varepsilon}$**：蛋白质任务最优为 0.2

### 关键发现

1. **粗糙轨迹（H<0.5）在蛋白质预测中更优**：RMSD<2Å 比例从 ABM 的 43% 提升到 48%，中位 RMSD 从 2.40Å 降至 2.12Å
2. **ABM 已是很强的基线**，优于此前所有方法（SBALIGN、Sesame）
3. **FDBM 保持耦合**（Proposition 5），与 SBALIGN 不同
4. **非配对设置中 H 远离 0.5 时微调不稳定**，表明 fBM SB 问题存在前向-后向不对称挑战
5. **计算开销极小**：FDBM 相比 ABM 仅增加了 MA-fBM 的输入/输出变换

## 亮点与洞察

- **将 fBM 引入扩散桥建模的首次工作**，概念上优雅：BM 只是 $H=0.5$ 的特例，现在可以调节 $H$ 来匹配数据特性
- **耦合保持性的理论证明**（Proposition 5）是重要的理论贡献
- **实验发现蛋白质构象变化适合粗糙轨迹**，暗示分子运动的记忆效应
- **架构复用设计精巧**：神经网络维度不变，仅通过缩放映射，最小化实现复杂度
- **Moons 用光滑 H，T-Shape 用粗糙 H**的发现表明最优 $H$ 的选择有可解释的结构

## 局限与展望

1. **非配对设置中 H 远离 0.5 不稳定**：SB 微调不收敛，限制了 FDBM 在非配对任务中的灵活性
2. **$H$ 需要手动搜索**：目前没有自动选择 $H$ 的方法，可考虑将其作为可学习参数
3. **MA-fBM 是近似**：$K=5$ 的近似精度有限，可能在某些场景下不够
4. **缺乏自适应 H**：不同区域可能需要不同的 $H$，全局固定可能次优
5. **蛋白质实验仅在 D3PM 子集上**：需更大规模验证

## 相关工作与启发

- 建立在 ABM (Bortoli et al.)、SBFlow (Bortoli et al.)、SBALIGN (Somnath et al.) 基础上
- MA-fBM 技术来自 Harms & Stefanovits 和 Daems et al.
- **核心启发**：选择驱动噪声的统计特性（记忆、粗糙度）应匹配目标数据的物理特性，而非仅追求数学方便

## 评分

⭐⭐⭐⭐ (4/5)

**理由**：概念新颖（首次将 fBM 引入扩散桥），理论扎实（耦合保持证明），实验全面（合成+蛋白质+图像），蛋白质任务上的提升有意义。非配对设置的不稳定性是明确的局限。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Remasking Discrete Diffusion Models with Inference-Time Scaling](remasking_discrete_diffusion_models_with_inference-time_scaling.md)
- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[NeurIPS 2025\] KLASS: KL-Guided Fast Inference in Masked Diffusion Models](klass_kl-guided_fast_inference_in_masked_diffusion_models.md)
- [\[ICLR 2026\] Unified Biomolecular Trajectory Generation via Pretrained Variational Bridge](../../ICLR2026/computational_biology/unified_biomolecular_trajectory_generation_via_pretrained_variational_bridge.md)
- [\[NeurIPS 2025\] Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra](atomic_diffusion_models_for_small_molecule_structure_elucidation_from_nmr_spectr.md)

</div>

<!-- RELATED:END -->
