---
title: >-
  [论文解读] JAPAN: Joint Adaptive Prediction Areas with Normalising-Flows
description: >-
  [ICLR 2026][时间序列][Conformal Prediction] JAPAN 用归一化流估计（条件）密度、并以对数密度作为共形得分，通过在密度上设阈值来构造几何无关、可不连通、随上下文自适应：的预测区域，在保持有限样本覆盖保证的同时把预测区域体积压到比一众残差类基线更紧。 领域现状：共形预测（Conformal…
tags:
  - "ICLR 2026"
  - "时间序列"
  - "Conformal Prediction"
  - "Normalising Flows"
  - "密度阈值"
  - "多元回归"
  - "时序预测"
  - "预测区域"
---

# JAPAN: Joint Adaptive Prediction Areas with Normalising-Flows

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=4SxAu9zMVC](https://openreview.net/forum?id=4SxAu9zMVC)  
**代码**: 待确认  
**领域**: 共形预测 / 不确定性量化 / 时间序列预测  
**关键词**: Conformal Prediction, Normalising Flows, 密度阈值, 多元回归, 时序预测, 预测区域  

## 一句话总结
JAPAN 用归一化流估计（条件）密度、并以对数密度作为共形得分，通过在密度上设阈值来构造**几何无关、可不连通、随上下文自适应**的预测区域，在保持有限样本覆盖保证的同时把预测区域体积压到比一众残差类基线更紧。

## 研究背景与动机

**领域现状**：共形预测（Conformal Prediction, CP）是一套模型无关的不确定性量化框架，给定显著性水平 $\epsilon$，它能构造预测集 $\Gamma_\epsilon(x)$ 使 $P(y\in\Gamma_\epsilon(x))\ge 1-\epsilon$，且这个覆盖保证是有限样本、无分布假设的，因此在安全攸关场景里很受欢迎。归纳式共形预测（ICP）把数据切成训练集与校准集，在校准集上算每个点的不一致得分，再取分位数当阈值，使 CP 可扩展到大模型。

**现有痛点**：传统 CP 几乎都用**残差类不一致得分**，比如绝对误差 $|y-\hat y|$。这在一维回归里很自然，但一旦响应变量是多维的，残差也变成多元向量，必须人为选一个几何把它压成标量——$\ell_2$ 范数给出球形区域、$\ell_1$ 范数给出矩形区域。这种几何约束未必反映真实不确定性的形状；更糟的是残差得分天然围绕单一均值（众数）堆积，当预测分布是**多峰**时会产出过度保守、罩住大片低密度区的臃肿区域（论文 Figure 1 的螺旋密度例子里，球形/椭圆/矩形方法都把区域撑得很大）。

**核心矛盾**：CP 给了覆盖保证（validity），但实用性还要求**效率**——即区域要尽量小。理论上 Lei et al. (2013) 早就指出，对条件密度 $p(y\mid x)$ 设阈值得到的区域是紧致的、甚至在阈值不依赖 $x$ 时是最小的；但真实密度未知，残差类得分根本无法逼近这种「按密度切」的最优形状。

**本文目标**：把不一致得分从「距离」切换到「密度」，在保留有限样本覆盖保证的前提下，构造贴合真实密度形状、能多峰、能不连通、能随输入自适应的紧致预测区域。

**核心 idea**：**用归一化流（NF）估计密度，把对数密度当共形得分。** NF 既能给出可处理的似然 $\log\hat p(y\mid x)$ 用作得分，又能借助其双射结构把「区域体积计算」转移到隐空间高效完成，从而避开 KDE/蒙特卡洛在高维下的昂贵采样。当 $\hat p$ 足够准，由它切出的区域就逼近真密度阈值下的最优区域，同时覆盖保证不丢。

## 方法详解

### 整体框架
JAPAN 把不确定性量化拆成三步：先用归一化流在训练集上学一个（条件）密度估计 $\hat p(y\mid x)$；再在校准集上把每个点的对数密度 $\alpha_j=\log\hat p(y_j\mid x_j)$ 当作共形得分、取 $(1-\epsilon)$ 分位数得到阈值 $\tau_\epsilon$；测试时预测区域就是所有对数密度高于阈值的 $y$ 构成的集合 $\Gamma_\epsilon(x)=\{y:\log\hat p(y\mid x)\ge\tau_\epsilon\}$，而这个区域的体积则通过流的隐空间采样高效估出。多元回归与「可交换时序轨迹」两类任务共用这套框架，只是上下文编码方式不同。

```mermaid
flowchart LR
    A[训练集 D_train] --> B[训练归一化流<br/>学 p̂ y|x]
    B --> C[校准集打分<br/>α_j = log p̂ y_j|x_j]
    C --> D[取 1-ε 分位数<br/>阈值 τ_ε]
    E[测试输入 x] --> F[密度阈值切区域<br/>Γ = y: log p̂≥τ_ε]
    D --> F
    F --> G[隐空间采样估体积<br/>Proposition 3]
```

### 关键设计

**1. 对数密度共形得分：把「距离」换成「密度」。** JAPAN 的根本动作是用归一化流通过变量替换公式算出对数密度 $\log p(y\mid x)=\log p_Z(h(y,x))+\Phi(y,x)$，其中 $h$ 是把数据映到标准高斯基分布的双射，$\Phi$ 是流映射带来的对数体积变化（离散流是雅可比行列式的对数，连续流是时间积分的散度项）。把这个对数密度直接当共形得分后，预测区域定义为 $\Gamma_\epsilon(x)=\{y:\log\hat p(y\mid x)\ge\tau_\epsilon\}$，$\tau_\epsilon$ 是校准集得分的 $(1-\epsilon)$ 分位数。这一步天然带来三个性质：**几何无关**（不再假设椭球/矩形）、**可不连通**（多峰时只罩高密度的几块、放掉中间的低密度谷）、**上下文自适应**（区域随 $x$ 变化）。由于得分仍是在可交换校准集上取分位数，覆盖保证 $1-\epsilon$ 由构造直接成立。

**2. 排序保持的最优性理论：为什么估计密度也够用。** 直接担心是 NF 估的密度不准会不会毁掉效率。论文用两个命题给出保证：若估计密度 $f_\theta$ 是真密度 $p$ 的某个严格单调变换 $f_\theta=g(p)$（即只要**保持排序**），则切出的区域面积与真密度阈值下的最优区域**完全相等** $\mathrm{Area}(\Gamma_\epsilon)=\mathrm{Area}(\Gamma^*_\epsilon)$（Proposition 1）；更现实地，若 $f_\theta$ 只在均匀误差 $\delta$ 内近似单调变换 $|f_\theta-g(p)|\le\delta$，则面积差被 $C(\delta)$ 控住且 $C(\delta)\to0$ 当 $\delta\to0$（Proposition 2）。换句话说，**JAPAN 不需要密度估得绝对准，只需把样本的密度高低排对**，这大大放宽了对流模型的要求。

**3. 隐空间体积估计：把昂贵的面积计算变便宜。** 评估覆盖很容易（算似然+比阈值），但算预测区域的**体积**很难——朴素做法是在标签空间做蒙特卡洛/拟蒙特卡洛，可标签空间支撑未知且高维时极贵。JAPAN 利用流的双射结构把积分搬到隐空间：从基分布 $z\sim p_Z$ 采样、经逆映射 $y=h^{-1}(z,x)$ 回到数据空间，则区域体积估计为
$$\widehat{\mathrm{Area}}(\Gamma_\epsilon(x))=\frac{1}{N}\sum_{i=1}^N \mathbf{1}\big(\hat p(h^{-1}(z_i,x)\mid x)\ge\tau_\epsilon\big)\cdot\frac{\exp(\phi(z_i,x))}{p_Z(z_i)}.$$
这里只统计逆变换后落在区域内的隐样本，权重 $\exp(\phi)/p_Z$ 修正隐→数据的体积形变。因为基分布是标准高斯、采样极快，该估计在中高维仍可行，且与重要性采样相联系、方差低（Proposition 3，附录 Table 10 显示比朴素 MC 快得多）。

**4. 时序专用架构 + 多种密度得分的统一视角。** 对「可交换时序轨迹」设置（每条完整轨迹是一个数据点、轨迹间 i.i.d.，因此得分仍可交换、覆盖保证不破），JAPAN 先用 RNN/Transformer 把历史窗口 $x^{(i)}_{1:T}$ 编成上下文向量 $c^{(i)}_T$，再让流以 $c^{(i)}_T$ 为条件建模未来轨迹的密度；为尊重时间因果结构，作者把图像领域的 TARFLOW 架构改造到时序上。更进一步，论文展示这套框架可换用多种密度得分——无条件 $p(\hat y)$、条件 $p(y\mid\hat y)$、后验 $p(x\mid y)$、隐空间密度（此时 CONTRA 成为 JAPAN 的特例）、以及自适应阈值 $\tau_\epsilon(x)$（趋近条件覆盖）——把一大批已有方法统一进「在某种密度上设阈值」的视角下。

## 实验关键数据

### 主实验：多元回归（25 次随机划分，目标覆盖 0.9）
表中报告覆盖率（Cov.）与预测区域面积（Area，越小越好）：

| 方法 | Energy Cov. / Area | RF2D Cov. / Area | RF4D Cov. / Area | SCM Cov. / Area(×10³) |
|------|------|------|------|------|
| CONTRA | 0.88 / 18.81 | 0.91 / 5.33 | 0.89 / 59.27 | 0.89 / 61.93 |
| PCP | 0.88 / 16.58 | 0.91 / 7.39 | 0.91 / 111.63 | 0.89 / 68.75 |
| NLE | 0.87 / 21.90 | 0.91 / 15.47 | 0.90 / 2732.15 | 0.90 / 102.13 |
| CQR | 0.88 / 31.12 | 0.91 / 12.50 | 0.91 / 1180.65 | 0.91 / 84.48 |
| CFRNN | 0.90 / 56.22 | 0.91 / 27.15 | 0.92 / 3322.47 | 0.91 / 83.60 |
| **JAPAN** | **0.88 / 16.32** | **0.91 / 5.06** | **0.91 / 24.11** | **0.90 / 61.28** |

JAPAN 在四个数据集上覆盖都达标，且面积几乎都是最小，**在较高维的 RF4D 上优势尤其夸张**（24.11 vs 次优 CONTRA 的 59.27，比 NLE/RCP 的上千小两个量级）。

### 主实验：时间序列预测（25 次随机划分）

| 方法 | COVID-19 Cov. / Area | Particle-1 Cov. / Area | Drone Cov. / Area | Pedestrian Cov. / Area |
|------|------|------|------|------|
| CONTRA | 0.92 / 563.64 | 0.87 / 0.90 | 0.89 / 1.51 | 0.89 / 0.55 |
| PCP | 0.91 / 610.56 | 0.87 / 1.71 | 0.89 / 3.21 | 0.88 / 1.60 |
| MCQR | 0.91 / 1276.50 | 0.91 / 2.89 | 0.86 / 3.32 | 0.86 / 1.89 |
| CFRNN | 0.92 / 927.09 | 0.97 / 3.39 | 0.99 / 5.53 | 0.97 / 2.47 |
| **JAPAN** | **0.91 / 400.94** | **0.91 / 0.89** | 0.88 / 1.47 | **0.89 / 0.50** |

JAPAN 区域普遍最紧，**COVID-19 上比基线小近一个数量级**（400.94 vs CFRNN 的 927）；Drone 上略逊 CONTRA 0.01 面积差但仍接近最优。RCP 在 COVID-19 上因数值不稳定/得分发散导致体积无穷（表中缺失），凸显 JAPAN 的稳健性。

### 关键发现
- **覆盖几乎都达标，胜负在「面积」**：所有方法都能压到 0.9 附近覆盖，但 JAPAN 的密度阈值区域系统性更小、信息量更高。
- **维度越高、分布越复杂，优势越大**：RF4D、COVID-19 这种高维/多峰场景里残差类方法区域爆炸，而 JAPAN 仍紧致。
- **稳健性**：JANET、CopulaCPTS 因用一半校准集训副模型、分母近零时区域会「爆炸」（高标准差）；RCP 数值发散；JAPAN 无此问题。
- **螺旋密度可视化**：只有 JAPAN 的区域紧贴螺旋，CONTRA（隐空间球）、PCP（局部球并）、椭圆/矩形法都罩进大片零密度区。

## 亮点与洞察
- **范式切换干净利落**：从「残差距离」到「密度阈值」的一步切换，同时拿下几何无关、可不连通、上下文自适应三个性质，而这些是残差类 CP 难以兼得的。
- **理论站得住**：排序保持命题说明「不需要密度估得准、只要排序对」，把对流模型精度的苛刻要求降到很务实的程度，是工程上敢用 NF 的底气。
- **隐空间体积估计是点睛之笔**：把 CP 里最难的「区域体积怎么算」用流的双射结构 + 重要性采样优雅解决，否则密度阈值法在高维根本不可用。
- **统一性强**：把 CONTRA（隐空间密度）等已有方法收编为特例，并给出无条件/条件/后验/自适应阈值等一族得分，框架延展性好。

## 局限与展望
- **依赖流模型质量**：虽然理论只要排序对，但严重欠拟合时密度排序也会乱（论文附录 A.7 专门讨论假设违背与欠拟合消融），此时区域可能失真。
- **只是边际覆盖**：主框架保证的是 marginal coverage，条件覆盖要靠自适应 $\tau_\epsilon(x)$ 扩展逼近，仍非严格条件有效。
- **时序设置受限**：可交换轨迹假设要求有「多条独立同分布的轨迹」，对「单条长时序、每个时刻是一个数据点」（可交换性被破坏）的常见场景不直接适用。
- **实验规模偏中小**：数据集维度与体量仍属中等，超高维标签空间下隐空间体积估计的方差与样本需求待进一步验证。

## 相关工作与启发
- **共形预测脉络**：CP 在多维响应上分三支——多元回归 i.i.d.、单条长时序（时间依赖破坏可交换性，如 Adaptive CP/EnbPI）、可交换轨迹集合；JAPAN 主攻第一与第三支。
- **多元回归 CP**：RCP 用协方差构椭圆区域、NLE 上下文自适应椭圆、PCP 用生成模型采样后并局部 $\ell_2$ 球；JAPAN 与它们的区别在「直接在数据空间密度上切」而非套几何。
- **生成式 UQ 的启发**：本文把归一化流（及其涵盖的 CNF、score-based diffusion via probability-flow ODE）当成 CP 的密度引擎，提示「任何有可处理似然 + 高效采样的生成模型」都能插入这套密度阈值共形框架，TARFLOW 等强流模型的进步会直接转化为更紧的预测区域。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把残差共形得分换成归一化流对数密度并配套隐空间体积估计，视角统一、把 CONTRA 等收为特例，思路清晰且有理论支撑。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 4 个多元回归 + 4 个时序数据集、11 个基线、25 次随机划分、含螺旋密度可视化与多种得分扩展；但数据规模偏中等、缺超高维压力测试。
- **写作质量**: ⭐⭐⭐⭐ 动机—理论—算法—实验—扩展层次分明，命题与算法表述清楚；个别记号略有滥用。
- **价值**: ⭐⭐⭐⭐ 给安全攸关场景提供了更紧、更贴合真实分布形状的有效预测区域，且框架可随生成模型进步持续受益，实用价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] DistDF: Time-series Forecasting Needs Joint-distribution Wasserstein Alignment](distdf_time-series_forecasting_needs_joint-distribution_wasserstein_alignment.md)
- [\[ICLR 2026\] ResCP: Reservoir Conformal Prediction for Time Series Forecasting](rescp_reservoir_conformal_prediction_for_time_series_forecasting.md)
- [\[ICLR 2026\] Online Time Series Prediction Using Feature Adjustment](online_time_series_prediction_using_feature_adjustment.md)
- [\[ICLR 2026\] Extreme Weather Nowcasting via Local Precipitation Pattern Prediction](extreme_weather_nowcasting_via_local_precipitation_pattern_prediction.md)
- [\[ICLR 2026\] SONATA: Synergistic Coreset Informed Adaptive Temporal Tensor Factorization](sonata_synergistic_coreset_informed_adaptive_temporal_tensor_factorization.md)

</div>

<!-- RELATED:END -->
