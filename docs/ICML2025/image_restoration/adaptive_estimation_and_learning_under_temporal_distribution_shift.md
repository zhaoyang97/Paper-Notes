---
title: >-
  [论文解读] Adaptive Estimation and Learning under Temporal Distribution Shift
description: >-
  [ICML 2025][图像恢复][distribution_shift] 提出基于小波软阈值的估计算法，在无需先验知识的情况下实现时间分布偏移下的最优逐点估计误差界，将序列非平稳性与小波域稀疏性建立联系，并应用于分布偏移下的二分类和全变分去噪问题。 传统统计估计假设数据是独立同分布（i.i.d.）的，但在股票价格预测、温度…
tags:
  - "ICML 2025"
  - "图像恢复"
  - "distribution_shift"
  - "去噪"
  - "nonstationary_estimation"
  - "binary_classification"
  - "total_variation"
---

# Adaptive Estimation and Learning under Temporal Distribution Shift

**会议**: ICML 2025  
**arXiv**: [2505.15803](https://arxiv.org/abs/2505.15803)  
**代码**: 未公开  
**领域**: 统计学习理论 / 分布偏移  
**关键词**: distribution_shift, wavelet_denoising, nonstationary_estimation, binary_classification, total_variation  

## 一句话总结

提出基于小波软阈值的估计算法，在无需先验知识的情况下实现时间分布偏移下的最优逐点估计误差界，将序列非平稳性与小波域稀疏性建立联系，并应用于分布偏移下的二分类和全变分去噪问题。

## 研究背景与动机

传统统计估计假设数据是独立同分布（i.i.d.）的，但在股票价格预测、温度预报、网络流量监控等实际场景中，数据分布往往随时间变化。本文放松了"同分布"假设，研究在时间分布偏移下如何利用历史观测序列估计最新时间步的真实值。

核心挑战在于：滑动窗口平均是最自然的估计方法，但窗口大小的选择面临经典的偏差-方差权衡——小窗口偏差小但方差大，大窗口方差小但偏差大。更关键的是，最优窗口取决于真实序列的平稳性/光滑度，而这在实践中通常是未知的。Mazzetto & Upfal (2023) 首先解决了这个问题，但他们基于"局部稳定性"的度量只能捕捉分段常数结构，对于更复杂的趋势模式表达能力有限。

## 方法详解

### 整体框架

本文提出用经典的小波软阈值去噪算法来解决时间分布偏移下的估计问题（Algorithm 1）。核心思路是：将时域中看似非平稳的信号变换到小波域，利用小波域的稀疏性实现高效估计。

算法流程：给定观测序列 $y_n, \ldots, y_1$，(1) 计算经验小波系数 $\tilde{\boldsymbol{\beta}} = \boldsymbol{W}\boldsymbol{y}$；(2) 对每个系数做软阈值操作 $\hat{\beta}_i = \text{sign}(\tilde{\beta}_i) \max\{|\tilde{\beta}_i| - \lambda, 0\}$，阈值 $\lambda = 2\sigma\sqrt{2\log(\log n / \delta)}$；(3) 逆小波变换重构信号并取最后一个坐标作为估计值。

与先前工作的根本区别在于：先前方法维护一个自适应窗口大小来平均相关历史观测，而小波方法不需要显式窗口，而是隐式地利用数据中最相关的部分。

### 关键设计一：基于小波稀疏性的通用估计误差界（Lemma 1）

Lemma 1 给出了算法的核心理论保证。对于观测模型 $y_i = \theta_i + \epsilon_i$（$\epsilon_i$ 为 i.i.d. $\sigma$-亚高斯噪声），设 $\mathcal{I}$ 为影响最终重构值 $\theta_1$ 的小波系数索引集，则以概率至少 $1 - \delta$：

$$|\hat{\theta}_1 - \theta_1| \leq \sum_{i \in \mathcal{I}} 6|W_{i,n}| \cdot (|\beta_i| \wedge \lambda)$$

其中 $\beta_i$ 为真实小波系数，$\lambda$ 为阈值参数。这个界的关键洞察是：估计误差由真实小波系数的稀疏性决定。当真实信号在小波域中稀疏时（即大部分 $\beta_i$ 很小），每一项 $|\beta_i| \wedge \lambda$ 取 $|\beta_i|$ 而非 $\lambda$，从而大幅降低误差。

### 关键设计二：Haar 小波下的变分界与最优性（Theorem 2）

当使用 Haar 小波系统时，Theorem 2 证明算法恢复了 Mazzetto & Upfal (2023) 的变分界：

$$|\hat{\theta}_1 - \theta_1| \leq \kappa \cdot U(r^*)$$

其中 $U(r) = \max_{t \in S(r)} |\bar{\theta}_{t:1} - \theta_1| \vee \sigma/\sqrt{r}$，$r^*$ 为使 $U(r)$ 最小化的最优时间点，$\kappa = (4\sqrt{2\log(\log n/\delta)} \vee 2\sqrt{2})(\log_2 n + 1)$。

$U(r)$ 刻画了一个偏差-方差权衡：$|\bar{\theta}_{t:1} - \theta_1|$ 是偏差项（过去均值偏离当前值），$\sigma/\sqrt{r}$ 是方差项。算法自动找到最优 $r^*$ 而无需事先知道非平稳性水平。

### 关键设计三：分布偏移下二分类的 ERM 算法（Theorem 9）

将小波估计器应用于二分类的 excess risk 控制。对假设类 $\mathcal{F}$ 中每个函数 $f$，用 Algorithm 1 估计其在最新分布上的损失，然后做 ERM。Excess risk 界为：

$$L_{\hat{f}} - L_{f_*} \leq 8\sqrt{\frac{2d\log(2n)}{n}} + 2\sqrt{\frac{2\log(3/\delta)}{n}} + \sqrt{3} \cdot \sup_{f \in \mathcal{F}} \sum_{i \in \mathcal{I}} 6|W_{i,n}| \cdot (|\beta_i^f| \wedge \lambda')$$

关键优势：只需一次 ERM 调用（先前方法需 $O(\log n)$ 次），且目标函数对可微代理损失保持可微性，支持梯度下降优化。

### 关键设计四：与全变分去噪的联系（Theorem 11）

证明了满足 Theorem 2 或 Corollary 3 形式界的任何算法，迭代运行后自动获得 TV 去噪的最优估计速率：

$$R_{\text{sq}} = \tilde{O}(n^{1/3} C^{2/3} \sigma^{4/3}), \quad R_{\text{abs}} = \tilde{O}(n^{2/3} C^{1/3} \sigma^{2/3})$$

这揭示了 Mazzetto & Upfal (2023) 和 Han et al. (2024) 的算法也是 TV 去噪的最优算法，这一事实此前未被发现。

## 实验关键数据

### 表1：已知噪声标准差下不同算法的 MSE（合成数据，Doppler 信号）

| 噪声水平 | AVG | HAAR (本文) | DB8 (本文) |
|:--------:|:---:|:----------:|:---------:|
| 0.2 | 0.511 | 0.068 | **0.007** |
| 0.5 | 0.509 | 0.155 | **0.032** |
| 1.0 | 0.508 | 0.155 | **0.129** |

> DB8 在 Doppler 信号上将 MSE 降低了一个数量级以上（噪声 0.2 时从 0.511 降至 0.007）。

### 表2：未知噪声标准差下不同算法的 MSE（合成数据，Doppler 信号）

| 噪声水平 | AVG | ARW | Aligator | HAAR (本文) | DB8 (本文) |
|:--------:|:---:|:---:|:--------:|:----------:|:---------:|
| 0.2 | 0.512 | 0.384 | 0.234 | 0.053 | **0.020** |
| 0.5 | 0.512 | 0.400 | 0.235 | 0.065 | **0.044** |
| 1.0 | 0.514 | 0.407 | 0.235 | **0.088** | 0.129 |

> 即使不知道噪声水平，小波方法仍显著优于基线。高噪声时 HAAR 优于 DB8，因高阶小波在噪声大时方差增大。

### 表3：真实数据（迪拜房产数据集）模型选择实验

| 方法 | 79%-1% 划分 | 75%-5% 划分 |
|:----:|:----------:|:----------:|
| ARW | 0.0790 | **0.0719** |
| HAAR (本文) | **0.0722** | 0.0736 |
| DB8 (本文) | 0.0762 | 0.0768 |

> 验证数据稀缺时（1% 验证集），HAAR 优于 ARW；非平稳性强的真实数据中低阶 Haar 小波反而更合适。

## 关键发现

1. **非平稳性与小波稀疏性的等价**：序列的时间非平稳程度直接对应其小波系数的稀疏程度，这一联系比"局部稳定性"概念更具一般性
2. **自适应性**：算法无需任何关于分布偏移程度的先验知识，自动适应序列的非平稳性水平
3. **计算效率提升**：二分类场景只需 1 次 ERM 调用（vs. 先前 $O(\log n)$ 次），代价仅为 excess risk 界膨胀 $O(\log n)$ 因子
4. **TV 去噪的新最优算法**：满足逐点误差界的算法自动成为 TV 去噪的最优算法，这一蕴含关系此前未知
5. **高阶小波的优势与局限**：DB8 在光滑趋势上远优于 HAAR，但在高度非平稳/高噪声场景中 HAAR 更稳健

## 亮点与洞察

- **视角转换的力量**：从"选最优窗口"到"小波域去噪"的视角转换，将一个难以直接解决的自适应问题化归为经典的小波理论，展示了信号处理工具在统计学习理论中的深层价值
- **理论与实践一致**：Lemma 1 预测的小波稀疏性优势在实验中得到清晰验证——Doppler 信号在 DB8 小波域真正稀疏，AVG 的 MSE 可达 0.51 而 DB8 仅 0.007
- **可微性保持**：基于软阈值的 ERM 目标保持可微性，这对深度学习时代的实际应用至关重要
- **小波选择类比核选择**：作者将小波基的选择类比为高斯过程中核函数的选择，都需要根据应用场景的先验知识来指导

## 局限性

1. **小波基选择**缺乏原则性方法，目前依赖对数据趋势的先验直觉
2. **噪声假设**要求 i.i.d. 亚高斯噪声，真实场景中噪声可能异方差或存在相关性
3. 缺少 Lemma 1 严格优于 Theorem 2 的**具体构造**，仅有经验性比较
4. 高阶小波引入数值不稳定性和更大方差，在**小样本高噪声**场景中可能失效
5. 未讨论如何将方法扩展到**变点检测**等相关问题

## 相关工作与启发

- **Mazzetto & Upfal (2023)**：首先解决该开放问题，基于局部稳定性度量，是本文的直接改进目标
- **Han et al. (2024) (ARW)**：另一自适应方法，不需要噪声标准差的先验知识
- **Donoho (1995)**：小波软阈值的经典工作，本文将其从信号去噪迁移到分布偏移估计
- **Baby et al. (2021) (Aligator)**：通过在线学习实现 TV 去噪的最优 MSE 率，但缺少逐点误差保证
- **启发**：信号处理中的经典工具（小波变换、软阈值）在现代机器学习理论问题中仍有深刻的应用价值

## 评分

| 维度 | 分数 (1-5) | 说明 |
|:----:|:---------:|:----:|
| 理论深度 | 5 | 多层次理论体系：通用界-Haar界-分类应用-TV去噪联系 |
| 新颖性 | 4 | 视角新颖但核心工具（小波阈值）是经典的，创新在于联系的建立 |
| 实验充分性 | 4 | 合成+真实数据验证，但真实数据实验规模较小 |
| 实用性 | 3 | 理论结果优美但小波选择的实践指导有限 |
| 写作质量 | 4 | 逻辑清晰，贡献明确，但符号较重 |

**总分**: 4.0/5 — 一篇理论扎实、视角独特的工作，成功将小波理论引入分布偏移估计问题并揭示了重要的跨领域联系。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] RegionFuse: Region-Adaptive Pixel Distribution Learning for Infrared and Visible Image Fusion](../../CVPR2026/image_restoration/regionfuse_region-adaptive_pixel_distribution_learning_for_infrared_and_visible_.md)
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](../../ICCV2025/image_restoration/learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)
- [\[ICCV 2025\] Metric Convolutions: A Unifying Theory to Adaptive Image Convolutions](../../ICCV2025/image_restoration/metric_convolutions_a_unifying_theory_to_adaptive_image_convolutions.md)
- [\[NeurIPS 2025\] Adaptive Discretization for Consistency Models](../../NeurIPS2025/image_restoration/adaptive_discretization_for_consistency_models.md)
- [\[NeurIPS 2025\] MAP Estimation with Denoisers: Convergence Rates and Guarantees](../../NeurIPS2025/image_restoration/map_estimation_with_denoisers_convergence_rates_and_guarantees.md)

</div>

<!-- RELATED:END -->
