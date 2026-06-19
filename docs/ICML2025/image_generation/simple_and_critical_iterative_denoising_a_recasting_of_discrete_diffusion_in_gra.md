---
title: >-
  [论文解读] Simple and Critical Iterative Denoising: A Recasting of Discrete Diffusion in Graph Generation
description: >-
  [ICML2025][图像生成][离散扩散] 提出 Simple Iterative Denoising (SID) 与 Critical Iterative Denoising (CID) 框架，通过假设中间噪声状态的条件独立性来消除离散扩散的复合去噪误差，并引入 Critic 网络自适应调节元素级重加噪概率，在图/分子生成任务上大幅超越标准离散扩散基线。
tags:
  - "ICML2025"
  - "图像生成"
  - "离散扩散"
  - "迭代去噪"
  - "Critic"
  - "图生成"
  - "分子生成"
  - "复合去噪误差"
  - "Flow Matching"
---

# Simple and Critical Iterative Denoising: A Recasting of Discrete Diffusion in Graph Generation

**会议**: ICML2025  
**arXiv**: [2503.21592](https://arxiv.org/abs/2503.21592)  
**代码**: [github.com/yoboget/sid](https://github.com/yoboget/sid)  
**领域**: 离散扩散 / 图生成  
**关键词**: 离散扩散, 迭代去噪, Critic, 图生成, 分子生成, 复合去噪误差, Flow Matching

## 一句话总结

提出 Simple Iterative Denoising (SID) 与 Critical Iterative Denoising (CID) 框架，通过假设中间噪声状态的条件独立性来消除离散扩散的复合去噪误差，并引入 Critic 网络自适应调节元素级重加噪概率，在图/分子生成任务上大幅超越标准离散扩散基线。

## 研究背景与动机

离散扩散模型（Discrete Diffusion Models, DDM）和离散流匹配（Discrete Flow Matching, DFM）在图结构等离散数据的生成建模中取得了显著进展。然而，这些模型面临一个核心挑战——**复合去噪误差（Compounding Denoising Error）**：

- **问题根源**：反向去噪过程中，中间噪声状态 $z_s$ 直接依赖于前一步的 $z_t$，形成马尔可夫链依赖。在生成初期，$Z_t$ 信息量有限，模型输出高熵分布，采样误差会沿时间步累积传播。
- **Mask 扩散尤其严重**：一旦元素被 unmask，后续步骤无法修正（$p_{s|t}(z|Z_t) = \delta_{z_t}(z)$，若 $z_t \neq \text{Mask}$），早期错误被永久锁定。
- **Marginal/Uniform 分布也受影响**：虽然每步元素均可改变，但由于去噪步中修改元素的概率受 $\Delta_t \cdot \dot{\alpha_t}/(1-\alpha_t)$ 约束（公式 5），修正概率极低，误差传播风险依然显著。

作者以在 Zinc250k 数据集上的分子有效性实验佐证：标准离散扩散最高只能达到约 80% 的分子有效率，而 SID 仅需少量去噪步即可接近 100%。

## 方法详解

### 1. Simple Iterative Denoising (SID)

**核心思想**：打破中间状态的马尔可夫链依赖，假设各时间步的噪声状态仅依赖于原始干净数据 $z_1$，而相互条件独立。

**加噪过程**（与 DDM 形式相同但语义不同）：

$$q_{t|1}(z|z_1) = \alpha_t \delta_{z_1}(z) + (1-\alpha_t) q_0(z)$$

其中 $\alpha_t \in [0,1]$ 为噪声调度参数，$q_0(z)$ 为噪声分布（可以是均匀分布、mask 分布或边缘分布）。

**关键假设（Assumption 3.1）**：条件独立性——

$$q_{t|1}(z|z_1) = q_{t|1}(z|z_s, z_1) \quad \forall\, t \neq s$$

即给定 $z_1$，任意两个时间步的噪声状态相互独立。

**去噪过程**：由条件独立假设推导出简化的去噪公式（Proposition 3.3）：

$$p_{s|t}(z|Z_t) = \alpha_s \cdot p_{1|t}(z|Z_t) + (1-\alpha_s) \cdot q_0(z)$$

可解释为两步操作：(1) 从噪声数据预测干净实例 $p_{1|t}(z_1|Z_t)$；(2) 对预测结果重新加噪 $q_{s|1}(z|z_1)$。关键区别在于 $z_s$ 仅依赖于预测的干净实例，而非当前噪声状态 $z_t$，从而切断了误差传播路径。

**等价性（Proposition 3.4）**：SID 的一步去噪等价于离散扩散的最大校正步（maximal corrector step）校正采样，因此可直接复用任何预训练的 DDM/DFM 去噪器，**无需重新训练**。

**训练损失**：与 DDM/DFM 完全一致，采用加权负对数似然：

$$\mathcal{L} = \mathbb{E}\left[\gamma \sum_{x_1^{(i)}} [-\log p_\theta(x_1^{(i)}|\mathcal{G}_t)] + (1-\gamma) \sum_{e_1^{(i,j)}} [-\log p_\theta(e_1^{(i,j)}|\mathcal{G}_t)]\right]$$

其中 $\gamma = n/(n+m)$ 平衡节点和边的权重。

### 2. Critical Iterative Denoising (CID)

**动机**：SID 中所有元素以相同概率 $1-\alpha_t$ 被重加噪，但不同元素在数据分布下的似然度不同——应对不太可能的元素提高重加噪概率。

**Critic 网络**：训练一个 GNN $f_\phi$ 估计元素级自适应噪声率 $\hat{\alpha}_t = p_\phi(a|\hat{Z}_{1|t})$，即预测每个去噪元素来自真实数据分布的概率。

**Critic 损失**：

$$\mathcal{L}_\phi = -\mathbb{E}_{t, \hat{Z}_{1|t}} \sum_i \log p_\phi(a_t^{(i)} | \hat{Z}_{1|t})$$

**最优 Critic（Theorem 4.1）**：

$$C^*({\hat{z}_{1|t}}) = \frac{\alpha_t \cdot p_{\text{data}}(\hat{z}_{1|t})}{\alpha_t \cdot p_{\text{data}}(\hat{z}_{1|t}) + (1-\alpha_t) \cdot p_{\text{pred}}(\hat{z}_{1|t})}$$

由此可得两个推论：若 $p_{\text{data}} = p_{\text{pred}}$，则 $\hat{\alpha}_t^* = \alpha_t$（退化为 SID）；若 $p_{\text{data}} > p_{\text{pred}}$，则 $\hat{\alpha}_t^* > \alpha_t$（保留高似然元素）；反之则增加重加噪概率。

**实现**：Critic 以残差 logit 参数化 $p_\phi(a|\hat{Z}_{1|t}) = \sigma(f_\phi(\hat{Z}_{1|t}, \alpha_t) + \sigma^{-1}(\alpha_t))$，在预训练去噪器上 post-hoc 训练，无需重训去噪器。

**推理流程**：每步 → (1) 去噪器预测 $\hat{Z}_{1|t}$ → (2) Critic 计算自适应 $\hat{\alpha}_{t+\Delta t}$ → (3) 以 $\hat{\alpha}_{t+\Delta t}$ 重加噪。

## 实验关键数据

### 分子生成（QM9 & ZINC250k）

| 模型 | QM9 Val.↑ | QM9 FCD↓ | ZINC Val.↑ | ZINC FCD↓ |
|:---|:---:|:---:|:---:|:---:|
| GDSS | 95.72 | 2.90 | 97.01 | 14.65 |
| DruM | 99.69 | 0.11 | 98.65 | 2.25 |
| DiGress | 98.19 | 0.10 | 94.99 | 3.48 |
| Marg. DDM | 95.73 | 1.09 | 80.40 | 8.50 |
| Mask DDM | 48.38 | 3.76 | 8.96 | 24.98 |
| **Marg. SID** | **99.67** | **0.50** | **99.50** | **2.01** |
| Mask SID | 96.43 | 1.80 | 93.85 | 9.05 |
| **Mask CID** | **99.92** | 1.76 | **99.97** | 3.46 |

- Marg. SID 在 ZINC250k 上 FCD 达到 2.01，优于所有基线（包括使用 1000 步的 DruM 的 2.25）
- Mask CID 在 ZINC250k 上有效率达 **99.97%**，无效分子数量仅为最优基线 DruM 的 1/50
- SID 仅使用 500 步，基线使用 1000 步

### 通用图生成（Planar & SBM）

| 模型 | Planar Spect.↓ | Planar V.U.N.↑ | SBM Spect.↓ | SBM V.U.N.↑ |
|:---|:---:|:---:|:---:|:---:|
| DruM | **6.2** | 90 | **5.0** | **85** |
| DiGress | 10.6 | 75 | 40.0 | 74 |
| Marg. DDM | 83.57 | 0.0 | 11.82 | 0.0 |
| **Marg. SID** | **7.62** | **91.3** | **5.93** | **63.5** |
| **Mask CID** | **6.40** | 66.0 | 11.94 | 19.0 |

- Marg. SID 在 Planar 上 V.U.N 达到 91.3%，超过 DruM 的 90%

### NFE 消融

- Mask CID 仅需 **32 步**即可达到 99% 以上的分子有效率
- 在 16 步极低 NFE 下，CID 在所有指标上仍优于其他模型

## 亮点与洞察

1. **条件独立假设简洁有效**：一个看似简单的假设（中间状态条件独立于彼此）从根本上消除了离散扩散的复合误差问题，理论清晰、实现简单
2. **训练免费升级**：SID 可直接复用任何预训练 DDM/DFM 去噪器，无需重训；Critic 也是 post-hoc 训练
3. **与校正采样的统一视角**：SID 等价于最大校正步的 corrector sampling，为理解不同采样策略提供了统一框架
4. **Critic 的 GAN 式动机**：最优 Critic 的形式类似 GAN 的判别器（比较 $p_{\text{data}}$ 与 $p_{\text{pred}}$），但用于指导重加噪而非对抗训练
5. **噪声分布的实证洞察**：边缘分布（marginal）一致优于 mask 分布用于图生成，mask 扩散的复合误差问题尤其严重

## 局限与展望

1. **仅在图/分子任务上验证**：未在文本、蛋白质等其他离散结构上进行实验，通用性有待验证
2. **Critic 在通用图上收益有限**：Planar/SBM 上 CID 相比 SID 提升不大，假设是因为通用图的数据分布偏差更分散
3. **SBM 数据集上 V.U.N 不及 DruM**：Marg. SID 的 63.5% 仍低于 DruM 的 85%，说明在某些图类型上方法仍有不足
4. **Critic 需要额外训练和推理开销**：每步需要两次前向传播（去噪器 + Critic），推理速度的绝对值未报告
5. **条件独立假设的理论限制**：虽然实验表现优异，但该假设不符合真实数据的马尔可夫性质，理论上不保证收敛到正确的数据分布

## 相关工作与启发

- **Cold Diffusion (Bansal et al., 2023)**：degrade-and-restore 策略的通用思路，本文可视为其在离散图结构上的实例化
- **DiGress (Vignac et al., 2023)**：使用边缘分布的离散扩散图生成基线
- **DruM (Jo et al., 2024)**：扩散桥模型，在通用图上仍是强基线
- **MDLM/UDLM (Sahoo et al., 2024)**：mask 去噪器的时间不变性，SID 继承了该性质
- **Planning 方法 (Liu et al., 2025)**：选择性去噪策略，与 CID 的自适应重加噪互补

## 评分

- 新颖性: ⭐⭐⭐⭐ — 条件独立假设简洁但有深度，Critic 的理论刻画优雅
- 实验充分度: ⭐⭐⭐⭐ — 4 个数据集 + 消融实验，公平对比设计好，但缺少非图领域验证
- 写作质量: ⭐⭐⭐⭐⭐ — 脉络清晰，理论推导完整，图示直观
- 价值: ⭐⭐⭐⭐ — 对离散扩散社区有重要贡献，训练免费升级的实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](../../ICCV2025/image_generation/lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [\[NeurIPS 2025\] Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction](../../NeurIPS2025/image_generation/toward_a_unified_geometry_understanding_riemannian_diffusion_framework_for_graph.md)
- [\[ICML 2025\] BRIDGE: Bootstrapping Text to Control Time-Series Generation via Multi-Agent Iterative Optimization and Diffusion Modeling](bridge_bootstrapping_text_to_control_time-series_generation_via_multi-agent_iter.md)
- [\[ICML 2025\] RestoreGrad: Signal Restoration Using Conditional Denoising Diffusion Models with Jointly Learned Prior](restoregrad_signal_restoration_using_conditional_denoising_diffusion_models_with.md)
- [\[ICLR 2026\] HOG-Diff: Higher-Order Guided Diffusion for Graph Generation](../../ICLR2026/image_generation/hog-diff_higher-order_guided_diffusion_for_graph_generation.md)

</div>

<!-- RELATED:END -->
