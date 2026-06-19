---
title: >-
  [论文解读] Risk and Cross Validation in Ridge Regression with Correlated Samples
description: >-
  [ICML2025][时间序列][岭回归] 利用随机矩阵理论和自由概率技术，为训练样本具有任意相关性的高维岭回归推导了精确的风险渐近公式，提出了修正的广义交叉验证估计器 CorrGCV，在样本相关条件下准确预测样本外风险。 - 经典统计假设失效：传统岭回归理论假设训练样本 i.i.d.，但时间序列数据（金融、气候、神经科学）…
tags:
  - "ICML2025"
  - "时间序列"
  - "岭回归"
  - "交叉验证"
  - "相关样本"
  - "随机矩阵理论"
  - "自由概率"
  - "时间序列预测"
  - "高维渐近"
---

# Risk and Cross Validation in Ridge Regression with Correlated Samples

**会议**: ICML2025  
**arXiv**: [2408.04607](https://arxiv.org/abs/2408.04607)  
**代码**: [Pehlevan-Group/S_transform](https://github.com/Pehlevan-Group/S_transform)  
**领域**: 时间序列  
**关键词**: 岭回归, 交叉验证, 相关样本, 随机矩阵理论, 自由概率, 时间序列预测, 高维渐近

## 一句话总结

利用随机矩阵理论和自由概率技术，为训练样本具有任意相关性的高维岭回归推导了精确的风险渐近公式，提出了修正的广义交叉验证估计器 CorrGCV，在样本相关条件下准确预测样本外风险。

## 研究背景与动机

- **经典统计假设失效**：传统岭回归理论假设训练样本 i.i.d.，但时间序列数据（金融、气候、神经科学）天然存在样本间相关性，这使已有理论不再适用
- **高维岭回归的理论进展**：近年来大量工作（Hastie et al. 2022; Bordelon et al. 2020; Canatar et al. 2021 等）在高维比例极限下给出了精确的样本外风险渐近，但**几乎全部假设 i.i.d. 样本**
- **GCV 估计器的局限**：经典广义交叉验证（GCV）在 i.i.d. 设定下渐近精确，但在样本相关时**系统性地估计偏差**——已有的修正尝试（Altman 1990; Carmack et al. 2012）也未在高维下证明渐近精确性
- **实际需求**：时间序列回归需要能正确考虑相关性的超参数调优方法（尤其是调节正则化强度 $\lambda$）

## 方法详解

### 问题建模

考虑岭回归，训练集 $\mathcal{D} = \{(\mathbf{x}_t, y_t)\}_{t=1}^T$，损失函数：

$$L(\mathbf{w}) = \frac{1}{T}\sum_{t=1}^T (y_t - \mathbf{x}_t^\top \mathbf{w})^2 + \lambda \|\mathbf{w}\|^2$$

数据模型：$\mathbf{y} = \mathbf{X}\bar{\mathbf{w}} + \boldsymbol{\epsilon}$，设计矩阵 $\mathbf{X} = \mathbf{K}^{1/2}\mathbf{Z}\boldsymbol{\Sigma}^{1/2}$，其中：

- $\boldsymbol{\Sigma} \in \mathbb{R}^{N \times N}$：特征-特征协方差
- $\mathbf{K} \in \mathbb{R}^{T \times T}$：样本-样本相关矩阵（**这是本文的核心新元素**）
- $\mathbf{Z}$：i.i.d. 标准高斯矩阵
- 噪声协方差：$\mathbb{E}[\epsilon_t \epsilon_s] = \sigma_\epsilon^2 K'_{ts}$

### 理论工具：确定性等价

核心技术是通过**自由概率的 S-变换**建立确定性等价。定义自由度：

$$\mathrm{df}_1 = \frac{1}{N}\mathrm{Tr}[\boldsymbol{\Sigma}(\boldsymbol{\Sigma}+\kappa)^{-1}], \quad \tilde{\mathrm{df}}_1 = \frac{1}{T}\mathrm{Tr}[\mathbf{K}(\mathbf{K}+\tilde{\kappa})^{-1}]$$

其中重正化岭参数 $\kappa, \tilde{\kappa}$ 通过 S-变换自洽方程确定，满足对偶关系 $\kappa\tilde{\kappa}/\lambda = 1/\tilde{\mathrm{df}}_1$。

**一点强确定性等价**（Lemma 2.2）：样本协方差的预解可用总体协方差的预解近似

$$\hat{\boldsymbol{\Sigma}}(\hat{\boldsymbol{\Sigma}}+\lambda)^{-1} \simeq \boldsymbol{\Sigma}(\boldsymbol{\Sigma}+\kappa)^{-1}$$

**两点强确定性等价**（Lemma 2.3, 2.4）：用于推导方差项的精确渐近。

### 主要结果

**结果一：匹配相关时的精确风险**（$\mathbf{K} = \mathbf{K}'$，Theorem 3.2）

$$R_g \simeq \frac{\kappa^2}{1-\gamma}\bar{\mathbf{w}}^\top \boldsymbol{\Sigma}(\boldsymbol{\Sigma}+\kappa)^{-2}\bar{\mathbf{w}} + \frac{\gamma}{1-\gamma}\sigma_\epsilon^2$$

其中 $\gamma = \frac{\mathrm{df}_2}{\mathrm{df}_1}\frac{\tilde{\mathrm{df}}_2}{\tilde{\mathrm{df}}_1}$（注意 $\gamma$ 现在同时依赖特征和样本两侧的自由度）。

**结果二：CorrGCV 估计器**

$$R_{out} = S(\mathrm{df}_1) \frac{\tilde{\mathrm{df}}_1}{\tilde{\mathrm{df}}_1 - \tilde{\mathrm{df}}_2} \hat{R}_{in}$$

- **渐近无偏、可集中**：在 $N, T \to \infty$ 下精确
- **仅需训练数据即可计算**：$S, \tilde{\mathrm{df}}_1, \tilde{\mathrm{df}}_2$ 均可从数据估计

**结果三：不匹配相关/协变量漂移的一般风险**（Theorem 3.3）

风险分解为三项：

$$R_g \simeq \underbrace{\kappa^2 \bar{\mathbf{w}}^\top(\boldsymbol{\Sigma}+\kappa)^{-1}\boldsymbol{\Sigma}'(\boldsymbol{\Sigma}+\kappa)^{-1}\bar{\mathbf{w}}}_{\mathrm{Bias}^2} + \underbrace{\mathrm{Var}_{\mathbf{X}}}_\text{协变量方差} + \underbrace{\mathrm{Var}_{\mathbf{X}\boldsymbol{\epsilon}}}_\text{噪声方差}$$

揭示了**协变量漂移与噪声-样本相关不匹配之间的对偶性**。

**结果四：时间序列中的相关测试点**

当测试点与训练集存在相关性时（近期预测），模型表现会过于乐观。本文精确刻画了**预测精度随预测距离的衰减**。

### 缩放律不变性

在幂律特征谱 $\lambda_k \sim k^{-\alpha}$ 下，最优风险的缩放率 $R_g \sim T^{-2\alpha\min(r,1)}$ 不受平稳过程相关结构的影响。

## 实验关键数据

### 主要对比：CorrGCV vs. Naïve GCV

| 估计器 | 弱相关 ($\xi=10^{-2}$) | 强相关 ($\xi=10^2$) |
|--------|----------------------|---------------------|
| Naïve GCV₁ $(1-q\cdot\mathrm{df}_1)^{-2}$ | ≈准确 | 严重偏差 |
| Naïve GCV₂ (Altman 1990) $S^2$ | ≈准确 | 低估风险 |
| Carmack et al. (2012) | ≈准确 | 高估风险 |
| **CorrGCV（本文）** | **准确** | **准确** |

### 超参数调优结果

- 仅 CorrGCV 能正确定位**最优正则化参数 $\lambda^*$**
- Naïve GCV 在强相关下错误定位最优 $\lambda$，导致次优泛化

### 时间序列实验

- 指数相关 $\mathbb{E}[\mathbf{x}_t \cdot \mathbf{x}_s] \propto e^{-|t-s|/\xi}$ 下，理论预测与 10 次重复实验精确吻合
- 相关测试点的风险随预测距离增大单调上升，验证了近期预测过于乐观的理论预言

### 关键消融

| 消融维度 | 发现 |
|---------|------|
| $\mathbf{K} = \mathbf{K}'$ vs. $\mathbf{K} \neq \mathbf{K}'$ | 匹配时 CorrGCV 精确；不匹配时需额外信息，无法仅从训练数据估计 |
| 过参数 $q>1$ vs. 欠参数 $q<1$ | 两种情况理论均成立 |
| 幂律协方差谱 | 缩放律指数不受相关结构影响 |

## 亮点与洞察

1. **填补重要理论空白**：首次为样本相关的高维岭回归给出渐近精确的风险公式和可计算的无偏风险估计器
2. **CorrGCV 的实用价值**：仅需训练数据即可计算，直接用于相关数据下的正则化参数调优
3. **协变量漂移-噪声相关不匹配的对偶性**：一个优雅的理论发现，统一了两类看似不同的问题
4. **重正化视角**：随机波动的效果被吸收为重正化岭参数 $\kappa$，即使 $\lambda \to 0$ 也可能有 $\kappa > 0$（隐式正则化）
5. **对时间序列预测的直接指导**：精确量化了近期预测的乐观偏差

## 局限与展望

- **高斯假设**：数据模型假设矩阵高斯分布，对非高斯时间序列（如金融收益的厚尾分布）的适用性还需验证
- **线性模型**：限于岭回归，未扩展到核方法或神经网络
- **已知相关结构**：CorrGCV 假设 $\mathbf{K}$ 已知或可准确估计，实际中相关结构估计本身可能不准确
- **平稳性假设**：主要结果假设时间序列平稳，非平稳场景需进一步研究
- **噪声-样本不匹配时无 GCV**：当 $\mathbf{K} \neq \mathbf{K}'$ 时，不存在仅从训练数据可估计的精确风险估计器

## 相关工作与启发

- **高维岭回归理论**：Hastie et al. (2022) 综述了比例极限下的岭回归；Bordelon et al. (2020) 和 Canatar et al. (2021) 建立了谱偏置理论；本文将这些结果推广到相关样本
- **GCV 理论**：Golub et al. (1979) 和 Craven & Wahba (1978) 提出经典 GCV；Jacot et al. (2020) 和 Atanasov et al. (2024) 证明其在高维下渐近精确；本文指出 GCV 与 S-变换的联系并推广
- **相关数据的 CV**：Altman (1990)、Carmack et al. (2012) 提出过修正，但均未证明高维渐近精确性
- **自由概率**：大量借鉴 Potters & Bouchaud (2020) 的技术框架
- 本文的理论框架为核回归、随机特征模型等在时间序列场景下的分析奠定了基础

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次为相关样本高维岭回归建立精确渐近和可计算估计器
- 实验充分度: ⭐⭐⭐⭐ 多种相关结构下验证充分，理论-实验匹配精确
- 写作质量: ⭐⭐⭐⭐ 数学严谨，物理直觉清晰，结构合理
- 价值: ⭐⭐⭐⭐⭐ 对时间序列回归的超参数调优有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] From Samples to Scenarios: A New Paradigm for Probabilistic Forecasting](../../ICLR2026/time_series/from_samples_to_scenarios_a_new_paradigm_for_probabilistic_forecasting.md)
- [\[ACL 2025\] CTPD: Cross-Modal Temporal Pattern Discovery for Enhanced Multimodal Electronic Health Records Analysis](../../ACL2025/time_series/ctpd_cross-modal_temporal_pattern_discovery_for_enhanced_multimodal_electronic_h.md)
- [\[AAAI 2026\] HydroDCM: Hydrological Domain-Conditioned Modulation for Cross-Reservoir Inflow Prediction](../../AAAI2026/time_series/hydrodcm_hydrological_domain-conditioned_modulation_for_cross-reservoir_inflow_p.md)
- [\[ICML 2026\] HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation](../../ICML2026/time_series/helix_hybrid_encoding_with_learnable_identity_and_cross-dimensional_synthesis_fo.md)
- [\[ICLR 2026\] Are Global Dependencies Necessary? Scalable Time Series Forecasting via Local Cross-Variate Modeling](../../ICLR2026/time_series/are_global_dependencies_necessary_scalable_time_series_forecasting_via_local_cro.md)

</div>

<!-- RELATED:END -->
