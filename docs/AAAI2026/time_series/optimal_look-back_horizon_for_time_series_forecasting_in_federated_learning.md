---
title: >-
  [论文解读] Optimal Look-back Horizon for Time Series Forecasting in Federated Learning
description: >-
  [AAAI 2026][时间序列][time series forecasting] 提出联邦学习场景下时间序列预测的最优回看窗口（look-back horizon）理论框架，通过合成数据生成器（SDG）和内禀空间表示，将预测损失分解为贝叶斯不可约误差和近似误差，证明总损失关于窗口长度是单峰的，最小充分窗口为最优解。
tags:
  - "AAAI 2026"
  - "时间序列"
  - "time series forecasting"
  - "联邦学习"
  - "look-back horizon"
  - "intrinsic space"
  - "Bayesian loss decomposition"
---

# Optimal Look-back Horizon for Time Series Forecasting in Federated Learning

**会议**: AAAI 2026  
**arXiv**: [2511.12791](https://arxiv.org/abs/2511.12791)  
**代码**: 无  
**领域**: 时间序列预测 / 联邦学习  
**关键词**: time series forecasting, federated learning, look-back horizon, intrinsic space, Bayesian loss decomposition

## 一句话总结

提出联邦学习场景下时间序列预测的最优回看窗口（look-back horizon）理论框架，通过合成数据生成器（SDG）和内禀空间表示，将预测损失分解为贝叶斯不可约误差和近似误差，证明总损失关于窗口长度是单峰的，最小充分窗口为最优解。

## 研究背景与动机

时间序列预测（TSF）中回看窗口 $H$ 的选择是核心建模决策，直接影响模型复杂度和预测精度。传统做法是将 $H$ 作为超参数通过交叉验证调优，缺乏理论指导。

Shi et al. (2024) 的 scaling law 理论将时间序列嵌入内禀表示空间，将预测损失分解为贝叶斯误差和近似误差，但**假设数据集中式、IID、同构模型架构**。在联邦学习（FL）中，数据分布在异构客户端上，具有不同的分布、序列长度和领域特征，全局固定窗口可能导致局部动态与模型输入不匹配。

核心问题：**如何在非 IID 联邦场景下，为每个客户端自适应地确定最优回看窗口？**

本文扩展 Shi 的理论到联邦非 IID 设置，引入 SDG 捕获客户端异构性，构建满足几何和统计性质的内禀表示空间，推导出最优窗口的闭式表达。

## 方法详解

### 整体框架

框架包含四个层次：(1) SDG 建模客户端时间序列的核心结构（AR + 季节性 + 趋势 + 噪声）；(2) 五步变换将时间序列窗口映射到内禀空间；(3) 损失分解为贝叶斯（不可约）+ 近似两项；(4) 证明总损失单峰性并给出最优窗口。

### 关键设计

**1. 合成数据生成器（SDG）**

为客户端 $k$、特征 $f$、时间步 $t$ 建模：

$$\hat{x}_{f,t,k} = \sum_{j=1}^{J} A_{f,j,k} \cdot \sin\left(\frac{2\pi t}{T_{f,j,k}} + \theta_{f,j,k}\right) + \sum_{i=1}^{p} \phi_{k,i} x_{f,t-i,k} + \beta_{f,k} t + \epsilon_{f,t,k}$$

其中季节性用正弦和表示（振幅 $A$、周期 $T$、相位 $\theta$），时间依赖用 AR(p) 建模（客户端特定系数 $\phi_{k,i}$），趋势为线性项，噪声 $\epsilon_{f,t,k} \sim \mathcal{N}(\mu_{f,k}, \sigma_{f,k}^2)$。

客户端异构性通过仿射变换引入特征偏斜：$x_{f,t,k} = \Lambda_{f,k} \tilde{x}_{f,t,k} + \delta_{f,k}$。

SDG 在真实温度数据上验证表现良好：均值偏差 $\approx 3.6 \times 10^{-3}$，ACF $L^2 \approx 3.7 \times 10^{-6}$，KS 统计量 = 0.042。

**2. 内禀空间构建与损失分解**

五步变换管线：(1) 客户端归一化消除仿射偏斜；(2) 窗口展平为向量；(3) 全局协方差估计 + 特征分解；(4) 基于 SDG 估计内禀维度；(5) PCA 投影到内禀空间。

内禀维度估计：

$$d_{I,k}(H) \approx F \cdot \left(\min\{H, \ell_{\mathrm{AR},k}\} + g_k(H) + 1\right)$$

其中有效 AR 记忆长度 $\ell_{\mathrm{AR},k} = \lceil \frac{\ln(1/(1-\epsilon))}{-\ln \rho_k} \rceil$，季节性复杂度 $g_k(H) = 2\sum_{j=1}^{J} w_{j,k} \cdot \min(1, H/T_{j,k}^*)$。

联邦损失分解（Theorem 1）：

$$L(H,S;m) = L_{\mathrm{Bayes}}(H,S) + L_{\mathrm{approx}}(H,S;m)$$

服务器级聚合：$L_{\mathrm{Bayes}}^{(\mathrm{server})} = \sum_{k=1}^{K} \pi_k L_{\mathrm{Bayes}}^{(k)}$。

**3. 最优窗口理论**

**贝叶斯损失随 $H$ 单调递减并饱和**（更多历史 → 更好识别季节性/AR 结构），客户端级分解为三项：

$$L_{\mathrm{Bayes}}^{(k)}(H,S) = L_{\mathrm{AR}}^{(k)}(S) + L_{\mathrm{seas}}^{(k)}(H) + L_{\mathrm{trend}}^{(k)}(H)$$

AR 项上界：$L_{\mathrm{AR}}^{(k)}(S) \leq \sum_f \sigma_{f,k}^2 \cdot \frac{1 - \rho_k^{2S}}{1-\rho_k^2}$

**近似损失随 $H$ 单调递增**（更高内禀维度 + 更少有效样本）：

$$L_{\mathrm{approx}}^{(k)}(H;m) \lesssim \left(K_2^2 d_{I,k}(H)^2\right)^{\frac{d_{I,k}(H)}{4+d_{I,k}(H)}} + \left(\frac{d_{I,k}(H) H}{D_k}\right)^{\frac{4}{4+d_{I,k}(H)}}$$

**Theorem 4（单峰性与最优窗口）**：总损失在 $[1, H_k^*(\delta)]$ 上严格递减，在 $[H_k^*(\delta), \infty)$ 上严格递增，最小充分窗口 $H_k^*(\delta)$ 即为最优解：

$$H_k^*(\delta) = \max\{\ell_{\mathrm{AR},k}, T_k^{(\tau)}\}$$

联邦全局窗口通过加权截尾均值聚合：$H_{\mathrm{server}}^* = \mathrm{TrimMean}_\alpha(\{H_k^*(\delta)\}; \{w_k\})$

### 损失函数 / 训练策略

本文为纯理论工作，不涉及具体的训练策略。损失分析基于平方损失 $\|V - m(U)\|^2$ 的期望，在内禀空间中推导。

## 实验关键数据

### 主实验

本文为理论分析论文，主要实验为 SDG 验证：

| 指标 | 数值 |
|------|------|
| 均值偏差 $\Delta\mu$ | $\approx 3.6 \times 10^{-3}$ |
| ACF $L^2$ gap（30 lags） | $\approx 3.7 \times 10^{-6}$ |
| 归一化 PSD $L^2$ gap | $\approx 8.2 \times 10^{-3}$ |
| KS 统计量 | 0.042 |
| 随机森林判别准确率（50步窗口） | 0.892 |

使用 2020 年 Jena 气象站温度数据（10 分钟分辨率，$N = 52,696$），AR 阶数 $p=30$，主要周期 $T_1 = 144$（日周期）。

### 消融实验

**内禀维度随窗口 $H$ 的变化**（理论分析）：

| $H$ 范围 | $d_{I,k}(H)$ 行为 |
|----------|-------------------|
| $H < \ell_{\mathrm{AR},k}$ | 线性增长（AR 信息未饱和） |
| $\ell_{\mathrm{AR},k} \leq H < \max T_{j,k}^*$ | 次线性增长（季节性逐步解析） |
| $H \geq H_{\mathrm{id}}$ | 饱和不再增长 |

**损失项随 $H$ 的行为**：

| 损失项 | 随 $H$ 增大 | 机制 |
|--------|-----------|------|
| $L_{\mathrm{Bayes}}$ | 单调递减 → 饱和 | 更多历史 → 更好识别 AR/季节性 |
| $L_{\mathrm{approx}}$ | 单调递增 | 内禀维度增大 + 有效样本减少 ($D_k/H$) |
| $L_{\mathrm{total}}$ | 先降后升（单峰） | 两者权衡在 $H_k^*$ 达到最优 |

### 关键发现

- 回看窗口存在理论最优值，过短（信息不足）和过长（过拟合）都会损害性能
- 最优窗口由两个因素决定：有效 AR 记忆长度和季节周期覆盖
- 不同客户端的最优窗口不同（取决于局部 AR 结构和季节性特征）
- 联邦场景需要鲁棒聚合（截尾均值）以避免极端客户端的影响

## 亮点与洞察

- **首个联邦 TSF 窗口选择理论**：将 Shi et al. 的中心化理论扩展到非 IID 联邦场景
- **可解读的最优窗口公式**：$H_k^* = \max\{\ell_{\mathrm{AR},k}, T_k^{(\tau)}\}$ 直接由信号参数（AR 记忆、季节周期）确定
- **严格的单峰性证明**：为窗口选择提供了理论保证，无需暴力搜索

## 局限与展望

- 纯理论工作，缺乏大规模真实数据集上的实验验证
- SDG 假设加性结构（AR + 季节 + 趋势 + 高斯噪声），不覆盖非线性交互、regime 切换等复杂模式
- 假设局部平稳性和稳定 AR 结构，在长记忆或接近单位根的场景下可能失效
- 联邦场景下全局协方差估计需要安全聚合，隐私性未讨论
- 重叠窗口被近似当作独立样本处理，可能高估有效样本量

## 相关工作与启发

- **vs Shi et al. (2024)**: Shi 在中心化 IID 设置下建立了 scaling law；本文扩展到联邦非 IID 设置，引入客户端特定的 SDG 和内禀空间
- **vs FedProx**: FedProx 通过正则化稳定联邦优化，但不涉及窗口选择；本文提供了窗口选择的理论基础
- **vs iTransformer/NLinear**: 实证发现不同模型有不同最优窗口（channel-dependent 模型偏短、线性模型偏长），本文给出这一现象的理论解释

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次在联邦 TSF 中建立最优窗口的理论框架
- 实验充分度: ⭐⭐⭐ 仅有 SDG 验证实验，无真实联邦数据集上的对比实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，符号体系一致
- 价值: ⭐⭐⭐⭐ 理论贡献有意义，但实际指导意义有待实验验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] FeDaL: Federated Dataset Learning for General Time Series Foundation Models](../../ICLR2026/time_series/fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)
- [\[AAAI 2026\] Detecting the Future: All-at-Once Event Sequence Forecasting with Horizon Matching](detecting_the_future_all-at-once_event_sequence_forecasting_with_horizon_matchin.md)
- [\[ICLR 2026\] A Unified Federated Framework for Trajectory Data Preparation via LLMs](../../ICLR2026/time_series/a_unified_federated_framework_for_trajectory_data_preparation_via_llms.md)
- [\[CVPR 2026\] Towards Uncertainty-aware Unsupervised Domain Adaptation for Videos and Time-Series with Causal Optimal Transport](../../CVPR2026/time_series/towards_uncertainty-aware_unsupervised_domain_adaptation_for_videos_and_time-ser.md)
- [\[CVPR 2026\] Real-Time Long Horizon Air Quality Forecasting via Group-Relative Policy Optimization](../../CVPR2026/time_series/real-time_long_horizon_air_quality_forecasting_via_group-relative_policy_optimiz.md)

</div>

<!-- RELATED:END -->
