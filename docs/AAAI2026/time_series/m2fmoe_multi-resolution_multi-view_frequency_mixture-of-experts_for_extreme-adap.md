---
title: >-
  [论文解读] M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting
description: >-
  [AAAI 2026][时间序列][极端事件预测] 提出 M2FMoE，通过傅里叶和小波双视角的频域混合专家建模常规与极端模式，结合跨视角共享频段分割器对齐两域语义、多分辨率自适应融合捕获多尺度信息、时序门控整合长短期特征，在 5 个水文极端事件数据集上无需极端事件标签即超越所有 SotA（含使用标签的方法），平均 RMSE 提升 22.3%。
tags:
  - "AAAI 2026"
  - "时间序列"
  - "极端事件预测"
  - "频域建模"
  - "混合专家"
  - "小波变换"
  - "傅里叶变换"
  - "多分辨率融合"
  - "水文预测"
---

# M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting

**会议**: AAAI 2026  
**arXiv**: [2601.08631](https://arxiv.org/abs/2601.08631)  
**代码**: [https://github.com/Yaohui-Huang/M2FMoE](https://github.com/Yaohui-Huang/M2FMoE)  
**领域**: 时间序列预测  
**关键词**: 极端事件预测, 频域建模, 混合专家, 小波变换, 傅里叶变换, 多分辨率融合, 水文预测

## 一句话总结
提出 M2FMoE，通过傅里叶和小波双视角的频域混合专家建模常规与极端模式，结合跨视角共享频段分割器对齐两域语义、多分辨率自适应融合捕获多尺度信息、时序门控整合长短期特征，在 5 个水文极端事件数据集上无需极端事件标签即超越所有 SotA（含使用标签的方法），平均 RMSE 提升 22.3%。

## 研究背景与动机

**领域现状**：时间序列预测在能源、交通、环境监测等领域至关重要。水文预测中的极端事件（暴雨、洪水、水位骤升）因稀有、突发、高方差而特别难以预测。

**现有痛点**：  
   - 主流深度学习模型（Transformer、MLP 等）聚焦于主导的规律性模式（周期、平滑趋势），对极端事件的不规则高频突变建模不力  
   - 频域方法（FFT / 小波）各有缺陷：FFT 提供全局频率但缺乏时间定位，小波有时频局部化但低频分辨率不足  
   - 已有极端自适应方法（DAN、MCANN 等）依赖极端事件标签作为辅助监督，限制了泛化能力  
   - 双视角策略引入跨视角频谱不对齐问题：FFT 的均匀频率轴与 CWT 的非线性尺度轴导致同一信号在两域中位置不一致

**核心矛盾**：极端事件与常规事件的频谱特性显著不同（极端：宽频多峰慢衰减 vs 常规：窄带低频集中），需要自适应聚焦不同频段，且单一频域视角无法兼顾全局与局部信息。

**本文目标** 在不依赖极端事件标签的前提下，通过多视角、多分辨率的频域建模统一捕获常规与极端两种时序模式。

**切入角度**：利用 MoE 的专家特化能力，将不同频段分配给不同专家；用共享频段分割器解决跨视角对齐问题。

**核心 idea**：FFT 专家看全局周期 + 小波专家看局部突变 → 共享频段分割器对齐两域 → 多分辨率融合从粗到细 → 门控整合长短期。

## 方法详解

### 整体框架
M2FMoE 包含三大模块：  
1. **多视角频率混合专家（MFMoE）**：FFT 分支 + 小波分支，各含 $E$ 个频段专家  
2. **多分辨率自适应融合（MAF）**：从多个时间分辨率聚合特征  
3. **时序门控整合（TGI）**：自适应融合短期预测与长期历史表示

输入信号先经**层次化时间分割**提取近期段 $\mathbf{X}_r$ 和完整历史 $\mathbf{X}$；近期段通过多分辨率平滑卷积生成不同粒度的差分序列 $\Delta \mathbf{X}_r^{(k)}$，送入 MFMoE。

### 关键设计 1：跨视角共享频段分割器（CSS）
- 学习共享频率边界 $\{\beta_1, \ldots, \beta_{E-1}\}$ 将频率范围 $[0,1]$ 分为 $E$ 个频段
- **FFT 视角**：边界直接缩放为频率索引 $\{\tilde{\beta}_e\}$
- **小波视角**：通过 Theorem 1 的逆映射（$a = \gamma/f$）将频率边界非线性映射为小波尺度索引 $\{\ddot{\beta}_e\}$
- **理论保证**：Theorem 1 证明频率-尺度映射保持信号能量守恒
- **核心价值**：使两个视角的专家在语义一致的频段上工作，消除跨视角不对齐问题

### 关键设计 2：双视角频率专家分支
**FFT 分支**（Fig. 2c）：
- 对差分序列逐通道做 FFT，用二值掩码 $\tilde{\mathbb{I}}_e$ 隔离每个专家的频段
- 轻量路由网络：对幅度谱通道平均 → 两层 Linear+ReLU+Softmax → 得到路由权重 $\boldsymbol{\alpha}$
- 按权重加权聚合各专家频段 → IFFT → 线性投影，输出 $\tilde{\mathbf{V}} \in \mathbb{R}^{T_p \times C}$

**小波分支**（Fig. 2d）：
- 用复高斯小波做 CWT 得功率谱 $\mathcal{P} = |\mathcal{W}(a,b)|^2$
- 用尺度掩码 $\ddot{\mathbb{I}}_e$ 隔离各专家的尺度范围
- 每个专家用 2 层卷积块处理：Conv → ReLU → Dropout → Conv
- 类似路由机制产生门控权重 $\boldsymbol{\eta}$，加权聚合后展平 → 两层线性投影

**互补性**：FFT 专家主攻低频全局趋势，小波专家主攻高频局部突变，路由机制根据输入谱特性动态调整权重。

### 关键设计 3：多分辨率自适应融合（MAF）+ 时序门控（TGI）
**MAF**：
- **多视角融合**：将 FFT 和小波分支输出沿通道拼接，加时间编码，经 BN + Linear 融合
- **多分辨率融合**：不同分辨率的融合结果经各自线性变换后累加：$\mathbf{H}_r = \sum_{i=1}^{R} \text{Linear}_i(\mathbf{H}_u^{(i)})$
- 加回最后观测值恢复原始值域

**TGI**：
- 历史输入 $\mathbf{X}$ 线性投影得 $\mathbf{H}_h$
- 门控系数 $\mathbf{G} = \sigma(\text{Linear}([\mathbf{H}_r; \mathbf{H}_h]))$
- 最终输出 $\hat{\mathbf{X}} = \mathbf{G} \odot \mathbf{H}_r + (1-\mathbf{G}) \odot \mathbf{H}_h$

### 损失函数
$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pred}} + \lambda \mathcal{L}_{\text{div}} + \mu \mathcal{L}_{\text{cons}}$

- $\mathcal{L}_{\text{pred}}$：MSE 预测损失
- $\mathcal{L}_{\text{div}}$：专家多样性损失，鼓励不同专家输出差异化（标准差惩罚）
- $\mathcal{L}_{\text{cons}}$：跨视角一致性损失，鼓励同一频段在 FFT 和小波视角的专家输出余弦相似

## 实验

### 主实验结果（5 个水库，预测 8h 与 72h）

| 数据集 | H(h) | M2FMoE RMSE | CATS RMSE | FreqMoE RMSE | iTrans RMSE | MCANN RMSE (w/ 标签) |
|--------|------|-------------|-----------|-------------|-------------|---------------------|
| Almaden | 8 | **7.99** | 16.09 | 14.73 | 32.13 | 8.45 |
| Coyote | 8 | **48.80** | 110.85 | 593.14 | 372.52 | 86.83 |
| Lexington | 8 | **251.96** | 618.99 | 387.00 | 690.43 | 253.0 |
| Stevens Cr. | 8 | **10.56** | 18.50 | 80.94 | 48.88 | 12.13 |
| Vasona | 8 | **5.13** | 6.91 | 14.32 | 12.18 | 5.35 |
| Coyote | 72 | **449.94** | 509.08 | 855.10 | 673.85 | 559.75 |
| Lexington | 72 | **772.84** | 906.53 | 1003.82 | 960.65 | 778.02 |

- 平均排名 1.4（vs 无标签基线最佳 CATS 3.7、有标签 MCANN 1.7）
- 对比无标签最佳基线平均 RMSE 提升 **22.30%**，最大提升 52.86%（Coyote, 8h）
- 对比使用极端事件标签的 MCANN 平均 RMSE 提升 **9.19%**，最大提升 43.8%

### 消融实验（预测 72h）

| 消融变体 | Almaden | Coyote | Lexington | Stevens Cr. | Vasona |
|---------|---------|--------|-----------|-------------|--------|
| Full M2FMoE | **54.12** | **449.94** | **772.84** | **76.94** | **19.57** |
| w/o 小波分支 | 57.70 | 555.64 | 827.39 | 87.19 | 19.84 |
| w/o FFT 分支 | 59.04 | 558.26 | 870.74 | 85.02 | 19.81 |
| w/o 多分辨率 | 59.48 | 483.22 | 855.24 | 85.00 | 19.51 |
| w/o CSS | 55.58 | 541.59 | 916.93 | 86.00 | 20.00 |
| w/o 对齐 | 59.12 | 516.73 | 872.03 | 85.80 | 19.28 |
| w/o 双视角 | 56.27 | 453.46 | 789.04 | 79.06 | 19.38 |

### 关键发现
1. **双视角互补有效**：移除任一分支都明显恶化，FFT 分支去除的影响稍大（失去全局趋势建模能力）
2. **CSS 的跨视角对齐至关重要**：移除 CSS 后 Lexington 上 RMSE 增加 18.6%，说明语义一致的频段分割是双视角协作的关键
3. **极端事件中小波专家激活更强**：t-SNE 可视化显示小波视角对高频局部突变更敏感，FFT 视角更擅长稳定周期模式
4. **专家数量 3-4 为最佳**：过多专家引入噪声和过拟合
5. **近期段长度有 sweet spot**：过短信息不足，过长引入噪声；完全不用近期段性能显著下降
6. 在 ETTh 等常规数据集上与 PatchTST/TimesNet 竞争力，但短期预测更突出

## 亮点
- **无需极端事件标签即超越使用标签的方法**：纯端到端学习，泛化性更强
- **严谨的跨域对齐理论**：Theorem 1 提供频率-尺度映射的能量守恒证明，CSS 基于此实现可学习的共享边界
- **从频谱异质性角度切入极端事件**：首次系统分析极端 vs 常规事件的频谱差异（Fig 1），为频域建模提供直觉
- **MoE + 多分辨率的嵌套设计**：频段级专家特化 → 视角级融合 → 分辨率级聚合 → 时序级门控，层次递进且各有明确功能

## 局限性
1. 仅在水文数据集上验证，未在金融、交通等其他极端事件场景测试，泛化性有待扩展
2. CWT 计算成本较 FFT 高（使用 PyWavelets 库），对超长序列可能成为瓶颈
3. 在长期预测（ETTh1/h2 的 96h→96h）上不如 PatchTST/TimesNet，说明该方法更适合短-中期极端事件场景
4. 专家数和分辨率等超参数需要逐数据集调优，自动化程度不足
5. 多样性损失和一致性损失的权重 $\lambda, \mu$ 需手动设置

## 相关工作
- **频域时序预测**：FEDformer（频率增强分解）、FreqMoE（频率分解 MoE）、U-Mixer（UNet 混合器）
- **极端事件方法**：NEC+/VIE/SADI/SEED（多阶段学习）、DAN（极值标签）、MCANN（聚类注意力）、EPL/EVL/GEVL（极值理论损失）
- **MoE 时序模型**：Time-MoE（大规模基础模型）、FreqMoE（频率分解）
- **多分辨率/多尺度**：TimeMixer（多尺度混合）、TimeMixer++（增强版）
- **注意力方法**：CATS（交叉注意力）、iTransformer（倒置 Transformer）

## 评分
⭐⭐⭐⭐ — 从频谱异质性角度切入极端事件预测非常新颖，CSS 提供了严谨的跨域对齐理论，无标签即超越有标签方法是强有力的结果。模块设计层次清晰、各有分工。不足在于验证场景偏窄（仅水文）、长期预测场景性能一般。总体是极端事件时序预测领域的扎实推进。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)
- [\[AAAI 2026\] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)
- [\[ICCV 2025\] VA-MoE: Variables-Adaptive Mixture of Experts for Incremental Weather Forecasting](../../ICCV2025/time_series/va-moe_variables-adaptive_mixture_of_experts_for_incremental_weather_forecasting.md)
- [\[AAAI 2026\] Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj](coherent_multi-agent_trajectory_forecasting_in_team_sports_with_causaltraj.md)
- [\[ICML 2026\] Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series](../../ICML2026/time_series/dynamic_tmoe_a_drift-aware_dynamic_mixture_of_experts_framework_for_non-stationa.md)

</div>

<!-- RELATED:END -->
