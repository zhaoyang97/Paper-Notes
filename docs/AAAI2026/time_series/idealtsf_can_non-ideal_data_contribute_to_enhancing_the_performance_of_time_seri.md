---
title: >-
  [论文解读] IdealTSF: Can Non-Ideal Data Contribute to Enhancing Time Series Forecasting?
description: >-
  [AAAI 2026][时间序列][时序预测] 提出 IdealTSF 框架，通过三阶段渐进式设计——负样本预训练模拟非理想数据增强鲁棒性、正样本训练用修复后数据学习趋势、ECOS 优化器引导到平坦极值——在含噪声/缺失的时序数据上 MSE 提升约 10%。 领域现状 领域现状：时序预测深度模型（Transformer、ML…
tags:
  - "AAAI 2026"
  - "时间序列"
  - "时序预测"
  - "负样本预训练"
  - "对抗训练"
  - "数据鲁棒性"
  - "非理想数据"
---

# IdealTSF: Can Non-Ideal Data Contribute to Enhancing Time Series Forecasting?

**会议**: AAAI 2026  
**arXiv**: [2512.05442](https://arxiv.org/abs/2512.05442)  
**代码**: [GitHub](https://github.com/LuckyLJH/IdealTSF)  
**领域**: 时间序列预测  
**关键词**: 时序预测, 负样本预训练, 对抗训练, 数据鲁棒性, 非理想数据

## 一句话总结
提出 IdealTSF 框架，通过三阶段渐进式设计——负样本预训练模拟非理想数据增强鲁棒性、正样本训练用修复后数据学习趋势、ECOS 优化器引导到平坦极值——在含噪声/缺失的时序数据上 MSE 提升约 10%。

## 研究背景与动机

### 领域现状

**领域现状**：时序预测深度模型（Transformer、MLP-based 等）通常假设输入数据完整无异常，在 ETT、Weather、ECL 等标准 benchmark 上取得优异性能。然而实际部署中的时序数据常包含缺失值、异常点和噪声。

### 现有痛点

**现有痛点**：(1) 简单插值方法（线性/样条）无法恢复复杂非线性结构，在大范围缺失时效果差；(2) 异常检测方法可能误删有意义的极端事件（如突发交通流量）；(3) 现有预测模型对输入扰动缺乏内在鲁棒性，测试时遇到分布偏移性能急剧下降。

### 核心矛盾

**核心矛盾**：非理想数据（噪声、缺失、异常）通常被视为需要"预处理修复"的障碍，但它包含有价值的信息——极端事件模式、系统故障特征等。单纯修复/丢弃这些数据意味着丢失信息，但直接使用又会污染训练。

### 解决思路

**本文目标**：将非理想数据转化为训练资产而非障碍。**切入角度**：将非理想数据视为"负样本"用于预训练增强模型免疫力，同时生成"正样本"用于正式训练，最后通过对抗优化进一步强化。**核心idea**：负样本预训练（合成非理想数据增强鲁棒性）+ 正样本训练（平滑修复后数据学习趋势）+ ECOS 对抗优化（引导到平坦极值提升泛化）。

## 方法详解

### 整体框架
三阶段渐进式训练：(1) **预训练**：从原始数据合成负样本（含跳跃/噪声/缺失），预训练 attention 模块学习对非理想数据的鲁棒表征；(2) **训练**：用混合插值 + Z-score/IQR 异常检测修复数据为正样本，用预训练后的 attention 提取特征并预测；(3) **优化**：ECOS 优化器注入对抗扰动引导参数到平坦极值区域。

### 关键设计

1. **负样本合成与预训练**:

    - 功能：让模型学会应对各种非理想数据模式
    - 核心思路：三种合成策略覆盖三种非理想场景：(a) 稳定分布跳跃——用 $\alpha$-stable 分布生成重尾跳跃增量 $\Delta x_i = R \cdot \cos(\theta)$ 叠加到时序中，$\alpha$ 控制尾部厚度模拟不同程度的突变；(b) 多尺度噪声——低频高强度 + 高频低强度的分层噪声注入，模拟传感器等多源扰动；(c) 结构性删除——随机选取连续段设为缺失，模拟数据采集中断
    - 设计动机：三种策略分别模拟了突变/跳跃（金融崩盘）、观测噪声（传感器漂移）、数据缺失（通信中断）三种典型异常，用合成数据做预训练可以在不需要真实标注的情况下增强鲁棒性

2. **正样本修复与训练**:

    - 功能：从非理想原始数据中恢复可靠的训练信号
    - 核心思路：先用 Z-score/IQR 检测异常点标记为缺失，再用混合平滑插值（线性 + 滑动平均加权）修复缺失值，生成"正样本"。用预训练后的 attention 模块（部分参数冻结）提取特征，MSE 损失训练预测头
    - 设计动机：预训练阶段已让 attention 学会对噪声的鲁棒表征，在正样本上训练确保模型学到干净的趋势和周期信号

3. **ECOS 生态系统优化器**:

    - 功能：引导模型收敛到平坦极值提升泛化能力
    - 核心思路：三阶段优化循环——(I) 沿梯度方向"上山"（加扰动 $e_\theta = \frac{\rho}{\|\nabla L\|} \cdot \nabla L$）探索损失面最坏方向；(II) 多步小学习率微调在扰动邻域内优化；(III) 恢复原参数 + 标准优化步骤。再叠加 FGSM/PGD 对抗训练
    - 设计动机：类似 SAM (Sharpness-Aware Minimization) 但更稳定——多步微调避免了 SAM 的单步回弹震荡问题，对抗训练进一步增加输入空间的鲁棒性

## 实验关键数据

### 主实验

在 ETTh1/h2/m1/m2、Weather、ECL、Traffic 等数据集上评测。

| 数据集 | 基线 Attention MSE | IdealTSF MSE | 提升 |
|--------|:---:|:---:|:---:|
| ETTh1 (720步) | 0.456 | **0.411** | 9.9% |
| ETTm1 (720步) | 0.400 | **0.361** | 9.8% |
| Weather (720步) | 0.259 | **0.234** | 9.7% |
| ECL (720步) | 0.214 | **0.193** | 9.8% |

平均跨数据集提升约 10% MSE。

### 消融实验

| 配置 | ETTh1 MSE | 说明 |
|------|:---:|------|
| Full IdealTSF | **0.411** | 完整模型 |
| w/o 负样本预训练 | 0.442 | 掉 7.5%，模型对噪声脆弱 |
| w/o 正样本修复 | 0.435 | 掉 5.8%，训练信号含噪 |
| w/o ECOS | 0.428 | 掉 4.1%，陷入尖锐极值 |
| w/o 对抗训练 | 0.421 | 掉 2.4%，泛化能力减弱 |

### 关键发现
- 负样本预训练贡献最大——即使不做正样本修复，预训练过的模型也比从零训练好 5%+
- ECOS 的"上山-下山"策略比纯 SAM 更稳定（多步微调避免震荡）
- 在人为注入 30% 缺失/噪声的对抗场景下优势翻倍（20%+ 提升）
- 三种负样本合成策略互补——单独使用任何一种贡献约 3%，联合使用达 7.5%

## 亮点与洞察
- **"非理想数据即负样本"** 的视角新颖——不是修复数据，而是用缺陷数据训练模型的免疫力
- **三阶段渐进设计** 逻辑清晰：预训练→训练→优化，每阶段解决不同层次问题
- **ECOS 优化器** 结合 SAM 平坦极值思想和对抗训练，是一个通用的优化策略，可推广到其他鲁棒学习场景

## 局限与展望
- 负样本合成的超参数（稳定分布 $\alpha, \beta, \gamma$、噪声尺度、删除长度）需要调节
- 仅在 attention 架构上验证，MLP/Transformer 等其他骨干未测试
- "正样本"生成依赖线性插值 + 滑动平均，对复杂非线性模式可能不够
- 未与专门的鲁棒时序预测方法（如 Robust-TSF）直接对比

## 相关工作与启发
- **vs RevIN/DLinear**: 通过归一化/分解处理非平稳性，但不处理缺失和异常；IdealTSF 直接从鲁棒性角度解决
- **vs SAM 优化器**: ECOS 扩展了 SAM 的思想，加入多步微调和对抗训练，更稳定
- **vs 数据增强方法**: 传统增强（裁剪/缩放）针对干净数据；IdealTSF 的负样本合成专门针对非理想场景设计
- 负样本预训练思路可推广到其他对噪声敏感的任务（如金融预测、医疗时序）

## 评分
- 新颖性: ⭐⭐⭐⭐ 负样本预训练+ECOS 优化器的组合有新意
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证+对抗场景测试+完整消融
- 写作质量: ⭐⭐⭐ 公式较多但部分符号不统一
- 价值: ⭐⭐⭐⭐ 为低质量数据场景的时序预测提供实用方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)
- [\[AAAI 2026\] Finding Time Series Anomalies using Granular-ball Vector Data Description](finding_time_series_anomalies_using_granular-ball_vector_data_description.md)
- [\[ICLR 2026\] Can we generate portable representations for clinical time series data using LLMs?](../../ICLR2026/time_series/can_we_generate_portable_representations_for_clinical_time_series_data_using_llm.md)
- [\[AAAI 2026\] Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios](scaling_llm_speculative_decoding_non-autoregressive_forecasting_in_large-batch_s.md)
- [\[ICML 2026\] Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting](../../ICML2026/time_series/parametric_prior_mapping_framework_for_non-stationary_probabilistic_time_series_.md)

</div>

<!-- RELATED:END -->
