---
title: >-
  [论文解读] Synthetic Series-Symbol Data Generation for Time Series Foundation Models
description: >-
  [NeurIPS 2025][时间序列][time series foundation model] 提出 Series-Symbol (S²) 数据生成机制和 SymTime 双模态基础模型，利用 Takens 定理和符号动力学理论生成无限规模的合成时序-符号配对数据（40M 对/50B token），通过跨模态对比学习预训练在 5 大时序任务上达到与真实数据预训练模型竞争的性能。
tags:
  - "NeurIPS 2025"
  - "时间序列"
  - "time series foundation model"
  - "synthetic data generation"
  - "symbolic expressions"
  - "对比学习"
  - "pre-training"
---

# Synthetic Series-Symbol Data Generation for Time Series Foundation Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.08445](https://arxiv.org/abs/2510.08445)  
**代码**: [GitHub](https://github.com/wwhenxuan/SymTime)  
**领域**: 时间序列  
**关键词**: time series foundation model, synthetic data generation, symbolic expressions, contrastive learning, pre-training

## 一句话总结
提出 Series-Symbol (S²) 数据生成机制和 SymTime 双模态基础模型，利用 Takens 定理和符号动力学理论生成无限规模的合成时序-符号配对数据（40M 对/50B token），通过跨模态对比学习预训练在 5 大时序任务上达到与真实数据预训练模型竞争的性能。

## 研究背景与动机

**领域现状**：时序基础模型（如 Moirai、Timer、TimeGPT）近年来取得了显著进展，但与 CV/NLP 领域相比，时序领域的训练数据面临严重的稀缺性和分布不平衡问题。现有的大规模时序数据集在金融、医疗等特定领域仍然不足，数据规模远小于 ImageNet 或 WebText 等基准。

**现有痛点**：根据 Neural Scaling Laws，训练数据的不平衡会导致模型在 OOD 数据上泛化能力下降，产生性能偏差。当前的时序预训练策略大多依赖真实数据收集，面临数据隐私限制和领域覆盖不全的双重瓶颈。少数使用合成数据的方法（如 Chronos）缺乏对时序本质的理论刻画。

**核心 idea**：基于 Takens 定理（时序是复杂动力系统的低维投影）和符号动力学理论（复杂系统可用符号表达式抽象表示），构建一套理论驱动的合成数据生成机制——通过随机构造多样的符号表达式来覆盖广泛的动力系统类型，由此生成的时序数据天然具备丰富的时序特性和语义对应关系。

## 方法详解

### 整体框架
整体分为两大部分：(1) Series-Symbol (S²) 数据生成机制，通过随机构造符号表达式并前向传播生成时序-符号配对数据，规模可无限扩展；(2) SymTime 基础模型，包含时序编码器和符号编码器，通过掩码建模和跨模态对比学习进行预训练，再在下游任务上微调。

### 关键设计

1. **S² 数据生成机制**:

    - 功能：生成无限规模的高质量合成时序数据及其对应的符号表达式
    - 核心思路：通过二叉树结构随机采样构建多变量符号表达式 f(·)——先选择二元运算符（+, −, ×）构建树骨架，再插入变量和常数到叶节点，添加一元运算符（sin, cos, log, exp, pow2 等），最后进行仿射变换增加多样性。输入 X 从混合分布和 ARMA 过程中采样，通过 Y=f(X) 前向传播得到输出序列
    - 设计动机：基于 Takens 定理和符号动力学的理论支撑，符号表达式与时序之间存在严格的语义对应。通过遍历所有输入/输出维度组合（M∈[1,6], N∈[1,12]），确保生成数据覆盖完整的时序表示空间。最终生成 40M 对配对数据，共 50B token 长度
    - 与先前方法区别：ForecastPFN 和 Chronos 的合成方法缺乏时序产生本质的理论刻画，S² 更贴合时序的生成机理

2. **SymTime 双模态预训练架构**:

    - 功能：利用符号信息增强时序表示学习
    - 核心思路：时序编码器（6 层 Transformer）通过 Masked Time Series Modeling (MTM) 重建被掩码的 patch；符号编码器（6 层 DistilBERT）通过 Masked Language Modeling (MLM) 学习符号表示。两个编码器通过 MoCo 风格的动量编码器进行跨模态对比学习，将语义相关的时序-符号对在表示空间中对齐
    - 设计动机：单纯的 MTM 预训练只能学到时序的统计模式，无法捕获时序背后的动力学语义。通过跨模态对比学习将符号的语义信息注入时序编码器，使其获得独特的归纳偏置

3. **动量蒸馏（Momentum Distillation）**:

    - 功能：对齐掩码数据的编码器输出与动量编码器的输出，减轻掩码噪声的影响
    - 核心思路：受 ALBEF 启发，将随机掩码视为噪声，使用动量编码器生成伪标签进行 KL 散度约束，使掩码后的表示更接近完整数据的表示
    - 设计动机：直接在掩码数据上做对比学习可能因信息缺失而产生噪声梯度，动量蒸馏通过软标签平滑了这一问题

### 损失函数 / 训练策略
总预训练目标：L = L_mtm + L_mlm + α·L_tsc + (1−α)·L_tsc^mod，其中 L_mtm 为 patch 重建的 MSE 损失，L_mlm 为掩码语言建模的交叉熵损失，L_tsc 为跨模态对比损失，L_tsc^mod 为动量蒸馏的 KL 散度损失。下游微调时，分类任务直接加线性头；重建类任务（预测/填补/异常检测）先分解趋势+周期分量再分别处理。

## 实验关键数据

### 主实验（Scaling 效果 - 长期预测）

| 预训练规模 | ETTm1 MSE | ETTm2 MSE | ETTh1 MSE | Weather MSE | Traffic MSE | Exchange MSE | Avg MSE |
|-----------|-----------|-----------|-----------|-------------|-------------|-------------|---------|
| 0B (无预训练) | 0.401 | 0.293 | 0.487 | 0.257 | 0.471 | 0.383 | 0.358 |
| 1B | 0.376 | 0.292 | 0.461 | 0.257 | 0.473 | 0.370 | 0.354 |
| 10B | 0.376 | 0.281 | 0.444 | 0.250 | 0.473 | 0.368 | 0.345 |
| 25B | 0.378 | 0.278 | 0.434 | 0.253 | 0.467 | 0.357 | 0.342 |
| **50B** | **0.371** | **0.274** | **0.430** | **0.247** | **0.457** | **0.359** | **0.336** |

### 消融实验

| 配置 | ETTh1 MSE | ETTh2 MSE | 说明 |
|------|-----------|-----------|------|
| Full SymTime | 最优 | 最优 | 完整预训练配置 |
| w/o Pre-train | 显著下降 | 显著下降 | 不预训练直接微调 |
| w/o Symbol | 下降 | 下降 | 去除符号编码器，仅用 MTM 预训练 |
| Real-Data | 下降 | 下降 | 在等规模真实数据上仅用 MTM 预训练 |
| w/o MTM | 下降 | 下降 | 去除掩码时序建模损失 |
| w/o Distill | 下降 | 下降 | 去除动量蒸馏 |
| Freeze | 最差 | 最差 | 冻结预训练参数不微调 |

### 关键发现
- 预训练数据从 0B 扩展到 50B，长期预测平均 MSE 从 0.358 持续降低到 0.336，验证了 S² 数据的 scaling 效果
- 短期预测 OWA 从 0.887 降至 0.849，填补 MSE 从 0.038 降至 0.026（ETTm2），效果显著
- 消融证明符号编码器和对比学习是关键——去除符号信息后性能下降，说明符号语义确实增强了时序表示
- 复杂度分析显示 SymTime 比 Time-LLM 等 LLM 基础模型参数更少、显存更低
- t-SNE 可视化显示预训练后时序编码器对不同运算符类型形成清晰聚类，确认了跨模态语义对齐的有效性

## 亮点与洞察
- 基于 Takens 定理和符号动力学的理论支撑使合成数据生成有严格的数学基础，而非启发式设计
- 数据可无限生成且覆盖全表示空间（Radviz 可视化验证 S² 数据的统计特性覆盖了 Monash 真实数据集）
- 在纯合成数据上预训练即可获得与真实数据竞争的下游性能，完全绕过数据隐私和稀缺问题
- 跨模态对比学习让时序编码器学到符号语义这一独特归纳偏置，是对时序预训练范式的有意义探索

## 局限与展望
- 符号表达式覆盖度受限于选定的运算符集合（未涵盖随机微分方程等）
- 模型规模较小（6 层 Transformer），未探索更大模型的 scaling behavior
- 微调后性能比较充分，但未展示零样本预测能力（与 Chronos、Moirai 的零样本对比不足）
- 当前仅支持确定性符号表达式，未考虑随机过程对时序生成的影响

## 相关工作与启发
- **Moirai / Timer / TimeGPT**：在真实数据上预训练的时序基础模型，SymTime 用纯合成数据达到竞争性能是有说服力的
- **Chronos**：合成+真实数据结合，但合成策略缺乏理论驱动，S² 的生成机制更本质
- **ALBEF / MoCo**：跨模态对比学习和动量蒸馏的方法论来源
- 核心启发：时序的本质是动态系统的投影，用符号表达式生成时序是从源头造数据的思路，可推广到其他科学数据生成场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 符号-时序双模态预训练思路新颖，理论驱动的合成数据生成有说服力
- 实验充分度: ⭐⭐⭐⭐ 5 大任务 + scaling 实验 + 消融 + 表示分析 + 复杂度分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，理论与实验衔接好，Takens 定理的引入自然
- 价值: ⭐⭐⭐⭐ 为时序基础模型提供了新的数据范式，纯合成预训练的可行性有启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SEMPO: Lightweight Foundation Models for Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] How Foundational are Foundation Models for Time Series Forecasting?](how_foundational_are_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] Towards Self-Supervised Foundation Models for Critical Care Time Series](towards_self-supervised_foundation_models_for_critical_care_time_series.md)
- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](mira_medical_time_series_foundation_model_for_real-world_health_data.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)

</div>

<!-- RELATED:END -->
