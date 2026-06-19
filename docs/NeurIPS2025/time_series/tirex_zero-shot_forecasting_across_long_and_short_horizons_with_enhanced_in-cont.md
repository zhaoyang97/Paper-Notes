---
title: >-
  [论文解读] TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning
description: >-
  [NeurIPS 2025][时间序列][时间序列预测] 提出基于xLSTM的预训练时间序列预测模型TiRex，通过连续片段掩码（CPM）策略和数据增强技术，在GiftEval和Chronos-ZS两大标准基准上以仅35M参数全面超越Chronos Bolt（200M）、TimesFM（500M）等大模型，同时在短期和长期零样本预测中均达到SOTA。
tags:
  - "NeurIPS 2025"
  - "时间序列"
  - "时间序列预测"
  - "零样本预测"
  - "xLSTM"
  - "上下文学习"
  - "数据增强"
  - "预训练模型"
---

# TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning

**会议**: NeurIPS 2025  
**arXiv**: [2505.23719](https://arxiv.org/abs/2505.23719)  
**作者**: Andreas Auer (NXAI/JKU Linz), Patrick Podest (JKU Linz), Daniel Klotz (ITUA Linz), Sebastian Böck (NXAI), Günter Klambauer (NXAI/JKU), Sepp Hochreiter (NXAI/JKU)  
**代码**: 未公开  
**领域**: 视频理解  
**关键词**: 时间序列预测, 零样本预测, xLSTM, 上下文学习, 数据增强, 预训练模型  

## 一句话总结

提出基于xLSTM的预训练时间序列预测模型TiRex，通过连续片段掩码（CPM）策略和数据增强技术，在GiftEval和Chronos-ZS两大标准基准上以仅35M参数全面超越Chronos Bolt（200M）、TimesFM（500M）等大模型，同时在短期和长期零样本预测中均达到SOTA。

## 研究背景与动机

### 问题背景
时间序列预测是能源、零售、医疗等领域的核心任务。近年来，受大语言模型启发，预训练时间序列模型通过上下文学习（in-context learning）实现零样本预测，使非专家用户也能使用高级预测工具，并在数据稀缺场景下显著提升性能。

### 已有工作的不足
- **Transformer主导但表现有限**：当前主流零样本预测模型（Chronos、TimesFM、Moirai等）均基于Transformer架构，但Transformer在时间序列预测中常不如预期，甚至被简单线性模型DLinear超越
- **LSTM缺乏上下文学习能力**：传统LSTM具备强大的状态追踪（state-tracking）能力，非常适合时间序列建模，但缺乏零样本泛化所需的上下文学习能力
- **长期预测中的误差累积**：现有方法在多步预测时采用自回归生成，用前一步的点估计作为下一步输入，导致不确定性传播中断、长期预测质量急剧下降
- **数据增强未被充分利用**：合成数据在预训练中已被使用，但系统性的数据增强策略（如视觉领域的成功实践）在时间序列预训练中几乎未被探索

### 核心动机
利用xLSTM（增强版LSTM），兼具LSTM的状态追踪优势和Transformer级别的上下文学习能力，构建同时擅长短期和长期零样本预测的预训练模型。

## 方法详解

### 整体架构
TiRex采用decoder-only模式，堆叠多个xLSTM块（使用sLSTM模块），在输入和输出层之间设置轻量级的残差块。

**输入层**：
1. 对每条时间序列进行z-score实例归一化，消除跨领域的尺度差异
2. 将时间序列分割为不重叠的窗口（patch），窗口大小 $m_{\text{in}}=32$
3. 每个窗口拼接一个二值掩码（标记缺失值），通过两层残差块映射到xLSTM隐藏维度 $d$

**xLSTM块**：
- 采用sLSTM（而非mLSTM）作为序列混合组件，保留**真实递归**以实现状态追踪
- 每个块包含sLSTM模块、前馈网络，均有RMSNorm和残差跳连
- sLSTM通过优化内核架构实现高效训练和推理

**输出层**：
- 预测9个等距分位数 $Q=\{0.1, 0.2, \ldots, 0.9\}$，实现概率预测
- 使用分位数损失（quantile loss）进行优化

### 连续片段掩码（Contiguous Patch Masking, CPM）
CPM是TiRex的核心创新之一，解决多步预测中的误差累积问题：

1. 对每个训练样本，随机采样连续掩码片段数 $c_{\text{mask}} \sim U(1, c_{\text{mask}}^{\text{max}})$
2. 随机采样掩码概率 $p_{\text{mask}} \sim U(0, p_{\text{mask}}^{\text{max}})$
3. 生成二值掩码并以 $c_{\text{mask}} \cdot m_{\text{out}}$ 为单位重复，掩码整段连续的patch
4. 被掩码的patch在输入中表示为"缺失值"

**关键设计思想**：训练时随机掩码连续片段，使模型在推理时能基于内部记忆状态传播预测信息和不确定性，而非依赖自回归点估计。这与传统自回归方法（如TimesFM）的本质区别在于：TiRex在多步预测时将未来输入视为缺失值，让xLSTM的内部状态自然地跨patch传播不确定性。

### 多步预测推理
当预测范围 $h$ 超出单个输出patch长度时：
- 现有方法：用前一步的中位数/均值作为下一步输入（自回归），每步重置概率分布
- TiRex：将未来输入标记为缺失值，让内部记忆跨patch传播预测状态和不确定性

### 数据增强策略
三种专门为时间序列预训练设计的增强技术：

1. **幅度调制（Amplitude Modulation）**：$y'_t = y_t \cdot a_t$，引入趋势和变化点，应用概率0.5
2. **截断增强（Censor Augmentation）**：$y'_t = \max/\min(y_t, c)$，在随机阈值处截断，应用概率0.5
3. **尖峰注入（Spike Injection）**：$y'_t = y_t + s_t$，添加短周期尖峰信号（tophat/RBF/线性核），应用概率0.05

### 训练细节
- 训练数据：Chronos训练集 + KernelSynth合成数据 + GiftEval预训练数据，共4750万条时间序列样本
- 上下文长度2048，窗口大小32
- 模型参数：35M（远小于竞争对手）

## 实验关键数据

### 实验1：GiftEval-ZS基准（长短期综合评估）

在排除与训练集重叠的16个评估设置后，共81个零样本评估设置：

| 模型 | 参数量 | CRPS (Overall) | CRPS (Short) | CRPS (Long) | Avg Rank |
|------|--------|---------------|-------------|------------|----------|
| **TiRex** | **35M** | **0.411±0.002** | **0.455±0.001** | **0.325±0.003** | **最优** |
| TimesFM-2.0 | 500M | 0.459 | — | — | 第二梯队 |
| TabPFN-TS | — | 0.463 | — | — | 第二梯队 |
| Chronos-Bolt-Base | 200M | 0.481 | — | — | 第二梯队 |

关键发现：
- TiRex以35M参数全面超越500M的TimesFM-2.0和200M的Chronos-Bolt
- 其他模型在短期或长期中各有优势，TiRex是唯一在两者中同时领先的模型
- 在长期预测中，TiRex成为首个超越PatchTST和TFT（task-specific模型）的零样本模型

### 实验2：消融研究

| 变体 | Gift-ZS Overall | Gift-ZS Long | Gift-ZS Short | Chronos-ZS WQL |
|------|----------------|-------------|-------------|---------------|
| **TiRex (完整)** | **0.411** | **0.325** | **0.455** | **0.592** |
| 朴素多步训练 | 0.424 ↓ | 0.335 ↓ | 0.475 ↓ | 0.650 ↓ |
| 无多步预测(自回归) | 0.445 ↓ | 0.370 ↓ | 0.471 ↓ | 0.589 |
| 无任何增强 | 0.430 ↓ | 0.339 ↓ | 0.473 ↓ | 0.623 ↓ |
| Transformer骨干 | 0.422 ↓ | 0.342 ↓ | 0.461 ↓ | 0.597 |
| mLSTM骨干 | 0.457 ↓ | 0.430 ↓ | 0.456 | 0.588 |
| Chronos Bolt Base | 0.454 ↓ | 0.418 ↓ | 0.458 | 0.627 ↓ |

关键结论：
- CPM对长期预测至关重要，移除CPM导致长期CRPS从0.325恶化到0.370
- sLSTM骨干显著优于mLSTM（长期0.325 vs 0.430）和Transformer（0.325 vs 0.342）
- 数据增强整体提升明显，无增强时Overall CRPS从0.411降至0.430
- 推理速度：TiRex比TimesFM-2.0快11×，比Chronos-Bolt Base快4×，比TabPFN-TS快2176×

## 亮点

- **小模型大性能**：仅35M参数全面超越200M-500M级别的竞争对手，在两大标准基准上同时达到短期和长期预测SOTA
- **CPM策略精巧有效**：连续片段掩码完美解决了自回归多步预测中的不确定性传播中断问题，训练时的随机掩码模式与推理时的缺失值输入自然对齐
- **架构选择有理论支撑**：选用sLSTM而非mLSTM或Transformer，利用其独特的状态追踪能力实现长期预测中的不确定性一致传播
- **首次系统性探索时间序列预训练的数据增强**：三种增强策略各有针对性，尤其是尖峰注入使模型能捕获稀有尖锐事件
- **推理效率极高**：比主要竞争对手快4-2176倍，GPU内存占用也大幅降低

## 局限与展望

- **仅支持单变量**：与大多数预训练预测模型一样，TiRex仅处理单变量时间序列，未建模变量间依赖关系
- **超参数调优不充分**：受计算资源限制，未进行广泛的超参数搜索，仅做了关键参数的敏感性分析
- **下游任务未探索**：模型学到的表示在分类、异常检测等其他时间序列任务上的迁移效果未知
- **领域分类存疑**：本文属于时间序列预测领域，当前归为3d_vision可能不够精确
- **评估基准中的数据泄露风险**：虽然作者排除了重叠数据集，但竞争模型（如Moirai与Chronos-ZS有82%重叠）的公平对比仍需注意

## 与相关工作的对比

- **Chronos/Chronos-Bolt (Amazon)**：基于encoder-decoder Transformer，Chronos-Bolt-Base 200M参数，在短期预测上有竞争力但长期预测明显弱于TiRex
- **TimesFM (Google)**：500M参数的decoder-only Transformer，v2.0在GiftEval-ZS上CRPS 0.459，被35M的TiRex大幅超越
- **Moirai (Salesforce)**：encoder-only设计+掩码建模，在Chronos-ZS上表现较好，但与测试集有82%重叠，零样本能力存疑
- **TabPFN-TS (Prior Labs)**：改进Transformer encoder，仅用合成数据预训练，在Chronos-ZS的MASE上略优于TiRex，但推理慢2176倍
- **xLSTM (Beck et al., 2024)**：TiRex的骨干架构，原论文展示了与Transformer LLM相当的上下文学习能力，但在时间序列通用预训练中的潜力未被充分挖掘
- **DLinear (Zeng et al., 2023)**：简单线性模型，证明了Transformer在时间序列中未必占优，间接支持了TiRex选择非Transformer架构的合理性

## 评分

- 新颖性: ⭐⭐⭐⭐ — CPM策略和xLSTM在零样本预测中的应用有新意，但整体思路是将已有架构应用于新任务
- 实验充分度: ⭐⭐⭐⭐⭐ — 两大标准基准、6种子重复、全面消融、推理效率分析、定性分析，实验设计非常扎实
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分，实验组织合理
- 价值: ⭐⭐⭐⭐⭐ — 以极小参数量统一短期和长期零样本预测SOTA，对实际应用有重大意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] In-Context Learning of Stochastic Differential Equations with Foundation Inference Models](in-context_learning_of_stochastic_differential_equations_with_foundation_inferen.md)
- [\[ICML 2025\] VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters](../../ICML2025/time_series/visionts_visual_masked_autoencoders_are_free-lunch_zero-shot_time_series_forecas.md)
- [\[ACL 2025\] Revisiting LLMs as Zero-Shot Time-Series Forecasters: Small Noise Can Break Large Models](../../ACL2025/time_series/revisiting_llms_as_zero-shot_time_series_forecasters_small_noise_can_break_large.md)
- [\[NeurIPS 2025\] OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales](omnicast_a_masked_latent_diffusion_model_for_weather_forecasting_across_time_sca.md)
- [\[ICLR 2026\] Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](../../ICLR2026/time_series/relational_transformer_toward_zero-shot_foundation_models_for_relational_data.md)

</div>

<!-- RELATED:END -->
