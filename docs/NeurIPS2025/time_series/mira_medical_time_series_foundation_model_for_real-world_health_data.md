---
title: >-
  [论文解读] MIRA: Medical Time Series Foundation Model for Real-World Health Data
description: >-
  [NeurIPS 2025][时间序列][时间序列基础模型] 提出 MIRA，一个专为医学不规则时间序列设计的基础模型，通过连续时间旋转位置编码、频率特定 MoE 和 Neural ODE 外推模块，在 4540 亿个观测点上预训练，零样本预测性能在 OOD 和 ID 场景中分别平均降低 8% 和 6% 的误差。
tags:
  - "NeurIPS 2025"
  - "时间序列"
  - "时间序列基础模型"
  - "不规则采样"
  - "医学信号"
  - "Neural ODE"
  - "Mixture-of-Experts"
---

# MIRA: Medical Time Series Foundation Model for Real-World Health Data

**会议**: NeurIPS 2025  
**arXiv**: [2506.07584](https://arxiv.org/abs/2506.07584)  
**代码**: [GitHub](https://github.com/Microsoft/MIRA)  
**领域**: 时间序列  
**关键词**: 时间序列基础模型, 不规则采样, 医学信号, Neural ODE, Mixture-of-Experts

## 一句话总结

提出 MIRA，一个专为医学不规则时间序列设计的基础模型，通过连续时间旋转位置编码、频率特定 MoE 和 Neural ODE 外推模块，在 4540 亿个观测点上预训练，零样本预测性能在 OOD 和 ID 场景中分别平均降低 8% 和 6% 的误差。

## 研究背景与动机

医学时间序列数据（ECG、EEG、生命体征、实验室检测）对理解患者动态生理状态至关重要。然而，构建有效的医学时间序列基础模型面临三个核心挑战：

**采样不规则**：ECG 以毫秒间隔记录，而实验室检测可能相隔数小时，时间间隔高度非均匀

**缺失值频繁**：临床工作流导致大量缺失观测值

**采样率异构**：不同设备和临床场景产生频率差异极大的信号

现有通用时间序列基础模型（Chronos、Moirai、TimesFM 等）通常假设均匀时间间隔，无法处理不规则临床数据。而新兴的医学时间序列基础模型仅在狭窄领域内（如 EEG 睡眠监测）实现跨数据集泛化，无法处理连续时间预测或不规则采样。

核心问题：**如何构建一个统一的医学时间序列基础模型，能够处理不规则采样、异构频率和频繁缺失值？**

## 方法详解

### 整体框架

MIRA 是一个 decoder-only 架构，包含三个核心组件：连续时间旋转位置编码（CT-RoPE）、稀疏时间混合专家层（MoE）、以及基于 Neural ODE 的连续动态外推模块。采用 channel-independent 设置处理单变量时间序列。

### 关键设计

1. **连续时间旋转位置编码（CT-RoPE）**：标准 RoPE 假设离散均匀的 token 索引，无法处理连续时间戳。CT-RoPE 将连续时间戳 $t \geq 0$ 直接离散化为旋转角度：

$$\theta_i(t) = \omega_i \cdot t, \quad \omega_i = 10000^{-2i/d}$$

对输入嵌入 $\mathbf{x} \in \mathbb{R}^d$ 的每对分量施加平面旋转。关键性质是注意力分数仅依赖时间差：

$$\langle q_m, k_n \rangle = x_m^\top (W^Q)^\top R_\Theta(t_n - t_m) W^K x_n$$

这使得模型能够捕捉连续时间的相对位置关系，保留标准点积注意力的效率。

2. **频率特定混合专家层（Frequency-Specific MoE）**：医学时间序列在多个时间频率上展现不同动态（平滑长期趋势与快速短期变化）。MoE 层替代标准 FFN 子层，每个 token 路由到 $K$ 个专家中：

$$\text{MoE}(\bar{\mathbf{u}}_t^l) = g_{N+1,t} \cdot \text{FFN}_{N+1}(\bar{\mathbf{u}}_t^l) + \sum_{i=1}^N g_{i,t} \cdot \text{FFN}_i(\bar{\mathbf{u}}_t^l)$$

其中 $\text{FFN}_{N+1}$ 是共享专家（全局残差路径），$g_{i,t}$ 通过 softmax + top-K 选择。共享专家权重通过独立 sigmoid 门控计算，用于捕获通用模式。

3. **连续动态外推模块（Neural ODE）**：自回归 Transformer 无法在推理时获取目标 token 的时间戳。该模块使用 Neural ODE 将隐状态从当前时间戳 $t_N$ 外推到目标时间戳 $t_{N+1}$：

$$h(t_{N+1}) = h(t_N) + \int_{t_N}^{t_{N+1}} f(s - t_N, h(s); \theta_{\text{ODE}}) \, ds$$

使用 Dormand-Prince (RK45) 方法进行数值积分，支持在任意未见时间点进行预测。

### 损失函数 / 训练策略

- **主损失**：Huber 损失，对异常值和噪声测量具有鲁棒性，通过阈值 $\delta$ 控制 L2/L1 切换
- **负载均衡损失**：$\mathcal{L}_{\text{aux}} = N \cdot \sum_{i=1}^N f_i r_i$，防止专家坍塌，促进均匀使用
- **预训练语料**：4540 亿时间点，来自 MIMIC-III/IV、PTB-XL、Sleep-EDF、WAVES 等公开数据集
- 从 Time-MoE checkpoint 继续训练，最多使用 8 张 NVIDIA H/A100 GPU

## 实验关键数据

### 主实验（OOD 零样本预测）

| 数据集 | 指标 | MIRA_large | Time-MoE_large | Moirai_large | Chronos_large | 最佳全监督 |
|--------|------|------------|----------------|-------------|--------------|-----------|
| Heart Rate | RMSE(×10⁻¹) | **1.392** | 0.833 | 2.098 | 1.218 | 0.774(Contiformer) |
| MIT-BIH | RMSE | **0.130** | 0.135 | 0.593 | 0.350 | 0.453(ODE-RNN) |
| CDC-IHA | RMSE(×10¹) | **4.401** | 4.748 | 6.788 | 15.986 | 5.211(Contiformer) |
| JH COVID-19 | RMSE(×10²) | **0.336** | 0.402 | 0.614 | 3.478 | 0.323(Contiformer) |
| ILI | RMSE | **1.041** | 0.951 | 1.499 | 1.870 | 0.391(Contiformer) |

MIRA_large 在 OOD 数据集上取得全面最优 RMSE，平均误差降低约 8%。

### ID 零样本预测

| 数据集 | 指标 | MIRA_large | Time-MoE_large | Chronos_large |
|--------|------|------------|----------------|--------------|
| SleepEDF | RMSE(×10²) | **0.189** | 0.244 | 0.413 |
| PTB-XL | RMSE | **0.121** | 0.109 | 0.229 |
| MIMIC-III | RMSE | **0.102** | 0.103 | 0.151 |
| MIMIC-IV | RMSE | **0.081** | 0.082 | 0.319 |
| WAVES | RMSE | **0.129** | 0.141 | 0.182 |

### 消融实验

| 配置 | 说明 |
|------|------|
| 去掉 CT-RoPE | 性能显著下降，位置编码对不规则时间至关重要 |
| 去掉 MoE → 标准 FFN | 多频率动态特征捕捉能力减弱 |
| 去掉 Neural ODE → 线性外推 | 无法准确预测任意时间点 |
| 医学预训练 vs 通用预训练 | 所有模型变体在医学语料继续预训练后均优于通用版本 |

### 关键发现

- **领域预训练至关重要**：在医学语料上继续预训练后，所有模型变体一致优于通用预训练版本，小模型甚至超过更大的通用模型
- **缺失率鲁棒性**：随着缺失率从 10% 增到 90%，MIRA 误差增长平缓，远优于 Moirai 等基线
- **扩展性良好**：从 MIRA_small (73M) 到 MIRA_large (455M)，性能稳步提升
- 在多个数据集上，MIRA 零样本性能接近甚至超过全监督微调模型

## 亮点与洞察

- **精准定位现实痛点**：不规则采样、异构频率、频繁缺失是临床数据的核心挑战，MIRA 是首个在统一架构中同时解决这三者的基础模型
- **CT-RoPE 设计优雅**：将 RoPE 自然推广到连续时间域，保留了相对位置建模的优良性质
- **数据规模可观**：4540 亿时间点的预训练语料，是目前最大的医学时间序列预训练数据集之一
- **MoE + Neural ODE 组合**：频率专家处理多尺度时间动态，ODE 实现连续时间外推，设计合理

## 局限与展望

- channel-independent 设计可能丢失多变量间相关信息，ICU 多参数监测场景中变量交互重要
- Neural ODE 的计算开销较大（自适应步长 ODE 求解器），可能限制实时应用
- 评估主要集中在预测任务，未涉及分类、异常检测等其他临床任务
- 预训练数据来源数量有限（5 个公开数据集），在更多样化的数据上可能进一步提升

## 相关工作与启发

与 Time-MoE、Moirai、Chronos 等通用时间序列基础模型相比，MIRA 的核心差异在于原生支持不规则时间。与 ContiFormer、Neural-CDE 等不规则时间序列模型相比，MIRA 是首个达到基础模型规模的。启发：医学领域的特殊性（不规则性、缺失值）需要架构层面的原生支持，而非后处理（如插值）。

## 评分

- 新颖性: ⭐⭐⭐⭐ CT-RoPE 和 MoE+ODE 组合有设计亮点，但各组件单独来看不完全新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 7个 OOD + 5个 ID 数据集，13个基线，消融充分
- 写作质量: ⭐⭐⭐⭐ 结构完整，技术细节充分
- 价值: ⭐⭐⭐⭐⭐ 填补医学时间序列基础模型空白，数据集和基准均开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Synthetic Series-Symbol Data Generation for Time Series Foundation Models](synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)
- [\[NeurIPS 2025\] Diffusion Transformers as Open-World Spatiotemporal Foundation Models](diffusion_transformers_as_open-world_spatiotemporal_foundation_models.md)
- [\[NeurIPS 2025\] SEMPO: Lightweight Foundation Models for Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] How Foundational are Foundation Models for Time Series Forecasting?](how_foundational_are_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)

</div>

<!-- RELATED:END -->
