---
title: >-
  [论文解读] LightGTS: A Lightweight General Time Series Forecasting Model
description: >-
  [ICML2025][模型压缩][时间序列基础模型] 提出 LightGTS，利用时间序列固有的尺度不变周期性归纳偏置，通过 Periodical Tokenization 和 Periodical Parallel Decoding 两个核心技术，仅用不到 500 万参数就在 9 个基准数据集上的 zero-shot 和 full-shot 设定中取得了 SOTA 性能，比现有时序基础模型小 10-100 倍。
tags:
  - "ICML2025"
  - "模型压缩"
  - "时间序列基础模型"
  - "周期性分词"
  - "轻量级预测"
  - "跨尺度泛化"
---

# LightGTS: A Lightweight General Time Series Forecasting Model

**会议**: ICML2025  
**arXiv**: [2506.06005](https://arxiv.org/abs/2506.06005)  
**代码**: [decisionintelligence/LightGTS](https://github.com/decisionintelligence/LightGTS)  
**领域**: 时间序列  
**关键词**: 时间序列基础模型, 周期性分词, 轻量级预测, 跨尺度泛化

## 一句话总结

提出 LightGTS，利用时间序列固有的尺度不变周期性归纳偏置，通过 Periodical Tokenization 和 Periodical Parallel Decoding 两个核心技术，仅用不到 500 万参数就在 9 个基准数据集上的 zero-shot 和 full-shot 设定中取得了 SOTA 性能，比现有时序基础模型小 10-100 倍。

## 研究背景与动机

**核心问题**：现有时间序列基础模型（TSFM）依赖大规模多源预训练和庞大模型参数来获取泛化能力，计算开销巨大，难以部署在资源受限场景。

**关键观察**：时间序列数据有两个独特属性——**尺度（scale）**指采样率（如每15分钟或每小时），**内禀周期（intrinsic period）**指模式重复出现的时间间隔（如日周期）。不同尺度下同一个内禀周期对应不同的**周期长度（cycle length）**，例如 ETTh2（小时采样）日周期长度为 24，ETTm2（15分钟采样）日周期长度为 96。

**现有方法的缺陷**：

- **固定分词（Fixed Tokenization）**：每个 token 包含固定数量数据点，导致不同尺度下 token 的信息密度不一致，且会破坏周期模式的连续性和结构完整性
- 作者的 case study 清楚地表明：在 ETTh1 上预训练后，固定分词方法能识别同尺度数据集的周期，但迁移到不同尺度数据集时性能显著下降
- 这迫使模型需要更多参数来弥补这一缺陷，导致计算成本增加

**核心思路**：利用时间序列中尺度不变的内禀周期这一归纳偏置，设计自适应的周期性分词和解码方案，从而压缩模型参数量同时保证高性能。

## 方法详解

### 整体架构

LightGTS 基于 Transformer Encoder-Decoder 结构，包含三个核心组件：

1. **Periodical Tokenization**（周期性分词）：自适应地将时间序列分割为周期对齐的 patch
2. **Flex Projection Layer**（弹性投影层）：处理不同长度的 patch 并映射到统一语义空间
3. **Periodical Parallel Decoding (PPD)**（周期性并行解码）：利用编码器最后一个 token 初始化解码器输入

### 1. Periodical Patching（周期性分块）

给定单变量时间序列 $\mathbf{x} \in \mathbb{R}^L$，首先通过周期检测确定周期长度：

$$P = \text{PeriodsFinding}(\mathbf{x})$$

当有先验知识（如已知采样率）时直接推算；否则用 FFT 自动检测。然后将序列分割为非重叠的周期 patch $\mathbf{X}_p \in \mathbb{R}^{P \times N}$，其中 $N = \lfloor L/P \rfloor$。每个 patch 精确对齐一个完整周期，确保不同尺度下 token 承载的语义一致。

### 2. Flex Projection Layer（弹性投影层）

**问题**：不同数据集的周期长度 $P$ 不同，固定权重的 patch embedding 无法处理变长 patch。简单线性插值调整权重会引入偏差，使得 $\mathbf{x} \cdot \theta \neq \mathbf{x'} \cdot \theta'$。

**解决方案**：将线性插值形式化为线性变换 $\text{Interp}(\mathbf{x})_P^{P'} = \mathbf{x} \cdot \mathbf{A}$，其中 $\mathbf{A} \in \mathbb{R}^{P \times P'}$。通过求解优化问题：

$$\theta' = \arg\min_{\theta'} \mathbb{E}_{x \sim \mathcal{X}} \left[ \|\mathbf{x} \cdot \theta - \mathbf{x}\mathbf{A} \cdot \theta'\|_F^2 \right]$$

理论推导（考虑 RevIN 归一化下分布一致性约束）得到闭式解：

$$\theta' = \delta^{-1} (\mathbf{A})^+ \theta, \quad \delta = \sqrt{P/P'}$$

其中 $(\mathbf{A})^+$ 是 Moore-Penrose 伪逆。该 **Flex-resize** 操作无需额外学习，仅通过数学变换保证不同 patch 尺寸下嵌入的等价性。

模型定义参考权重 $\theta_e, \theta_d \in \mathbb{R}^{P^* \times D}$（默认 $P^*=48$），前向传播时动态 resize 适配当前序列的周期长度。

### 3. Encoding（编码）

使用标准 Transformer Encoder，注意力机制中引入 **RoPE（旋转位置编码）**建模 token 间的相对位置关系：

$$\mathbf{S}_{ij} = (\mathbf{W}_Q \mathbf{x}_e^i)^T \mathbf{R}_{i-j} (\mathbf{W}_K \mathbf{x}_e^j)$$

$$\mathbf{Attn}_i = \sum_j \frac{\exp(\mathbf{S}_{ij})}{\sum_k \exp(\mathbf{S}_{ik})} (\mathbf{W}_V \mathbf{x}_e^j)$$

### 4. Periodical Parallel Decoding (PPD)

**核心设计**：将编码器最后一个 token $\mathbf{e}^N$ 复制 $K = \lceil F/P \rceil$ 次作为解码器输入。关键 insight 包括：

- 最后一个 token 与未来预测保持时间连续性
- 利用历史和预测区间之间的周期结构一致性
- 非自回归方式避免累积误差，降低计算开销

对复制的 token 施加指数衰减权重 $\omega(\tau) = 1/e^\tau$，然后并行送入解码器：

$$\mathbf{Z} = \text{Decoder}(\{\omega(j) \mathbf{h}^j\}, \mathbf{E})$$

$$\hat{\mathbf{Y}} = \text{Flex-resize}(\theta_d)_{P}^{P^*} \cdot \mathbf{Z}$$

### 5. 损失函数

标准 MSE 损失：$\mathcal{L}_{\text{MSE}} = \|\mathbf{Y} - \hat{\mathbf{Y}}\|_F^2$

### 模型配置

| 变体 | Encoder层数 | Decoder层数 | 隐藏维度 | FFN维度 | 参数量 |
|------|------------|------------|---------|--------|-------|
| LightGTS-tiny | 1 | 1 | 256 | 512 | 1.3M |
| LightGTS-mini | 3 | 3 | 256 | 512 | 4M |

预训练配置：历史 token 数 $N=10$，预测 token 数 $K=4$，参考 patch 大小 $P^*=48$，batch size = 8192，学习率 $5 \times 10^{-4}$，Adam 优化器 + StepLR decay。

## 实验关键数据

### 预训练与评估数据集

- **预训练**：涵盖能源、自然、健康、交通、Web、经济等领域的 30+ 公开数据集（Monash、UEA、UCR 等），采样频率从毫秒到月级
- **评估**：9 个基准数据集（ETTh1/h2/m1/m2、Weather、Traffic、Electricity、Solar、Exchange），与预训练数据集严格不重叠
- 预测长度：$F \in \{96, 192, 336, 720\}$

### Zero-shot 预测结果（Table 1，各预测长度平均）

| 数据集 | LightGTS-mini | Timer | MOIRAI | Chronos | TimesFM | Time-MoE |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| ETTm1 | **0.327** | 0.768 | 0.390 | 0.551 | 0.435 | 0.376 |
| ETTm2 | **0.247** | 0.315 | 0.276 | 0.293 | 0.347 | 0.315 |
| ETTh1 | **0.388** | 0.562 | 0.510 | 0.533 | 0.479 | 0.394 |
| Weather | **0.208** | 0.292 | 0.260 | 0.288 | - | 0.270 |
| Traffic | **0.561** | 0.613 | - | 0.615 | - | - |
| Solar | **0.191** | 0.771 | 0.714 | 0.393 | 0.500 | 0.411 |
| Electricity | 0.213 | 0.297 | **0.188** | - | - | - |

- LightGTS-mini 平均 MSE 降低超过 **30%**（vs 最强 baseline）
- 即便是 LightGTS-tiny（1.3M 参数）也实现 27% 的平均 MSE 降低

### Full-shot 预测结果（Table 2，LightGTS-mini vs SOTA 小模型）

| 数据集 | LightGTS (full-shot) | PDF | iTransformer | PatchTST |
|--------|:---:|:---:|:---:|:---:|
| ETTm1 | **0.321** | 0.342 | 0.347 | 0.349 |
| Traffic | **0.393** | 0.395 | 0.397 | 0.397 |
| Solar | **0.179** | 0.200 | 0.202 | 0.200 |
| Electricity | **0.156** | 0.160 | 0.163 | 0.171 |

Full-shot 设定下比 6 个 SOTA baseline 平均 MSE 降低 **7%**。在 5 个数据集上 zero-shot 即超过 baseline 的 full-shot 结果。

### 效率对比（Table 3，ETTm1 F=720）

| 模型 | 参数量 | MACs | 最大内存(MB) | 推理时间(s) |
|------|--------|------|-------------|-----------|
| Time-MoE | 453M | 5252.9G | 14131 | 2.13 |
| Chronos | 700M | 92327.9G | 10269 | 34.33 |
| MOIRAI | 300M | 97.36G | 2009 | 0.10 |
| Timer | 67.4M | 52.6G | 1435 | 0.08 |
| PatchTST | 6.3M | 225M | 672 | 0.01 |
| **LightGTS** | **4M** | **213M** | 713 | **0.01** |

LightGTS 参数量比 Timer 小 17 倍，比 Chronos 小 175 倍，MACs 比 MOIRAI 小 450+ 倍。

### 消融实验（Table 4，zero-shot 平均）

| 解码方式 | 分词方式 | ETT-avg | Weather | Electricity | Traffic |
|---------|---------|:---:|:---:|:---:|:---:|
| PPD | Periodical | **0.328** | **0.208** | **0.213** | **0.561** |
| PPD | Fixed | 0.436 | 0.262 | 0.226 | 0.621 |
| AR | Periodical | 0.341 | 0.226 | 0.229 | 0.634 |
| AR | Fixed | 0.442 | 0.265 | 0.231 | 0.630 |
| MAE | Periodical | 0.388 | 0.260 | 0.322 | 0.746 |

- Periodical Tokenization 在所有解码方式下均稳定优于 Fixed Patching
- PPD 持续优于 AR 和 MAE，与 Periodical Patching 结合时增益更显著
- MAE 解码表现最差，可能因重建目标与预测任务存在 gap

### 解码器输入选择（Table 6）

| 初始化方式 | ETT-avg | Weather | Electricity | Traffic |
|-----------|:---:|:---:|:---:|:---:|
| Last token | **0.328** | **0.208** | **0.213** | **0.561** |
| Learnable | 0.342 | 0.278 | 0.231 | 0.627 |
| CLS token | 0.343 | 0.341 | 0.234 | 0.634 |
| Mean token | 0.404 | 0.328 | 0.273 | 0.703 |

Last token 由于与周期性最对齐且与预测任务最相关，效果最佳。

## 亮点与洞察

1. **归纳偏置驱动的设计哲学**：不追求模型规模，而是深挖时间序列的周期性归纳偏置，用正确的 inductive bias 替代暴力参数堆叠，设计理念值得借鉴
2. **Flex Projection 的理论推导**：通过 SVD 和 Moore-Penrose 伪逆给出闭式解，无需额外训练即可适配不同 patch 尺寸，兼具优雅与实用
3. **跨尺度一致性**：在不同采样粒度（如10分钟/30分钟/1小时）下，LightGTS 保持稳定性能，而 Timer 和 Time-MoE 波动很大（Fig. 4）
4. **可作为插件**：Periodical Tokenization 可直接应用到其他 TSFM（如 Timer），无需重训练即获 19.23% 的 MSE 提升（Table 11）

## 局限与展望

1. **依赖周期性假设**：对于缺乏明显周期性的数据（如 Exchange 汇率数据），周期分词的增益有限，FFT 检测到的周期可能不准确
2. **周期长度需先验知识或足够数据**：当无先验且数据量不足时，FFT 检测的周期长度可能偏离真实值，影响 Periodical Patching 效果
3. **channel-independent 范式**：预训练和微调均将多元时序拆成单变量处理，未建模变量间依赖
4. **评估数据集范围**：9 个基准数据集集中在能源、交通、天气等具有明显周期性的领域，缺乏金融、社交等弱周期/非平稳数据的验证
5. **参考 patch 大小 $P^*$**：虽然实验显示对 $P^*$ 不敏感，但默认选 48 的经验性较强，缺乏系统性选择指南

## 相关工作与启发

- **TimesNet (Wu et al., 2022)**：同样利用 FFT 发现周期性，但 LightGTS 在 tokenization 层面利用周期，更本质
- **PatchTST (Nie et al., 2023)**：固定 patch 大小的 baseline，LightGTS 的 periodical patching 是对其的关键改进
- **Timer (Liu et al., 2024)**：同为 Transformer 架构的 TSFM，LightGTS 直接对比并证明了固定分词的局限性
- **MOIRAI (Woo et al., 2024)**：基于采样频率预定义多种 patch 大小，但仍是离散选择而非连续自适应
- **TTMs**：使用 CV 启发的 patch merging 做自适应，但受限于预定义 patch 大小

## 评分
- 新颖性: ⭐⭐⭐⭐ — Periodical Tokenization + Flex Projection 的组合巧妙，理论推导完整
- 实验充分度: ⭐⭐⭐⭐⭐ — 9 个数据集 × zero-shot/full-shot + 详细消融 + 效率对比 + 跨分辨率鲁棒性分析
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，case study 直观，理论和实验组织良好
- 价值: ⭐⭐⭐⭐⭐ — 4M 参数达到 SOTA，对资源受限部署极具实际意义，Periodical Tokenization 可迁移到其他 TSFM

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs](../../AAAI2026/model_compression/xlinear_a_lightweight_and_accurate_mlp-based_model_for_long-term_time_series_for.md)
- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](../../ICLR2026/model_compression/enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)
- [\[ICCV 2025\] SDKD: Frequency-Aligned Knowledge Distillation for Lightweight Spatiotemporal Forecasting](../../ICCV2025/model_compression/frequency-aligned_knowledge_distillation_for_lightweight_spatiotemporal_forecast.md)
- [\[ICML 2025\] Joker: Joint Optimization Framework for Lightweight Kernel Machines](joker_joint_optimization_framework_for_lightweight_kernel_machines.md)
- [\[ICML 2025\] Neutral Residues: Revisiting Adapters for Model Extension](neutral_residues_revisiting_adapters_for_model_extension.md)

</div>

<!-- RELATED:END -->
