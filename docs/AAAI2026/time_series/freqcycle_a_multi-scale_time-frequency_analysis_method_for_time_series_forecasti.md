---
title: >-
  [论文解读] FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting
description: >-
  [AAAI 2026][时间序列][时间序列预测] 提出FreqCycle框架，通过FECF模块显式学习共享周期模式、SFPL模块增强中高频能量占比，并扩展为MFreqCycle处理耦合多周期性，在7个基准上达到SOTA性能与效率的最优平衡。 时间序列预测（TSF）中，挖掘时频特征至关重要。现有研究存在三个关键不足： 过度依…
tags:
  - "AAAI 2026"
  - "时间序列"
  - "时间序列预测"
  - "频域分析"
  - "周期性建模"
  - "中高频增强"
  - "多尺度分解"
---

# FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting

**会议**: AAAI 2026  
**arXiv**: [2603.09661](https://arxiv.org/abs/2603.09661)  
**代码**: [github.com/boya-zhang-ai/FreqCycle](https://github.com/boya-zhang-ai/FreqCycle)  
**领域**: 时间序列预测  
**关键词**: 时间序列预测, 频域分析, 周期性建模, 中高频增强, 多尺度分解

## 一句话总结

提出FreqCycle框架，通过FECF模块显式学习共享周期模式、SFPL模块增强中高频能量占比，并扩展为MFreqCycle处理耦合多周期性，在7个基准上达到SOTA性能与效率的最优平衡。

## 研究背景与动机

时间序列预测（TSF）中，挖掘时频特征至关重要。现有研究存在三个关键不足：

**过度依赖复杂架构提取周期模式**：深度学习模型试图通过注意力机制等复杂结构学习长程依赖中的周期规律，但实际上许多时间序列（如电力、交通）存在明显的日周期和周周期模式（图1展示了ETTm2数据集中的日周期和经平均池化后的周周期），可以更直接地显式建模

**忽视中高频成分**：现有方法（如DLinear等基于MLP的模型）擅长学习低频周期成分，但其基础架构本质上难以有效表示中高频成分。从频谱分析来看，中高频成分的能量占比极低（图2左），但它们包含关键的短期波动和非周期特征信息

**缺乏多尺度嵌套周期的专门处理**：时间序列数据常展现耦合多周期性（如日周期嵌套在周周期内），简单使用MLP模型无法自适应地在多层级协调建模这些嵌套结构

## 方法详解

### 整体框架

FreqCycle（图3a）由两个互补模块组成：
1. **FECF（滤波增强周期预测）**：在时域显式学习共享周期模式，处理低频成分
2. **SFPL（分段频域模式学习）**：在频域增强中高频能量占比，处理残差中的非周期成分

整体流程：输入 → FECF提取并去除周期成分 → 残差送入SFPL → 残差预测 + 滤波后周期预测 = 最终预测

对于长回看窗口场景，进一步扩展为**MFreqCycle**，通过多尺度并行框架解耦嵌套周期特征。

### 关键设计

#### 1. **FECF（Filter-Enhanced Cycle Forecasting）模块**

FECF是对CycleNet的改进，核心思想是**显式学习全局共享的周期基底**：

- 定义可学习的周期基底 $Q \in \mathbb{R}^{W \times D}$（初始化为零矩阵），$W$ 为基周期长度
- 通过周期复制操作生成与输入/输出等长的周期分量 $c_{t-L+1:t}$ 和 $c_{t+1:t+H}$
- **自适应滤波增强**：对预测周期分量进行频域滤波，放大低频信息、衰减中高频干扰：

$$c'_{t+1:t+H} = \text{IFFT}(\text{Filter}(\text{FFT}(c_{t+1:t+H})))$$

$$\text{Filter}(\xi) = \xi \odot \theta_c$$

其中 $\theta_c$ 是可学习的滤波器参数，$\odot$ 为逐元素乘法。

完整FECF流程：
1. 提取残差：$r_{t-L+1:t} = x_{t-L+1:t} - c_{t-L+1:t}$
2. 残差预测：$r_{t+1:t+H} = \text{SFPL}(r_{t-L+1:t})$
3. 重建预测：$\bar{x}_{t+1:t+H} = r_{t+1:t+H} + c'_{t+1:t+H}$

基周期 $W$ 由数据固有特性决定：1小时间隔数据取 $W=24$（日周期）或 $W=168$（周周期）。

#### 2. **SFPL（Segmented Frequency-domain Pattern Learning）模块**

SFPL专门设计用于**增强中高频成分的能量贡献**，其灵感来自短时傅里叶变换（STFT）：

1. **分段**：将输入 $r_{t-L+1:t}$ 通过滑动窗口分为 $s$ 个子段，两端零填充后堆叠得 $R \in \mathbb{R}^{s \times L \times D}$
2. **频域变换**：对 $R$ 沿时间维做FFT得 $F \in \mathbb{C}^{s \times \lfloor L/2+1 \rfloor \times D}$
3. **可学习滤波与自适应加权**：

$$\theta'_1, \ldots, \theta'_s = \text{softmax}(\theta_1, \ldots, \theta_s)$$
$$F' = F \odot \Theta'_F, \quad f = \sum_{i=1}^{s} F'(i)$$

4. **重建**：$f$ 经iFFT后通过FFN层产生残差预测输出

分段操作实质上是STFT的变体：缩短时域窗口实现更高的频域局部性，从而更精确地定位瞬态频率成分。经过可学习滤波和自适应加权整合后的频域表示 $f$ 有效增强了中高频能量（图2右 vs 左）。

#### 3. **MFreqCycle（多尺度扩展）**

为处理耦合多周期性（如日-周嵌套），MFreqCycle采用多尺度并行架构：

- **基周期模块**：捕获最小显著周期（如日周期）及其非周期特征
- **周周期模块**：建模宏观尺度周期模式（如周周期），使用更长输入窗口
    - 包含池化层 + 特征学习模块 + Linear层
    - 池化+投影设计在提取关键趋势的同时保持计算效率

多尺度预测融合：

$$\theta'_0, \theta'_1 = \text{softmax}(\theta_0, \theta_1)$$
$$\bar{x}_{t+1:t+H} = \bar{x}^0_{t+1:t+H} \odot \theta'_0 + \bar{x}^1_{t+1:t+H} \odot \theta'_1$$

### 损失函数 / 训练策略

- 使用标准MSE损失进行端到端训练
- 周期基底 $Q$ 和自适应滤波器通过梯度反向传播联合训练
- 理论复杂度为 $O(L \log L)$，避免了注意力机制的二次复杂度

## 实验关键数据

### 主实验（7个数据集，L=96，H∈{96,192,336,720}，取平均）

| 数据集 | FreqCycle(MSE/MAE) | CycleNet | DLinear | iTransformer | PatchTST |
|--------|-------------------|----------|---------|-------------|----------|
| ETTm1 | **0.372/0.389** | 0.386/0.395 | 0.403/0.407 | 0.407/0.410 | 0.387/0.400 |
| ETTm2 | **0.263/0.311** | 0.272/0.315 | 0.350/0.401 | 0.288/0.332 | 0.281/0.326 |
| ETTh1 | **0.428/0.427** | 0.432/0.427 | 0.456/0.452 | 0.454/0.448 | 0.469/0.455 |
| ETTh2 | 0.371/0.399 | 0.383/0.404 | 0.559/0.515 | 0.383/0.407 | 0.387/0.407 |
| Weather | 0.243/0.270 | 0.254/0.279 | 0.265/0.317 | 0.258/0.278 | 0.259/0.281 |
| ECL | **0.168/0.259** | 0.170/0.260 | 0.212/0.300 | 0.178/0.270 | 0.205/0.290 |
| Traffic | **0.448/0.261** | 0.485/0.313 | 0.625/0.383 | 0.428/0.282 | 0.481/0.304 |

FreqCycle在14个指标中取得10个第一、2个第二。

### 消融实验

| 配置 | ETTh1 MSE | ETTh2 MSE | Traffic MAE | Weather MSE |
|------|-----------|-----------|-------------|-------------|
| FreqCycle完整 | 0.369 | 0.282 | 0.262 | 0.159 |
| 去掉FECF | 0.375(+1.65%) | 0.292(+3.26%) | 0.318(+21.30%) | 0.179(+12.59%) |
| SFPL替换为MLP | 0.379(+2.66%) | 0.302(+6.91%) | 0.262(+0.04%) | 0.161(+1.51%) |

| 频域处理方法对比 | ETTh2 MSE | ECL MSE | Traffic MSE |
|----------------|-----------|---------|-------------|
| FreqCycle(SFPL) | **0.282** | **0.139** | **0.438** |
| SFPL→LPF(FITS) | 0.294(+4.00%) | 0.143(+3.39%) | 0.449(+2.63%) |
| SFPL→PaiFilter | 0.293(+3.68%) | 0.143(+2.96%) | 0.445(+1.81%) |

### MFreqCycle长回看窗口实验

| 数据集 | L | FreqCycle MSE | CycleNet MSE | FITS MSE |
|--------|---|--------------|-------------|----------|
| ETTh2 | 96 | 0.371 | 0.383 | 0.383 |
| ETTh2 | 168 | **0.241** | 0.388 | 0.333 |
| ETTm2 | 96 | 0.263 | 0.267 | 0.286 |
| ETTm2 | 672 | **0.167** | 0.262 | 0.250 |

MFreqCycle在长回看窗口下性能大幅提升（ETTm2从0.263降至0.167）。

### 关键发现

1. **FECF和SFPL具有互补性**：FECF在周期性强的数据集（Traffic, Weather）提升最大，SFPL在非平稳性强的数据集（ETTh系列）表现突出
2. **SFPL优于传统频域滤波**：相比低通滤波（FITS）和PaiFilter，SFPL通过分段增强更有效地提取中高频信息
3. **效率优势**：FreqCycle在预测性能、峰值内存消耗和训练速度三个维度上达到最优平衡（图4）
4. **可视化验证**：频谱分析明确显示SFPL增强了中高频能量占比同时保持低频成分结构（图5）

## 亮点与洞察

1. **显式周期建模 vs 隐式学习**：放弃用复杂架构隐式捕获周期性，转而用参数化周期基底直接学习，简单高效
2. **中高频成分的重要性**：首次系统性地论证并解决了MLP-based方法在中高频建模上的固有缺陷
3. **STFT启发的分段策略**：将信号处理中的短时傅里叶变换思想引入深度学习，理论基础扎实
4. **多尺度设计的实用性**：MFreqCycle在长回看窗口下的巨大提升证明了解耦嵌套周期特征的价值

## 局限与展望

1. **基周期W需要手动设定**：依赖数据的先验知识（如采样频率），自动确定最优周期组合是一个开放问题
2. **MFreqCycle仅验证到周周期**：年周期等更长周期因数据集限制未能验证
3. **通道独立（Channel-Independent）**：未显式建模变量间的相关性
4. 可考虑与Transformer结合，利用注意力机制捕获跨通道依赖
5. 分段策略中子段数s的选择与数据特性的关系有待进一步研究

## 相关工作与启发

- **CycleNet**（Lin et al., 2024a）：本文FECF的前身，显式建模周期模式但缺乏频域增强
- **FITS**（Xu et al., 2024）：频域低通滤波方法，但仅关注低频成分
- **FilterNet**（Yi et al., 2024）：可学习滤波器方法，本文SFPL在此基础上进一步改进
- 启发：时间序列预测中，时域和频域方法各有所长，将二者结合（FECF处理低频+SFPL处理中高频）可以实现互补

## 评分

- 新颖性: ⭐⭐⭐⭐ — SFPL的中高频增强是有意义的创新，但整体是组合式改进
- 实验充分度: ⭐⭐⭐⭐⭐ — 7个数据集、多层消融、效率分析、可视化验证齐全
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，理论动机充分，但部分公式推导可以更精简
- 价值: ⭐⭐⭐⭐ — 性能与效率的良好平衡，对MLP-based TSF方法有实际推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting](m2fmoe_multi-resolution_multi-view_frequency_mixture-of-experts_for_extreme-adap.md)
- [\[ICML 2026\] Generalizing Multi-scale Time-Series Modeling with a Single Operator](../../ICML2026/time_series/generalizing_multi-scale_time-series_modeling_with_a_single_operator.md)
- [\[AAAI 2026\] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)
- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](../../ICLR2026/time_series/learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[AAAI 2026\] Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification](counterfactual_explainable_ai_xai_method_for_deep_learning-based_multivariate_ti.md)

</div>

<!-- RELATED:END -->
