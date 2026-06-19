---
title: >-
  [论文解读] AirDDE: Multifactor Neural Delay Differential Equations for Air Quality Forecasting
description: >-
  [AAAI2026][时间序列][air quality forecasting] 首个将神经延迟微分方程（NDDE）引入空气质量预测的框架，通过记忆增强注意力模块和物理引导的延迟演化函数，对污染物连续时间传播中的延迟效应进行建模，在三个数据集上平均 MAE 降低 8.79%。 空气质量预测对公共健康和环境可持续性至关重要…
tags:
  - "AAAI2026"
  - "时间序列"
  - "air quality forecasting"
  - "neural delay differential equations"
  - "physics-guided"
  - "spatiotemporal graph"
---

# AirDDE: Multifactor Neural Delay Differential Equations for Air Quality Forecasting

**会议**: AAAI2026  
**arXiv**: [2603.17529](https://arxiv.org/abs/2603.17529)  
**代码**: [github.com/w2obin/airdde-aaai](https://github.com/w2obin/airdde-aaai)  
**领域**: 时间序列  
**关键词**: air quality forecasting, neural delay differential equations, physics-guided, spatiotemporal graph  

## 一句话总结

首个将神经延迟微分方程（NDDE）引入空气质量预测的框架，通过记忆增强注意力模块和物理引导的延迟演化函数，对污染物连续时间传播中的延迟效应进行建模，在三个数据集上平均 MAE 降低 8.79%。

## 背景与动机

空气质量预测对公共健康和环境可持续性至关重要，但污染物动力学的复杂性使得精确预测极具挑战。现有方法存在两个关键局限：

1. **离散时间建模**：传统 STGNN 和注意力方法将污染物动态建模为离散时间过程，无法捕捉真实世界中污染物的连续时间演化特性。
2. **忽视传播延迟**：即便近年来基于 Neural ODE 的方法（如 AirPhyNet、AirDualODE）将建模推进到连续时间，但它们仍采用"瞬时假设"——系统演化仅依赖当前状态，忽略了污染物从源头传播到下游地区所需的非零时间延迟。

实际场景中，延迟效应无处不在：**某地排放的污染物可能需要数小时才能被风输送到下游区域**，源与观测影响之间存在不可忽略的时滞。此外，延迟具有空间异质性——不同站点间的延迟由气象因素（风速风向）和地理距离共同调制，现有 NDDE 方法只能建模全局统一延迟，无法刻画这种位置特异性。

## 核心问题

如何在连续时间污染物演化框架中有效建模**多因素调制的异质性延迟效应**？具体需解决：

- 延迟受风场、地理距离等多因素动态调制，非固定常数
- 某地某时的浓度是周围多个位置以不同延迟到达的污染物累积叠加的结果
- 这种时空累积效应植根于大气动力学过程，纯数据驱动方法难以有效捕捉

## 方法详解

AirDDE 整体架构包含四个核心部分：

### 1. 时空编码器（Spatiotemporal Encoder）

采用 GNN-GRU 架构：通过可学习节点嵌入 $\boldsymbol{E}_1, \boldsymbol{E}_2$ 构建自适应邻接矩阵 $\boldsymbol{A}$，并用 GNN 替换 GRU 门控中的 MLP，将图拓扑融入时序更新：

$$\boldsymbol{h}_e^t = \text{GNN-GRU}(\boldsymbol{X}^t, \boldsymbol{h}_e^{t-1}, \boldsymbol{A})$$

### 2. 扩散-平流图构建（Diffusion-Advection Graph）

- **扩散图** $\boldsymbol{A}_{\text{diff}}$：基于站点间 Haversine 地理距离 + 高斯核归一化，建模无风/弱风条件下的扩散传播
- **平流图** $\boldsymbol{A}_{\text{adv}}^t$：**每个时间步动态构建**，若位置 $j$ 在 $t_2$ 时刻的风速和方向能使气团在 $t_1$ 时刻到达位置 $i$，则建立有向边。时滞 $\tau = t_1 - t_2$ 即为污染物传输时间。这比全局统一延迟假设更灵活

### 3. 记忆增强注意力模块（MAA）

针对污染传播的双尺度历史模式设计双重注意力：

- **全局记忆建模**：引入可学习全局记忆单元 $\boldsymbol{M}_g \in \mathbb{R}^{m \times d_e}$，通过注意力机制让当前特征 $\boldsymbol{h}_e^t$ 自适应检索全局历史模式（如持续高 PM2.5 区域）
- **局部记忆建模**：基于平流图定义动态邻域 $\mathcal{N}(i)^t$，对每个位置 $i$ 在时滞 $\tau$ 窗口内对自身及邻居的历史特征做注意力聚合，捕捉局部瞬态事件（如沙尘暴引发的 AQI 突变）

最终拼接编码器特征、全局记忆特征和局部记忆特征，经 MLP 得到延迟感知的初始状态 $\boldsymbol{h}_m^t$。

### 4. 物理引导的延迟演化函数（PDE）

以扩散-平流方程为物理先验，演化函数建模三个分项：

$$\frac{d\boldsymbol{h}^t}{dt} = \underbrace{D \cdot \text{GNN}_{\text{diff}}(\boldsymbol{A}_{\text{diff}}, \boldsymbol{h}^t)}_{\text{扩散项}} + \underbrace{\text{GNN}_{\text{adv}}(\boldsymbol{A}_{\text{adv}}^t, \boldsymbol{h}^{t-\tau})}_{\text{延迟平流项}} + \underbrace{f(\boldsymbol{h}^t || \boldsymbol{M})}_{\text{源/汇项}}$$

- **扩散项**：通过扩散图上的 GNN 消息传递近似 Chebyshev 多项式
- **延迟平流项**：关键创新——使用**历史状态** $\boldsymbol{h}^{t-\tau}$ 而非当前状态，在平流图上传播，显式建模传输延迟
- **源/汇项**：用 MLP 从多因素特征中学习隐含的源（如风输入）和汇（如降水清除）

DDE 求解器维护历史状态缓冲区，采用四阶 Runge-Kutta 积分求解未来状态，最后通过 GNN-GRU 解码器输出预测。训练损失为 Huber Loss，对离群值鲁棒。

## 实验关键数据

### 数据集

| 数据集 | 因素数 | 站点数 | 时间范围 | 粒度 |
|---------|--------|--------|----------|------|
| KnowAir | 18 | 184 | 2015-2018 | 3h |
| China-AQI | 8 | 209 | 2017-2019 | 1h |
| US-PM | 8 | 175 | 2020-2021 | 1h |

### 主实验（vs 19 个 baseline）

| 数据集 | AirDDE MAE | 次优 MAE | MAE 降幅 |
|--------|-----------|----------|---------|
| KnowAir | **16.92** | 18.64 (AirDualODE) | -9.23% |
| China-AQI | **17.03** | 18.89 (AirDualODE) | -9.85% |
| US-PM | **3.53** | 3.81 (PDFormer) | -7.3% |

在 China-AQI 上提升最大——该数据集时间粒度更细、污染水平更高、站点更多，动力学最复杂，AirDDE 的延迟建模优势最为显著。

### 消融实验（KnowAir，3 天平均 MAE）

| 变体 | AVG MAE | 对比完整模型 |
|------|---------|-------------|
| 完整 AirDDE | **16.92** | — |
| 去掉 MAA 模块 | 19.16 | +13.2% |
| 去掉全局记忆 | 17.80 | +5.2% |
| 去掉局部记忆 | 17.44 | +3.1% |
| 去掉 PDE 函数 | 19.39 | +14.6% |
| 去掉源/汇项 | 18.18 | +7.4% |
| 用注意力替换物理先验 | 18.78 | +11.0% |

PDE 函数和 MAA 模块均为关键组件；物理引导变体始终优于纯数据驱动变体。

### 鲁棒性（KnowAir）

随数据质量下降（缺失率 10%→50%、噪声 80dB→40dB），AirDDE 相对次优方法的 MAE 改善从 9.23% 扩大到 15.42%，表明全局记忆和物理引导演化在数据退化条件下有更强的恢复和去噪能力。

### 效率

在 China-AQI 上，AirDDE 训练时间 9.24 min/epoch、GPU 显存 10.46 GB，均优于对手 AirDualODE（10.09 min、11.14 GB），且 MAE 降低 9.85%。相比 AirFormer 和 PDFormer，训练时间略长但 MAE 分别降低 13.11% 和 10.70%。

## 亮点

1. **首创性**：首次将 NDDE 引入空气质量预测并结合物理引导，将延迟建模从"全局统一"推进到"位置-时间特异性"
2. **物理可解释性强**：PDE 函数直接对应扩散-平流方程的三个分项，各项物理意义清晰
3. **延迟建模的完整性**：MAA 捕捉多因素调制的延迟初始状态，PDE 在演化过程中维持延迟感知，形成闭环
4. **实验全面**：19 个 baseline、3 个数据集、消融/长期/鲁棒/效率/超参/案例研究，覆盖面极好
5. **案例分析直观**：城市级和区域级平流延迟案例清晰展示了模型对风驱动延迟传输的捕捉能力

## 局限与展望

1. **延迟状态维护效率**：DDE 求解器需维护历史状态缓冲区，随站点和时间窗口增长计算开销增大
2. **延迟建模为确定性的**：风场本身具有随机性，未来可引入不确定性量化到延迟估计中
3. **未建模复合延迟**：现实中污染物可能经中间区域中转，形成多跳复合传输路径，当前框架未显式建模
4. **平流图构建依赖风场数据质量**：若气象数据缺失或不准确，动态平流图质量会下降
5. **时滞 $\tau$ 为离散超参数**：从 {0,1,2,3} 中选取，未来可探索连续/自适应延迟学习

## 与相关工作的对比

| 方法 | 时间建模 | 延迟建模 | 物理引导 | 多因素 |
|------|---------|---------|---------|--------|
| STGNN 类（GAGNN 等） | 离散 | ✗ | ✗ | 部分 |
| AirFormer | 离散 | ✗ | ✗ | ✓ |
| PDFormer | 离散 | 统一延迟 | ✗ | ✗ |
| AirPhyNet | 连续 (NODE) | ✗ | ✓ | ✓ |
| AirDualODE | 连续 (NODE) | ✗ | ✓ | ✓ |
| STDDE | 连续 (NDDE) | 统一延迟 | ✗ | ✗ |
| **AirDDE** | **连续 (NDDE)** | **异质延迟** | **✓** | **✓** |

AirDDE 是唯一同时具备连续时间建模、异质延迟、物理引导和多因素融合的方法。

## 启发与关联

- **NDDE + 物理先验**是对 Neural ODE 领域的有价值扩展思路，可推广到交通流预测、传染病传播等存在延迟效应的时空预测任务
- 动态平流图构建方式（基于风速风向和距离的条件边）可迁移到其他需要建模传播路径的场景
- 全局记忆 + 局部记忆的双尺度设计对处理数据缺失/噪声场景有启发，可与其他时空预测框架结合

## 评分

- 新颖性: ⭐⭐⭐⭐ （首次将 NDDE + 物理引导用于空气质量，延迟异质性建模有原创性）
- 实验充分度: ⭐⭐⭐⭐⭐ （19 baseline + 6 类实验 + 案例分析，非常全面）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，物理动机阐述充分）
- 价值: ⭐⭐⭐⭐ （对时空预测中延迟建模有方法论贡献，可推广性好）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Random Controlled Differential Equations](../../ICLR2026/time_series/random_controlled_differential_equations.md)
- [\[ICML 2026\] Learning Manifold and Itô Dynamics with Branched Neural Rough Differential Equations](../../ICML2026/time_series/learning_manifold_and_itô_dynamics_with_branched_neural_rough_differential_equat.md)
- [\[CVPR 2026\] Real-Time Long Horizon Air Quality Forecasting via Group-Relative Policy Optimization](../../CVPR2026/time_series/real-time_long_horizon_air_quality_forecasting_via_group-relative_policy_optimiz.md)
- [\[NeurIPS 2025\] In-Context Learning of Stochastic Differential Equations with Foundation Inference Models](../../NeurIPS2025/time_series/in-context_learning_of_stochastic_differential_equations_with_foundation_inferen.md)
- [\[AAAI 2026\] Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting](sonnet_spectral_operator_neural_network_for_multivariable_time_series_forecastin.md)

</div>

<!-- RELATED:END -->
