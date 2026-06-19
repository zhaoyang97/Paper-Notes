---
title: >-
  [论文解读] CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models
description: >-
  [NeurIPS 2025][时间序列][causal discovery] 提出 CausalDynamics——迄今最大规模的动力系统因果发现 benchmark（14000+ 图、5000 万+ 样本），涵盖从 3 维混沌 ODE/SDE 到层级耦合系统再到拟真气候模型的三层渐进复杂度体系，并全面评估了 10 种 SOTA 因果发现算法，揭示当前深度学习方法在高维非线性动力系统上的不足。
tags:
  - "NeurIPS 2025"
  - "时间序列"
  - "causal discovery"
  - "dynamical systems"
  - "benchmark"
  - "time series"
  - "ODE/SDE"
  - "causal graph"
---

# CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.16620](https://arxiv.org/abs/2505.16620)  
**代码**: [kausable/CausalDynamics](https://github.com/kausable/CausalDynamics)  
**领域**: 时间序列  
**关键词**: causal discovery, dynamical systems, benchmark, time series, ODE/SDE, causal graph

## 一句话总结

提出 CausalDynamics——迄今最大规模的动力系统因果发现 benchmark（14000+ 图、5000 万+ 样本），涵盖从 3 维混沌 ODE/SDE 到层级耦合系统再到拟真气候模型的三层渐进复杂度体系，并全面评估了 10 种 SOTA 因果发现算法，揭示当前深度学习方法在高维非线性动力系统上的不足。

## 研究背景与动机

- **因果发现需求广泛**：气候科学、生物学、金融等领域的非线性动力系统中，直接干预往往不可行，需要从观测时间序列数据中推断因果关系
- **现有 benchmark 不足**：大多数因果发现 benchmark 基于静态因果图或自回归模型，缺乏对连续状态空间演化、复杂反馈环路、随机性和 regime shift 的刻画
- **真实数据缺乏 ground truth**：现有真实世界数据集（如 CausalRivers、MoCap）缺少完全可解析的因果 ground truth，无法隔离算法局限性与数据特性
- **合成数据局限**：已有合成 benchmark 通常只包含少量图（如 Netsim、DREAM3&4），且多为弱非线性系统，无法全面评估算法在复杂动力系统上的表现
- **关键挑战未被系统涵盖**：噪声、未观测混淆变量 (unobserved confounders)、时间延迟 (time-lag)、varsortability 等实际问题需要统一框架进行系统评估
- **类似 CASP/ImageNet 的范式推动**：蛋白质结构预测的 CASP13 催生了 AlphaFold，ImageNet 催生了 AlexNet，作者认为因果发现领域同样需要一个大规模标准化 benchmark 来推动方法创新

## 方法详解

### 整体框架：三层渐进复杂度体系

CausalDynamics 采用分层设计，从简单到复杂递进：

| 层级 | 数据来源 | 挑战 | 图数量 |
|------|----------|------|--------|
| Tier 1 (Simple) | 59 个 3 维混沌 ODE/SDE 系统 | 混淆变量、噪声 | 585 |
| Tier 2 (Coupled) | 层级耦合 ODE/SDE (N=3,5,10) | 混淆、时延、标准化、forcing | 14096 |
| Tier 3 (Climate) | MAOOAM + ENSO 气候模型 | 高维度 | 12 |

### 关键设计 1：结构动力因果模型 (SDCM)

将经典结构因果模型 (SCM) 与微分方程结合，定义 SDCM：

$$\frac{d}{dt}x_{k,t} := f^k(\boldsymbol{x}_{\text{PA}_k, t}, \delta), \quad x_{k,0} = x_k(0)$$

其中 $\delta$ 控制噪声幅度：$\delta=0$ 为 ODE，$\delta>0$ 为 SDE。每个系统的因果图由邻接矩阵 $\mathcal{A}$ 表示。

### 关键设计 2：基于 GNR 的层级耦合图生成

Tier 2 采用 Growing Network with Redirection (GNR) 模型生成无标度 DAG：

- **节点（因果单元）**：每个节点代表 $d$ 维时间序列，根节点可选动力系统驱动（Lorenz/Rössler）、周期驱动（$A\sin(\omega t + \phi)$）或线性驱动
- **边（耦合函数）**：使用 MLP 实现，激活函数从 {identity, sin, sigmoid, tanh, ReLU} 中随机采样，权重以 dropout 概率 $p_{\text{zero}}$ 稀疏化
- **信息聚合**：非根节点的值由父节点信号经 MLP 变换后求和得到：$x_{v_k}(t) = \sum_{k \in \text{pa}_i} f_{(k,i)}(x_k(t))$

### 关键设计 3：系统化因果挑战注入

- **混淆变量**：采样两个邻接矩阵，对第二个矩阵的非对角项旋转 90°后合并，确保混淆图仍为无标度 DAG
- **时间延迟**：以概率 $p_t$ 在边上引入固定延迟 $\tau$：$x_{v_k}(t) = f_{(k,i)}(x_k(t-\tau))$，形成时间上的循环图
- **标准化**：对节点值沿时间维度标准化以消除 varsortability 伪影

### 关键设计 4：Tier 3 拟真气候模型

- **ENSO 模型 (XRO)**：融合 Hasselmann 随机框架与 recharge oscillator 动力学，基于观测 SST 初始化，可调节跨洋盆耦合强度
- **MAOOAM 模型 (qgs)**：准地转两层模型，求解正压/斜压相互作用，模拟高维大气-海洋耦合动力学

## 损失函数与训练

本文为 benchmark 论文，不涉及新模型训练。评估指标包括：

- **AUROC**（越高越好）：衡量图重建的分类能力
- **AUPRC**（越高越好）：对稀疏图更敏感，惩罚假阳性
- **SHD**（越低越好）：结构汉明距离，衡量预测图与真实图的编辑距离

## 实验关键数据

### 表 1：AUROC / AUPRC 结果（部分，10 种算法）

| 实验 | PCMCI+ | F-PCMCI | VARLiNGAM | DYNOTEARS | NGC | TSCI |
|------|--------|---------|-----------|-----------|-----|------|
| Simple-Default | **.52/.71** | .51/.70 | .50/.69 | .43/.67 | .50/.69 | .46/.68 |
| Coupled-Default | .67/.25 | **.67/.27** | .60/.19 | .59/.21 | .50/.15 | .60/.23 |
| Coupled-Confounder | **.58/.20** | .55/.19 | .51/.17 | .49/.17 | .50/.16 | .51/.18 |
| Climate-MAOOAM | **.69/.88** | .50/.81 | .50/.81 | .64/.86 | .50/.81 | .58/.84 |
| Climate-ENSO | .57/.70 | **.57/.70** | .56/.69 | .55/.69 | .50/.67 | .50/.67 |

### 表 2：SHD 结果（部分，越低越好）

| 实验 | PCMCI+ | F-PCMCI | NGC | CUTS+ | RCD |
|------|--------|---------|-----|-------|-----|
| Simple-Default | 41.04 | 35.30 | **28.91** | 48.11 | 61.85 |
| Coupled-Default | 224.80 | 192.90 | 840.95 | **152.00** | 157.05 |
| Coupled-Time-lag | 327.72 | 350.61 | 793.67 | 247.22 | **201.11** |
| Climate-MAOOAM | 80.00 | 130.00 | **31.00** | 130.00 | 130.00 |
| Climate-ENSO | 529.36 | 530.27 | **337.09** | 608.73 | 665.36 |

**核心发现**：

- 非 DL 方法（PCMCI+、F-PCMCI）在大多数场景下优于 DL 方法，尤其在高维耦合系统上
- 基于拓扑的 TSCI 在有混淆变量和高维系统时优于纯 DL 方法
- 所有方法在耦合动力系统上表现不佳：普遍出现虚假自相关推断和在非平稳动力学下过度密集的邻接矩阵预测
- 标准化（消除 varsortability）对方法提升有限，尤其是 VARLiNGAM 等依赖拓扑排序的方法

## 亮点

- **规模空前**：14000+ 图、5000 万+ 样本，远超现有同类 benchmark
- **三层渐进设计**：从简单混沌系统到拟真气候模型，系统覆盖不同复杂度
- **可扩展框架**：plug-and-play 式工作流，用户可自定义耦合结构、噪声水平、延迟等参数生成新数据
- **全面评估**：同时评估 10 种涵盖 5 大类（Granger、约束、噪声、得分、拓扑）的因果发现算法
- **揭示重要 insight**：DL 方法在自称擅长的高维非线性场景中反而不如简单的非 DL 方法，指出了研究方向

## 局限性

- 当前仅处理固定参数（时延、噪声水平）和 3 维基础系统，未覆盖更高维或变参数场景
- Tier 3 只有 12 个气候图，统计显著性有限
- 生成数据与真实观测之间仍有 gap，框架虽可扩展但尚未对接真实气候再分析数据
- 评估指标（SHD）不考虑图大小和边密度，跨层级比较困难
- 未纳入近期基于 Transformer 或扩散模型的因果发现方法作为 baseline

## 相关工作对比

| 方面 | CausalDynamics | CausalTime | CauseMe | CausalBench |
|------|---------------|------------|---------|-------------|
| 数据类型 | ODE/SDE + 气候模型 | DL 生成 | SAVAR + 气候 | 单细胞 RNA |
| 图数量 | 14000+ | 少量 | 少量 | 数千（静态图）|
| 动力系统 | ✓（混沌、随机） | 部分 | 部分 | ✗ |
| Ground truth | ✓（解析推导） | ✗（缺乏可靠验证） | 部分 | ✓ |
| 可扩展性 | ✓（plug-and-play） | 有限 | 有限 | 有限 |

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个大规模动力系统因果发现 benchmark，三层设计和 GNR 耦合图生成方法新颖
- 实验充分度: ⭐⭐⭐⭐ — 10 种 SOTA 方法全面评估，但 Tier 3 图数量偏少
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，理论推导和实验呈现规范
- 价值: ⭐⭐⭐⭐ — 有望成为因果发现领域的标准 benchmark，推动方法发展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Causal Discovery from Conditionally Stationary Time Series](../../ICML2025/time_series/causal_discovery_from_conditionally_stationary_time_series.md)
- [\[ICLR 2026\] Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research](../../ICLR2026/time_series/omni-ieeg_a_large-scale_comprehensive_ieeg_dataset_and_benchmark_for_epilepsy_re.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)
- [\[ICCV 2025\] VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models](../../ICCV2025/time_series/vlrmbench_a_comprehensive_and_challenging_benchmark_for_vision-language_reward_m.md)
- [\[NeurIPS 2025\] Causal Masking on Spatial Data: An Information-Theoretic Case for Learning Spatial Datasets with Unimodal Language Models](causal_masking_on_spatial_data_an_information-theoretic_case_for_learning_spatia.md)

</div>

<!-- RELATED:END -->
