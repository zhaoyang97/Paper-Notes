---
title: >-
  [论文解读] KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks
description: >-
  [ICML 2025][目标检测][时间序列异常检测] KAN-AD 将时间序列异常检测重新建模为用光滑单变量函数逼近序列，用截断傅里叶展开替代 KAN 中的 B 样条避免局部扰动敏感性，以不到 1000 个参数在 4 个基准上平均提升 15% 检测精度。 领域现状： 时间序列异常检测 (TSAD) 是云服务和 Web 系统…
tags:
  - "ICML 2025"
  - "目标检测"
  - "时间序列异常检测"
  - "KAN"
  - "Kolmogorov-Arnold网络"
  - "B样条"
  - "傅里叶展开"
---

# KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks

**会议**: ICML 2025  
**arXiv**: [2411.00278](https://arxiv.org/abs/2411.00278)  
**代码**: 无  
**领域**: Time Series  
**关键词**: 时间序列异常检测, KAN, Kolmogorov-Arnold网络, B样条, 傅里叶展开

## 一句话总结
KAN-AD 将时间序列异常检测重新建模为用光滑单变量函数逼近序列，用截断傅里叶展开替代 KAN 中的 B 样条避免局部扰动敏感性，以不到 1000 个参数在 4 个基准上平均提升 15% 检测精度。

## 研究背景与动机
**领域现状**: 时间序列异常检测 (TSAD) 是云服务和 Web 系统实时监控的核心能力。主流方法基于预测模型（预测下一步，大偏差→异常）。

**现有痛点**: (a) 预测模型倾向于过拟合小波动，对局部扰动过于敏感；(b) 有效的 TSAD 应关注"正常"行为的全局平滑模式，而非细节抖动；(c) 直接使用 KAN (Kolmogorov-Arnold Network) 虽然理论上能用单变量函数逼近，但 B 样条的局部性使其对扰动敏感。

**核心矛盾**: 精确拟合 vs 鲁棒检测——过于精确的拟合反而降低了异常检测能力。

**本文切入**: 从 Kolmogorov-Arnold 表示定理出发，将时间序列建模为光滑单变量函数的组合。

**核心 idea**: 用截断傅里叶展开替代 B 样条作为 KAN 的基函数，傅里叶的全局性天然免疫局部扰动，加上轻量学习机制强调全局模式。

## 方法详解

### 整体框架
输入：时间序列窗口 → KAN-AD (傅里叶基函数 + 轻量学习机制) → 预测下一步值 → 计算预测误差 → 超阈值即为异常。

### 关键设计

1. **傅里叶 KAN 替代 B 样条 KAN**:

    - Kolmogorov-Arnold 表示定理：$f(\mathbf{x}) = \sum_{q=0}^{2n} \Phi_q(\sum_{p=1}^n \phi_{q,p}(x_p))$
    - 标准 KAN 用 B 样条参数化 $\phi_{q,p}$，但 B 样条是局部基函数——对输入小扰动敏感
    - KAN-AD 用截断傅里叶级数：$\phi(x) = a_0 + \sum_{k=1}^K (a_k \cos(kx) + b_k \sin(kx))$
    - 每个单变量函数的光滑性由傅里叶截断阶 $K$ 控制
    - 设计动机：傅里叶基函数是全局的，单个系数的改变影响整条曲线，天然抗局部噪声

2. **轻量学习机制**:

    - 强调全局模式的低频信息
    - 限制网络容量，避免拟合高频噪声
    - 极少参数（<1000 个可训练参数）
    - 设计动机：小模型天然具有正则化效果，学到的只能是最显著的模式

3. **异常检测策略**:

    - 正常数据训练：拟合"正常"行为的光滑模式
    - 测试时：异常点偏离光滑模式 → 大预测误差 → 检测为异常
    - 设计动机：模型无法拟合异常模式，从而产生大误差信号

### 损失函数 / 训练策略
- 预测损失：均方误差 (MSE) 用于下一步预测
- 仅在正常数据上训练（无监督异常检测范式）

## 实验关键数据

### 主实验

| 基准 | 指标 | KAN-AD | 之前SOTA | 提升 |
|------|------|--------|---------|------|
| 基准1 | F1 / AUC | 最优 | - | 显著 |
| 基准2 | F1 / AUC | 最优 | - | 显著 |
| 基准3 | F1 / AUC | 最优 | - | 峰值超27% |
| 基准4 | F1 / AUC | 最优 | - | 显著 |
| 4基准平均 | 检测精度 | - | - | **+15%** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 傅里叶 KAN (KAN-AD) | **最优** | 全局基函数抗局部扰动 |
| B 样条 KAN (原始 KAN) | 较差 | 局部基函数对噪声敏感 |
| 不同截断阶 $K$ | 性能曲线 | 中等 $K$ 最优，过大过拟合 |
| 推理速度 | 用时 | 比原始 KAN 快 50% |

### 关键发现
- 傅里叶基函数比 B 样条在 TSAD 任务上大幅领先
- 不到 1000 个参数即可达到 SOTA 性能——极致的参数效率
- 推理速度比原始 KAN 快 50%，得益于傅里叶变换的高效实现
- 傅里叶截断阶 $K$ 类似于频率带宽，控制拟合精度 vs 鲁棒性的权衡

## 亮点与洞察
- **重新思考 TSAD 本质**: 检测异常 ≠ 精确预测，而是识别偏离光滑模式的偏差
- **KAN 的实际应用**: 将 KAN 从理论工具转化为实用的轻量模型
- **极致参数效率**: <1000 参数挑战万参数级 SOTA，暗示 TSAD 的内在低维性
- **工程价值**: 小模型 + 快推理 = 非常适合实时监控场景

## 局限与展望
- 傅里叶展开假设一定的周期性/规律性，对完全非周期的时间序列可能不适用
- 单变量函数组合可能不足以表达高维时间序列中的复杂依赖
- 截断阶 $K$ 目前需要手动选择
- 仅在 4 个标准基准上验证，实际生产环境的多样性未覆盖

## 相关工作与启发
- 原始 KAN (Liu et al. 2024) 提出了 Kolmogorov-Arnold 网络架构
- TSAD 经典方法：LSTM-based, Transformer-based, 图神经网络
- 统计方法如 STL 分解也利用光滑性假设
- 启发：好的归纳偏置比大模型更重要——TSAD 需要的是光滑性而非容量

## 评分
- 新颖性: ⭐⭐⭐⭐ KAN+傅里叶在 TSAD 中的应用新颖且有理论支撑
- 实验充分度: ⭐⭐⭐⭐ 4 个基准 + 完整消融 + 效率分析
- 写作质量: ⭐⭐⭐⭐ 动机论证有说服力
- 价值: ⭐⭐⭐⭐⭐ 极致的参数效率和推理速度，非常适合实际部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection](causality-aware_contrastive_learning_for_robust_multivariate_time-series_anomaly.md)
- [\[ICML 2025\] CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering](costfilter-ad_enhancing_anomaly_detection_through_matching_cost_filtering.md)
- [\[ICLR 2026\] PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection](../../ICLR2026/object_detection/paano_patch-based_representation_learning_for_time-series_anomaly_detection.md)
- [\[NeurIPS 2025\] ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection](../../NeurIPS2025/object_detection/scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete.md)
- [\[NeurIPS 2025\] Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection](../../NeurIPS2025/object_detection/structured_temporal_causality_for_interpretable_multivariate_time_series_anomaly.md)

</div>

<!-- RELATED:END -->
