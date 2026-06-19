---
title: >-
  [论文解读] FedQS: Optimizing Gradient and Model Aggregation for Semi-Asynchronous Federated Learning
description: >-
  [NeurIPS 2025][优化/理论][联邦学习] 提出 FedQS，首个同时优化半异步联邦学习（SAFL）中梯度聚合和模型聚合策略的框架，通过将客户端分为四类并自适应调整训练策略，在准确率、收敛速度和稳定性上全面超越基线。 联邦学习中，半异步模式（SAFL）在同步和异步之间取得了平衡，但面临关键挑战： 两种聚合策略的性…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "联邦学习"
  - "半异步"
  - "梯度聚合"
  - "模型聚合"
  - "分治策略"
---

# FedQS: Optimizing Gradient and Model Aggregation for Semi-Asynchronous Federated Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.07664](https://arxiv.org/abs/2510.07664)  
**代码**: [GitHub](https://github.com/bkjod/FedQS_)  
**领域**: 优化  
**关键词**: 联邦学习, 半异步, 梯度聚合, 模型聚合, 分治策略

## 一句话总结

提出 FedQS，首个同时优化半异步联邦学习（SAFL）中梯度聚合和模型聚合策略的框架，通过将客户端分为四类并自适应调整训练策略，在准确率、收敛速度和稳定性上全面超越基线。

## 研究背景与动机

联邦学习中，半异步模式（SAFL）在同步和异步之间取得了平衡，但面临关键挑战：

**两种聚合策略的性能差距**：
   - **梯度聚合**（FedSGD）：收敛更快、准确率更高，但波动严重
   - **模型聚合**（FedAvg）：更稳定，但收敛慢、准确率低
   - 当同时存在过时更新 + 非 IID 数据时，两者差距飙升至 11.52%

**缺乏理论理解**：现有分析仅是经验性的

**服务器端 vs 客户端的局限**：
   - 服务器端方法：与特定聚合策略紧耦合
   - 客户端方法：缺乏全局信息

## 方法详解

### 整体框架

FedQS 包含三个模块：
- **Mod①**（全局聚合估计）：部署在客户端，估计全局梯度方向
- **Mod②**（本地训练自适应）：部署在客户端，根据客户端类型调整训练策略  
- **Mod③**（全局模型聚合）：部署在服务器，自适应加权聚合

### 关键设计

**Mod①：全局聚合估计**
- 客户端存储最近两个全局模型，计算伪全局梯度：$L_g(w_g^t) = w_g^t - w_g^{t-1}$
- 计算本地-全局梯度相似度 $s_i^t$（如余弦相似度）
- 核心创新：从客户端视角获取全局信息，实现与聚合策略的解耦

**Mod②：本地训练自适应（分治策略）**

根据更新速度 $f_i^t$ 和梯度相似度 $s_i^t$ 将客户端分为四类：

| 类型 | 速度 | 相似度 | 策略 |
|------|------|--------|------|
| FBC（快但偏） | $f_i^t > \bar{f}^t$ | $s_i^t < \bar{s}^t$ | 保持学习率，触发反馈机制提高聚合权重 |
| FUC（快且正） | $f_i^t > \bar{f}^t$ | $s_i^t > \bar{s}^t$ | 降低学习率 $\eta_i^t = \eta_i^{t-1} - a\mathcal{F}$，加入动量 |
| SUC（慢但正） | $f_i^t < \bar{f}^t$ | $s_i^t > \bar{s}^t$ | 提高学习率 $\eta_i^t = \eta_i^{t-1} + a\mathcal{F}$，加入动量 |
| SBC（慢且偏） | $f_i^t < \bar{f}^t$ | $s_i^t < \bar{s}^t$ | 提高学习率 + 根据验证集判断是滞后还是分布偏差 |

**动量更新公式**：
$$w_{i,e}^t = w_{i,e-1}^t - \eta_i^t \left[\sum_{r=1}^{e}(m_i^t)^r \nabla F_{i,e-r}(w_{i,e-r-1}^t) + \nabla F_{i,e}(w_{i,e-1}^t)\right]$$

**Mod③：全局模型聚合**
- 对触发反馈机制的客户端调整权重：$p_i = \frac{\exp(\phi - \mathcal{F})}{2\phi - \mathcal{F}} \cdot \frac{(1+\mathcal{G})}{2K}$
- 归一化后进行加权聚合

### 损失函数 / 训练策略

**收敛保证**（定理 4.2 和 4.3）：
- FedQS-SGD 和 FedQS-Avg 都实现指数收敛率
- 收敛界包含三项：$\mathcal{V}^t$（指数收敛项）、$\mathcal{U} = O(\delta^2)$（数据异质性）、$\mathcal{W} = O(G_c^2)$（梯度变化）
- 假设条件：$L$-光滑、梯度有界、异质度有界

## 实验关键数据

### 主实验

**准确率与收敛速度**（三种任务类型）：

| 方法 | CV (x=0.1) | CV (x=0.5) | CV (x=1) | NLP (R=200) | NLP (R=600) | RWD-Gender | RWD-Ethnicity |
|------|-----------|-----------|----------|------------|------------|------------|---------------|
| FedAvg | 56.05 | 73.71 | 77.86 | 47.04 | 45.52 | 77.10 | 77.25 |
| M-step | 62.17 | 80.49 | 82.46 | 49.38 | 48.12 | 78.20 | 78.01 |
| **FedQS-Avg** | **63.91** | **80.26** | **82.74** | **50.43** | **50.08** | **78.94** | **78.85** |
| FedSGD | 65.71 | 83.87 | 85.42 | 48.04 | 49.64 | 77.15 | 78.33 |
| WKAFL | 64.66 | 85.14 | 86.02 | 50.49 | 50.09 | 78.96 | 76.97 |
| **FedQS-SGD** | **68.88** | **86.11** | **86.79** | **52.22** | **52.49** | **78.74** | **79.24** |

**运行时间对比**（秒）：

| 方法 | CV (x=0.1) | NLP (R=200) | RWD-Gender |
|------|-----------|------------|------------|
| FedAvg (同步) | 78,048 | 22,417 | 30,149 |
| FedQS-Avg (SAFL) | 32,827 | 6,023 | 5,701 |
| FedQS-SGD (SAFL) | 32,784 | 5,248 | 5,523 |

相比同步基线，FedQS 平均减少约 70% 运行时间。

### 消融实验

**模块消融**（CV 任务平均）：

| 模块 | 配置 | Avg 准确率 | SGD 准确率 | Avg 收敛 | SGD 收敛 |
|------|------|-----------|-----------|---------|---------|
| Mod① | Cosine | 74.14 | 80.59 | 251 | 230 |
| Mod① | Euclidean | 75.69 | 79.55 | 244 | 232 |
| Mod① | Manhattan | 76.56 | 80.28 | 228 | 221 |
| Mod② | w/o momentum | 73.21 | 78.88 | 269 | 242 |
| Mod② | with momentum | 74.14 | 80.59 | 251 | 230 |
| Mod③ | w/o feedback | 68.35 | 78.83 | 284 | 268 |
| Mod③ | with feedback | 74.14 | 80.59 | 251 | 230 |

**关键消融发现**：
- 去除动量：平均准确率下降 4.3%，收敛需多 6% epoch
- 去除反馈机制：FedQS-Avg 准确率暴跌 7.81%
- 相似度函数选择对性能影响较小（余弦/欧几里得/曼哈顿差异不大）

**系统设置鲁棒性**：

| 场景 | FedAvg | FedQS-Avg | FedSGD | FedQS-SGD |
|------|--------|-----------|--------|-----------|
| N=50, 1:20 | 70.1 | 79.2 | 77.4 | **80.7** |
| N=200, 1:100 | 49.4 | 64.7 | 74.4 | **80.1** |

在极端异构场景（200 客户端，速度比 1:100）下，FedQS-SGD 仍达到 80.1%。

### 关键发现

1. **梯度与模型聚合差距的根源**：过时更新在梯度聚合中仅影响方向/幅度，在模型聚合中则重置优化轨迹；非 IID 数据进一步放大这一差距
2. **分治策略的有效性**：四类客户端分类覆盖了所有异质性组合，各有针对性的优化策略
3. **动量和反馈机制互补**：动量加速局部收敛，反馈机制改善全局聚合
4. **超参数敏感性**：$a$（学习率变化率）影响最大，$k$（动量变化速度）影响最小

## 亮点与洞察

1. **首次统一框架**：同时优化梯度和模型两种聚合策略，而非仅关注其中之一
2. **客户端自适应**：不需要服务器预先了解客户端特性，客户端可动态调整策略应对变化的资源
3. **理论保证**：提供了两种聚合策略的指数收敛证明
4. **极低额外开销**：客户端仅需额外一次相似度计算和两次比较，通信仅增加 1-bit 信号和几个浮点数
5. **实验覆盖广**：CV（CIFAR-10）、NLP（Shakespeare）、真实世界（UCI Adult）三种任务

## 局限与展望

1. 模型聚合模式下引入了少量振荡（oscillation）
2. 引入了三个新超参数（$a, m_0, k$），增加了实现和复现难度
3. 实验限于中等规模模型（ResNet-18, LSTM, FCN），未验证大模型场景
4. 未来可探索超参数自动调整机制

## 相关工作与启发

- WKAFL 利用余弦相似度加权聚合过时梯度，启发了 FedQS 的 Mod①
- FedAT 使用分层异步框架，但需要预知客户端性能分布
- SCAFFOLD 的梯度校正思想在 FedAC 中有应用，FedQS 从正交角度（分治）解决问题
- 动量机制在联邦优化中的作用得到进一步验证

## 评分

- 新颖性：⭐⭐⭐⭐ — 首次统一优化两种聚合策略
- 实验完整度：⭐⭐⭐⭐⭐ — 三种任务 + 8 基线 + 消融 + 超参分析 + 系统设置分析
- 实用性：⭐⭐⭐⭐ — 低额外开销，可扩展性好
- 写作质量：⭐⭐⭐⭐ — 结构清晰，图表丰富

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Layer-wise Update Aggregation with Recycling for Communication-Efficient Federated Learning](layer-wise_update_aggregation_with_recycling_for_communication-efficient_federat.md)
- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](streaming_federated_learning_with_markovian_data.md)
- [\[CVPR 2025\] Model Poisoning Attacks to Federated Learning via Multi-Round Consistency](../../CVPR2025/optimization/model_poisoning_attacks_to_federated_learning_via_multi-round_consistency.md)
- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[NeurIPS 2025\] Personalized Subgraph Federated Learning with Differentiable Auxiliary Projections](personalized_subgraph_federated_learning_with_differentiable_auxiliary_projectio.md)

</div>

<!-- RELATED:END -->
