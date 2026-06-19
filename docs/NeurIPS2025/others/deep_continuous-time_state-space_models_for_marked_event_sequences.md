---
title: >-
  [论文解读] Deep Continuous-Time State-Space Models for Marked Event Sequences
description: >-
  [NeurIPS 2025 Spotlight][标记时间点过程] S2P2 将线性 Hawkes 过程与深度状态空间模型结合，通过堆叠多层隐式线性 Hawkes (LLH) 层 + 非线性激活构建高表达力的连续时间 MTPP 模型，利用并行扫描实现线性复杂度和亚线性时间，在 8 个真实数据集上平均提升 33% 预测似然。
tags:
  - "NeurIPS 2025 Spotlight"
  - "标记时间点过程"
  - "状态空间模型"
  - "Hawkes过程"
  - "并行扫描"
  - "连续时间建模"
---

# Deep Continuous-Time State-Space Models for Marked Event Sequences

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2412.19634](https://arxiv.org/abs/2412.19634)  
**代码**: 集成于 [EasyTPP](https://github.com/ant-research/EasyTPP)  
**领域**: others (时序点过程 / 状态空间模型)  
**关键词**: 标记时间点过程, 状态空间模型, Hawkes过程, 并行扫描, 连续时间建模

## 一句话总结
S2P2 将线性 Hawkes 过程与深度状态空间模型结合，通过堆叠多层隐式线性 Hawkes (LLH) 层 + 非线性激活构建高表达力的连续时间 MTPP 模型，利用并行扫描实现线性复杂度和亚线性时间，在 8 个真实数据集上平均提升 33% 预测似然。

## 研究背景与动机

**领域现状**：标记时间点过程（MTPP）建模不规则事件序列，应用于电商、医疗、金融等。基于 RNN 的方法 $O(N)$ 串行；Transformer 方法 $O(N^2)$ 二次缩放。

**现有痛点**：
   - RNN-MTPP：串行计算、长程依赖弱
   - Transformer-MTPP：二次复杂度，长序列（如患者医疗史）不可承受
   - 经典 Hawkes 过程：可解释但表达力有限

**核心矛盾**：如何同时实现高表达力、长程依赖捕获和高效并行计算？

**切入角度**：SSM（状态空间模型）在离散序列建模中已展示高效并行+长程依赖优势，但直接应用于 MTPP 面临事件间断性（跳跃输入）问题。

**核心idea**：将 Hawkes 过程的跳跃微分方程与 SSM 的状态递推统一，构建既保留事件序列归纳偏置又能用并行扫描高效计算的 LLH 层。

## 方法详解

### 从 Hawkes 到 LLH 的统一

经典线性 Hawkes 过程（LHP）的强度微分形式：

$$d\bm{\lambda}_t = -\bm{\beta}(\bm{\lambda}_{t-} - \bm{\nu})dt + \bm{\alpha} d\mathbf{N}_t$$

SSM 的状态方程：

$$d\mathbf{x}(t) = \mathbf{A}\mathbf{x}(t)dt + \mathbf{Bu}(t)dt$$

两者的结构类比：$\bm{\lambda}_t \leftrightarrow \mathbf{x}(t)$，$-\bm{\beta} \leftrightarrow \mathbf{A}$，$\bm{\nu}_t \leftrightarrow \mathbf{Bu}(t)$。但 LHP 限于 $K$ 维（标记数）、SSM 缺少事件脉冲项。

**LLH 层**统一两者：

$$d\mathbf{x}_t = -\mathbf{A}\mathbf{x}_{t-}dt + \mathbf{A}\mathbf{Bu}_{t-}dt + \mathbf{E}\bm{\alpha} d\mathbf{N}_t$$

$$\mathbf{y}_t = \mathbf{C}\mathbf{x}_t + \mathbf{Du}_t$$

其中 $\mathbf{A} \in \mathbb{R}^{P \times P}$ 是通用动态矩阵（比 LHP 的 $\bm{\beta}$ 表达力更强），$\mathbf{E}\bm{\alpha} d\mathbf{N}_t$ 是事件脉冲项，$P$ 可以任意大。

### 对角化与并行计算

1. **对角化**：令 $-\mathbf{A} = \mathbf{V}\bm{\Lambda}\mathbf{V}^{-1}$，在复平面上直接参数化 $\bm{\Lambda}$（约束实部为负确保稳定性），避免矩阵指数计算
2. **ZOH 离散化**：零阶保持假设得到闭式更新：

$$\tilde{\mathbf{x}}_{t'} = \bar{\bm{\Lambda}}\tilde{\mathbf{x}}_t + (\bar{\bm{\Lambda}} - \mathbf{I})\tilde{\mathbf{B}}\mathbf{u}_{t'-} + \tilde{\mathbf{E}}\bm{\alpha}_k$$

其中 $\bar{\bm{\Lambda}} = \exp(\bm{\Lambda}(t'-t))$ 是逐元素指数。

3. **并行扫描**：更新式是线性递推 $\mathbf{z}_{i+1} = \mathbf{R}_i \mathbf{z}_i + \mathbf{b}_i$ 的标准形式——可用并行扫描在 $O(\log N)$ 时间完成。

### 输入依赖动态

受 Mamba 启发，允许动态矩阵依赖输入：

$$\bm{\Lambda}_i = \text{diag}(\text{softplus}(\mathbf{W}'\mathbf{u}_{t_i} + \mathbf{b}'))\bm{\Lambda}$$

这是条件线性的（$\bm{\Lambda}_i$ 仅依赖输入 $\mathbf{u}$，不依赖状态 $\mathbf{x}$），仍然可用并行扫描。

### S2P2 架构

堆叠 $L$ 层 LLH + 逐位置非线性（GELU）+ LayerNorm + 残差连接：

$$\mathbf{u}_t^{(l+1)} = \text{LayerNorm}^{(l)}(\sigma(\mathbf{y}_t^{(l)}) + \mathbf{u}_t^{(l)})$$

最终强度：$\bm{\lambda}_t = \mathbf{s} \odot \text{softplus}((\mathbf{W}\mathbf{u}_{t-}^{(L+1)} + \mathbf{b}) \odot \mathbf{s}^{-1})$

训练目标：最大化对数似然 $\mathcal{L}(\mathcal{H}_T) = \sum_{i=1}^{N_T} \log \lambda_{t_i}^{k_i} - \int_0^T \lambda_s ds$（积分项 MC 估计）。

关键特点：不需要参数化解码头，强度直接由连续演化的隐状态计算。

## 实验关键数据

### 综合排名（8 个数据集，5 随机种子平均）

| 模型 | 似然排名 | 标记预测 | 时间预测 | 标记校准 | 时间校准 | **综合排名** |
|------|---------|---------|---------|---------|---------|------------|
| RMTPP | 6.8 | 6.9 | 5.0 | 5.8 | 6.1 | 6.1 |
| NHP | 2.4 | 1.8 | 2.4 | 4.9 | 3.6 | 2.9 |
| AttNHP | 2.4 | 2.9 | 6.6 | 3.4 | 3.7 | 3.7 |
| IFTPP | 4.1 | 5.0 | 4.0 | 1.8 | 2.6 | 3.6 |
| **S2P2** | **1.9** | **1.9** | **2.3** | **3.0** | **2.8** | **2.1** |

S2P2 以近一个整数排名的优势领先所有基线。

### 总对数似然（nats/event，部分数据集）

| 模型 | Amazon | Taxi | StackOverflow | MIMIC-II | EHRSHOT |
|------|--------|------|---------------|----------|---------|
| NHP | 0.129 | 0.514 | -2.241 | 0.060 | -3.966 |
| AttNHP | 0.484 | 0.493 | -2.194 | -0.170 | OOM |
| IFTPP | 0.496 | 0.453 | -2.233 | 0.317 | -6.596 |
| **S2P2** | **0.781** | **0.522** | **-2.163** | **0.919** | **-2.512** |

S2P2 在 EHRSHOT（大规模医疗数据集，最长序列最多标记）上优势尤为突出——AttNHP 直接 OOM。

### 合成实验验证
- 经典 Hawkes/自纠正过程：几乎完美恢复真实强度
- 非齐次 Poisson 方波强度：NHP 等受限于参数化形式失败，S2P2 完美捕获
- 长程依赖：S2P2 恢复 98% 真实似然 vs NHP 88%

## 亮点与洞察
- **框架优雅**：从 Hawkes → SSM 的统一视角揭示了两类模型的内在联系
- **理论上 SSM 的表达力结果**（Muca Cirone et al., 2024）直接适用于 S2P2
- **效率优势实质性**：$O(\log N)$ 并行时间 vs RNN 的 $O(N)$，Transformer 的 $O(N^2)$
- S2P2 是**真正的连续时间模型**——不像 MHP（Mamba for TPP）那样用离散编码+参数化解码头
- 能处理 EHRSHOT 此类超长序列（其他 Transformer 模型 OOM）

## 局限与展望
- 对角化假设要求系统可对角化，极端条件下可能受限
- MC 估计积分项引入方差
- 未与 intensity-free 方法（normalizing flow TPP）在效率方面对比
- 未测试离散事件生成/模拟任务质量
- ZOH 假设对非常密集的事件窗口可能欠精确

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Hawkes+SSM 的统一框架是全新连接，脉冲跳跃微分方程设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 8 数据集 × 6 指标 × 合成验证 × 完整基线
- 写作质量: ⭐⭐⭐⭐ 推导细致清晰，图示直观
- 价值: ⭐⭐⭐⭐⭐ 提供了 MTPP 领域的新标准工具，效率与性能兼得

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Deep Learning for Continuous-Time Stochastic Control with Jumps](deep_learning_for_continuous-time_stochastic_control_with_jumps.md)
- [\[ICML 2025\] UnHiPPO: Uncertainty-Aware Initialization for State Space Models](../../ICML2025/others/unhippo_uncertainty-aware_initialization_for_state_space_models.md)
- [\[NeurIPS 2025\] Continuous Thought Machines](continuous_thought_machines.md)
- [\[NeurIPS 2025\] Addressing Mark Imbalance in Integration-free Neural Marked Temporal Point Processes](addressing_mark_imbalance_in_integrationfree_neural_marked_t.md)
- [\[NeurIPS 2025\] SPACE: SPike-Aware Consistency Enhancement for Test-Time Adaptation in Spiking Neural Networks](space_spike-aware_consistency_enhancement_for_test-time_adaptation_in_spiking_ne.md)

</div>

<!-- RELATED:END -->
