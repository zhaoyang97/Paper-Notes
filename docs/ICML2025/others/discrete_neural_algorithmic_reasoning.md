---
title: >-
  [论文解读] Discrete Neural Algorithmic Reasoning
description: >-
  [ICML2025][离散化] 提出离散神经算法推理器(DNAR)，通过特征离散化、硬注意力和连续/离散数据流分离三大组件，迫使神经网络沿有限预定义状态执行算法轨迹，在 BFS/DFS/Dijkstra/Prim/MIS 等任务上实现100%完美测试得分：，并可形式化证明所学算法的正确性。 神经算法推理(Neural Alg…
tags:
  - "ICML2025"
  - "离散化"
  - "图神经网络"
  - "算法模拟"
  - "泛化"
  - "可解释性"
  - "硬注意力"
---

# Discrete Neural Algorithmic Reasoning

**会议**: ICML2025  
**arXiv**: [2402.11628](https://arxiv.org/abs/2402.11628)  
**代码**: [yandex-research/dnar](https://github.com/yandex-research/dnar)  
**领域**: 其他  
**关键词**: 离散化, 图神经网络, 算法模拟, 泛化, 可解释性, 硬注意力

## 一句话总结

提出离散神经算法推理器(DNAR)，通过特征离散化、硬注意力和连续/离散数据流分离三大组件，迫使神经网络沿有限预定义状态执行算法轨迹，在 BFS/DFS/Dijkstra/Prim/MIS 等任务上实现**100%完美测试得分**，并可形式化证明所学算法的正确性。

## 研究背景与动机

神经算法推理(Neural Algorithmic Reasoning)旨在用神经网络模拟经典算法的执行过程。CLRS-30 基准将多种经典算法统一编码为图上的操作，训练模型逐步模拟算法的状态转移(hints)。

**核心挑战**：当前基于 GNN 的神经推理器在分布外(OOD)数据上泛化能力差，尤其是在测试图远大于训练图时性能急剧下降。根本原因在于：

1. 经典算法的状态转移是**离散且确定性**的，不受分布偏移影响；而神经网络在连续潜空间中操作，遇到 OOD 输入时容易累积误差
2. 即使在最简单的实数加法场景中，梯度优化也难以精确模拟目标计算(Klindt, 2023)
3. 对于大图，softmax 注意力权重会被稀释(annealing)，导致消息传递退化

**动机**：既然算法的正确性来源于有限离散状态之间的确定性转移，那么能否强制神经推理器也维持离散状态？这样做不仅能提升泛化能力，还能让模型天然可解释，甚至可以形式化证明其正确性。

## 方法详解

### 整体框架：Encode-Process-Decode

模型遵循编码-处理-解码范式。输入图 $G$ 的节点/边特征经线性编码器映射为高维向量，处理器(单层 GNN)循环更新特征：

$$X^{t+1}, E^{t+1} = \text{Processor}(X^t, E^t, A, S^t)$$

$$S^{t+1} = \text{ScalarUpdate}(X^t, E^t, A, S^t)$$

其中 $X^t$ 为节点特征，$E^t$ 为边特征，$S^t$ 为连续标量输入，$A$ 为邻接矩阵。

### 三大核心组件

**1. 特征离散化(Feature Discretization)**

在处理器输出端添加离散瓶颈层，将连续特征映射到有限状态集合：

$$x_i^{t+1} = \text{Discretize}_{\text{nodes}}(\hat{x}_i^{t+1})$$

$$e_{ij}^{t+1} = \text{Discretize}_{\text{edges}}(\hat{e}_{ij}^{t+1})$$

具体实现：将特征投影为 $k$ 维状态 logits，推理时取 argmax 得到 one-hot 离散状态。训练时使用 teacher forcing 或 Gumbel-Softmax(退火温度从 3.0 降至 0.01)。

**2. 硬注意力(Hard Attention)**

标准 softmax 注意力在大图上会被稀释。硬注意力确保每个节点仅关注最重要的邻居：

$$\alpha_{ij} = \frac{\langle Q_j, K_i + K_{ij} \rangle}{\sqrt{h}}$$

取 argmax 而非 softmax，严格限制每个节点可接收的消息集合，保证在任意图规模下行为一致。

**3. 连续/离散数据流分离**

大多数算法涉及连续输入(如边权重)。直接离散化会丢失信息。本文将标量输入 $S$ 与离散状态分开维护：

- **读取接口**：标量仅作为注意力中的边优先级，用于在相同离散状态的邻居间打破平局(tie-breaking)
- **更新接口**：通过离散操作(increment/keep/push)更新标量，避免学习高精度连续运算：

$$s_i^{t+1} = \text{inc}(x_i^t) + \text{keep}(x_i^t) \cdot s_i^t + \sum_{j \in \mathcal{N}(i)} \text{push}(e_{ji}^t) \cdot s_{ji}^t$$

其中 $\text{inc}$、$\text{keep}$、$\text{push}$ 为 0-1 离散函数，由节点/边特征经线性投影+离散化得到。

## 实验关键数据

### 数据集

使用 SALSA-CLRS 基准，覆盖 BFS、DFS、最短路径(Dijkstra)、最小生成树(Prim)、最大独立集(MIS)、偏心率(Eccentricity) 6 个任务。训练集为 ≤16 节点的 ER 随机图，测试集扩展到 16-1600 节点(最多 100 倍放大)。

### 主要结果(SALSA-CLRS)

| 任务 | 图规模 | GIN (节点/图) | PGN (节点/图) | **DNAR (节点/图)** |
|------|--------|---------------|---------------|---------------------|
| BFS | 16 | 98.8 / 92.5 | 100. / 100. | **100. / 100.** |
| BFS | 1600 | 86.5 / 0.0 | 98.5 / 0.0 | **100. / 100.** |
| DFS | 16 | 41.5 / 0.0 | 82.0 / 19.9 | **100. / 100.** |
| DFS | 1600 | 17.8 / 0.0 | 23.1 / 0.0 | **100. / 100.** |
| SP | 1600 | 36.9 / 0.0 | 84.5 / 0.0 | **100. / 100.** |
| Prim | 1600 | 43.2 / 0.0 | 66.8 / 0.0 | **100. / 100.** |
| MIS | 1600 | 79.2 / 0.0 | 98.9 / 5.2 | **100. / 100.** |
| Ecc. | 1600 | NA / 16.0 | NA / 83.0 | **NA / 100.** |

DNAR 在**所有任务、所有测试规模**上均达到 100% 节点级和图级准确率。

### CLRS-30 对比(图规模 64)

| 任务 | Hint-ReLIC | G-ForgetNet | **DNAR** |
|------|-----------|-------------|----------|
| BFS | 99.00 | 99.96 | **100.** |
| DFS | — | — | **100.** |
| Dijkstra | — | — | **100.** |
| Prim | — | — | **100.** |

### 多任务实验

单一处理器 + 任务相关编解码器，同时执行所有 6 个算法，仍然达到**完美泛化**。训练仅需 A100 单卡，单任务 <1h，多任务 5-6h。

## 亮点与洞察

1. **完美泛化 + 可证明正确性**：首次在神经算法推理中实现 100% 测试准确率，且可形式化证明所学算法对任意输入正确。这是该领域的里程碑式成果
2. **设计哲学深刻**：从"算法的泛化来源于离散确定性状态转移"这一洞察出发，系统性地设计了三个互补组件
3. **连续/离散分离巧妙**：将标量仅用于注意力 tie-breaking，避免了离散化导致的信息损失，同时通过 inc/keep/push 离散操作完成标量更新
4. **训练极其高效**：1000 步即可收敛，单卡 A100 不到 1 小时
5. **多任务能力**：单一架构同时完美执行 6 种不同算法，展示了强大的通用性

## 局限与展望

1. **依赖 hint 监督**：模型依赖算法执行轨迹的逐步监督(teacher forcing)，无 hint 设置下的探索仍初步
2. **任务覆盖有限**：目前仅验证了 6 个图算法任务，缺少排序、字符串匹配等非图算法的验证
3. **标量操作集固定**：inc/keep/push 三种操作可能不足以覆盖更复杂的算法(如需乘法、除法的场景)
4. **未涵盖边级推理**：缺少 edge-based reasoning、graph-level hints 等更复杂的数据流交互
5. **实用场景不明**：完美模拟经典算法的实际应用价值尚不清楚——如果已知算法，为何不直接执行？作者未充分讨论 neural reasoner 相对于直接算法执行的优势场景

## 相关工作与启发

- **CLRS-30** (Veličković et al., 2022)：算法推理的标准基准
- **SALSA-CLRS** (Minder et al., 2023)：更严格的大规模 OOD 评估基准
- **Transformer Programs** (Friedman et al., 2023)：可训练→可读程序转换的可解释 Transformer
- **RASP/RASP-L** (Weiss et al., 2021; Zhou et al., 2024)：Transformer 计算模型的形式化语言

**启发**：离散化 + 可证明正确性的思路可能对 LLM 的推理能力提升有参考价值——如果能将推理步骤约束在可验证的离散状态空间中，或许能提高 LLM 在数学/逻辑推理中的可靠性。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 离散化 + 可证明正确性在算法推理领域具有开创性
- 实验充分度: ⭐⭐⭐⭐ — 完美结果令人信服但任务覆盖范围可进一步扩大
- 写作质量: ⭐⭐⭐⭐⭐ — 逻辑清晰，动机→方法→实验链条完整
- 价值: ⭐⭐⭐⭐ — 在算法推理子领域意义重大，但实用性待验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Optimal Sensor Scheduling and Selection for Continuous-Discrete Kalman Filtering with Auxiliary Dynamics](optimal_sensor_scheduling_and_selection_for_continuous-discrete_kalman_filtering.md)
- [\[ACL 2025\] Implicit Reasoning in Transformers is Reasoning through Shortcuts](../../ACL2025/others/implicit_reasoning_in_transformers_is_reasoning_through_shortcuts.md)
- [\[ICML 2026\] MetaDNS: Enhancing Exploration in Discrete Neural Samplers via Well-Tempered Metadynamics](../../ICML2026/others/metadns_enhancing_exploration_in_discrete_neural_samplers_via_well-tempered_meta.md)
- [\[ICLR 2026\] Neural Force Field: Few-shot Learning of Generalized Physical Reasoning](../../ICLR2026/others/neural_force_field_few-shot_learning_of_generalized_physical_reasoning.md)
- [\[ACL 2025\] Unifying Continuous and Discrete Text Diffusion with Non-simultaneous Diffusion Processes](../../ACL2025/others/neodiff_unified_text_diffusion.md)

</div>

<!-- RELATED:END -->
