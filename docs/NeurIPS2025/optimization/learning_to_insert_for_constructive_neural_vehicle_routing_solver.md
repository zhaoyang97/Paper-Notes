---
title: >-
  [论文解读] Learning to Insert for Constructive Neural Vehicle Routing Solver
description: >-
  [NeurIPS 2025][优化/理论][车辆路径问题] 提出 L2C-Insert，首个基于学习的插入式构造范式用于神经组合优化，通过允许在部分解的任意合法位置插入节点（而非仅追加到末尾），显著提升 TSP/CVRP 的构造质量和灵活性。 领域现状： 神经组合优化（NCO）方法通过神经网络自动学习启发式策略求解 VRP…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "车辆路径问题"
  - "神经组合优化"
  - "插入式构造"
  - "TSP"
  - "CVRP"
---

# Learning to Insert for Constructive Neural Vehicle Routing Solver

**会议**: NeurIPS 2025  
**arXiv**: [2505.13904](https://arxiv.org/abs/2505.13904)  
**代码**: [GitHub](https://github.com/CIAM-Group/L2C_Insert)  
**领域**: 优化  
**关键词**: 车辆路径问题, 神经组合优化, 插入式构造, TSP, CVRP

## 一句话总结

提出 L2C-Insert，首个基于学习的插入式构造范式用于神经组合优化，通过允许在部分解的任意合法位置插入节点（而非仅追加到末尾），显著提升 TSP/CVRP 的构造质量和灵活性。

## 研究背景与动机

**领域现状**: 神经组合优化（NCO）方法通过神经网络自动学习启发式策略求解 VRP，其中构造式方法最受关注。现有方法几乎全部采用追加式（appending）范式——依序将未访问节点追加到当前解的末尾。

**现有痛点**: 追加式范式有严格的操作约束——新节点只能添加到解的末端，这阻止了更有效的修改，早期贪心决策常导致路径交叉等次优结果。

**核心矛盾**: 追加式范式在 NLP 的 seq2seq 框架中效果很好，但 VRP 的解空间结构与语言序列截然不同——解的质量高度依赖节点的全局位置关系而非局部顺序。

**本文目标**: 探索插入式范式在 NCO 中的潜力，设计学习驱动的插入策略、训练方案和推理技术。

**切入角度**: 从运筹学中成熟的插入式启发策略出发，将其与神经网络学习结合，克服传统手工策略的局限。

**核心idea**: 将构造过程从"选择节点追加到哪"改为"选择节点插入到部分解的哪个位置"，用注意力机制学习最优插入位置。

## 方法详解

### 整体框架

L2C-Insert 采用 Encoder-Decoder 架构：Encoder 将节点特征编码为嵌入向量，Decoder 在每步维护三类嵌入（当前节点、未访问节点、部分解中的插入位置），通过多层注意力计算插入概率。

### 关键设计

1. **插入位置嵌入 (Position Embedding)**: 将部分解中相邻已访问节点的嵌入水平拼接形成插入位置表示。→ 位置 $(π_i, π_{i+1})$ 的嵌入为 $\mathbf{e}_{\pi_i} = W_2[\mathbf{h}_{\pi_i}, \mathbf{h}_{\pi_{i+1}}]$。→ 将 $2d$ 维拼接投影到 $d$ 维保持一致性。→ 与节点选择嵌入不同，这里编码的是"间隙"的特征。

2. **多注意力层 Decoder**: 将当前节点嵌入、未访问节点嵌入和位置嵌入三者一起送入 $L$ 层堆叠注意力层。→ 每层做自注意力和前馈网络。→ 最终通过线性投影 + masked softmax 计算各位置的插入概率：$p_i = \text{Softmax}(X_i^{(L)} W_f + b_f)$。→ 掩码将非法位置（当前节点、未访问节点对应位置）设为 $-\infty$。

3. **监督学习训练方案**: 给定实例及其标注最优解 $\pi^*$，随机选择初始节点，按标注解的顺序重新插入未访问节点到目标位置。→ 目标位置由标注解中相邻的两个已访问节点定义。→ 损失函数：$\mathcal{L}(\theta) = -\log p_\theta(\vec{e}^* \mid \vec{e}_{\pi_{1:l}}, \pi^*)$。→ 为何不用 RL：重型 decoder 下 SL 比 RL 更稳定有效。

4. **插入式局部重构 (Insertion-based Local Reconstruction)**: 推理阶段的迭代优化技术。→ **破坏阶段**：采用基于距离的邻近破坏（而非传统的基于序列位置的破坏），随机选一个节点并移除其 $\alpha$ 个最近邻。→ **重构阶段**：模型将被移除的节点重新插入部分解。→ 关键优势：基于距离的破坏策略能更有效逃离局部最优——两个几何距离相近但序列距离远的节点在序列式破坏下难以同时考虑，但在距离式破坏下可以。

### 损失函数 / 训练策略

- 使用 Adam/AdamW 优化器
- 初始学习率 $1 \times 10^{-4}$，每 epoch 衰减 0.97
- 在 100 万个 100 节点实例上训练 50 epochs（TSP）/ 15 epochs（CVRP）
- 推理时使用贪心 rollout + 局部重构迭代优化

## 实验关键数据

### 主实验 (TSP)

| 方法 | TSP100 Gap | TSP1K Gap | TSP10K Gap |
|------|:-:|:-:|:-:|
| LKH3 | 0.000% | 0.000% | 0.000% |
| POMO aug×8 | 0.138% | 40.577% | OOM |
| LEHD RRC1000 | 0.0016% | 0.731% | 12.709% |
| BQ bs16 | 0.010% | 1.354% | OOM |
| **L2C-Insert (II=200)** | **0.0014%** | **0.952%** | **4.117%** |
| **L2C-Insert (II=1000)** | **0.0002%** | **0.481%** | **2.079%** |

### 主实验 (CVRP)

| 方法 | CVRP100 Gap | CVRP1K Gap | CVRP10K Gap |
|------|:-:|:-:|:-:|
| HGS | 0.000% | 0.000% | 0.000% |
| LEHD RRC1000 | 0.420% | 3.028% | 9.916% |
| INViT-3V | 6.282% | 13.230% | 21.892% |
| **L2C-Insert (II=1000)** | **0.413%** | **4.836%** | **8.646%** |

### 消融实验：Append vs Insert

| 方法 | TSP100 Gap | TSP1K Gap | TSP10K Gap |
|------|:-:|:-:|:-:|
| L2C-Append (II=1000) | 0.00144% | 0.974% | 12.881% |
| **L2C-Insert (II=1000)** | **0.00017%** | **0.481%** | **2.079%** |
| 差距缩减 | 88.2% | 50.6% | 83.9% |

### 消融实验：破坏策略

| 破坏策略 | TSP1K Gap | TSP10K Gap |
|------|:-:|:-:|
| 基于序列 (Sequence-based) | 1.182% | 6.980% |
| **基于距离 (Distance-based)** | **0.481%** | **2.079%** |

### 关键发现

- **插入式显著优于追加式**：在完全相同架构下，L2C-Insert 比 L2C-Append 的 gap 减少 50%-88%。
- **强大的规模扩展能力**：在 100 节点上训练的模型可直接泛化到 100K 节点。
- **基于距离的破坏策略是关键**：比基于序列的策略在 TSP10K 上提升 70%。
- **在 TSPLIB/CVRPLIB 真实实例上同样最优**：L2C-Insert 在所有实例组上超越所有神经求解器。
- **高效推理**：100K 节点 TSP 仅需 26 分钟（vs LKH3 需 400 小时）。

## 亮点与洞察

- 首次将插入式范式引入 NCO，填补了过去十年的研究空白。
- 插入位置嵌入的设计巧妙——通过相邻节点拼接自然编码了"间隙"的特征。
- 基于距离的局部重构策略与插入范式天然适配，是追加式方法无法充分利用的。
- 监督学习训练方案中"目标位置由标注解中相邻已访问节点定义"的思路简洁有效。

## 局限与展望

- 节点选择策略使用简单的最近邻启发式，未学习——可以训练联合的节点选择+位置选择策略。
- 仅验证了 TSP 和 CVRP 两种问题，未扩展到更多组合优化问题类型。
- 混合追加-插入求解器是有前景的方向但尚未实现。
- 在 CVRP 大规模实例上与 HGS 的差距仍较大（CVRP100K ~8%）。
- 训练数据需要最优解标注，限制了在更大规模上的直接训练。

## 相关工作与启发

- Pointer Networks (Vinyals et al.) 开创了 seq2seq 框架在 NCO 中的应用，但建立的追加范式成为后续研究的思维定式。
- LEHD 和 BQ 是近期强大的追加式基准，L2C-Insert 在保持类似架构的前提下通过范式转变大幅超越。
- 运筹学中的 cheapest insertion 等经典启发策略是本工作的灵感来源。
- 与 NeurOpt 等改进式方法不同，L2C-Insert 属于构造式方法，可与改进式方法结合。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首创插入式构造范式在 NCO 中的应用，思路清晰且有力
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 100-100K 节点、合成+真实数据集、充分消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观，问题定义准确
- 价值: ⭐⭐⭐⭐⭐ 为 NCO 领域开辟了新范式，实验结果全面领先

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees](rethinking_neural_combinatorial_optimization_for_vehicle_routing_problems_with_d.md)
- [\[NeurIPS 2025\] Neural Thermodynamics: Entropic Forces in Deep and Universal Representation Learning](neural_thermodynamics_entropic_forces_in_deep_and_universal_representation_learn.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)

</div>

<!-- RELATED:END -->
