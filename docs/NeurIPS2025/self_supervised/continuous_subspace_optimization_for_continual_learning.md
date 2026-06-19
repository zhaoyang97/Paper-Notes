---
title: >-
  [论文解读] Continuous Subspace Optimization for Continual Learning (CoSO)
description: >-
  [NeurIPS 2025][自监督学习][continual learning] 提出 CoSO 框架，通过从每步梯度的 SVD 动态导出连续子空间（而非 LoRA 的固定子空间），结合历史任务正交投影防止干扰和 Frequent Directions 高效聚合梯度信息，在 ImageNet-R 20 任务上以 78.19% 最终准确率超越最佳 baseline 2.77 个百分点。
tags:
  - "NeurIPS 2025"
  - "自监督学习"
  - "continual learning"
  - "catastrophic forgetting"
  - "Dynamic Subspace"
  - "Orthogonal Projection"
  - "Frequent Directions"
  - "LoRA"
  - "GaLore"
---

# Continuous Subspace Optimization for Continual Learning (CoSO)

**会议**: NeurIPS 2025  
**arXiv**: [2505.11816](https://arxiv.org/abs/2505.11816)  
**作者**: Quan Cheng, Yuanyu Wan, Lingyu Wu, Chenping Hou, Lijun Zhang（南京大学、浙江大学、国防科技大学）  
**领域**: 持续学习 / 参数高效微调  
**关键词**: continual learning, catastrophic forgetting, Dynamic Subspace, Orthogonal Projection, Frequent Directions, LoRA, GaLore

## 一句话总结

提出 CoSO 框架，通过从每步梯度的 SVD 动态导出连续子空间（而非 LoRA 的固定子空间），结合历史任务正交投影防止干扰和 Frequent Directions 高效聚合梯度信息，在 ImageNet-R 20 任务上以 78.19% 最终准确率超越最佳 baseline 2.77 个百分点。

## 研究背景与动机

**领域现状**：持续学习的核心挑战是灾难遗忘——模型在学习新任务时对旧任务的性能急剧下降。近年来基于预训练 ViT 的 PEFT 方法（如 LoRA）成为主流方案，将参数更新限制在固定低秩子空间中以减轻任务间干扰。

**现有痛点**：(a) LoRA 的固定秩约束导致性能不如全秩微调，学习容量受限；(b) InfLoRA 在预设子空间内减轻干扰，SD-LoRA 解耦幅度与方向，但二者都将权重更新限定在单一低秩子空间内；(c) 长任务序列下性能退化明显，固定子空间的表达能力不足以适应持续变化的梯度结构。

**核心矛盾**：如何在保持高学习容量（灵活性）的同时有效缓解灾难遗忘（稳定性）？固定子空间的"安全"和动态子空间的"灵活"之间存在根本张力。

**本文切入角度**：受 GaLore（梯度低秩投影用于离线学习）启发，从梯度本身的 SVD 动态导出子空间而非预先固定，同时通过维护历史任务子空间的正交基来隔离不同任务的更新方向。

**核心 idea**：每 K 步从当前梯度 SVD 导出投影矩阵进行低秩优化，强制投影到历史任务子空间的正交补上，用 Frequent Directions 高效维护历史信息。

## 方法详解

### 整体框架

对每个新任务 $\tau$，在每个训练步执行以下流程：

1. 计算当前梯度 $G_{\tau,t}$
2. 正交投影：$G'_{\tau,t} = G_{\tau,t} - \mathcal{M}_{\tau-1}\mathcal{M}_{\tau-1}^T G_{\tau,t}$（去除与历史子空间对齐的分量）
3. 截断 SVD：$P_{\tau,t} = U[:, :r_1]$（得到当前低秩投影矩阵）
4. 前向投影 → Adam 优化 → 反向投影更新参数
5. 同步用 Frequent Directions 增量聚合梯度信息到 sketch 矩阵 $S_{\tau,t}$
6. 任务结束后，对 $S_{\tau,T}$ 做 SVD 提取主方向，附加到历史正交基 $\mathcal{M}_\tau$

### 关键设计

**设计一：连续子空间优化**

- 功能：动态导出低秩投影矩阵取代 LoRA 的固定矩阵
- 核心思路：每 K 步对当前正交化梯度做截断 SVD 得到 rank-$r_1$ 投影矩阵 $P_{\tau,t}$，在此子空间内用 Adam 优化。与 LoRA 不同，子空间随梯度演化而连续变化，使得最终学到的权重可以是全秩的
- 具体流程：$R_{\tau,t} = P_{\tau,t}^T G'_{\tau,t}$（前向投影）→ $N_{\tau,t} = \text{Adam}(R_{\tau,t})$（低维优化）→ $\tilde{G}_{\tau,t} = P_{\tau,t} N_{\tau,t}$（反向投影）→ $W_{\tau,t} = W_{\tau,t-1} - \eta \tilde{G}_{\tau,t}$
- 设计动机：固定子空间无法适应训练过程中梯度方向的变化；通过在多个连续子空间中优化，突破低秩约束的学习容量上界
- 内存优势：相比 LoRA 类方法，内存需求从 $mn + 3mr_1 + 3nr_1$ 降至 $mn + mr_1 + 2nr_1$

**设计二：历史任务正交投影**

- 功能：确保新任务的参数更新不干扰旧任务
- 核心思路：维护正交基矩阵 $\mathcal{M}_{\tau-1}$，融汇所有历史任务的梯度子空间。每步将当前梯度投影到正交补：$G'_{\tau,t} = G_{\tau,t} - \mathcal{M}_{\tau-1}\mathcal{M}_{\tau-1}^T G_{\tau,t}$
- 原理：由于 $P_{\tau,t}$ 从 $G'_{\tau,t}$ 导出，所有参数更新都在历史子空间的零空间中进行，对先前任务的线性层输出不产生影响
- 设计动机：提供对遗忘的原则性保护。消融实验显示移除正交投影导致 20 任务上最终准确率下降 8.52 个百分点

**设计三：Frequent Directions 梯度聚合**

- 功能：高效维护任务特定的梯度协方差信息
- 核心思路：用 FD 算法以 $O(mnr_2T)$ 复杂度（而非直接计算协方差矩阵的 $O(m^2nT)$）增量聚合所有训练步的梯度信息，生成 sketch 矩阵 $S_{\tau,T}$
- 具体流程：先对梯度做 rank-$r_2$ 截断 SVD 得到 $Q_{\tau,t}$，再用 FD 增量更新 $S_{\tau,t} = \text{FD}([S_{\tau,t-1}, Q_{\tau,t}])$
- 任务结束时：对 $S_{\tau,T}$ 做 SVD，按 $\sum_{i=1}^k \sigma_i^2 / \sum_{j=1}^{r_2} \sigma_j^2 \leq \epsilon_{th}$ 选取 $k$ 个主方向，附加到 $\mathcal{M}_\tau = [\mathcal{M}_{\tau-1}, U_\tau[:, :k]]$
- 理论保证：Proposition 1 给出了近似误差上界，当 $r_2$ 超过梯度的内在秩时误差可忽略

### 训练策略

- **损失函数**：标准交叉熵，温度参数设为 3 以防止过拟合
- **Backbone**：ViT-B/16（ImageNet-21K 预训练 + ImageNet-1K 微调），也测试了 DINO 自监督预训练的 ViT-B/16
- **优化范围**：仅优化 Multi-Head Attention 的 output projection 层（而非 QKV 变换）
- **优化器**：Adam（$\beta_1=0.9, \beta_2=0.999$）
- **关键超参数**：$r_1$（投影秩）、$r_2$（FD 秩，设置 > $r_1$）、$K$（SVD 更新间隔）、$\epsilon_{th}$（信息保留阈值，统一 0.98）

| 超参数 | CIFAR100 | ImageNet-R | DomainNet |
|:---:|:---:|:---:|:---:|
| $r_1$ | 15 | 50 | 70 |
| $r_2$ | 100 | 120 | 160 |
| $K$ | 1 | 1 | 20 |
| 训练轮数 | 20 | 40 | 5 |

## 实验结果

### 主要对比（ImageNet-R）

在 ImageNet-R 上与 6 个 SOTA 方法对比（L2P、DualPrompt、CODA-Prompt、InfLoRA、VPT-NSP², SD-LoRA），3 次独立运行取均值和标准差：

| 设置 | CoSO 最终准确率 | 最佳 Baseline | 提升 |
|:---:|:---:|:---:|:---:|
| 5 Tasks | — | — | +2.38% |
| 10 Tasks | — | — | +3.23% |
| 20 Tasks | **78.19%** | 75.42% (SD-LoRA) | **+2.77%** |

- 20 任务的平均准确率：CoSO 83.69% vs 最佳 baseline 81.32%（+2.37%）
- 任务数越多，CoSO 的优势越显著，证明其在长序列挑战性场景下的鲁棒性
- 学习过程曲线显示 CoSO 在中间阶段和训练尾声均保持最优，准确率下降速度显著慢于竞争方法

### CIFAR100 和 DomainNet

- DomainNet（5 Tasks）：CoSO 最终准确率超越最佳 baseline 1.75%，平均准确率超 1.37%
- CIFAR100（10 Tasks）：同样取得最优表现

### 消融实验（ImageNet-R）

| 变体 | 5 Tasks 降幅 | 10 Tasks 降幅 | 20 Tasks 降幅 |
|:---:|:---:|:---:|:---:|
| w/o Orth（去除正交投影） | — | — | **-8.52%** |
| w/o FD（仅用最终子空间替代 FD 聚合） | -1.65% | -1.89% | -1.59% |

- 正交投影是核心贡献，去除后性能急剧下降，说明任务干扰是灾难遗忘的主因
- FD 聚合也不可或缺，相比仅使用任务结束时的单一子空间，全过程梯度聚合能捕获更丰富的任务信息

### 计算与内存开销（ImageNet-R 10 Tasks）

| 方法 | GFLOPs | 内存 (GB) |
|:---:|:---:|:---:|
| L2P / DualPrompt / CODA-P | 70.24 | 12.90-12.97 |
| InfLoRA | 35.12 | 13.44 |
| SD-LoRA | 35.12 | 15.62 |
| **CoSO** | **35.12** | **13.61** |

- 计算量仅为 Prompt 类方法的一半（无需两次前向传播）
- 内存与 InfLoRA 相当，远低于 SD-LoRA

### DINO 自监督 Backbone

在 DINO 预训练的 ViT-B/16 上（ImageNet-R 10 Tasks），CoSO 同样以显著优势超越所有 baseline，验证了方法的通用性。

## 创新点与贡献

1. **连续子空间优化**：从根本上突破 LoRA 固定子空间的学习容量限制，通过动态梯度 SVD 实现在多个低秩子空间中的连续优化，等效支持全秩权重更新
2. **正交约束的原则性遗忘防护**：通过维护历史任务子空间的正交基并将新任务梯度投影到其正交补，提供对任务干扰的数学保证
3. **Frequent Directions 高效聚合**：用流式矩阵 sketch 算法以线性复杂度聚合整个任务训练过程的梯度信息，附带理论误差上界

## 不足与局限

1. **适用范围受限**：目前仅验证了视觉分类任务，未在多模态、生成、NLP 等场景测试。作者明确指出扩展到多模态是开放问题
2. **SVD 计算开销**：虽然每 K 步才做一次 SVD，但在高维参数矩阵上 SVD 本身的计算量仍不可忽视
3. **正交基持续增长**：随任务数增加，$\mathcal{M}_\tau$ 的列数持续增长，可能逐步挤压新任务的可用优化空间
4. **超参数敏感性**：$r_1, r_2, K$ 需要按数据集调整，虽然 $\epsilon_{th}$ 统一为 0.98，但不同数据集的最优投影秩差异很大（15 vs 70）

## 个人思考

1. **GaLore 到持续学习的迁移非常自然**：GaLore 的核心思想是"梯度内在低秩可以用于内存高效优化"，而持续学习恰好需要在子空间层面隔离不同任务。CoSO 将这两个需求优雅地统一——正交投影解决遗忘，动态 SVD 解决容量限制
2. **正交约束的可扩展性值得关注**：当任务数非常多时（如 100+），正交补空间会被逐步压缩，新任务可用的"垂直方向"越来越少。这可能需要引入子空间遗忘或压缩机制
3. **仅优化 output projection 的设计很克制**：不调整 QKV 变换，仅调 attention 输出投影层，参数量极小但效果已经很好，暗示预训练 ViT 的中间表示具有很强的通用性
4. **与 OGD（Orthogonal Gradient Descent）的联系**：CoSO 的正交投影思路与 OGD 一脉相承，关键改进在于用 FD 聚合整个训练过程（而非某个检查点）的梯度信息来估计任务子空间
5. **应用前景**：这种"连续子空间 + 正交约束"的范式可能推广到 LLM 的持续指令微调、多任务适配器等场景

## 实验关键数据

### 主实验

| 数据集 | 任务数 | CoSO Final Acc | 最佳Baseline | 提升 |
|--------|------|---------------|------------|------|
| ImageNet-R | 5 | **82.10%** | VPT-NSP² 79.72% | +2.38 |
| ImageNet-R | 10 | **81.10%** | 77.87% | +3.23 |
| ImageNet-R | 20 | **78.19%** | 75.42% | +2.77 |
| CIFAR100 | 10 | **88.77%** | 88.09% | +0.68 |
| DomainNet | 5 | **74.27%** | 72.52% | +1.75 |

### 消融实验（ImageNet-R 20任务）

| 配置 | Final Acc | Avg Acc |
|------|----------|---------|
| CoSO完整 | **78.27%** | **83.62%** |
| w/o 正交投影 | 69.75% (-8.52) | 78.88% |
| w/o Frequent Directions | 76.68% (-1.59) | 82.41% |

### 关键发现
- **正交投影是核心**：去掉后下降8.52个百分点，任务干扰被完全释放
- **任务越多提升越大**：5任务+2.38% → 20任务+2.77%，说明方法在长序列上优势更明显
- **Frequent Directions贡献稳定的1.5-2%**：聚合所有中间梯度比仅用最终步梯度更好
- **计算开销与InfLoRA相当**：GFLOPs相同，内存略高0.17G

## 亮点与洞察
- **动态子空间+正交约束的有机结合**：动态保证学习容量（表达力），正交保证抗遗忘（安全性），两者不矛盾而是互补
- **Frequent Directions的巧妙应用**：将流式算法从数据分析引入持续学习，解决了"如何高效总结一个任务的全部梯度信息"的关键问题
- **理论清晰的任务隔离机制**：正交投影有明确的几何意义——在参数空间中为每个任务"预留"互不干扰的子空间

## 局限与展望
- 仅在类增量学习上评估，多模态/域增量等更复杂场景未覆盖
- 需要调4个超参数（$r_1, r_2, K, \epsilon_{th}$），不同数据集需不同配置
- 简单数据集上改进有限（CIFAR100仅+0.68%），收益-成本比随任务复杂度变化

## 相关工作与启发
- **vs LoRA/InfLoRA/SD-LoRA**：固定子空间→动态子空间，学习容量质的提升
- **vs OGD（梯度正交投影）**：CoSO在低秩子空间内操作正交投影，更高效
- **启发**：正交性+动态子空间的思路可推广到LLM的持续学习

## 评分
- 新颖性: ⭐⭐⭐⭐ 动态子空间+正交投影+FD的组合新颖，几何直觉清晰
- 实验充分度: ⭐⭐⭐⭐ 多数据集+多任务数+详细消融
- 写作质量: ⭐⭐⭐⭐ 方法推导清晰，算法伪代码完整
- 价值: ⭐⭐⭐⭐ 对持续学习社区有重要贡献，尤其是长任务序列场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Subspace Alignment for CLIP-based Continual Learning via Canonical Correlation Analysis](../../CVPR2026/self_supervised/subspace_alignment_for_clip-based_continual_learning_via_canonical_correlation_a.md)
- [\[NeurIPS 2025\] M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](m-grpo_stabilizing_self-supervised_reinforcement_learning_for_multimodal_underst.md)
- [\[ECCV 2024\] Revisiting Supervision for Continual Representation Learning](../../ECCV2024/self_supervised/revisiting_supervision_for_continual_representation_learning.md)
- [\[ECCV 2024\] STSP: Spatial-Temporal Subspace Projection for Video Class-Incremental Learning](../../ECCV2024/self_supervised/stsp_spatial-temporal_subspace_projection_for_video_class-incremental_learning.md)
- [\[CVPR 2026\] Beyond Myopic Alignment: Lookahead Optimization for Online Class-Incremental Learning](../../CVPR2026/self_supervised/beyond_myopic_alignment_lookahead_optimization_for_online_class-incremental_lear.md)

</div>

<!-- RELATED:END -->
