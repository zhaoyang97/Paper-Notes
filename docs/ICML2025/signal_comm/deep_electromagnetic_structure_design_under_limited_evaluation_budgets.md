---
title: >-
  [论文解读] Deep Electromagnetic Structure Design Under Limited Evaluation Budgets
description: >-
  [ICML 2025][信号/通信][电磁结构设计] 提出 Progressive Quadtree-based Search (PQS) 方法，通过四叉树层次化表示压缩电磁结构的高维设计空间，并利用基于一致性的样本选择机制在有限仿真预算下高效搜索优质设计，相比生成式方法节省 75~85% 的评估成本。
tags:
  - "ICML 2025"
  - "信号/通信"
  - "电磁结构设计"
  - "四叉树搜索"
  - "代理模型"
  - "有限评估预算"
  - "样本选择"
---

# Deep Electromagnetic Structure Design Under Limited Evaluation Budgets

**会议**: ICML 2025  
**arXiv**: [2506.19384](https://arxiv.org/abs/2506.19384)  
**代码**: 无  
**领域**: 信号通信  
**关键词**: 电磁结构设计, 四叉树搜索, 代理模型, 有限评估预算, 样本选择

## 一句话总结

提出 Progressive Quadtree-based Search (PQS) 方法，通过四叉树层次化表示压缩电磁结构的高维设计空间，并利用基于一致性的样本选择机制在有限仿真预算下高效搜索优质设计，相比生成式方法节省 75~85% 的评估成本。

## 研究背景与动机

电磁结构 (EMS) 设计在天线、频率选择表面、超材料等领域至关重要。然而该问题面临两大核心挑战：

**设计空间巨大**：以 $12 \times 24$ 的网格为例，每个单元有金属/空两种状态，总搜索空间为 $2^{288} \approx 10^{86}$，远超 NAS ($10^4 \sim 10^{18}$) 和分子设计 ($10^6$)。

**评估代价极高**：每次评估需要求解 Maxwell 方程的全波电磁仿真，单次仿真耗时 660 到 42,780 秒不等，无法用简单解析近似替代。

现有方法主要分两类：
- **基于预测器的方法**：训练 DNN 代理模型近似仿真函数，但仍在像素级高维空间中搜索，且高精度代理需要大量训练数据（通常 1 万~200 万样本）。
- **生成式方法**：用 cGAN、cVAE 等直接生成满足约束的设计，但同样需要大规模数据集训练。

与 NAS 和分子设计不同，EMS 设计**缺乏公开数据集、预训练模型和数据增强手段**，且工业场景中每个任务高度定制化，需从零开始。因此急需一种在有限仿真预算（如 1000 次仿真）下仍能找到高质量设计的方法。

## 方法详解

### 整体框架

PQS 框架包含两个互补模块：
1. **Quadtree-based Search Strategy (QSS)**：用四叉树表示 EMS 布局，将像素级搜索转化为层次化的渐进式搜索，从全局模式到局部细节逐步细化。
2. **Consistency-based Sample Selection (CSS)**：基于预测一致性度量动态分配仿真预算，在利用（exploitation）和探索（exploration）之间取得平衡。

整体流程：初始化数据集 $D_0$ → 训练预测器 $f_\theta$ → QSS 搜索候选设计 → CSS 选择仿真样本 → 执行仿真获取反馈 → 更新数据集和预测器 → 迭代直至预算耗尽。

### 关键设计

#### 1. **四叉树表示 (Quadtree-based Representation)**

将传统像素级布局矩阵转换为四叉树结构。核心思路：同质区域用单个叶节点表示（只需 1 bit），复杂区域递归细分为 4 个子节点。

每个节点 $n$ 对应布局矩阵的一个矩形子区域，由行索引 $[r_n^{\text{start}}, r_n^{\text{end}}]$ 和列索引 $[c_n^{\text{start}}, c_n^{\text{end}}]$ 决定。需要细化时，按中点分裂：

$$r_{\text{mid}} = \left\lfloor \frac{r_n^{\text{start}} + r_n^{\text{end}}}{2} \right\rfloor, \quad c_{\text{mid}} = \left\lfloor \frac{c_n^{\text{start}} + c_n^{\text{end}}}{2} \right\rfloor$$

每个叶节点存储一个二值状态 $s_n \in \{0, 1\}$，完整矩阵通过所有叶节点重建：

$$x_{i,j} = \sum_{n \in L} s_n \cdot \mathbb{I}_n(i,j)$$

其中 $L$ 为叶节点集合，$\mathbb{I}_n(i,j)$ 为指示函数。设计空间从原始的 $2^{m \times n}$ 压缩为 $2^{|L|}$，通过控制 $|L|$ 即可渐进管理设计空间的复杂度。

**设计动机**：像素级搜索面临维度灾难，四叉树利用 EMS 布局的空间局部性，同质区域无需逐像素搜索，大幅降低有效搜索维度。

#### 2. **渐进式树搜索 (Progressive Tree Search)**

从最简单的设计空间（根节点）开始，逐步增加复杂度。每次迭代随机选择一个叶节点，以 0.5 概率执行两种操作之一：
- **重采样**：改变叶节点状态 $s_n$，不扩展空间
- **分裂**：将叶节点细分为 4 个子节点，增加搜索粒度

维护 Top-K 最优设计列表，持续更新直到叶节点数达到上限 $N_{\max}$。

#### 3. **深度感知重要性分配 (Depth-wise Importance Assignment)**

树搜索阶段后，对 Top-K 设计进行进一步优化。不同于初始分裂时的均匀分布，此阶段调优四叉树的分区参数（每个节点的行列范围），以更精确地刻画关键区域：

$$\max_{\mathbf{s'} \in \mathcal{S'}} O(f_\theta(\mathbf{x_{s'}}))$$

其中 $\mathcal{S'} = \{(r_n^{\text{start}}, r_n^{\text{end}}, c_n^{\text{start}}, c_n^{\text{end}}) \mid n \in Q\}$。

#### 4. **基于一致性的样本选择 (CSS)**

使用 Kendall's tau 系数 $\tau$ 度量相邻迭代间预测排序的一致性：

$$\tau = \frac{2}{n(n-1)} \sum_{i<j} \text{sign}(O(f_{\theta_{t-1}}(\mathbf{x}_i)) - O(f_{\theta_{t-1}}(\mathbf{x}_j))) \cdot \text{sign}(O(f_{\theta_t}(\mathbf{x}_i)) - O(f_{\theta_t}(\mathbf{x}_j)))$$

$\tau$ 接近 1 表示预测一致性高，接近 0 或负数表示一致性低。

**设计动机**：排序准确性比数值精度更重要——能正确排序候选者的模型即使预测值有偏差也能有效指导优化。

### 损失函数 / 训练策略

**目标函数**采用多准则的最小值，确保所有性能指标均不低于阈值：

$$O(S(\mathbf{x})) = \min_{1 \leq k \leq p} S_k(\mathbf{x})$$

**混合选择策略**：根据 $\tau$ 值动态分配仿真预算：
- $R_p = \tau \times R$ 个样本由预测器推荐（利用）
- $R_r = (1-\tau) \times R$ 个样本随机选取（探索）

当模型预测不可靠时自动增加随机探索比例，防止陷入由不准确预测导致的偏差。

预测器采用 ResNet50 架构，初始数据集 300 个样本，总仿真预算限制为 1000 次。

## 实验关键数据

### 主实验

在两个真实工程任务上评估：双层频率选择表面 (DualFSS, $12 \times 12 \times 2$, 空间 $10^{86}$) 和高增益天线 (HGA, $15 \times 20$, 空间 $10^{90}$)。

| 方法 | DualFSS Agg Obj↑ | #仿真 | HGA Agg Obj↑ | #仿真 |
|------|-----------------|-------|-------------|-------|
| Random Sampling | 7.28 | 1000 | 0.63 | 1000 |
| Surrogate-RS | 5.81 | 1000 | 3.09 | 1000 |
| Surrogate-GA | 4.19 | 1000 | 1.58 | 1000 |
| TS-DDEO | 5.56 | 1000 | 0.52 | 1000 |
| cGAN | 3.13 | 7000 | -1.09 | 4000 |
| cVAE | 8.93 | 7000 | -1.37 | 4000 |
| InvGrad | 2.89 | 7000 | 3.18 | 4000 |
| GenCO | 1.18 | 7000 | -5.30 | 4000 |
| **PQS (Ours)** | **15.20** | **1000** | **3.66** | **1000** |

PQS 仅用 1000 次仿真即超越所有基线。在 DualFSS 上比随机搜索提升 109%，比使用 7000 次仿真的生成式方法 cVAE 提升 70%。

### 消融实验

在 HGA 任务上进行。

**变量数量 $N_{\max}$ 的影响：**

| $N_{\max}$ | Agg Obj↑ | Obj1↑ | Obj2↑ | Kendall's Tau↑ |
|-----------|---------|-------|-------|---------------|
| 16 | 3.09 | 3.09 | 6.03 | 0.284 ± 0.053 |
| 32 | **3.66** | **3.66** | **6.48** | 0.232 ± 0.052 |
| 64 | 3.02 | 3.02 | 5.18 | 0.126 ± 0.030 |

$N_{\max} = 32$ 取得最优。太小容易陷入局部最优，太大导致预测器在有限数据下难以准确建模。

**QSS 和 CSS 模块效果：**

| QSS | CSS | Agg Obj↑ | Obj1↑ | Obj2↑ |
|-----|-----|---------|-------|-------|
| ✗ | ✗ | 3.09 | 3.09 | 3.18 |
| ✓ | ✗ | 3.22 | 3.22 | 6.34 |
| ✓ | ✓ | **3.66** | **3.66** | **6.48** |

两个模块均有积极贡献，CSS 在 QSS 基础上进一步提升 13.7%。

### 关键发现

1. **基于预测器的方法在少样本场景下表现不佳**：Surrogate-RS (5.81 dB) 甚至低于纯随机搜索 (7.28 dB)，说明 1000 个样本不足以训练可靠的预测器来指导搜索。
2. **生成式方法需要大量预算但仍不如 PQS**：使用 7 倍仿真预算的 cGAN/cVAE/InvGrad 等均不及 PQS，凸显层次化表示和一致性驱动机制的优势。
3. **鲁棒性优异**：10 次独立运行中，PQS 的 Agg Obj 为 4.34 ± 0.34，方差最低，而 IDN 为 -17.08 ± 4.23，cVAE 为 -4.04 ± 1.09。
4. **CSS 提升搜索效率 50%**：在寻找数据集中最优样本的实验中，CSS 比 Top-K 和 Random 策略效率提升约 50%。

## 亮点与洞察

1. **四叉树 + 渐进搜索的设计思路非常精巧**：将维度灾难问题转化为层次化的渐进细化问题，从粗粒度探索全局结构再细化局部细节，既符合工程直觉也有效降低了搜索复杂度。
2. **一致性度量用 Kendall's tau 而非数值误差**：认识到排序正确性比预测精度更重要，这一洞察使得即使代理模型不够精确也能有效指导搜索。
3. **评估成本节省显著**：相比生成式方法节省 75~85% 仿真成本，折算为 20~39 天的产品设计周期缩短，工业应用价值突出。
4. **问题建模清晰**：与 NAS、分子设计的类比分析（Table 1）精准定位了 EMS 设计的独特挑战——无公开数据、无预训练模型、无数据增强。

## 局限与展望

1. **二值设计空间限制**：当前方法假设每个单元仅有 0/1 两种状态，未考虑多材料或连续参数化设计。
2. **对称性未充分利用**：许多 EMS 结构具有旋转或镜像对称性，四叉树表示未显式编码这一先验，可结合对称性约束进一步压缩搜索空间。
3. **预测器架构固定为 ResNet50**：未探索更轻量或更适合结构化输入的模型（如 GNN），可能在更小数据量下获得更好的排序能力。
4. **可扩展性验证不足**：仅在两个工程任务上测试，未在更大规模（如 $50 \times 50$ 网格）或更多设计目标的场景下验证。
5. **缺乏与贝叶斯优化方法的比较**：如 BOHB、TPE 等在少样本优化中有成熟应用，未作对比。

## 相关工作与启发

- **与 NAS 的联系**：PNAS (Liu et al., 2018a) 的渐进式搜索思想与本文的渐进四叉树搜索有异曲同工之妙，但 NAS 有 weight sharing 等数据增强手段，EMS 设计缺乏这些基础设施。
- **代理辅助优化**：SAHSO (Li et al., 2022) 和 TS-DDEO (Zheng et al., 2023b) 代表了代理辅助进化算法的最新进展，但在少样本场景下仍然困难。
- **GenCO (Ferber et al., 2024)**：用 VAE 生成 + 梯度上升优化的组合策略，但在 EMS 场景下性能不佳，说明生成模型在缺乏大量训练数据时难以捕捉高维结构规律。
- **启发**：本文的层次化搜索思想可迁移到其他需要在有限预算下优化高维离散结构的场景，如 FPGA 布局、电路设计、3D 打印结构优化等。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 四叉树层次化表示用于 EMS 设计是首创，渐进搜索 + 一致性选择的组合设计新颖。
- **实验充分度**: ⭐⭐⭐⭐ 两个真实工程任务、11 个基线、消融实验和鲁棒性分析均很充分，但缺少更大规模验证。
- **写作质量**: ⭐⭐⭐⭐ 问题定义清晰，与 NAS/分子设计的类比分析有助于理解问题定位，整体结构流畅。
- **实用价值**: ⭐⭐⭐⭐⭐ 大幅减少仿真成本（节省 20~39 天），对工业界 EMS 设计有直接应用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Meta-learning Structure-Preserving Dynamics](../../ICML2026/signal_comm/meta-learning_structure-preserving_dynamics.md)
- [\[ICML 2025\] Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization](fourier_position_embedding_enhancing_attentions_periodic_extension_for_length_ge.md)
- [\[ICML 2025\] Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization](large_language_model_llm-enabled_in-context_learning_for_wireless_network_optimi.md)
- [\[ICCV 2025\] Boosting Multimodal Learning via Disentangled Gradient Learning](../../ICCV2025/signal_comm/boosting_multimodal_learning_via_disentangled_gradient_learning.md)
- [\[CVPR 2025\] Continuous Space-Time Video Resampling with Invertible Motion Steganography](../../CVPR2025/signal_comm/continuous_space-time_video_resampling_with_invertible_motion_steganography.md)

</div>

<!-- RELATED:END -->
