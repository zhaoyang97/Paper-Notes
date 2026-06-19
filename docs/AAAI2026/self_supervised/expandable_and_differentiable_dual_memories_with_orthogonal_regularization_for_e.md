---
title: >-
  [论文解读] Expandable and Differentiable Dual Memories with Orthogonal Regularization for Exemplar-free Continual Learning
description: >-
  [AAAI 2026][自监督学习][持续学习] 提出 **EDD（Expandable and Differentiable Dual Memory）**，一种**无需存储旧样本**的持续学习方法，通过**可微分的共享记忆和任务特定记忆**将数据分解为可复用的子特征，结合**记忆扩展-剪枝**和**正交正则化**机制，在 CIFAR-10/100 和 Tiny-ImageNet 上超越 14 种 SOTA 方法，最终准确率分别达到 55.13%、37.24% 和 30.11%。
tags:
  - "AAAI 2026"
  - "自监督学习"
  - "持续学习"
  - "灾难性遗忘"
  - "双记忆系统"
  - "正交正则化"
  - "无样本回放"
---

# Expandable and Differentiable Dual Memories with Orthogonal Regularization for Exemplar-free Continual Learning

**会议**: AAAI 2026  
**arXiv**: [2511.09871](https://arxiv.org/abs/2511.09871)  
**代码**: [https://github.com/axtabio/EDD](https://github.com/axtabio/EDD)  
**领域**: 其他  
**关键词**: 持续学习, 灾难性遗忘, 双记忆系统, 正交正则化, 无样本回放

## 一句话总结

提出 **EDD（Expandable and Differentiable Dual Memory）**，一种**无需存储旧样本**的持续学习方法，通过**可微分的共享记忆和任务特定记忆**将数据分解为可复用的子特征，结合**记忆扩展-剪枝**和**正交正则化**机制，在 CIFAR-10/100 和 Tiny-ImageNet 上超越 14 种 SOTA 方法，最终准确率分别达到 55.13%、37.24% 和 30.11%。

## 研究背景与动机

### 问题场景

**持续学习（Continual Learning, CL）** 面临的核心挑战是**灾难性遗忘（Catastrophic Forgetting）**：在学习新任务时忘记旧任务的知识。这在无样本回放（exemplar-free）设置下尤为严峻，因为无法存储旧样本来回顾。

### 现有方法的根本缺陷

现有方法虽能部分缓解遗忘，但存在一个被忽视的根本问题——**它们将未来任务视为与过去完全独立的，忽视了任务间的有用关系**：

**正则化方法**（如 EWC、LwF）：约束参数更新来保护旧知识
   - 缺点：限制了模型可塑性，且无法利用任务间共享知识
   
**架构扩展/参数隔离方法**（如 PNN、DEN）：为每个任务分配专用神经元并冻结
   - 缺点：模型不可控增长，且冻结参数无法被新任务重用

**共同缺陷**：所有方法都是通过"惩罚新信息"或"隔离新旧参数"来保护旧知识，而非主动**复用旧知识来帮助学习新数据**。

### 核心动机：从"防御性保护"到"积极复用"

EDD 的核心理念是**将过去的数据特征分解为小的子特征，存入记忆中，使得新数据可以检索和复用这些已有的子特征**（Figure 1c）。这是一个范式转变：
- 旧方法：过去 vs 未来 → 隔离/惩罚
- EDD：过去 + 未来 → 共享/复用

灵感来自**互补学习系统理论（CLS）**：人脑通过海马体（快速学习新知识）和新皮层（缓慢积累共享知识）的协作来实现持续学习。

## 方法详解

### 整体框架

EDD 在 ResNet-18 的中间层插入两个互补记忆：
- **共享记忆 $M^s$**：编码可跨任务复用的通用表示（对应 CLS 中的新皮层）
- **任务特定记忆 $M^t$**：基于共享知识，捕获每个任务独有的细粒度判别特征（对应海马体）

两个记忆都是**完全可微分的 key-value 记忆**，与编码器和分类器端到端联合优化。

### 关键设计

#### 1. **可微分 Key-Value 记忆**

- **结构**：每个记忆 $M^\ell$ 包含 $L_\ell$ 个可学习的 slot，每个 slot 有 key $\mathbf{k}_j^\ell$ 和 value $\mathbf{v}_j^\ell$
- **读取机制**：给定中间特征 $\mathbf{h}$，通过余弦相似度注意力读取记忆：
$$w_j = \frac{\exp(\langle \mathbf{k}_j^\ell, \mathbf{h} \rangle)}{\sum_{i=1}^{L_\ell} \exp(\langle \mathbf{k}_i^\ell, \mathbf{h} \rangle)}$$
$$\hat{\mathbf{h}} = \sum_{j=1}^{L_\ell} w_j \mathbf{v}_j^\ell$$
- **设计动机**：
    - 完全可微分使得网络可以自主学习每个样本的最优潜在表示
    - key-value 分离允许灵活的知识组织和检索
    - 记忆输出 $\hat{\mathbf{h}}$ 直接替代原始特征继续前向传播

#### 2. **记忆扩展与知识剪枝**

训练每个任务后，记忆执行自组织：

**剪枝（Freezing）**：识别对当前任务最重要的 slot 并冻结：
$$\Delta_j^\ell = |K_j^\ell - (K_j^\ell)^{(t-1)}|_2 + |V_j^\ell - (V_j^\ell)^{(t-1)}|_2$$
变化最大的 slot 被视为最关键的，它们会被冻结以保护已学知识。

**冻结 slot 数量**与类别比例成正比：
$$|\mathcal{F}_t^\ell| \approx \frac{|\mathcal{C}_t|}{|\mathcal{C}_{1:t}|} \cdot \text{unfrozen slots}$$

**扩展**：冻结后添加同等数量的新 slot（随机初始化），确保后续任务有足够的学习容量。

- **设计动机**：
    - 剪枝保证稳定性（旧知识不被覆盖）
    - 扩展保证可塑性（新知识有空间学习）
    - 增长速度与类别多样性匹配，避免不受控膨胀

#### 3. **正交正则化**

**仅应用于任务特定记忆 $M^t$**，强制冻结 slot 和新 slot 在几何上正交：

$$\mathcal{L}_{\text{orth}} = \|K_F^t (K_U^t)^\top\|_F^2 + \|V_F^t (V_U^t)^\top\|_F^2$$

即惩罚冻结 slot 和活跃 slot 之间的任何对齐，鼓励新的任务特定特征占据与旧特征**正交的子空间**。

- **设计动机**：虽然冻结防止了直接参数更新导致的遗忘，但新旧知识的**表示重叠**仍可能造成干扰。正交约束确保子空间分离。
- **仅在任务特定记忆上**：共享记忆不需要正交约束，因为其目标就是跨任务共享。

#### 4. **记忆引导的表示对齐**

使用前一模型的记忆激活模式来引导当前模型：

$$\mathcal{L}_{\text{align}} = \sum_{\ell \in \{s,t\}} \mathbb{E}_{x \sim \mathcal{T}_t} [1 - \cos(A_{\text{new}}^\ell(x), A_{\text{old}}^\ell(x))]$$

其中 $A$ 是记忆的注意力权重向量。这确保当前模型对同一输入检索与前一模型相似的记忆 slot 组合。

- **设计动机**：不存储旧样本，而是通过记忆激活模式的一致性来传递知识——一种"无样本蒸馏"。

### 损失函数 / 训练策略

**总损失**：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}^{(t)} + \lambda_{\text{mem}} \mathcal{L}_{\text{align}} + \lambda_{\text{orth}} \mathcal{L}_{\text{orth}}$$

**训练流程**：
1. 第一个任务：仅用 $\mathcal{L}_{\text{CE}}$ 从头训练
2. 后续任务：
    - 冻结前一模型，将其 BN 层校准到新数据分布（batch adaptation）
    - 复制前一模型为当前模型初始化
    - 用总损失在新任务数据上训练
    - 训练后执行扩展-剪枝
    - 当前模型成为下一任务的前一模型

**超参数**：$\mathcal{L}_\ell = 1000$ slots, $\lambda_{\text{mem}} = 20$, $\lambda_{\text{orth}} = 10$, pruning ratio = 0.15, Adam 优化器, lr=0.001

## 实验关键数据

### 数据集设置
- **S-CIFAR-10**：5 tasks × 2 classes
- **S-CIFAR-100**：10 tasks × 10 classes 和 20 tasks × 5 classes
- **S-Tiny-ImageNet**：10 tasks × 20 classes 和 20 tasks × 10 classes

### 主实验（Table 1, 与 14 种方法对比）

| 方法 | S-CIFAR-10 | S-CIFAR-100 (10T) | S-CIFAR-100 (20T) | S-TinyImgNet (10T) | S-TinyImgNet (20T) |
|------|-----------|-------------------|-------------------|--------------------|--------------------|
| JT (上界) | 83.38 | 70.44 | 70.44 | 59.99 | 59.99 |
| FT (下界) | 18.35 | 4.43 | 2.91 | 5.84 | 2.51 |
| PEC (prev SOTA) | 52.19 | 21.82 | 18.29 | 15.97 | 13.51 |
| DualNet (buffer) | 41.69 | 28.96 | 15.91 | 24.48 | 12.56 |
| LUCIR (buffer) | 42.78 | 30.57 | 19.99 | 25.84 | 14.51 |
| **EDD (ours)** | **55.13** | **37.24** | **21.68** | **30.11** | **18.34** |

**关键对比**：
- vs PEC（前 SOTA）：CIFAR-10 +5.6%, TinyImageNet-20T **+35.8%**
- vs DualNet（带 buffer=500）：EDD 在**无样本**条件下仍全面超越
- **随着任务复杂度增加，EDD 的优势从 5.6% 扩大到 26.4%**

### 消融实验（Table 2）

| 配置 | CIFAR-100 (10T) | TinyImageNet (10T) |
|------|-----------------|--------------------| 
| Naive $\mathcal{L}_{\text{CE}}$ | 32.47 ± 0.62 | 25.38 ± 0.54 |
| + $\mathcal{L}_{\text{align}}$ | 34.82 ± 0.58 (+2.35) | 27.12 ± 0.47 (+1.74) |
| + $\mathcal{L}_{\text{orth}}$ | 33.95 ± 0.71 (+1.48) | 26.55 ± 0.52 (+1.17) |
| + $\mathcal{L}_{\text{align}}$ + $\mathcal{L}_{\text{orth}}$ | 35.67 ± 0.49 (+3.20) | 28.03 ± 0.61 (+2.65) |
| + BA + $\mathcal{L}_{\text{align}}$ + $\mathcal{L}_{\text{orth}}$ | **37.24 ± 0.29** (+4.77) | **30.11 ± 0.22** (+4.73) |

**关键发现**：
- 记忆对齐和正交正则化的效果是互补的（组合 > 单独之和）
- Batch Adaptation 额外贡献 +1.57% / +2.08%，通过缓解分布偏移

### 与联合学习的特征对齐分析

四种度量（余弦相似度↑、KL散度↓、Wasserstein距离↓、特征距离↓）上，**EDD 在所有指标上都最接近联合学习**。

**CIFAR-10 的逐类余弦相似度（Figure 7）**：
- EDD 在所有类和所有任务步骤中保持一致的高相似度
- 其他方法（如 RPC、GSS）到最后一个任务时相似度降到接近 0
- 特别地：第二个任务中学到的第四个类别达到了**最高相似度分数**，即使在所有后续任务学完之后

### 令人惊奇的发现：未来任务提升过去知识

训练 T18 后重新测试 T5 的准确率**反而提升了**（Figure A.2）。这种反直觉现象（后续任务增强了此前任务的性能）源于 EDD 的记忆机制——T18 训练期间冻结的高重要性记忆 slot 被 T5 测试数据重新利用。这有力支持了 EDD 促进跨任务特征共享和复用的核心论点。

### 计算复杂度
- 空间：最多为基础模型的 2 倍（前一模型的完整拷贝）
- 每步计算成本恒定（不随任务数增长）
- CIFAR-100 上运行时间接近最小值，TinyImageNet 上每 epoch < 1 分钟

## 亮点与洞察

1. **范式转变**：从"保护旧知识不被覆盖"转向"复用旧知识来帮助新学习"。这不仅是技术创新，更是对持续学习问题的重新定义。
2. **CLS 理论的优雅实现**：共享记忆 = 新皮层（慢学习通用知识），任务特定记忆 = 海马体（快速学习独特特征），两者端到端可微分。
3. **无样本也能超越有样本**：在严格的 exemplar-free 设置下超越了使用 buffer=500 的方法（DualNet, LUCIR），颠覆了"回放必要性"的直觉。
4. **发现了"前向知识增强"现象**：未来任务的学习可以反向提升旧任务性能，这在传统 CL 中几乎未被观察到。

## 局限与展望

1. **长任务序列扩展性**：超过 50 个任务时，记忆管理的累积开销可能成为瓶颈。
2. **极端任务偏移**：当连续任务几乎没有共享结构时，共享记忆可能难以捕获有用的公共特征。
3. **高维输入/大输出空间**：扩展到极高分辨率图像或大规模分类可能面临记忆占用问题。
4. **仅使用 ResNet-18**：未在 Vision Transformer 等其他骨干上验证。
5. **记忆 slot 数初始值**：$\mathcal{L}_\ell = 1000$ 是固定的，更优的初始化策略值得探索。

## 相关工作与启发

- **DualNet（Pham et al. 2021）**：CLS 启发的快慢网络，但依赖 buffer + 自监督学习，EDD 通过可微记忆消除了这些依赖。
- **CLS-ER（Arani et al. 2022）**：短期/长期语义记忆 + episodic buffer，但通过学习率调整损失间接整合知识。
- **PNN（Rusu et al. 2016）**：每任务一列参数 + 横向连接，但增长不可控。
- **启发**：
    - "分解-存储-复用"是一个强大的范式，将特征分解为原子化子特征后存入记忆，可以实现任意粒度的知识共享。
    - 正交约束在任务特定记忆上的选择性施加（而非全局施加）是一个精细的设计选择。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 可微分双记忆 + 扩展剪枝 + 正交正则的组合是全新的，且有认知科学理论支撑
- 实验充分度: ⭐⭐⭐⭐⭐ — 14 种对比方法、3 个数据集 5 种设置、充分消融、特征对齐分析、计算复杂度分析
- 写作质量: ⭐⭐⭐⭐ — 方法描述清晰系统，但部分符号定义冗长
- 价值: ⭐⭐⭐⭐⭐ — 在严格的 exemplar-free 设置下大幅推进了 SOTA，对 CL 社区有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Exemplar-Free Continual Learning for State Space Models](../../CVPR2026/self_supervised/exemplar-free_continual_learning_for_state_space_models.md)
- [\[ECCV 2024\] Exemplar-Free Continual Representation Learning via Learnable Drift Compensation](../../ECCV2024/self_supervised/exemplar-free_continual_representation_learning_via_learnable_drift_compensation.md)
- [\[CVPR 2026\] Exemplar-Free Class Incremental Learning via Preserving Class-Discriminative Structure](../../CVPR2026/self_supervised/exemplar-free_class_incremental_learning_via_preserving_class-discriminative_str.md)
- [\[CVPR 2026\] Is Parameter Isolation Better for Prompt-Based Continual Learning?](../../CVPR2026/self_supervised/is_parameter_isolation_better_for_prompt-based_continual_learning.md)
- [\[ACL 2025\] AnalyticKWS: Towards Exemplar-Free Analytic Class Incremental Learning for Small-footprint Keyword Spotting](../../ACL2025/self_supervised/analytickws_towards_exemplar-free_analytic_class_incremental_learning_for_small-.md)

</div>

<!-- RELATED:END -->
