---
title: >-
  [论文解读] Magnitude-Modulated Equivariant Adapter for Parameter-Efficient Fine-Tuning of Equivariant Graph Neural Networks
description: >-
  [AAAI 2026][图学习][等变图神经网络] 提出 MMEA (Magnitude-Modulated Equivariant Adapter)，一种用于球谐基等变 GNN 的轻量参数高效微调方法，通过标量门控按"阶-多重度"通道独立调制特征幅度，在严格保持等变性的前提下，以更少参数量实现了超越 ELoRA 和全参数微调的 SOTA 分子势能预测精度。
tags:
  - "AAAI 2026"
  - "图学习"
  - "等变图神经网络"
  - "参数高效微调"
  - "分子势能预测"
  - "球谐函数"
  - "PEFT"
---

# Magnitude-Modulated Equivariant Adapter for Parameter-Efficient Fine-Tuning of Equivariant Graph Neural Networks

**会议**: AAAI 2026  
**arXiv**: [2511.06696](https://arxiv.org/abs/2511.06696)  
**代码**: [https://github.com/CLaSLoVe/MMEA](https://github.com/CLaSLoVe/MMEA)  
**领域**: 图学习  
**关键词**: 等变图神经网络, 参数高效微调, 分子势能预测, 球谐函数, PEFT

## 一句话总结

提出 MMEA (Magnitude-Modulated Equivariant Adapter)，一种用于球谐基等变 GNN 的轻量参数高效微调方法，通过标量门控按"阶-多重度"通道独立调制特征幅度，在严格保持等变性的前提下，以更少参数量实现了超越 ELoRA 和全参数微调的 SOTA 分子势能预测精度。

## 研究背景与动机

### 等变 GNN 与分子势能预测

密度泛函理论 (DFT) 是化学和材料科学的标准计算方法，但其立方级计算复杂度限制了大规模仿真。深度学习分子势能模型（如 MACE、NequIP、Equiformer 等）通过学习原子间势能来加速模拟，同时保持量子力学精度。

其中，**基于球谐函数的等变 GNN** 特别强大：
- 内在尊重旋转、平移和置换对称性
- 能建模高阶物理信息（不仅限于标量，还包括向量和更高阶张量）
- 样本效率极高：几百到几千个局部结构即可达到化学精度

### 微调的必要性与挑战

当目标系统在预训练数据中代表性不足时（如稀有化学构型），预训练模型精度下降。微调可以在小规模任务数据上恢复精度，但：

**全参数微调的风险**：过拟合、灾难性遗忘

**传统 PEFT（LoRA、Adapter）的致命缺陷**：这些方法会混合不同张量阶 (order) 的不可约表示，**破坏等变性**，导致模型失去对称性保证

### ELoRA 的局限

ELoRA 是首个等变 PEFT 方法，通过引入路径依赖的低秩适配器到每个张量通道中实现等变微调。但 ELoRA 在每个张量阶内仍保留了较高的自由度——它允许不同多重度 (multiplicity) 通道之间的混合。

**核心洞察**：在一个训练良好的等变 GNN 中，每个阶的多重度通道已经构成了一个稳健的基。让它们在微调过程中自由混合可能会扭曲预训练特征空间的几何结构。

### MMEA 的物理直觉

既然预训练模型已经为每个阶学到了良好的基表示，那么**仅调节每个通道的幅度（而非混合通道）就足以适应新的化学环境**。这就像调节收音机各频率的音量，而不是重新编排频率。

## 方法详解

### 整体框架

MMEA 在等变线性层后插入轻量门控模块，通过标量增益调制每个"阶×多重度"通道的幅度，严格不混合不同多重度通道。

### 关键设计

#### 1. **节点特征空间分解**：理解等变表示结构

节点特征空间是不可约表示的直和：
$$\mathcal{H} := \bigoplus_{\ell=0}^{L} \mathcal{H}^{(\ell)}, \quad \mathcal{H}^{(\ell)} := V^{(\ell)} \otimes \mathbb{R}^{1 \times m_\ell}$$

其中 $V^{(\ell)}$ 是阶 $\ell$ 的不可约表示（维度 $d_\ell = 2\ell+1$），$m_\ell$ 是多重度。SO(3) 群作用 $g$ 仅作用于 $V^{(\ell)}$：
$$g \cdot (v \otimes a) := (\rho^{(\ell)}(g)v) \otimes a$$

**关键观察**：群作用只在 $V^{(\ell)}$ 上旋转，多重度空间 $\mathbb{R}^{1 \times m_\ell}$ 是不变的。因此，在多重度维度上进行标量缩放不会破坏等变性。

#### 2. **轻量门控网络**：从标量通道生成调制增益

**功能**：仅以 $\ell=0$（标量）特征 $\mathbf{h}^{(0)}$ 为输入，通过两层 MLP 生成所有阶所有多重度的增益标量。

**瓶颈投影**：
$$\mathbf{z} = \text{SiLU}(W_\downarrow \mathbf{h}^{(0)} + \mathbf{b}_\downarrow), \quad W_\downarrow \in \mathbb{R}^{r \times m_0}$$

**扩展**：
$$[\boldsymbol{\gamma}^{(0)}, \boldsymbol{\gamma}^{(1)}, \ldots, \boldsymbol{\gamma}^{(L)}] = W_\uparrow \mathbf{z} + \mathbf{b}_\uparrow, \quad W_\uparrow \in \mathbb{R}^{(\sum_\ell m_\ell) \times r}$$

每个 $\boldsymbol{\gamma}^{(\ell)} \in \mathbb{R}^{m_\ell}$ 为阶 $\ell$ 的每个多重度通道分配一个标量增益。$r$ 为瓶颈维度。

**设计动机**：仅使用标量通道作为输入以保持参数效率和等变性（标量在 SO(3) 变换下不变）。

#### 3. **等变调制**：按通道缩放特征幅度

对于标量阶（$\ell = 0$）：加法调制
$$\mathcal{A}_\Gamma^{(0)}(\mathbf{h}^{(0)}) = \mathbf{h}^{(0)} + \boldsymbol{\gamma}^{(0)}$$

对于高阶（$\ell \geq 1$）：乘法缩放
$$\mathcal{A}_\Gamma^{(\ell)}(\mathbf{h}^{(\ell)}) = \sum_{k=1}^{m_\ell} \phi(\gamma_k^{(\ell)}) \mathbf{v}_k^{(\ell)} \otimes \mathbf{e}_k^{(\ell)}$$

其中 $\phi(x) = 1+x$（残差缩放）或 $\phi(x) = e^x$（正缩放）。

最终调制特征：
$$\mathbf{h}' := \mathcal{A}_\Gamma(\mathbf{h}) = \bigoplus_{\ell=0}^{L} \mathcal{A}_\Gamma^{(\ell)}(\mathbf{h}^{(\ell)})$$

**核心区别 MMEA vs ELoRA**：
- ELoRA 允许在每个阶内部不同多重度通道之间的低秩混合
- MMEA 仅对每个通道进行独立的标量缩放，完全不混合不同多重度

### 等变性证明（简述）

论文给出了严格的数学证明 $\mathcal{A}_\Gamma(g \cdot \mathbf{h}) = g \cdot \mathcal{A}_\Gamma(\mathbf{h})$：
1. **增益不变性**：$\boldsymbol{\gamma}^{(\ell)}$ 仅由标量 $\mathbf{h}^{(0)}$ 产生，而 $\mathbf{h}^{(0)}$ 在 SO(3) 下不变
2. **调制等变性**：标量缩放与 Wigner-D 矩阵旋转可交换，因为缩放作用在多重度空间（不变空间），旋转作用在表示空间

### 损失函数 / 训练策略

- 使用加权能量-力联合损失（ef loss）
- 能量权重 1，力权重 1000
- Adam 优化器，学习率 0.005
- EMA 衰减 0.995，梯度裁剪 100
- 与 ELoRA 使用相同的预训练模型（MACE-OFF）、数据集和超参数

## 实验关键数据

### 主实验

#### rMD17 数据集（10 个有机分子，50 样本训练）

| 分子 | 指标 | Full | ELoRA | **MMEA** | MMEA 相对 Full 改进 |
|------|------|------|-------|---------|-------------------|
| Aspirin | E/F | 9.7/23.9 | 8.0/18.3 | **7.3/16.4** | ↓25%/↓31% |
| Azobenzene | E/F | 4.6/14.8 | 4.0/12.6 | **3.9/11.9** | ↓15%/↓20% |
| Benzene | E/F | 0.3/2.4 | 0.2/1.6 | **0.2/1.4** | ↓33%/↓42% |
| Naphthalene | E/F | 1.8/8.1 | 1.4/6.0 | **1.2/5.7** | ↓33%/↓30% |
| Paracetamol | E/F | 6.5/20.3 | 4.7/14.5 | **4.3/13.3** | ↓34%/↓34% |
| Salicylic | E/F | 4.3/17.2 | 3.2/13.3 | **2.9/12.0** | ↓33%/↓30% |
| Toluene | E/F | 1.8/8.8 | 1.4/6.2 | **1.2/5.4** | ↓33%/↓39% |
| Uracil | E/F | 2.9/15.8 | 2.1/12.3 | **2.0/10.7** | ↓31%/↓32% |

MMEA 在所有 10 个分子上全面超越 Full 和 ELoRA。相比 ELoRA，能量 MAE 平均额外提升 ~6.6%，力 MAE 平均额外提升 ~8.7%。

#### 3BPA 数据集（300K 训练，多温度泛化测试）

| 条件 | 指标 | Scratch | Full | ELoRA | **MMEA** |
|------|------|---------|------|-------|---------|
| 300K | E/F | 3.0/8.8 | 3.3/7.8 | 3.0/7.5 | **2.7/7.5** |
| 600K | E/F | 9.7/21.8 | 7.3/16.6 | 6.5/15.5 | **6.5/15.4** |
| 1200K | E/F | 29.8/62.0 | 20.3/48.7 | 17.6/42.0 | **17.1/39.7** |
| Dihedral | E/F | 7.8/16.5 | 7.3/12.3 | 5.9/11.4 | **5.6/10.6** |

在高温（分布外）条件下 MMEA 泛化能力更优，1200K 力误差从 ELoRA 的 42.0 降至 39.7，二面角切片也显著改善。

#### AcAc 数据集

| 条件 | 指标 | Scratch | Full | ELoRA | MMEA(r=16) | MMEA(r=32) |
|------|------|---------|------|-------|------------|------------|
| 300K | E/F | 0.9/5.1 | 1.0/5.1 | 0.8/4.5 | **0.7/4.4** | **0.7/4.2** |
| 600K | E/F | 4.6/22.4 | 5.8/16.4 | 3.9/13.6 | 3.6/13.2 | **3.2/13.0** |

### 消融实验

#### rMD17-Aspirin 上的消融

| 配置 | Energy MAE | Forces MAE | 说明 |
|------|-----------|------------|------|
| **MMEA (完整)** | **7.3** | **16.4** | 最优 |
| Full fine-tuning | 9.7 | 23.9 | 基线 |
| w/o 非线性激活 | 7.6 | 16.4 | SiLU 有一定贡献 |
| w/o input-head 复用 | 9.2 | 16.7 | 标量/高阶分开处理有害 |
| w/o 标量调制 | 12.9 | 30.5 | **标量通道调制至关重要** |
| w/o 高阶调制 | 8.3 | 16.6 | 高阶信息调制有价值 |
| 共享高阶调制 | 7.6 | 16.6 | 独立调制优于共享 |
| Readout only | 23.8 | 36.8 | 仅调 0.3% 参数，能力不足 |
| Adapter (传统) | 11.0 | 26.3 | **破坏等变性导致性能下降** |

**最关键发现**：移除标量调制导致性能灾难性下降（Energy 从 7.3 到 12.9），说明标量通道是调制的核心。传统 Adapter 因破坏等变性而比 Full 还差。

### 参数效率

| 方法 | rank r | 可训练参数量 | 占 Full 比例 |
|------|--------|------------|-------------|
| Full | / | 751,896 | 100% |
| ELoRA | 16 | 175,880 | 23.4% |
| **MMEA** | **16** | **151,354** | **20.1%** |
| MMEA | 32 | 201,258 | 26.7% |

MMEA (r=16) 仅用 Full 的 20.1% 参数，约为 ELoRA 的 85%，却实现了更优性能。

### 关键发现

1. 在等变 GNN 中，按通道调制幅度足以适应新化学环境，无需混合通道
2. 减少自由度反而提升性能：MMEA 比 ELoRA 更少参数但更优，核心原因是更好地保留了预训练知识
3. 标量通道 ($\ell=0$) 的调制是最关键的组件
4. 收敛速度大幅提升：MMEA 在 epoch 58 达到 ELoRA 在 ~epoch 200 的损失水平
5. 当目标系统严重偏离预训练分布时（如用无机预训练模型预测有机分子），MMEA 不如 ELoRA 和 Full

## 亮点与洞察

1. **"少即是多"的深刻洞察**：在参数高效微调领域，MMEA 证明了更少的自由度（仅标量缩放）可以带来更好的泛化，因为保留了预训练模型的几何结构
2. **物理直觉驱动的设计**：从"等变表示空间已经学到了好的基"出发，仅调节幅度不改变方向
3. **严格的等变性证明**：不同于工程经验设计，MMEA 有完整的数学保证
4. **实用性**：集成到广泛使用的 e3nn 框架中，社区可直接使用
5. **训练效率**：收敛速度比 ELoRA 快约 3-4 倍

## 局限与展望

1. **分布外场景受限**：当目标分布与预训练数据差异较大时，MMEA 不如 ELoRA（因为更少自由度在此情况下反而是劣势）
2. **无法参数合并**：不像 ELoRA 可以将学到的权重合并回 backbone，MMEA 在推理时有约 2.1% 的额外延迟
3. **全连接张量积的门控**：如何有效为多输入的张量积设计门控仍是开放问题
4. **收敛速度仍有提升空间**：尽管比 ELoRA 快，但绝对收敛速度仍不够快
5. 可以探索自适应 rank 选择策略（不同层/不同阶使用不同 rank）

## 相关工作与启发

- **ELoRA**：首个等变 PEFT 方法，MMEA 的直接改进对象
- **FiLM (Feature-wise Linear Modulation)**：MMEA 的标量门控与 FiLM 的思想类似，但在等变框架下有特殊约束
- **MACE**：基础 backbone 模型，多体交互的等变 GNN
- **LoRA**：参数高效微调的开创性工作，但不适用于等变网络
- **BitFit**：仅更新 bias 的极简 PEFT 方法，与 MMEA 的"极简调制"理念相通

## 评分

- 新颖性: ⭐⭐⭐⭐ — 洞察独到（保留多重度基仅调幅度），但方法本身较简单
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个标准数据集、10个分子、多温度、详细消融和参数分析
- 写作质量: ⭐⭐⭐⭐⭐ — 物理动机清晰、等变性证明严谨、实验设置公平透明
- 价值: ⭐⭐⭐⭐ — 对分子模拟社区有直接实用价值，但适用范围限于分布内微调

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning](../../NeurIPS2025/graph_learning/smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)
- [\[ICLR 2026\] AdS-GNN - a Conformally Equivariant Graph Neural Network](../../ICLR2026/graph_learning/ads-gnn_-_a_conformally_equivariant_graph_neural_network.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](adaptive_riemannian_graph_neural_networks.md)
- [\[AAAI 2026\] Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)
- [\[AAAI 2026\] EchoLess: Label-Based Pre-Computation for Memory-Efficient Heterogeneous Graph Learning](echoless_label-based_pre-computation_for_memory-efficient_heterogeneous_graph_le.md)

</div>

<!-- RELATED:END -->
