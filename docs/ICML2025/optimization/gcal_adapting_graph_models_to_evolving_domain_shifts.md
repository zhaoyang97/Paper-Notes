---
title: >-
  [论文解读] GCAL: Adapting Graph Models to Evolving Domain Shifts
description: >-
  [ICML 2025][优化/理论][图持续学习] 提出 Graph Continual Adaptive Learning (GCAL)，通过"适应+生成记忆"双层优化策略，在图模型面对持续演变的 OOD 图序列时，利用信息最大化进行无监督域适应，同时基于信息瓶颈理论设计变分记忆图生成模块来压缩历史图知识，有效缓解灾难性遗忘。
tags:
  - "ICML 2025"
  - "优化/理论"
  - "图持续学习"
  - "域适应"
  - "灾难性遗忘"
  - "记忆重放"
  - "信息瓶颈"
---

# GCAL: Adapting Graph Models to Evolving Domain Shifts

**会议**: ICML 2025  
**arXiv**: [2505.16860](https://arxiv.org/abs/2505.16860)  
**代码**: [https://github.com/joe817/GCAL](https://github.com/joe817/GCAL)  
**领域**: 优化  
**关键词**: 图持续学习, 域适应, 灾难性遗忘, 记忆重放, 信息瓶颈

## 一句话总结

提出 Graph Continual Adaptive Learning (GCAL)，通过"适应+生成记忆"双层优化策略，在图模型面对持续演变的 OOD 图序列时，利用信息最大化进行无监督域适应，同时基于信息瓶颈理论设计变分记忆图生成模块来压缩历史图知识，有效缓解灾难性遗忘。

## 研究背景与动机

图神经网络（GNN）在社交网络分析、生物信息学、推荐系统等领域取得了显著成功，但面对持续到来的新分布（OOD）图数据时存在关键挑战：

**单步适应的局限性**：现有图域适应方法（如 MMD、对抗学习）仅处理单步适应场景，无法应对连续域漂移

**灾难性遗忘**：当模型适应新图域时，会丢失对先前图域的适应能力。作者通过实验验证，SOTA 方法 EERM 在四个 OOD 图数据集上性能持续严重下滑

**无标签限制**：现有持续学习方法大多依赖标签数据进行记忆选择和重放，但在图域适应场景中标签通常不可获得

**研究空白**：缺少针对图模型的无监督记忆生成与重放的持续适应方法

核心动机是：如何在无标签条件下，让图模型在面对持续到来的不同分布图数据时，既能适应新域，又能保留旧域知识？

## 方法详解

### 整体框架

GCAL 采用"适应并生成记忆（Adapt and Generate Memory）"的双层优化策略。当第 $t$ 个目标图 $G_t$ 到来时：

1. **内层循环（Adapt）**：使用信息最大化方法在新图 $G_t$ 上适应模型，同时对记忆池中的历史记忆图进行重放以防止遗忘，更新模型参数 $\Theta_{t-1} \to \Theta_t$
2. **外层循环（Generate Memory）**：基于适应后的模型 $f(\Theta_t)$，利用变分记忆图生成器学习一个压缩的记忆图 $\hat{G}_t$，加入记忆池供未来重放使用

### 关键设计

#### 1. 带记忆重放的适应（Adaptation with Memory Replay）

由于目标域无标签，采用 Information Maximization（IM）作为自监督适应目标。核心思想是：一个有效的模型应该对目标数据的预测具有高置信度（输出接近 one-hot），同时保持类别多样性。适应损失包含两项：

- **熵最小化**：$-\mathbb{E}_{v \sim \mathcal{V}}[\sum_k p_{v,k} \log p_{v,k}]$，使模型对每个节点的预测更确定
- **多样性正则化**：$\sum_k \hat{p}_k \log \hat{p}_k$，防止所有节点收敛到同一伪标签

适应时同时在新图 $G_t$ 和记忆池 $\mathcal{G} = \{\hat{G}_1, ..., \hat{G}_{t-1}\}$ 上施加 IM 损失：

$$\mathcal{L}_{AMR} = \mathcal{L}_{Adp}(G_t; \Theta_{t-1}) + \sum_{i=1}^{t-1} \mathcal{L}_{Adp}(\hat{G}_i; \Theta_{t-1})$$

#### 2. 变分记忆图生成（Variational Memory Graph Generation）

记忆图需满足三个条件：(1) 尺寸远小于原图 ($K \ll N_t$)；(2) 保留充分的任务相关信息；(3) 能跨分布泛化。

**信息瓶颈推导**：基于图信息瓶颈原理，优化目标为最大化记忆图与任务信号的互信息、同时最小化与原图的互信息。通过引入变分隐变量 $Z_t$ 和链式法则，推导出可优化的下界（Theorem 3.1）：

$$\mathcal{L}(\Phi) \geq \mathbb{E}[\log P_f(\hat{Y}_t|\hat{G}_t)] - \beta \mathbb{E}[\text{KL}(P_g(\hat{G}_t|G_t,Z_t) \| Q(\hat{G}_t))] + \beta \mathbb{E}[\log P_g(\hat{G}_t|G_t,Z_t)]$$

**变分图生成器架构**：

- **变分 GNN 编码器**：将输入图 $G_t$ 映射为隐变量分布 $[\mu; \log\sigma] = \text{TopKSelector}(\text{GNN}_{\mu,\sigma}(A_t, X_t))$
- **TopK 选择器**：使用可训练参数 $\mathbf{p}$ 对节点分布评分并选择 Top-K 个最重要的节点，实现图压缩
- **节点特征生成**：通过重参数化技巧从高斯分布采样：$\hat{z}_i = \mu_i + \sigma_i^2 \odot \varepsilon$
- **边生成**：假设边服从独立伯努利分布，使用 MLP 计算边权重，并通过 Gumbel-Max 重参数化实现可微采样：
    - $w_{i,j} = (\text{MLP}([\hat{z}_i; \hat{z}_j]) + \text{MLP}([\hat{z}_j; \hat{z}_i])) / 2$
    - $a_{i,j} = \text{Sigmoid}((w_{i,j} + \log\frac{\delta}{1-\delta}) / \tau)$

这样的对称设计保证了边的无向性，Gumbel-Sigmoid 提供了连续松弛使梯度可回传。

### 损失函数 / 训练策略

总体目标为双层优化，外层优化记忆图生成器参数 $\Phi$，内层优化模型参数 $\Theta$。记忆图生成的总损失包含三项：

#### (1) 记忆图学习损失 $\mathcal{L}_{MGL}$（梯度匹配）

通过梯度匹配策略实现图蒸馏，对齐记忆图与原图在模型参数上的梯度方向：

$$\mathcal{L}_{MGL} = D\left(\frac{\partial \mathcal{L}_{Adp}(\hat{G}_t; f(\Theta_t))}{\partial \Theta_t}, \frac{\partial \mathcal{L}_{Adp}(G_t; f(\Theta_t))}{\partial \Theta_t}\right)$$

其中 $D$ 为逐层梯度的余弦距离之和：$D(\hat{\mathbf{g}}, \mathbf{g}) = \sum_i (1 - \frac{\hat{\mathbf{g}}_i \cdot \mathbf{g}_i}{\|\hat{\mathbf{g}}_i\| \|\mathbf{g}_i\|})$

#### (2) 正则化损失 $\mathcal{L}_{Reg}$（KL 散度）

将生成分布约束到先验分布，节点特征先验为标准正态分布 $\mathcal{N}(0, I)$，边先验为伯努利分布 $\text{Bernoulli}(q)$：

$$\mathcal{L}_{Reg} = \frac{1}{2}\sum_{i,j}(\mu_{i,j}^2 + \sigma_{i,j}^2 - \log\sigma_{i,j}^2 - 1) + \sum_{i,j}(w_{i,j}\log\frac{w_{i,j}}{q} + (1-w_{i,j})\log\frac{1-w_{i,j}}{1-q})$$

#### (3) 生成损失 $\mathcal{L}_{Gen}$（分布对齐）

使用 L2 距离度量记忆图与原图在模型隐层表示空间中的分布差异：

$$\mathcal{L}_{Gen} = \|\sum_{i=1}^{K} \hat{u}_i(\Theta) - \sum_{i=1}^{N_t} u_i(\Theta)\|_2$$

**总优化目标**：

$$\min_{\hat{G}_t, \Phi} \mathcal{L}_{MGL} + \lambda_1 \mathcal{L}_{Reg} + \lambda_2 \mathcal{L}_{Gen} \quad \text{s.t.} \quad \Theta_t = \arg\min_\Theta \mathcal{L}_{AMR}$$

此外，使用指数移动平均（EMA）策略平滑模型参数更新，每个时间步使用新的生成器并仅保留生成的记忆图。

## 实验关键数据

### 主实验

在四个图数据集上与八个基线方法比较（AP = 平均性能，AF = 平均遗忘率，↑ 越高越好）：

| 数据集 | 指标 | GCAL | CoTTA (最强基线) | 提升 |
|--------|------|------|-----------------|------|
| Twitch-explicit | AP-AUC/% | **55.65±0.09** | 53.94±0.36 | +1.71 |
| Facebook-100 | AP-ACC/% | **52.72±0.36** | 50.12±0.16 | +2.60 |
| Elliptic | AP-F1/% | **56.57±0.14** | 54.08±0.05 | +2.49 |
| OGB-Arxiv | AP-ACC/% | **45.22±0.17** | 40.28±0.01 | +4.94 |

遗忘指标方面，GCAL 在全部四个数据集上均为正值（AF > 0），说明不仅避免了遗忘，甚至在适应新域后对旧域有一定提升。

### 消融实验

| 配置 | Twitch | Facebook | Elliptic | OGB-Arxiv |
|------|--------|----------|----------|-----------|
| w/o $\mathcal{L}_{Reg}$ & $\mathcal{L}_{Gen}$ | 54.03±2.63 | 52.05±0.31 | 46.53±0.01 | 44.70±0.06 |
| w/o $\mathcal{L}_{Reg}$ | 55.34±0.41 | 52.37±0.56 | 55.23±0.32 | 44.76±0.48 |
| w/o $\mathcal{L}_{Gen}$ | 55.37±0.33 | 52.14±0.32 | 55.64±0.57 | 44.91±0.11 |
| w/o EMA | 54.79±0.04 | 47.66±0.06 | 53.83±0.20 | 43.19±0.08 |
| **GCAL (完整)** | **55.65±0.09** | **52.72±0.36** | **56.57±0.14** | **45.22±0.17** |

### 关键发现

1. **全面超越 SOTA**：GCAL 在所有四个数据集上均显著超越所有基线方法，尤其在 OGB-Arxiv 上提升最为明显（+4.94% AP）
2. **抗遗忘能力强**：AF 指标全部为正，表明模型在持续适应中不仅避免了遗忘，还能反向提升旧域性能
3. **EMA 是关键组件**：消融实验中去掉 EMA 导致最大性能下降（Facebook-100 下降 5.06%），表明平滑参数更新对持续适应至关重要
4. **同时去掉 $\mathcal{L}_{Reg}$ 和 $\mathcal{L}_{Gen}$ 导致严重退化**：Elliptic 数据集上从 56.57 降至 46.53，说明信息瓶颈理论推导的三项损失合力工作不可或缺
5. **记忆压缩比极低即有效**：仅用原图 1%~9% 的节点生成记忆图即可达到良好效果

## 亮点与洞察

1. **问题定义新颖**：首次系统化定义了图模型在持续演变 OOD 图序列上的无监督持续适应问题，填补了图域适应与持续学习的交叉空白
2. **理论驱动设计**：从信息瓶颈理论出发推导出可优化的下界（Theorem 3.1），使记忆图生成有坚实的理论基础，而非纯启发式设计
3. **无监督记忆生成**：不依赖任何标签信息生成记忆图，通过信息最大化提供伪监督信号，大幅拓宽了应用场景
4. **高度压缩的经验池**：记忆图仅需原图 1%~9% 的节点，存储和计算开销极小
5. **端到端可微生成**：节点通过 VAE 重参数化、边通过 Gumbel-Sigmoid 重参数化，整个记忆图生成流程完全可微

## 局限与展望

1. **任务类型限制**：当前仅验证了节点分类任务，未涉及链接预测、图分类等其他图任务
2. **类别数不变假设**：假设源域和所有目标域共享相同的类别集合，不适用于开放世界或类增量场景
3. **记忆池线性增长**：记忆池大小随时间步线性增长，长序列场景下重放开销会累积
4. **每步需新生成器**：每个时间步使用新的生成器，生成器的知识未被跨步复用
5. **数据集规模有限**：实验主要在社交网络和引用网络上验证，缺少更大规模或更复杂结构的图数据验证

## 相关工作与启发

- **图域适应**：EERM (Wu et al., 2022b) 和 GTrans (Jin et al., 2022) 是代表方法，但每步独立训练无法避免遗忘
- **持续测试时适应**：CoTTA (Wang et al., 2022) 使用 EMA 更新但缺少记忆机制；EATA (Niu et al., 2022) 使用样本选择但未考虑图结构
- **图蒸馏/压缩**：梯度匹配策略源自 Dataset Condensation (Jin et al., 2021)，GCAL 将其创新性地应用于记忆图学习
- **信息瓶颈**：Graph Information Bottleneck (Wu et al., 2020b; Sun et al., 2022) 启发了记忆图的理论推导
- **启发**：将变分记忆生成与梯度匹配蒸馏结合是一个值得在其他持续学习场景推广的范式

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 首次将持续学习与图域适应深度结合，理论推导完整 |
| 技术深度 | 4.5 | 信息瓶颈下界推导严谨，双层优化+三损失设计完善 |
| 实验充分性 | 4 | 4个数据集、8个基线、完整消融，但缺少大规模验证 |
| 写作质量 | 4 | 结构清晰，问题-方法-实验逻辑通顺 |
| 实用价值 | 3.5 | 代码已开源，但适用场景（无标签图序列适应）较窄 |
| **综合** | **4** | 扎实的理论驱动工作，在图持续适应新领域建立了强基线 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Sparse Causal Discovery with Generative Intervention for Unsupervised Graph Domain Adaptation](sparse_causal_discovery_with_generative_intervention_for_unsupervised_graph_doma.md)
- [\[ICML 2025\] Nonparametric Teaching for Graph Property Learners](nonparametric_teaching_for_graph_property_learners.md)
- [\[ICML 2025\] Subspace Optimization for Large Language Models with Convergence Guarantees](subspace_optimization_for_large_language_models_with_convergence_guarantees.md)
- [\[CVPR 2025\] Federated Learning with Domain Shift Eraser](../../CVPR2025/optimization/federated_learning_with_domain_shift_eraser.md)
- [\[NeurIPS 2025\] MergeBench: A Benchmark for Merging Domain-Specialized LLMs](../../NeurIPS2025/optimization/mergebench_a_benchmark_for_merging_domain-specialized_llms.md)

</div>

<!-- RELATED:END -->
