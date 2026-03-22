# GraphTOP: Graph Topology-Oriented Prompting for Graph Neural Networks

**会议**: NeurIPS 2025  
**arXiv**: [2510.22451](https://arxiv.org/abs/2510.22451)  
**代码**: https://github.com/xbfu/GraphTOP (有)  
**领域**: 图学习 / Graph Prompting  
**关键词**: Graph Prompting, Topology, Edge Rewiring, Pre-training, Gumbel-Softmax

## 一句话总结
提出首个图拓扑导向的 prompting 框架 GraphTOP，通过将 topology-oriented prompting 建模为边重连问题并用 Gumbel-Softmax 松弛到连续空间，在 5 个数据集 4 种预训练策略下超越 6 个基线方法。

## 研究背景与动机

1. **领域现状**："预训练-适配" 范式在图学习中广泛使用。Graph prompting 通过修改输入图的数据（保持预训练 GNN 冻结）来适配下游任务，是一种高效的适配策略。
2. **现有痛点**：现有 graph prompting 方法几乎都是 feature-oriented 的——只操作节点特征或隐层表示（如 GPF、All-in-one、GraphPrompt），完全忽视了图拓扑这一图数据的本质特征。然而图表示不仅依赖特征，还由拓扑结构决定。
3. **核心矛盾**：如何通过修改图拓扑来实现 prompting？边选择是离散优化问题，难以直接用梯度下降。
4. **本文要解决什么？**：设计 topology-oriented 的 graph prompting 框架，让预训练 GNN 模型通过拓扑修改适配下游节点分类任务。
5. **切入角度**：将 topology prompting 建模为 edge rewiring 问题，通过 Bernoulli 重参数化 + Gumbel-Softmax 松弛到连续概率空间。
6. **核心 idea 一句话**：学习每条边的存在概率作为拓扑 prompt，通过 Gumbel-Softmax 使其可微训练，用熵正则化保证松弛紧密性和图稀疏性。

## 方法详解

### 整体框架
给定预训练 GNN 模型 $f_{\theta^*}$ 和输入图 $\mathcal{G} = (\mathbf{A}, \mathbf{X})$，GraphTOP 学习一个 prompted 图 $\tilde{\mathcal{G}} = (\mathbf{S}, \mathbf{X})$，其中 $\mathbf{S}$ 是学习得到的邻接矩阵。通过 $\mathbf{S}$ 替换原始 $\mathbf{A}$，冻结的 GNN 能为下游任务生成更合适的表示。

### 关键设计

1. **Edge Rewiring 建模**:
   - 做什么：将 topology prompt 建模为每对节点间是否存在边的二元决策 $s_{ij} \in \{0,1\}$
   - 核心思路：每个 $s_{ij}$ 视为 Bernoulli 随机变量，参数为 $p_{ij}$，通过 Gumbel-Softmax 重参数化使采样可微。利用 Theorem 1 证明 $\Pr(G_1 - G_2 + \log(p/(1-p)) \geq 0) = p$
   - 设计动机：将离散的边选择问题松弛为连续的概率优化，使得梯度下降可用

2. **共享可训练 Projector**:
   - 做什么：通过一个 2 层 MLP 从预训练 GNN 的节点表示计算边概率 $p_{ij} = \sigma(W_2(\text{ReLU}(W_1(h_i + h_j))))$
   - 核心思路：不独立学每个 $p_{ij}$（参数爆炸），而是用共享 projector 从节点表示推断边概率
   - 设计动机：大幅减少参数量，且利用预训练表示中的语义信息指导拓扑修改

3. **子图约束 (Subgraph-constrained)**:
   - 做什么：将 edge rewiring 限制在每个目标节点的 $\rho$-hop 局部子图内（默认 $\rho=2$）
   - 核心思路：只重连目标节点与子图内其他节点的边，保持其余边不变，复杂度从 $O(D^{2\rho})$ 降到 $O(D^{\rho})$
   - 设计动机：GNN 表示主要依赖局部子图，全局 rewiring 不必要且计算成本高

### 损失函数 / 训练策略
- 总损失：$\mathcal{L}_P + \lambda_1 \mathcal{L}_E + \lambda_2 \mathcal{L}_S$
- $\mathcal{L}_P$: 节点分类交叉熵
- $\mathcal{L}_E$: 熵正则化（鼓励概率趋向 0 或 1，保证松弛紧密性）
- $\mathcal{L}_S$: 稀疏性正则化（控制 prompted 图的边密度，参数 $\gamma=0.5$）
- 温度退火 $\tau$ 线性递减，从概率近似过渡到确定性输出

## 实验关键数据

### 主实验 — 5-shot Node Classification
| 预训练 | 方法 | Cora | PubMed | Amazon | Minesweeper | Flickr |
|--------|------|------|--------|--------|-------------|--------|
| GraphCL | Linear Probe | 55.69 | 67.30 | 23.19 | 67.59 | 29.31 |
| GraphCL | ProNoG (SOTA) | 60.01 | 68.17 | 23.26 | 65.48 | 26.17 |
| GraphCL | **GraphTOP** | **65.12** | **69.42** | **27.15** | **70.23** | **31.84** |

### 消融实验
| 配置 | 关键发现 |
|------|---------|
| w/o 熵正则化 $\mathcal{L}_E$ | 推理时拓扑不稳定，性能下降 |
| w/o 稀疏正则化 $\mathcal{L}_S$ | 图变得过密，效果下降 |
| Feature prompt vs Topology prompt | 拓扑 prompt 与特征 prompt 互补 |

### 关键发现
- 在 4 种预训练策略（GraphCL, SimGRACE, LP-GPPT, LP-GraphPrompt）下一致超越所有基线
- Theorem 2 理论证明：edge rewiring 可增大类间表示距离 $\text{Dist}' = \frac{p+q}{|p-q|} \text{Dist} > \text{Dist}$
- 拓扑 prompt 额外计算开销很小：$O(D^2 d_l^2)$ vs GNN 的 $O(KD^2 d_l^2)$

## 亮点与洞察
- **首次探索拓扑维度的 graph prompting**：之前所有工作都只做 feature prompt，本文填补了拓扑方向的空白。
- **Gumbel-Softmax + 熵正则化的搭配很巧妙**：用 Gumbel-Softmax 实现可微边选择，用熵正则化保证训练后边概率趋向确定性，推理时拓扑稳定。
- **理论保证**：基于 CSBM 模型证明 topology prompting 可增大类间距离，是严格的理论支撑。

## 局限性 / 可改进方向
- **仅支持节点分类**：当前框架聚焦于节点级任务，图级和边级任务的拓扑 prompt 设计待探索
- **5-shot 设定**：极低标注量下效果好，但全监督场景下是否仍有优势未验证
- **子图约束带来的信息损失**：$\rho=2$ 的限制可能错过重要的长程连接
- **可改进**：将 topology prompt 与 feature prompt 联合优化；探索自适应 $\rho$

## 相关工作与启发
- **vs GPF/All-in-one**: 这些方法只修改节点特征，GraphTOP 修改拓扑结构，两者正交互补
- **vs 图结构学习 (GSL)**: GSL 同样修改图结构但需要端到端训练 GNN，GraphTOP 保持 GNN 冻结更高效

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将 graph prompting 拓展到拓扑维度
- 实验充分度: ⭐⭐⭐⭐ 5 数据集 × 4 预训练策略，但仅做了节点分类
- 写作质量: ⭐⭐⭐⭐ 理论分析清晰，符号较多但组织合理
- 价值: ⭐⭐⭐⭐ 为 graph prompting 开辟了新方向，有实际应用潜力
