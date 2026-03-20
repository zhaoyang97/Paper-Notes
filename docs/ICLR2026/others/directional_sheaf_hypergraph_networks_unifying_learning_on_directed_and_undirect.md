# Directional Sheaf Hypergraph Networks: Unifying Learning on Directed and Undirected Hypergraphs

**会议**: ICLR 2026
**arXiv**: [2510.04727](https://arxiv.org/abs/2510.04727)
**代码**: [GitHub](https://github.com/EmaMule/DirectionalSheafHypergraphs)
**领域**: 图神经网络 / 超图学习
**关键词**: sheaf neural networks, directed hypergraphs, Laplacian, spectral methods, heterophily

## 一句话总结

本文提出 Directional Sheaf Hypergraph Networks (DSHN)，通过将 Cellular Sheaf 理论与有向超图的方向信息结合，构造了一种复值 Hermitian Laplacian 算子，统一并推广了现有的图和超图 Laplacian，在 7 个真实数据集上相对准确率提升 2%–20%。

## 研究背景与动机

1. **超图的高阶交互建模**：许多真实系统存在多实体间的高阶关系，传统图只能表达成对关系。超图通过超边连接多个节点来建模多路交互。
2. **无向超图的局限**：大多数 HGNN 仅处理无向超图，忽略了超边中可能存在的方向性（化学反应中反应物→产物、因果交互的发起方→接收方）。
3. **Sheaf 理论缓解异质性**：通过为节点和边分配向量空间及可学习的 restriction map，能有效缓解过平滑和异质性问题。但已有 Sheaf 超图方法不支持有向超图。
4. **已有 SHN 的谱性质缺陷**：Duta et al. (2023) 的 Sheaf Hypergraph Laplacian 不满足正半定性，无法作为合格卷积算子。
5. **有向图方法的成功经验**：Magnetic Laplacian 用复数相位编码方向，但未推广到超图。

## 方法详解

### 整体框架

DSHN 三步走：(1) 定义 Directed Hypergraph Cellular Sheaf（复值 restriction map）；(2) 构造 Directed Sheaf Hypergraph Laplacian（Hermitian 正半定算子）；(3) 基于该 Laplacian 的扩散卷积网络。

### 关键设计

1. **方向性矩阵 $\mathcal{S}^{(q)}$（Definition 1）**
   - **做什么**：为节点-超边对分配复值系数编码方向角色
   - **核心思路**：头节点系数 1，尾节点系数 $e^{-2\pi i q}$，参数 $q$ 控制方向信息强度
   - **设计动机**：$q=0$ 退化为无向；$q=1/4$ 时方向编码在虚部，与 Magnetic Laplacian 对齐

2. **Directed Sheaf Hypergraph Laplacian**
   - **做什么**：$\mathbf{L}^{\vec{\mathcal{F}}} = \mathbf{D}_V - \mathbf{B}^{(q)\dagger} \mathbf{D}_E^{-1} \mathbf{B}^{(q)}$
   - **核心思路**：对角块实值（自身信息），非对角块有向时为复值，修正了 Duta et al. 的对角项系数
   - **设计动机**：确保正半定性等所有谱卷积所需性质

3. **谱性质保证**
   - **做什么**：证明可对角化、实非负特征值、正半定、谱上界 1
   - **核心思路**：通过 Dirichlet 能量非负性证明正半定
   - **设计动机**：保证 Fourier 变换良定义和多项式滤波器稳定

4. **统一泛化**
   - **做什么**：证明在特殊情况下退化为 Graph Sheaf Laplacian、Magnetic Laplacian、Zhou 超图 Laplacian、GeDi Laplacian 等
   - **设计动机**：一个框架统一所有已有算子

5. **DSHNLight**
   - **做什么**：detach Laplacian 构建的梯度，固定 restriction map 参数
   - **设计动机**：大幅降低计算成本，多个数据集上性能相当甚至更好

### 损失函数 / 训练策略

- 标准交叉熵节点分类损失
- 复值输出通过 unwind（拼接实部虚部）转实值后送分类头
- Restriction map 通过 MLP 学习，输入为节点和超边特征的拼接

## 实验关键数据

### 主实验

7 个数据集上对比 13 个 baseline 的节点分类准确率：

| 数据集 | DSHN 相对最佳 baseline 提升 |
|--------|--------------------------|
| Cora (co-auth) | ~2% |
| Citeseer (co-auth) | ~5% |
| Senate-committees | ~8% |
| House-committees | ~4% |
| Walmart-trips | ~20% |
| Zoo | ~3% |
| 20Newsgroups | ~2% |

### 消融实验

| 变体 | 效果 |
|------|------|
| $q=0$（无方向） | 退化为无向 sheaf 方法，性能下降 |
| $q=1/4$（标准相位） | 有向数据上表现最佳 |
| Trivial sheaf ($\mathcal{F}=I$) | 退化为有向超图 Laplacian，性能大幅下降 |
| DSHNLight | 计算效率高，多数数据集性能接近 |

### 关键发现

- 方向性 + Sheaf 联合使用效果显著优于单独使用任一
- Duta et al. (2023) 的 Laplacian 确实存在负特征值（附录给出反例）
- DSHNLight 的"随机投影"策略出乎意料地有效
- 在异质性数据集上优势最为明显

## 亮点与洞察

1. 一个复值 Hermitian 算子统一了多种已有 Laplacian 定义
2. 严格纠正了 Duta et al. (2023) 的谱性质错误
3. 方向信息用复数相位编码的思路从有向图自然推广到超图
4. DSHNLight 与极限学习机思想呼应，说明随机特征在图学习中有效

## 局限性 / 可改进方向

- $nd \times nd$ Laplacian 的可扩展性问题
- $q$ 作为全局参数，未能为每条超边学习不同 $q$
- 实验仅覆盖节点分类
- 有向超图真实数据集稀缺
- 缺少 WL 层级等表达能力分析

## 相关工作与启发

- **Hansen & Gebhart (2020)**：Graph Sheaf NN → 本文推广到有向超图
- **Zhang et al. (2021)**：Magnetic Laplacian → 本文推广到超图 + sheaf
- **Duta et al. (2023)**：SheafHyperGNN → 本文修正其谱性质缺陷
- **启发**：复值 Hermitian + Sheaf 范式可推广到 simplicial complex 等更一般拓扑结构

## 评分

- **新颖性**: ⭐⭐⭐⭐ Sheaf + 有向超图的结合和统一性结果有重要理论价值
- **实验充分度**: ⭐⭐⭐⭐ 7 个数据集、13 个 baseline、完整消融
- **写作质量**: ⭐⭐⭐⭐ 数学推导清晰，符号系统较重但定义精确
- **价值**: ⭐⭐⭐⭐ 修正了已有方法缺陷并提供统一框架
