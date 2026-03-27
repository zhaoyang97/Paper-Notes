# Cooperative Sheaf Neural Networks

**会议**: ICLR2026  
**arXiv**: [2507.00647](https://arxiv.org/abs/2507.00647)  
**代码**: 待确认  
**领域**: 图学习  
**关键词**: Sheaf Neural Networks, 有向图, 协作行为, 过压缩, 异质图  

## 一句话总结

提出 Cooperative Sheaf Neural Network (CSNN)，通过在有向图上定义 cellular sheaf 的入度/出度 Laplacian，使节点能独立选择是否广播 (PROPAGATE) 或监听 (LISTEN) 信息，从而缓解过压缩并提升异质图节点分类性能。

## 研究背景与动机

Sheaf Neural Networks (SNNs) 利用 cellular sheaf 在图上构造灵活的扩散过程，能处理异质图任务并缓解过平滑。然而作者发现 SNNs 存在根本性限制：节点无法独立选择信息流方向。具体而言，若节点 $i$ 想阻止接收邻居信息（不 LISTEN），必须令 $\mathcal{F}_{i \unlhd e} = 0$，但这同时阻止了 $i$ 向外广播（不 PROPAGATE），导致 PROPAGATE 蕴含 LISTEN，退化为 ISOLATE。这一限制阻碍了对长程依赖中过压缩问题的有效处理。

## 方法详解

### 整体框架

CSNN 将无向图的每条边替换为一对有向边，在有向图上定义 cellular sheaf，引入入度和出度 sheaf Laplacian，然后组合两个 Laplacian 进行扩散，实现协作行为。

### 关键设计

**1. 有向 Cellular Sheaf (Definition 3.2)**：对有向图 $G=(V,E)$，每个节点 $i$ 关联两类 restriction map：源映射 $\mathcal{F}_{i \unlhd ij}$（$i$ 作为边 $ij$ 的源端）和目标映射 $\mathcal{F}_{i \unlhd ji}$（$i$ 作为边 $ji$ 的目标端），从而支持非对称通信。

**2. 入度/出度 Sheaf Laplacian (Definition 3.3)**：
- 出度 Laplacian：$L_{\mathcal{F}}^{\text{out}}(\mathbf{X})_i = \sum_{j \in N(i)} (\mathbf{S}_i^\top \mathbf{S}_i \mathbf{x}_i - \mathbf{T}_i^\top \mathbf{S}_j \mathbf{x}_j)$
- 入度 Laplacian 转置：$((L_{\mathcal{F}}^{\text{in}})^\top(\mathbf{X}))_i = \sum_{j \in N(i)} (\mathbf{T}_i^\top \mathbf{T}_i \mathbf{x}_i - \mathbf{T}_i^\top \mathbf{S}_j \mathbf{x}_j)$

**3. Flat Vector Bundle 高效实现**：每个节点仅需一对 conformal map $\mathbf{S}_i$（源）和 $\mathbf{T}_i$（目标），用 Householder 反射参数化正交矩阵乘以可学习正标量，参数量仅 $O(n)$ 而非 $O(m)$。

**4. 协作行为**：$\mathbf{T}_i = 0 \Rightarrow$ LISTEN 关闭；$\mathbf{S}_i = 0 \Rightarrow$ PROPAGATE 关闭；两者都非零为 STANDARD；两者都为零为 ISOLATE。

**5. 扩展感受野 (Proposition 4.2)**：CSNN 每层可影响 $2t$ 跳邻居（通常 GNN 只有 $t$ 跳），且可选择性忽略路径上中间节点直接接收远端信息 (Proposition 4.3)，有效缓解过压缩。

## 实验关键数据

| 实验类型 | 关键结果 |
|---------|---------|
| 合成过压缩测试 | CSNN 显著优于现有 SNN 和 Cooperative GNN，能有效处理长程依赖 |
| 异质图节点分类 (11 个数据集) | CSNN 通常优于已有 SNN（NSD、SheafHNN）和 Cooperative GNN |
| 长程图分类 (2 个任务) | CSNN 展现出强劲性能 |
| 总计 | 超过 13 个真实世界任务中 CSNN 通常表现最优 |

核心发现：
- 合成实验验证 CSNN 能建模长程依赖并避免过压缩
- 在异质图任务上，协作行为带来的非对称信息流是性能提升的关键
- Flat vector bundle 使 CSNN 保持与 NSD 相当的计算效率

## 亮点与洞察

1. **理论清晰**：Proposition 3.1 严格证明了传统 SNN 无法实现协作行为，动机自然
2. **数学优雅**：通过有向图上的 sheaf 结构统一了协作 GNN 和 sheaf 扩散两条研究线
3. **双倍感受野**：每层 $2t$ 跳的理论保证是缓解过压缩的核心优势
4. **参数高效**：flat vector bundle 使参数仅 $O(n)$，远低于一般 sheaf 的 $O(m)$

## 局限性

- 将无向边替换为有向边对使边数翻倍，在稠密图上内存开销增大
- 理论分析主要针对线性扩散，非线性情况下的行为未充分讨论
- Householder 反射的参数化可能限制 conformal map 的表达能力
- 缺少对超大规模图（百万节点级）的可扩展性验证

## 相关工作与启发

- **Neural Sheaf Diffusion (NSD, Bodnar et al. 2022)**：CSNN 的直接前驱，但 NSD 在无向图上无法实现协作行为
- **Cooperative GNN (Finkelshtein et al. 2024)**：通过 Gumbel-Softmax 实现离散动作选择，但需额外 action network；CSNN 用连续的 sheaf 结构更自然
- **SheafHNN (Bamberger et al. 2025)**：同为 sheaf 方法，但仍受限于无向图框架
- 启发：有向图结构天然适合建模非对称关系（如社交网络的关注/被关注），该框架可推广至知识图谱推理

## 评分

- 新颖性: ⭐⭐⭐⭐ (有向 sheaf + 协作行为的结合具有理论深度)
- 实验充分度: ⭐⭐⭐⭐ (13+ 真实任务 + 合成验证)
- 写作质量: ⭐⭐⭐⭐ (理论刻画精确，逻辑清晰)
- 价值: ⭐⭐⭐⭐ (统一了 sheaf 扩散与协作 GNN 两条线)
