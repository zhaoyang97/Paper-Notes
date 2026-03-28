# What Expressivity Theory Misses: Message Passing Complexity for GNNs

**会议**: NeurIPS 2025
**arXiv**: [2509.01254](https://arxiv.org/abs/2509.01254)
**代码**: https://www.cs.cit.tum.de/daml/message-passing-complexity/
**领域**: GNN 理论 / 表达力
**关键词**: 消息传递复杂度, lossyWL, GNN表达力, 过压缩, 连续度量

## 一句话总结
批判 GNN 的二值表达力理论无法解释实际性能差异，提出 MPC——基于概率性 lossyWL 的连续、任务特定复杂度度量，与准确率的 Spearman 相关性达 -1（传统 WLC 恒为零），成功解释了 GCN+虚拟节点为何在长程任务上优于更高表达力的高阶模型。

## 研究背景与动机

1. **领域现状**：GNN 表达力理论以 WL 同构测试为核心。大量工作追求超越 WL 的更高表达力。

2. **现有痛点**：表达力理论是二值的（能/不能），无法解释同等表达力模型的性能差异；benchmark 中几乎所有图对已被 WL 区分，更高表达力不应带来额外好处。

3. **核心矛盾**：GCN+虚拟节点（不增加 WL 表达力）在长程任务上优于超越 WL 的高阶模型——表达力理论无法解释。

4. **本文要解决什么？** 提出既保留不可能性结论又能解释实际性能差异的连续复杂度框架。

5. **切入角度**：将 WL 测试概率化——每条消息以随机游走概率独立存活（lossyWL）。

6. **核心 idea**：lossyWL 的信息保留概率的负对数 = MPC = 任务特定、架构特定、连续的学习难度度量。

## 方法详解

### 整体框架
lossyWL：标准 WL 的概率化版本，每条消息 $m_{u \to v}$ 以概率 $I_{vu}$ 存活。MPC = $-\log P[lossyWL_v^L(G) \vDash f_v(G)]$。

### 关键设计

1. **lossyWL**:
   - 做什么：建模实际 MPNN 中消息传递的有损性
   - 核心思路：每层每条消息独立以 $I_{vu}$ 概率存活，节点颜色成为随机变量
   - 设计动机：标准 WL 假设无损传播，但实际中过压缩/过平滑/under-reaching 导致信息丢失

2. **MPC 的理论性质**:
   - 不可行性保留：$MPC = \infty$ 当且仅当 WL 也无法区分
   - 函数精细化：更精细的任务复杂度更高
   - 过压缩下界：$MPC \geq -\log(I^L_{vu})$

3. **推广到任意架构**:
   - VN/rewiring/高阶图等在变换后的消息传递图上执行 lossyWL

## 实验关键数据

### 保持初始特征任务（过平滑代理）

| 架构 | MPC vs Accuracy Spearman ρ | WLC ρ |
|------|--------------------------|--------|
| GCN/GIN/GCN-VN/GSN | **ρ = -1**（完美负相关） | **ρ = 0** |

### 长程信息传播任务

| 架构 | 表达力 | MPC | 性能 |
|------|--------|-----|------|
| GCN | WL-等价 | 高 | 差 |
| GCN-VN | WL-等价 | **低** | **好** |
| CIN | 超越 WL | 高 | 差 |
| FragNet | 超越 WL | 高 | 差 |

### 关键发现
- MPC 完美负相关于准确率，WLC ρ = 0
- GCN+VN 在长程任务上超越更高表达力的模型

## 亮点与洞察
- “追求更高表达力可能是误导性的”——成功的关键是最小化任务特定的 MPC
- lossyWL 的设计优雅：只修改一个假设就从二值理论变为连续理论

## 局限性
- 不建模学习动态（梯度/优化难度）
- Monte Carlo 模拟对大图可能计算昂贵
- 仅关注节点级任务

## 相关工作
- **vs WL 表达力理论**: 二值理论，本文证明其对实际性能解释力不足
- **vs 过压缩文献**: 过压缩是 MPC 的一个下界

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ lossyWL + MPC 是 GNN 理论的重要进展
- 实验充分度: ⭐⭐⭐⭐⭐ 多任务 × 多架构 × ZINC 验证
- 写作质量: ⭐⭐⭐⭐⭐ 论点犌利，理论严谨
- 价值: ⭐⭐⭐⭐⭐ 重新定义了理解和改进 GNN 的方式

### 补充技术细节
- lossyWL 中 Bernoulli 存活概率 $I_{vu}$ 即归一化邻接矩阵元素，对应随机游走转移概率
- MPC 的不可行性定理：$MPC = \infty$ 当且仅当存在 $G', w$ 使得 $f_v(G) \neq f_w(G')$ 但所有 $M \in \mathcal{M}$ 都有 $M_v(G) = M_w(G')$
- 函数精细化定理：如果 $f \vDash g$（f 比 g 更精细），则 $MPC(f_v, G) \geq MPC(g_v, G)$
- 任务三角不等式：$MPC(f_v \| g_v, G) \leq MPC(f_v, G) + MPC(g_v, G)$
- 在 3-正则随机图上验证，结果可迁移到 ZINC 和 Long Range Graph Benchmark
- 保持初始特征任务的复杂度至少线性增长：$\mathbb{E}[MPC] \in \Omega(L)$（Lemma retaining）
- 为广泛架构分析：包括 MLP、GCN、GIN、GraphSage、GCN-VN、GSN、FragNet、CIN 等 8 种
- CIN 和 GraphSage 由于显式残差连接保持了完美准确率——MPC 框架抽象了实现细节
- Table 1 显示 WL 已经区分了流行基准中几乎所有图对，质疑了更高表达力的实际必要性
- 可通过 Monte Carlo 模拟计算 MPC，对每个 (graph, node, task) 元组给出复杂度值
- 虚拟节点不增加 WL 表达力但降低了 MPC（所有节点通过 VN 一步可达），这解释了其实际效果
- MPC 的计算对单个 (graph, node, task) 元组非常快，但大规模评估需要采样
- 结果在 ZINC 和 Long Range Graph Benchmark 上进一步验证了迁移性
- MPC 框架自然推广到任意 MP 图变换（包括 rewiring/高阶图），见附录 B.1
