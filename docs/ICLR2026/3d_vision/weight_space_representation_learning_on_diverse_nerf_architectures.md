# Weight Space Representation Learning on Diverse NeRF Architectures

**会议**: ICLR 2026  
**arXiv**: [2502.09623](https://arxiv.org/abs/2502.09623)  
**代码**: 有（论文提供链接）  
**领域**: 3D 视觉 / NeRF  
**关键词**: NeRF, weight space, graph meta-network, contrastive learning, architecture-agnostic

## 一句话总结
提出首个能处理多种 NeRF 架构（MLP/tri-plane/hash table）权重的表示学习框架，通过 Graph Meta-Network 编码器 + SigLIP 对比损失构建架构无关的潜在空间，在 13 种 NeRF 架构上实现分类、检索和语言任务，并能泛化到训练时未见的架构。

## 研究背景与动机

1. **领域现状**：NeRF 将 3D 信息编码到网络权重中，nf2vec 和 Cardace 等方法通过处理 NeRF 权重进行下游任务（分类、检索），但限定于单一 NeRF 架构（仅 MLP 或仅 tri-plane）。
2. **现有痛点**：NeRF 架构多样化快速发展（MLP→tri-plane→hash table），每种新架构都需要重新设计处理框架，限制了实用性。
3. **核心矛盾**：不同 NeRF 架构的权重结构差异巨大（MLP 权重矩阵 vs 平面特征 vs hash table），如何构建统一的表示空间？
4. **本文要解决什么？** 设计架构无关的 NeRF 权重处理框架，使同一物体的不同 NeRF 表示被映射到相近的潜在向量。
5. **切入角度**：利用 Graph Meta-Network 将任意 NeRF 转为参数图（parameter graph），然后用 GNN 处理。
6. **核心 idea 一句话**：用 SigLIP 对比损失对齐同一物体不同架构 NeRF 的 embedding，使 GMN 编码器产生架构无关的潜在空间。

## 方法详解

### 整体框架
将 NeRF 权重转为参数图（graph），用 GMN 编码器提取潜在向量，用 nf2vec 解码器重建辐射场。训练使用渲染损失 $\mathcal{L}_R$ + SigLIP 对比损失 $\mathcal{L}_C$。推理时编码器输出的向量直接用于分类/检索/语言任务。

### 关键设计

1. **参数图构建（NeRF→Graph）**:
   - 做什么：将三种 NeRF 架构转为统一的图表示
   - 核心思路：MLP 用标准参数图表示（权重为边特征）；tri-plane 沿用 Lim 等人的空间参数网格表示；**hash table（本文新贡献）**——为每个 table entry 和每个特征维度各创建节点，entry-feature 对应的值作为边特征，避免了显式建模底层体素网格的立方级复杂度
   - 设计动机：hash table 是当前最流行的 NeRF 架构，必须支持

2. **GMN 编码器**:
   - 做什么：从参数图提取架构无关的潜在向量
   - 核心思路：标准消息传递 GNN，节点和边特征通过邻域聚合更新，最终对边特征做平均池化得到 embedding
   - 设计动机：GNN 天然对节点排列等变，可处理任意图结构，因而可处理任意 NeRF 架构

3. **SigLIP 对比损失**:
   - 做什么：在潜在空间中对齐同一物体的不同架构 NeRF
   - 核心思路：$\mathcal{L}_C = -\frac{1}{|\mathcal{B}|} \sum_{j,k} \ln \frac{1}{1+e^{-\ell_{jk}(t \mathbf{u}_j \cdot \mathbf{v}_k + b)}}$，其中 $\ell_{jk}=1$ 表示同一物体，$-1$ 表示不同物体
   - 设计动机：仅用渲染损失会导致不同架构的 NeRF 在潜在空间中形成按架构聚类（而非按内容聚类），对比损失打破架构壁垒

### 损失函数
$\mathcal{L}_{R+C} = \mathcal{L}_R + \lambda \mathcal{L}_C$，$\lambda = 2 \times 10^{-2}$

## 实验关键数据

### 主实验（多架构分类准确率）

| 设置 | 训练架构 | 测试架构 | 准确率 |
|------|---------|---------|--------|
| 单架构 MLP | MLP | MLP | ~82% |
| 单架构 TRI | TRI | TRI | ~84% |
| 单架构 HASH | HASH | HASH | ~83% |
| 多架构 ALL ($\mathcal{L}_{R+C}$) | MLP+TRI+HASH | MLP+TRI+HASH | ~83% |
| 多架构→未见架构 | MLP+TRI+HASH | 10种未见变体 | ~78% |

### 消融实验

| 损失 | 多架构分类 | 跨架构检索 | 说明 |
|------|-----------|-----------|------|
| $\mathcal{L}_R$ only | 架构内聚类 | 极低 | 不同架构形成独立簇 |
| $\mathcal{L}_C$ only | ~79% | 高 | 缺乏渲染约束 |
| $\mathcal{L}_{R+C}$ | ~83% | 最高 | 最优组合 |

### 关键发现
- **仅渲染损失导致架构聚类**：t-SNE 可视化清楚显示不同架构的 NeRF 即使表示同一物体也被分到不同簇
- **对比损失是关键**：添加 SigLIP 后潜在空间按物体类别组织而非按架构
- **泛化到未见架构**：在 10 种未见超参架构上保持 ~78% 准确率
- **首次处理 hash table NeRF**：验证了参数图表示的通用性

## 亮点与洞察
- **参数图对 hash table 的设计很精巧**：避免立方级复杂度，保持 hash table 本身的内存效率
- **对比损失打破架构壁垒的洞察深刻**：渲染损失只学"内容"但会混入"架构"信息，SigLIP 显式约束"同物体不同架构应相近"
- **对 NeRF 数据格式标准化有推动意义**：如果不同架构的 NeRF 可以统一检索，那么 NeRF 可能成为 3D 数据的通用存储格式

## 局限性 / 可改进方向
- 仅在 ShapeNet 合成数据上验证，真实场景的 NeRF 更复杂
- 三种架构族之间的跨族泛化未充分测试（如 MLP 训练→HASH 测试）
- hash table 的参数图不保留空间邻接关系
- 未涉及 3DGS 这一重要新表示

## 相关工作与启发
- **vs nf2vec**: 只能处理固定 MLP，本文扩展到任意架构
- **vs Cardace et al.**: 只能处理 tri-plane，本文统一三族
- 对元学习（meta-network）在 3D 领域的应用有开创意义

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个架构无关的 NeRF 权重处理框架
- 实验充分度: ⭐⭐⭐⭐ 13 种架构覆盖广泛，但仅 ShapeNet 数据
- 写作质量: ⭐⭐⭐⭐ 方法清晰，消融充分
- 价值: ⭐⭐⭐⭐ 对 NeRF 统一处理有重要推动
