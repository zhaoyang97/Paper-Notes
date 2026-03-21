# RoadPainter: Points Are Ideal Navigators for Topology Transformer

**会议**: ECCV 2024
**arXiv**: [2407.15349](https://arxiv.org/abs/2407.15349)
**代码**: 无
**领域**: 自动驾驶
**关键词**: 拓扑推理, 车道中心线检测, BEV感知, 实例分割, SD地图

## 一句话总结

提出 RoadPainter，通过先回归车道中心线点再利用实例 mask 精炼的两阶段策略，结合混合注意力机制和真实-虚拟车道分离策略，在 OpenLane-V2 数据集上实现 SOTA 的拓扑推理性能。

## 研究背景与动机

1. **领域现状**：自动驾驶中的拓扑推理旨在从多视图图像中提取车道中心线及其拓扑连接关系，为下游轨迹预测和规划提供路由信息。近年来从 2D 车道检测发展到 BEV 空间的在线向量化地图构建。

2. **现有痛点**：TopoNet 等方法通过直接回归预测中心线点，在高曲率区域（如弯道、匝道）的几何精度不足。回归方法倾向于学习直线形状的中心线，难以捕捉复杂的几何细节。同时，交叉路口中的虚拟车道与实际车道的特性差异很大，统一处理导致性能受限。

3. **核心矛盾**：回归提供稳定的初始定位但几何精度差，分割提供精确的几何细节但定位不稳定——如何结合两者优势？

4. **本文要解决什么**：提升高曲率区域的中心线检测精度，同时改善拓扑推理性能。

5. **切入角度**：先用 transformer decoder 回归粗略中心线点，再用这些点引导生成实例 mask，从 mask 中采样新点与回归点融合精炼。

6. **核心 idea**："Points Are Ideal Navigators"——用回归的点引导 mask 生成，再用 mask 反馈优化点的位置，实现回归与分割的优势互补。

## 方法详解

### 整体框架

多视图图像 → 图像骨干（ResNet-50）提取多尺度特征 → BEVFormer 构建 BEV 特征 → （可选）SD Map Interaction 增强 BEV 特征 → 混合注意力 Transformer Decoder 回归中心线点 + 拓扑关系 → Points-Guided Mask Generation 生成实例 mask → Points-Mask Fusion 精炼中心线 → 输出精确中心线与拓扑矩阵

### 关键设计

1. **混合注意力 Transformer Decoder + 真实-虚拟分离策略（RVS）**:
   - 做什么：从 BEV 特征中检测车道中心线实例并建立拓扑关联
   - 核心思路：decoder 包含三种注意力——masked cross-attention 聚合 mask 区域特征、deformable cross-attention 聚合可学习采样点特征、self-attention 促进 query 交互。关键创新是 RVS 自注意力：实际车道和虚拟车道使用独立 query，在自注意力中虚拟 query 可以看到实际 query，但实际 query 不看虚拟 query：
   $$\text{RVSelfAttn} = \text{softmax}\left(\frac{\begin{bmatrix} Q^r Q^{rT} & -\infty \\ Q^v Q^{rT} & Q^v Q^{vT} \end{bmatrix}}{\sqrt{C}}\right) \begin{bmatrix} Q^r \\ Q^v \end{bmatrix}$$
   - 设计动机：虚拟车道（交叉路口连接线）的位置依赖于实际车道，但实际车道位置不依赖虚拟车道。通过非对称注意力 mask 编码这种先验知识

2. **Points-Guided Mask Generation（PGM）**:
   - 做什么：利用回归的中心线点引导生成每条中心线的实例分割 mask
   - 核心思路：将回归出的中心线点 $\mathbf{l}_i \in \mathbb{R}^{K \times 3}$ 通过位置编码 MLP 和 query 编码 MLP 生成 mask query $\mathbf{Q}_i'$，然后与 BEV 特征做点积：$\mathbf{M}_i = \mathbf{B} \cdot \mathbf{Q}_i'$。相比 Mask2Former 使用无位置先验的可学习 query，PGM 利用回归点的空间位置信息引导 mask 生成
   - 设计动机：分割 mask 可以捕获中心线的精细几何形状（尤其是高曲率区域），但纯分割方法定位不稳定。使用回归点作为 mask 的位置先验，兼顾稳定性与精度

3. **Points-Mask Fusion（PMF）**:
   - 做什么：从 mask 中采样点，与回归点融合得到精炼的中心线
   - 核心思路：分两步——(1) Mask Points Sampling：对 mask 逐列做 softmax 回归一个点 $\mathbf{C}_{i,j} = [0,1,...,H-1]^T \cdot \text{softmax}(\mathbf{M}_i(:,j))$，同时预测每个点的存在概率 $\mathbf{P}_i$ 和方向概率 $D_i$；(2) Points Fusion：过滤异常点（距邻居点 >1.5m），对有效 mask 点重采样为 $K$ 个点后与回归点取平均。注意虚拟车道不做 mask 精炼（因缺乏视觉信息）
   - 设计动机：mask 提供更精细的几何信息弥补回归的不足，简单取平均即可有效融合两种表示

4. **SD Map Interaction（可选模块）**:
   - 做什么：利用标清地图（SD Map）的先验信息增强 BEV 特征
   - 核心思路：将 SD Map 的向量化实例转换为 BEV 语义特征 $\mathbf{E}_S$，通过 transformer decoder 与在线 BEV 特征交互：$\hat{\mathbf{B}} = \text{TrDec}(\mathbf{B}, \mathbf{E}_S + \mathbf{E}_P)$
   - 设计动机：解决遮挡和有限感知距离的问题，SD Map 提供超视距的道路形状先验

5. **拓扑关联头**:
   - 做什么：预测中心线之间的拓扑连接矩阵 $\mathbf{A}_{ll} \in [0,1]^{N_L \times N_L}$
   - 核心思路：将 query 与位置信息融合 $\mathbf{E}_i = \psi_1(\mathbf{Q}_i) + \psi_2(\mathbf{l}_i)$，拼接后通过二值分类器预测拓扑关系

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{top}(\mathbf{A}_{ll}) + \mathcal{L}_{cls}(\mathbf{S}) + \mathcal{L}_{det}(\mathbf{L}_V, \mathbf{L}_R) + \mathcal{L}_{mask}(\mathbf{M}) + \mathcal{L}_{mp}(\mathbf{C}, \mathbf{P}, D)$$

- $\mathcal{L}_{top}$, $\mathcal{L}_{cls}$: Focal Loss 监督拓扑关系和中心线置信度
- $\mathcal{L}_{det}$: L1 Loss 监督中心线几何形状
- $\mathcal{L}_{mask}$: BCE + Dice Loss 监督实例 mask
- $\mathcal{L}_{mp}$: L1 + BCE + Focal Loss 监督 mask 采样点、存在概率和方向
- 实际/虚拟中心线分别做二部图匹配和损失计算
- AdamW 优化器，初始学习率 $2 \times 10^{-4}$，训练 24 epochs，cosine annealing
- 梯度裁剪（max norm = 35），backbone 学习率为其他模块的十分之一

## 实验关键数据

### 主实验

| 数据集/方法 | DET_l ↑ | DET_t ↑ | TOP_ll ↑ | TOP_lt ↑ | OLS ↑ |
|------------|---------|---------|----------|----------|-------|
| **subset_A** | | | | | |
| TopoNet | 28.5 | 48.1 | 4.1 | 20.8 | 35.6 |
| TopoMLP | 28.3 | 50.0 | 7.2 | 22.8 | 38.2 |
| RoadPainter | **30.7** | 47.7 | **7.9** | **24.3** | **38.9** |
| SMERF* (SD map) | 33.4 | 48.6 | 7.5 | 23.4 | 39.4 |
| RoadPainter* (SD map) | **36.9** | 47.1 | **12.7** | **25.8** | **42.6** |
| **subset_B** | | | | | |
| TopoNet | 24.3 | 55.0 | 2.5 | 14.2 | 33.2 |
| TopoMLP | 26.6 | 58.3 | 7.6 | 17.8 | 38.7 |
| RoadPainter | **28.7** | 54.8 | **8.5** | 17.2 | 38.5 |

### 消融实验

| PGM | PMF | SD | DET_l | TOP_ll | OLS | AP_l | 说明 |
|-----|-----|-----|-------|--------|-----|------|------|
| ✗ | ✗ | ✗ | 26.9 | 7.7 | 37.2 | - | Baseline |
| ✓ | ✗ | ✗ | 28.1 | 7.9 | 37.6 | 13.5 | +PGM: mask 监督提升 DET_l +1.2 |
| ✓ | ✓ | ✗ | 30.7 | 7.9 | 38.9 | 14.1 | +PMF: mask 精炼提升 DET_l +2.6 |
| ✓ | ✓ | ✓ | 36.9 | 12.7 | 42.6 | 15.4 | +SD Map: 大幅提升 DET_l +6.2 |

| 注意力消融 | DET_l | TOP_ll | 说明 |
|-----------|-------|--------|------|
| RoadPainter (full) | 30.7 | 7.9 | - |
| w/o hybrid attention | 29.6 | 7.2 | 混合注意力贡献 DET_l +1.1 |
| w/o real-virtual self-attn | 29.6 | 7.5 | RVS 对 TOP_ll 影响明显 (-0.4) |

### 关键发现

- **Points-Mask Fusion 是核心贡献**：PMF 模块（从 mask 采样点与回归点融合）贡献了 DET_l +2.6 的提升，远超 PGM 的 +1.2，说明"回归+分割互补"策略确实有效
- **SD Map 提升最为显著**：DET_l +6.2, TOP_ll +4.8，超视距信息对拓扑推理价值巨大
- **RVS 策略主要提升拓扑推理**：对 TOP_ll 贡献 +0.4，符合虚拟车道依赖实际车道的先验
- **分割 mask 在高曲率场景优势明显**：可视化结果表明弯道和交叉口的中心线精度显著提升
- **Baseline 的 TOP_ll (7.7) 已超过 TopoNet (4.1)**，说明拓扑关联头的设计本身就很有效

## 亮点与洞察

- **回归与分割互补的思路**非常通用，可迁移到其他需要精确几何预测的任务（如 3D 物体检测、关键点检测等）
- **真实-虚拟车道分离**是对领域知识的精确建模，通过非对称注意力 mask 实现，简单优雅
- **按列回归 mask 点**的设计巧妙解决了从热力图到有序点集的转换，避免了复杂的后处理
- **存在概率 + 方向概率**的预测解决了 mask 点数量可变和方向不确定的问题，实现了端到端训练
- **SD Map 交互模块**可插拔设计，为没有 SD Map 的场景也保持了竞争力

## 局限性 / 可改进方向

- DET_t（交通标志检测）和 TOP_lt（车道-标志拓扑）指标不如 TopoMLP，因为使用了 3 层 BEVFormer 而非 6 层 PETR，后者更适合检测交通标志
- mask 精炼仅对实际车道有效，虚拟车道（缺乏视觉信息）未能受益
- 完全垂直的中心线需要额外的按行采样机制处理，增加了复杂性
- FPS 仅 6.5（RTX3090, FP32），实时性仍有提升空间
- Points-Mask Fusion 使用简单平均，更自适应的加权策略可能进一步提升

## 相关工作与启发

- **vs TopoNet**: RoadPainter 在 DET_l 上提升 +2.2（+7.7%），关键在于 mask 精炼弥补了纯回归的不足
- **vs TopoMLP**: OLS 上略优但 DET_t 和 TOP_lt 弱于 TopoMLP，原因在于 BEV 构建方式不同（BEVFormer vs PETR）
- **vs Mask2Former**: 借鉴了 mask attention 作为 BEV 特征聚合方式，但创新了 points-guided mask query（比可学习 query 有更好的位置先验）
- **vs MapTR**: MapTR 关注向量化地图元素但不处理拓扑关系，RoadPainter 在此基础上增加拓扑推理

## 评分

- 新颖性: ⭐⭐⭐⭐ 回归与分割互补的思路虽非首创，但 points-guided mask + mask-based refinement 的闭环设计在拓扑推理中是创新的
- 实验充分度: ⭐⭐⭐⭐ 两个子集对比、多组消融、注意力消融、可视化分析均有，但缺少与更多方法的对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观，公式规范
- 价值: ⭐⭐⭐⭐ 实际提升显著且思路可迁移，SD Map 集成对工业界有参考价值
