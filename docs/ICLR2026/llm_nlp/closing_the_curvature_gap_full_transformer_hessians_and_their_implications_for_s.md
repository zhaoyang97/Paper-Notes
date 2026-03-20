# Closing the Curvature Gap: Full Transformer Hessians and Their Implications for Scaling Laws

**会议**: ICLR 2026  
**arXiv**: [2510.16927](https://arxiv.org/abs/2510.16927)  
**代码**: https://github.com/modernTalker/transformer_hessian (有)  
**领域**: LLM/NLP 理论  
**关键词**: Transformer Hessian, LayerNorm, scaling laws, loss landscape, optimization theory

## 一句话总结
首次推导完整 Transformer block（含 LayerNorm 和 FFN）的显式 Hessian 表达式及谱范数上界，建立了损失面随数据量增加以 $O(1/k)$ 速率收敛的理论框架，为 scaling laws 和曲率感知训练提供了数学基础。

## 研究背景与动机

1. **领域现状**：Transformer 的经验成功背后有 neural scaling laws 描述的可预测改善规律。已有工作推导了 self-attention 的 Hessian 表达式，但 LayerNorm 和 FFN 的二阶分析一直缺失。
2. **现有痛点**：没有完整的 Transformer block Hessian 意味着：(1) 无法完整理解优化地形如何随数据量变化；(2) 无法从理论上解释曲率在不同子层间的传播；(3) 缺乏 scaling laws 的数学基础。
3. **核心矛盾**：LayerNorm 和 FFN 的非线性使得二阶导数推导非常复杂，此前的理论工作只能分析 self-attention，留下了"曲率空白"。
4. **本文要解决什么**：推导包含 LayerNorm 和 FFN 的完整 Transformer block 的 Jacobian 和 Hessian，建立损失面收敛的理论界。
5. **切入角度**：使用行向量化 $\text{vec}_r(\cdot)$ 框架和 Gauss-Newton 分解，将 Hessian 系统地分解为各子层的贡献，逐层推导。
6. **核心 idea 一句话**：通过显式推导 LayerNorm 和 FFN 的 Hessian 来补全 Transformer 的二阶理论，并用 Taylor 展开分析损失面随数据量的收敛行为。

## 方法详解

### 整体框架

理论推导链：Self-Attention Hessian（已有） → LayerNorm Jacobian/Hessian（Theorem 2-3） → ReLU FFN 导数（Lemma 1） → 完整 Transformer Block Hessian（Theorem 4-5） → 谱范数上界（Theorem 1, 6） → 损失面收敛定理（Theorem 7）。

Transformer block 定义（post-norm）：
$$\mathbf{Y} = \text{LayerNorm}(\mathbf{X} + \mathbf{F}(\mathbf{X}))$$
$$\mathbf{Z} = \text{LayerNorm}(\mathbf{Y} + \text{FFN}(\mathbf{Y}))$$

### 关键设计

1. **LayerNorm 的 Jacobian 和 Hessian（Theorem 2-3）**:
   - 做什么：推导 LayerNorm 相对于输入的一阶和二阶导数
   - 核心思路：将 LayerNorm 分解为 $\text{LN}(\mathbf{X}) = \mathbf{P}(\mathbf{X})\mathbf{M}(\mathbf{X})$，其中 $\mathbf{M}$ 是中心化（减均值），$\mathbf{P}$ 是逆标准差对角矩阵。利用乘积法则得 Jacobian = 两项之和（$\mathbf{P}$ 对 centering 的缩放 + $\mathbf{M}$ 对 $\mathbf{P}$ 变化的贡献）。Hessian 通过进一步对 Jacobian 求导得到，$\frac{\partial^2 \mathbf{M}}{\partial \mathbf{X}^2} = 0$（centering 是线性的），但 $\mathbf{P}$ 的二阶导不为零。
   - 设计动机：LayerNorm 的 Hessian 此前未被推导过，它通过 per-row variance 贡献曲率，是理解 Transformer 优化地形的关键缺失组件。

2. **完整 Transformer Block 的 Hessian（Theorem 4-5）**:
   - 做什么：组装含 Self-Attention + LayerNorm + FFN + residual 的完整 block Hessian
   - 核心思路：设 $\mathbf{S} = \text{ReLU}(\mathbf{Y}\mathbf{W}_1)\mathbf{W}_2 + \mathbf{Y}$（FFN + residual），$\mathbf{Z} = \text{LN}(\mathbf{S})$。利用链式法则：
     $$\mathbf{H}_{\text{tr}}^{(i,j)} = (\mathbf{J}_Z \otimes \mathbf{I}_{n_i})\bm{\xi}_{ij} + (\mathbf{I}_{Ld_V} \otimes \mathbf{B}_i^\top)\mathbf{H}_Z\mathbf{B}_j$$
     其中 $\mathbf{J}_Z$ 是 LN 的 Jacobian，$\mathbf{H}_Z$ 是 LN 的 Hessian，$\bm{\xi}_{ij}$ 是 $\mathbf{S}$ 的二阶混合导数，$\mathbf{B}_i$ 是 $\mathbf{S}$ 对参数的 Jacobian。
   - 设计动机：Gauss-Newton 分解允许将损失的 Hessian 分解为"外积项"（一阶信息）+"函数 Hessian 项"（二阶信息），两者分别对应不同的优化特性。

3. **谱范数上界（Theorem 1, 6）**:
   - 做什么：为 Self-Attention 和完整 Transformer Block 的 Hessian 提供谱范数的显式上界
   - 核心思路：利用 Kronecker 积和矩阵范数的亚乘性，将 Hessian 范数界分解为输入范数 $\|\mathbf{X}\|_2$、权重范数 $\|\mathbf{W}\|_2$、序列长度 $L$、维度 $d_V, d_K$ 等因素的函数。完整 block 的界 $\leq 5 \max_{i,j}(\cdots)$（5 = $\sqrt{m_b n_b}$，5 个参数组）。
   - 设计动机：显式上界揭示了各子层对整体曲率的贡献——Value 和 Key 相关项通过 softmax 导数占主导，FFN 由 ReLU 的分段线性性控制（Hessian 几乎处处为零），LayerNorm 通过 per-row variance 贡献。

4. **损失面收敛定理（Theorem 7）**:
   - 做什么：建立损失函数随数据量增加的收敛速率
   - 核心思路：利用 Taylor 展开和 Hessian 界证明：
     $$|\mathcal{L}_{k+1}(\mathbf{w}) - \mathcal{L}_k(\mathbf{w})| \leq \frac{2L}{k+1} + \frac{M\|\mathbf{w} - \mathbf{w}^*\|_2^2}{k+1}$$
     其中 $M$ 来自 Theorem 1/6 的 Hessian 谱范数界。这说明损失面的变化以 $O(1/k)$ 速率衰减。
   - 设计动机：从理论上解释了数据量增加时损失地形趋于稳定的经验观察，为"何时从数据扩展转向模型扩展"提供了判断依据。

### 损失函数 / 训练策略

- 理论分析使用 MSE loss：$l(\cdot, \text{Target}) = \frac{1}{Ld_V}\|\cdot - \text{Target}\|_F^2$
- 实验在 ViT 上验证，在 MNIST（1 block, dim=16）和 CIFAR-100（8 blocks, dim=128）上训练

## 实验关键数据

### 主实验

Hessian 结构验证（MNIST, 1 Transformer block）：

| 观察 | 结果 |
|------|------|
| 初始化模型 Hessian | 条目幅度高度不均匀，Value 相关 block 最大 |
| 训练后 Hessian | 所有 block 幅度增大，Value-Value block 仍占主导 |
| 参数 block 范数排序 | Key, Value >> Query, W1, W2 |

### 消融实验

损失面收敛验证（CIFAR-100, 8 blocks, log-log scale）：

| 数据量 k | $|\mathcal{L}_{k+1} - \mathcal{L}_k|$ 趋势 |
|---------|--------------------------------------|
| 小 k | 变化大，不稳定 |
| 大 k | 近似线性下降（log-log），符合 $O(1/k)$ |

### 关键发现
- **Value-Value Hessian 占主导**：训练前后都是 Hessian 中幅度最大的 block，说明 Value 矩阵的曲率最大，对优化影响最深——这从理论上解释了为什么 Adam 对 Transformer 重要（不同参数的曲率差异巨大，需要自适应学习率）。
- **FFN Hessian 受 ReLU 控制**：ReLU 的二阶导几乎处处为零，FFN 的曲率主要来自一阶项的组合而非自身的非线性。
- **LayerNorm 通过 per-row variance 贡献曲率**：variance 越小（特征越相似），LayerNorm 的曲率越大，可能导致训练不稳定。
- **$O(1/k)$ 收敛速率实验验证**：CIFAR-100 上的 log-log 图显示损失差的 EMA 接近线性下降，符合理论预测。

## 亮点与洞察
- **填补了关键理论空白**：Self-Attention 的 Hessian 已有，但 LayerNorm 和 FFN 的缺失使得此前的分析不完整。本文补齐后可以做端到端的曲率分析。
- **Block 异质性 Hessian 的实践意义**：不同参数 block 的曲率差异巨大（Value >> Query），这从理论上支持了 per-block learning rate（不同参数用不同学习率）和 curvature-aware preconditioning 的必要性。
- **$O(1/k)$ 收敛为数据预算提供依据**：当曲率趋于平稳时，增加数据的边际收益递减，理论上可以在此拐点从数据扩展切换到模型扩展。

## 局限性 / 可改进方向
- **局部分析**：Taylor 展开和 Assumption 1（共享最小值）仅在局部成立，对全局优化地形的描述有限。
- **单 block 分析**：理论推导针对单个 Transformer block，未扩展到多层堆叠（层间 Hessian 传播未分析）。
- **post-norm + MSE only**：理论基于 post-norm（现代 LLM 多用 pre-norm）和 MSE loss（实际用 cross-entropy）。虽然论文声称可扩展到 CE loss，但未在理论中给出。
- **实验规模偏小**：MNIST 和 CIFAR-100 上的 ViT 远非现代 LLM 的规模，理论预测在大规模模型上是否仍准确有待验证。
- **$M$ 不是真常数**：Theorem 7 中的 $M$ 依赖于 $\|\mathbf{X}\|_2$，随数据变化，严格来说不是 $O(1/k)$。

## 相关工作与启发
- **vs Zhang et al. (NeurIPS 2024，"Why Transformers Need Adam")**：同样从 Hessian 角度分析 Transformer 优化，但本文推导了完整 block（含 LN 和 FFN），更全面。
- **vs Ormaniec et al. (2024)**：分析 Self-Attention block 的 Hessian 分解，本文在此基础上扩展到完整 Transformer。
- **vs Kaplan et al. / Hoffmann et al. (Scaling Laws)**：经验性的 scaling laws，本文提供了从曲率角度理解 scaling 的理论工具。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次完成 Transformer 完整 Hessian 推导，填补明确的理论空白。
- 实验充分度: ⭐⭐⭐ 实验仅在 MNIST/CIFAR-100 小模型上验证，缺少大规模实验支撑。
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，结构清晰；但公式密集，可读性对非理论方向读者不友好。
- 价值: ⭐⭐⭐⭐ 为 Transformer 优化理论奠定新基础，但需在大规模模型上进一步验证实用性。
