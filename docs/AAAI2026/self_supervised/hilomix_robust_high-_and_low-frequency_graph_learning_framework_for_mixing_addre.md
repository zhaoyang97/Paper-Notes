---
title: >-
  [论文解读] HiLoMix: Robust High- and Low-Frequency Graph Learning Framework for Mixing Address Association
description: >-
  [AAAI 2026][自监督学习][图神经网络] 提出 HiLoMix，一种针对混币地址关联任务的鲁棒图学习框架，通过异质属性混合交互图（HAMIG）、频率感知图对比学习和基于置信度的标签加权监督学习，分别解决图稀疏、标签稀缺和标签噪声三大挑战，在 F1、AUC、MRR 上分别超越次优基线 5.69%、7.34% 和 15.61%。
tags:
  - "AAAI 2026"
  - "自监督学习"
  - "图神经网络"
  - "区块链分析"
  - "频率感知对比学习"
  - "标签噪声"
  - "Tornado Cash"
---

# HiLoMix: Robust High- and Low-Frequency Graph Learning Framework for Mixing Address Association

**会议**: AAAI 2026  
**arXiv**: [2511.07759](https://arxiv.org/abs/2511.07759)  
**代码**: 无  
**领域**: 自监督  
**关键词**: 图神经网络, 区块链分析, 频率感知对比学习, 标签噪声, Tornado Cash

## 一句话总结

提出 HiLoMix，一种针对混币地址关联任务的鲁棒图学习框架，通过异质属性混合交互图（HAMIG）、频率感知图对比学习和基于置信度的标签加权监督学习，分别解决图稀疏、标签稀缺和标签噪声三大挑战，在 F1、AUC、MRR 上分别超越次优基线 5.69%、7.34% 和 15.61%。

## 研究背景与动机

随着混币服务（如 Tornado Cash）被恶意行为者用于非法资金转移，**混币地址关联（Mixing Address Association）**成为关键的安全研究任务。Tornado Cash 允许用户将资产存入资金池并匿名提取，切断交易链路，但也被用于洗钱——据 Elliptic 报告，其在被制裁前已帮助转移超过 15 亿美元非法资金。

现有方法面临三大挑战：

**标签噪声（Label Noise）**：可用标签通常由启发式规则生成，不可避免地包含噪声，在训练中引入歧义，损害泛化能力

**标签稀缺（Label Scarcity）**：地面真值关联极其有限，模型无法建立全面的决策边界

**图结构稀疏**：地址关联稀少导致图结构高度稀疏，许多节点孤立，限制了消息传递和特征聚合的效率

现有方法的具体不足：
- **启发式方法**（基于时间、gas 价格等交易特征）：可扩展性差，只能捕获粗心用户，无法检测主动规避检测的恶意行为者
- **BERT4ETH**：忽略账户间的结构关系，只建模交易序列
- **MixBroker**：将地址交互建模为 MIG 图，但受限于图稀疏和标签噪声

## 方法详解

### 整体框架

HiLoMix 包含四个核心组件：
1. **HAMIG 构建**：异质属性混合交互图，丰富拓扑结构
2. **频率感知图对比学习**：从高低频视图对比学习鲁棒节点表示
3. **置信度监督**：为噪声标签分配自适应权重
4. **Stacking 集成**：融合多个异质模型的预测

### 关键设计

#### 1. **HAMIG 构建（解决图稀疏）**

将 MixBroker 提出的 MIG（仅有地址关联边）扩展为异质图：

$$\mathcal{G} = (\mathcal{V}_a, \mathcal{V}_t, \mathcal{E}_{at}, \mathcal{E}_{aa}, \mathcal{X})$$

- $\mathcal{V}_a$：与 Tornado Cash 交互的账户地址节点
- $\mathcal{V}_t$：Tornado Cash 智能合约节点
- $\mathcal{E}_{at}$：混币交易边（账户↔合约）——**新增边**
- $\mathcal{E}_{aa}$：地址关联边
- $\mathcal{X} \in \mathbb{R}^{n \times d}$：节点特征矩阵（197维），编码交互频率、时间戳、gas 价格统计

**效果**：平均节点度提升约 45 倍，聚类系数和图密度提升数个数量级。

#### 2. **频率感知图对比学习（解决标签稀缺）**

核心思想：将图信号分解为高频和低频分量，分别用高通和低通 GNN 处理，然后对比学习。

**Step 1: 频率分解**
用 MLP 估计边平滑概率 $s_{ij}$，将图分为低频视图 $\mathcal{G}^{(LF)}$ 和高频视图 $\mathcal{G}^{(HF)}$：

$$A_{ij}^{(LF)} = s_{ij}, \quad A_{ij}^{(HF)} = 1 - s_{ij} \quad \text{where } A_{ij} = 1$$

**Step 2: 双分支 GNN**
- 低通 GNN（捕获平滑全局语义）：$\mathcal{H}_l^{(LF)} = (I + \tilde{A}^{(LF)})\mathcal{H}_{l-1}^{(LF)}$
- 高通 GNN（保留判别局部信号）：$\mathcal{H}_l^{(HF)} = (I - \alpha \cdot \tilde{A}^{(HF)})\mathcal{H}_{l-1}^{(HF)}$

**Step 3: InfoNCE 对比损失**
对比同一节点在两个频率视图中的表示：

$$\mathcal{L}_{con} = \frac{1}{2|\mathcal{E}_{aa}|}\sum_{e \in \mathcal{E}_{aa}}[\mathcal{L}^{(LF)} + \mathcal{L}^{(HF)}]$$

#### 3. **置信度监督（解决标签噪声）**

利用深度网络的记忆效应（先学习干净样本，后拟合噪声），动态将标签分为三类：

**Step 1: 干净标签集提取**
利用 mutual loss（两个网络的联合预测损失）和动态阈值：

$$\mathcal{L}_{thre}^t = \text{Percentile}(\mathcal{L}_{mul}^{ij}, 1 - \frac{t}{2 \cdot T_{\max}})$$

**Step 2: 置信翻转标签集提取**
当两个网络一致地、高置信地预测出与标签不同的结果时：

$$\mu(e_{ij}) = \sqrt{p_{ij,c}^{(LF)} \cdot p_{ij,c}^{(HF)}}$$

**Step 3: 三类标签差异化权重**
- $\mathcal{E}_{cl}$（干净）：权重 $\xi=1$，使用原始标签
- $\mathcal{E}_{cf}$（置信翻转）：权重 $\xi=\mu(i)$，使用翻转后标签
- $\mathcal{E}_{re}$（剩余）：权重 $\xi=0.5$，使用原始标签

**Step 4: 总训练损失**

$$\mathcal{L} = \mathcal{L}_{con} + \lambda \cdot \mathcal{L}_{sup}$$

#### 4. **Stacking 集成预测**

高通和低通 GNN 本质上是同构的，直接组合效果有限。引入三个额外异质模型（随机森林、逻辑回归、MLP）进行 Stacking 集成，丰富决策边界和表示能力。

### 损失函数 / 训练策略

- 联合训练高通和低通 GNN，使用对比损失 + 置信度监督损失
- Adam 优化器，学习率 0.003，batch size 128，训练 50 epoch
- 标签划分阈值动态调整：早期宽松（保留更多样本），逐步严格
- 5-fold 交叉验证用于 Stacking 集成的元学习器

## 实验关键数据

### 主实验

数据集：554,589 条 Tornado Cash 混币交易（2019-12-2025-01），106,982 个账户节点，375,255 条边。

| 方法 | F1 | AUC | MRR | Hits@5 | Hits@10 |
|------|-----|-----|-----|--------|---------|
| DeepWalk | 0.6844 | 0.7078 | 0.0866 | 0.0932 | 0.1940 |
| GCN | 0.7639 | 0.7795 | 0.1011 | 0.1233 | 0.3151 |
| GIN | 0.7869 | 0.8167 | 0.1415 | 0.1900 | 0.3802 |
| Co-teaching+ | 0.7690 | 0.8230 | 0.2712 | 0.4281 | 0.6169 |
| NRGL | 0.7866 | 0.8513 | 0.3989 | 0.5896 | 0.7263 |
| BERT4ETH | 0.6990 | 0.7897 | 0.3366 | 0.4694 | 0.5714 |
| MixBroker | 0.7930 | 0.8182 | 0.1062 | 0.1231 | 0.3323 |
| **HiLoMix** | **0.8382** | **0.9137** | **0.4612** | **0.6556** | **0.8037** |
| **提升%** | **5.69%** | **7.34%** | **15.61%** | **11.18%** | **10.65%** |

HiLoMix 在所有指标上全面领先，MRR 提升最显著（15.61%），表明模型能有效将真实关联排在候选列表前端。

### 消融实验

| 配置 | F1 | AUC | MRR | 说明 |
|------|-----|-----|-----|------|
| w/o 异质模型 | 0.7393 | 0.7948 | 0.2977 | **最大退化**，F1↓11.80%，MRR↓35.45% |
| w/o HiLo GNNs | 0.8075 | 0.8761 | 0.3507 | 移除双分支 GNN |
| w/o Hi GNN | 0.8169 | 0.8977 | 0.4202 | 仅低通 |
| w/o Lo GNN | 0.8286 | 0.9046 | 0.4129 | 仅高通 |
| w/o 异质图 | 0.8158 | 0.8931 | 0.4419 | 仅保留关联边 |
| w/o HiLo 学习 | 0.8083 | 0.8911 | 0.4228 | 用两个 GCN 替代 |
| w/o 标签划分 | 0.8238 | 0.8938 | 0.4260 | 不做置信度分类 |
| w/o stacking | 0.7882 | 0.8547 | 0.4061 | 不用集成 |
| **HiLoMix** | **0.8382** | **0.9137** | **0.4612** | 完整模型 |

### 关键发现

1. **Stacking 中异质模型至关重要**：移除异质基模型导致最大性能退化，说明同构的高低通 GNN 无法单独提供足够互补信息
2. **频率分解优于标准 GCN**：w/o HiLo learning（用 GCN 替代）导致显著退化
3. **HAMIG 有效缓解稀疏性**：平均节点度提升 45 倍，图密度提升数个数量级
4. **置信度标签划分有效处理噪声**：动态阈值策略使模型在训练后期逐步过滤错误标签
5. **BERT4ETH 不如标准 GNN**：序列建模忽视结构信息，在关联任务上表现不佳
6. **训练效率良好**：每 epoch 6.91 秒，远快于 NRGL（15.27 秒），推理 <0.01 秒

## 亮点与洞察

- **系统性解决方案**：针对标签噪声、标签稀缺、图稀疏三大挑战分别提出对应模块，架构设计有针对性
- **频率视角的引入**非常有洞察力——混币场景中，低频捕获全局社区结构，高频捕获异常局部模式
- **置信度三类划分**（干净/翻转/剩余）比简单的噪声过滤更精细，充分利用了"可能错标但可矫正"的样本
- **实用价值高**：随着 Tornado Cash 制裁解除，混币地址关联技术的需求可能增大
- 构建了最新的地面真值数据集（2019-2025），覆盖完整的 Tornado Cash 运营历史

## 局限与展望

- **仅在 Tornado Cash 上验证**，虽然它占以太坊混币 95% 交易量，但方法对其他混币协议的适用性未知
- **启发式标签作为基础**：即使有噪声处理，初始标签质量仍然是性能上限
- **特征工程依赖**：197维特征需要了解 Tornado Cash 合约细节，对新协议需要重新设计
- **图的静态建模**：未考虑交易时间序列和图的动态演化
- **隐私伦理考量**：混币分析可能侵犯合法用户隐私，需要在安全和隐私之间权衡

## 相关工作与启发

- 与 MixBroker 的关系：HiLoMix 在 MIG 基础上构建 HAMIG，并增加了无监督和鲁棒学习机制
- 频率感知 GCL 启发自 S3GCL 等频谱方法，但在分解前增加了信号解耦步骤
- 置信度标签学习参考了 Co-teaching 和 NRGL 的思想，但多了"置信翻转"类别
- 对其他区块链分析任务（交易所监控、DeFi 异常检测）的启发：标签噪声和稀缺是普遍问题

## 评分

- 新颖性: ⭐⭐⭐⭐ — 频率感知+置信度标签的组合在该领域是首创
- 实验充分度: ⭐⭐⭐⭐⭐ — 11 个基线对比，完整消融，超参敏感性，效率分析，图结构分析
- 写作质量: ⭐⭐⭐⭐ — 问题分析透彻，方法模块化清晰
- 价值: ⭐⭐⭐⭐ — 实际安全应用价值高，新数据集构建有社区贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Explanation-Preserving Augmentation for Semi-Supervised Graph Representation Learning](explanation-preserving_augmentation_for_semi-supervised_graph_representation_lea.md)
- [\[ICML 2026\] Learning Graph Foundation Models on Riemannian Graph-of-Graphs](../../ICML2026/self_supervised/learning_graph_foundation_models_on_riemannian_graph-of-graphs.md)
- [\[AAAI 2026\] Robust Tabular Foundation Models](robust_tabular_foundation_models.md)
- [\[CVPR 2025\] Order-Robust Class Incremental Learning: Graph-Driven Dynamic Similarity Grouping](../../CVPR2025/self_supervised/order-robust_class_incremental_learning_graph-driven_dynamic_similarity_grouping.md)
- [\[CVPR 2026\] Learning by Analogy: A Causal Framework for Compositional Generalization](../../CVPR2026/self_supervised/learning_by_analogy_a_causal_framework_for_compositional_generalization.md)

</div>

<!-- RELATED:END -->
