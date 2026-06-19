---
title: >-
  [论文解读] Weisfeiler and Leman Go Gambling: Why Expressive Lottery Tickets Win
description: >-
  [ICML2025][计算生物][图彩票假说] 首次从理论上将 GNN 的表达力（Weisfeiler-Leman 测试）与彩票假说（LTH）联系起来，提出并证明了强表达力彩票假说（SELTH），证明稀疏初始化的 GNN 中存在保持 1-WL 表达力的可训练子网络，且表达力更强的稀疏初始化更可能成为"中奖彩票"，同时展示了不当剪枝导致的不可恢复表达力损失在药物发现等场景中的严重后果。
tags:
  - "ICML2025"
  - "计算生物"
  - "图彩票假说"
  - "图神经网络"
  - "Weisfeiler-Leman测试"
  - "稀疏初始化"
  - "关键路径"
  - "图剪枝"
---

# Weisfeiler and Leman Go Gambling: Why Expressive Lottery Tickets Win

**会议**: ICML2025  
**arXiv**: [2506.03919](https://arxiv.org/abs/2506.03919)  
**作者**: Lorenz Kummer, Samir Moustafa, Anatol Ehrlich, Franka Bause, Nikolaus Suess, Wilfried N. Gansterer, Nils M. Kriege (University of Vienna)
**代码**: [lorenz0890/wl2025lottery](https://github.com/lorenz0890/wl2025lottery)  
**领域**: 计算生物  
**关键词**: 图彩票假说, GNN表达力, Weisfeiler-Leman测试, 稀疏初始化, 关键路径, 图剪枝

## 一句话总结

首次从理论上将 GNN 的表达力（Weisfeiler-Leman 测试）与彩票假说（LTH）联系起来，提出并证明了强表达力彩票假说（SELTH），证明稀疏初始化的 GNN 中存在保持 1-WL 表达力的可训练子网络，且表达力更强的稀疏初始化更可能成为"中奖彩票"，同时展示了不当剪枝导致的不可恢复表达力损失在药物发现等场景中的严重后果。

## 研究背景与动机

彩票假说（LTH）认为大型随机初始化的神经网络中包含更小的可训练子网络（"中奖彩票"），这些子网络能匹配完整网络的性能。LTH 在 CNN 领域已有深入的理论研究（如 SLTH 证明了无需训练的子网络存在性），但在 GNN 领域主要停留在经验验证层面，理论基础严重缺乏。

现有 GNN 彩票工作的核心局限：
- **UGS**：统一剪枝邻接矩阵和模型权重以发现图彩票（GLT），但完全忽略表达力保持
- **wang2022searching**：通过分层稀疏化和正则化剪枝搜索 GLT，同样不考虑表达力
- **yan2024multicoated**：基于 SLTH 的 Multicoated Supermasks，关注内存效率而非表达力

**关键空白**：LTH 与 GNN 表达力之间的联系完全未被探索。已有工作关注减少参数和计算，但训练前剪枝对 GNN 区分非同构图能力的影响被忽视。

**药物发现的实际动机**：分子性质预测中，结构相似的分子可能对同一蛋白靶点表现出截然不同的活性（活性悬崖 activity cliffs）。立体异构体仅在三维空间排列上不同但生物活性差异巨大。如果剪枝导致 GNN 无法区分这些关键分子结构，可能导致有毒药物被误判为安全。

## 方法详解

### 整体框架

论文建立了一套从表达力角度理解 GNN 剪枝的理论框架。分析对象是基于矩（moment-based）的 GNN 架构（如 GIN、GCN），其更新规则为：

$$\mathbf{h}_v^{(k)} = \text{MLP}^{(k)}\left(\mathbf{h}_v^{(k-1)} + \sum_{u \in N(v)} \mathbf{h}_u^{(k-1)}\right)$$

图级嵌入通过跨层拼接 readout 得到：$\mathbf{h}_G = \|_{k=0}^{n} \sum_{v \in V(G)} \mathbf{h}_v^{(k)}$

### 设计1：表达力保持准则与关键路径

**准则 1**：对于 $\Phi^{(k+1)}$（含分类头的 GNN）要正确分类数据集 $D$ 中的所有图，$\Phi^{(k)}$（消息传递层）必须区分所有属于不同类别的非同构图对。任何丧失该区分能力的剪枝掩码都不可能是中奖彩票。

**关键路径（Definition 3.1）**：设 $\mathcal{W}_{\Phi^{(k)}} = \bigcup_i^k \mathcal{W}_i$ 为 GNN 所有 MP 层 MLP 权重的并集。如果移除某路径集合后，不存在任何权重配置能使剪枝后网络达到原始性能，则该路径集合是关键的。关键路径是最大表达力子网络的子集——保留所有关键路径的剪枝理论上可构成中奖彩票。

### 设计2：强表达力彩票假说（SELTH）

**定理 3.2（核心定理）**：在基于矩的 GNN 中，存在稀疏初始化的可训练子网络，其表达力与 1-WL 测试匹配。对于使用单射激活函数的 MLP（每层输出维度不小于输入维度），存在稀疏权重配置使每层 MLP 在输入域上保持单射性，从而保证 1-WL 等价表达力。

这是经典 SLTH 向 GNN 的首个形式化推广，增加了"表达力"约束——不仅子网络可训练，还须保持最大图区分能力。

### 设计3：表达力与收敛/泛化

**定理 3.3**：若两个图在初始化时获得部分相同的嵌入，这些共同嵌入的梯度贡献是共方向的（codirectional），导致训练中两图嵌入持续纠缠。表达力更强的初始化（嵌入更独特）提供更多样的梯度信号，潜在地加速收敛和改善泛化。

### 设计4：表达力损失的不可恢复性

**引理 3.6**：给定数据集含 $I$ 种同构类型，若剪枝使 $U$ 种类型不可区分，则分类准确率存在严格上界。训练过程无法恢复这种损失——这在分子性质预测等安全关键场景中意味着不可接受的风险。

## 实验关键数据

### 实验设置
- **架构**：GIN 和 GCN
- **数据集**：10 个标准图分类基准（含 MoleculeNet 等分子数据集）
- **总运行次数**：超过 **13,500 次**
- **表达力度量 $\tau$**：非同构图对中仍可区分的比例（$\tau_{pre}$ 训练前，$\tau_{post}$ 训练后）
- **中奖判定**：稀疏模型准确率下降不超过 5% 即为中奖彩票
- **总计算时间**：约 8 周

### Table 1: 表达力转变概率

| 剪枝比例 | $P(\tau_{post} > \tau_{pre})$ | 含义 |
|---|---|---|
| 低（<30%） | 极低（个位数%） | 训练几乎不提升表达力 |
| 中（30-60%） | 更低 | 表达力下降趋势更明显 |
| 高（>60%） | 接近 0 | 表达力大幅不可逆下降 |

**核心结论**：GNN 从低 $\tau_{pre}$ 恢复到高 $\tau_{post}$ 的概率极低。训练无法弥补剪枝造成的表达力损失。

### Figure 2+3: 中奖概率与表达力的关系

| 分析维度 | 结果 |
|---|---|
| 中奖概率 vs. 表达力阈值 $\vartheta$ | 单调递增：表达力越高，中奖概率越大 |
| Pearson 相关 $\tau_{pre}$ vs. 训练后准确率 (GIN) | 正相关，统计显著 |
| Pearson 相关 $\tau_{pre}$ vs. 训练后准确率 (GCN) | 正相关，统计显著 |
| $\tau_{pre}$ vs. $\tau_{post}$ 散点 (Figure 4) | 所有点在对角线下方：$\tau_{post} \leq \tau_{pre}$ |

在所有 10 个数据集和不同剪枝比例上，初始化表达力越高，成为中奖彩票的概率越大。

## 关键发现

1. **表达力是中奖关键**：13,500+ 次实验一致表明，初始化时保持 1-WL 表达力的稀疏子网络显著更可能成为中奖彩票
2. **训练不弥补剪枝损失**：$\tau_{post} \leq \tau_{pre}$ 在所有实验中成立，不当剪枝造成的表达力损失不可逆
3. **表达力-性能正相关**：Pearson 相关分析确认训练前表达力与训练后准确率正相关
4. **药物发现警示**：分子性质预测中不当剪枝导致活性悬崖和立体异构体不可区分，可能引发安全隐患

## 亮点与洞察

- **理论开创性**：首次将 GNN 表达力（WL 测试层面）与 LTH 形式化联系，提出 SELTH 作为 GNN 特有的理论扩展，填补了 LTH 在 GNN 领域的理论空白
- **关键路径概念**：将剪枝讨论从"哪些权重可去掉"转变为"哪些计算路径必须保留以维持图区分能力"
- **实践指导**：为 GNN 剪枝提供理论指导——剪枝策略应检查是否破坏了 MLP 的单射性，比随机或幅度剪枝更有原则性
- **实验规模**：超过 13,500 次实验运行，横跨 10 个数据集、多种稀疏度和两种架构，验证充分

## 局限性

- **架构覆盖有限**：仅验证 GIN 和 GCN，未涉及 GAT 等注意力 GNN 或 max/min 聚合架构（审稿人一致指出）
- **理论假设较强**：假设单射连续可微零固定激活函数，虽实验表明对 ReLU 等也成立但形式化保证有差距
- **引理 3.6 实用性不足**：需预知数据集所有同构类型且假设均匀类分布，实际中难以满足
- **缺少实用剪枝算法**：论文为理论分析，未提出完整的表达力感知剪枝方法（仅在 rebuttal 中描述了简单思路）
- **仅图分类**：节点分类等场景需额外调整
- **实验描述仓促**：审稿人一致指出实验部分可读性和清晰度有待提升

## 相关工作与启发

| 维度 | 本文 SELTH | 经典 LTH/SLTH | 现有 GNN LTH |
|---|---|---|---|
| 理论层面 | 表达力保持条件 + WL 等价 | 子网络存在性证明 | 仅经验验证 |
| 关注焦点 | 图区分能力（表达力） | 函数逼近能力 | 参数/计算效率 |
| 剪枝对象 | MLP 参数权重 | 模型参数 | 图结构 + 模型参数 |
| 应用示例 | 药物发现（活性悬崖） | CV/NLP | 分子图/社交网络 |

**启发**：表达力保持思想可推广到设计更好的 GNN 剪枝算法——每层剪枝后检查聚合函数单射性，选择保持最多图区分对的配置。同一作者组 KDD 2024 的"WL Go Indifferent"分析了比特翻转攻击对 GNN 表达力的影响，与本文形成攻击/防御的互补视角。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将 WL 表达力与 LTH 形式化联系，SELTH 假说是全新理论贡献
- 实验充分度: ⭐⭐⭐ — 13,500+ 次实验规模大，但仅两种架构，缺少实用算法对比
- 写作质量: ⭐⭐⭐ — 理论严谨但符号密集，实验部分描述仓促（审稿人一致反馈）
- 价值: ⭐⭐⭐⭐ — 为 GNN 剪枝提供重要理论基础，药物安全领域有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Random Search Neural Networks for Efficient and Expressive Graph Learning](../../NeurIPS2025/computational_biology/random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)
- [\[CVPR 2025\] Semantic and Expressive Variation in Image Captions Across Languages](../../CVPR2025/computational_biology/semantic_and_expressive_variations_in_image_captions_across_languages.md)
- [\[NeurIPS 2025\] Why Masking Diffusion Works: Condition on the Jump Schedule for Improved Discrete Diffusion](../../NeurIPS2025/computational_biology/why_masking_diffusion_works_condition_on_the_jump_schedule_for_improved_discrete.md)
- [\[CVPR 2026\] FEAST: Fully Connected Expressive Attention for Spatial Transcriptomics](../../CVPR2026/computational_biology/feast_fully_connected_expressive_attention_for_spatial_transcriptomics.md)
- [\[ICML 2025\] Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning](neural_graph_matching_improves_retrieval_augmented_generation_in_molecular_machi.md)

</div>

<!-- RELATED:END -->
