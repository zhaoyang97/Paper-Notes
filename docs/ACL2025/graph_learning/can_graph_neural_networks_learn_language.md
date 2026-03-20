# Morpher: Can Graph Neural Networks Learn Language with Extremely Weak Text Supervision?

**会议**: ACL 2025  
**arXiv**: [2412.08174](https://arxiv.org/abs/2412.08174)  
**代码**: https://github.com/Violet24K/Morpher  
**领域**: 其他  
**关键词**: GNN, LLM alignment, multi-modal prompt learning, graph-text, zero-shot graph classification

## 一句话总结
提出 Morpher，首个图-文多模态 prompt learning 范式——在冻结 GNN 和 LLM 参数的前提下，同时学习图 prompt 和文本 prompt + 跨模态投影器，用极弱文本监督（仅类别名几个词）将 GNN 表征对齐到 LLM 语义空间，首次实现 GNN 的 CLIP 式零样本图分类。

## 研究背景与动机

1. **领域现状**：CLIP 在视觉领域成功证明了与文本对齐可增强编码器的迁移性，已扩展到视频/3D/语音。但图数据的 CLIP 式对齐仅在分子领域和文本属性图上探索过（因为配对数据充足）。
2. **现有痛点**：一般图数据跃三大挑战：(1) 数据极稀缺且文本监督极弱（标签仅几个词）；(2) 任务跨节点/边/图多层级；(3) 同结构在不同领域含义不同。联合预训练不现实。
3. **核心矛盾**：如何在有限下游数据（少样本 + 弱文本）下将预训练 GNN 适配到语义嵌入空间？
4. **切入角度**：(1) LLM 的嵌入空间天然是高质量语义空间（无需联合预训练）；(2) 数据有限时 prompt learning 比 fine-tuning 更优。→ 用 prompt 桥接独立预训练的 GNN 和 LLM。

## 方法详解

### 整体框架
冻结 GNN $f^g_\phi$ + 冻结 LLM $f^t_\phi$ → 同时学习图 prompt $P^g_\theta \in \mathbb{R}^{n_g \times d}$（作为新节点插入原图）+ 文本 prompt $P^t_\theta \in \mathbb{R}^{n_t \times d_t}$（拼接到类别名 embedding 前）+ 跨模态投影器 → 少样本对比学习对齐。

### 关键设计

1. **改进的图 prompt**:
   - 现有图 prompt (Sun et al. 2023) 将 prompt token 作为新节点插入图中，但跨连接（prompt-输入）远多于内连接（prompt-prompt），导致训练不稳定。
   - Morpher 平衡跨连接和内连接数量（类似 NLP 中注意力的平衡），稳定优化。

2. **双模态 prompt learning**:
   - **图侧**：图 prompt 插入原图 → GNN 编码 → mean pooling 得图表征。
   - **文本侧**：文本 prompt 拼接类别名 embedding → LLM 编码 → 类别表征。
   - 跨模态投影器将图表征投影到 LLM 空间 → cosine 相似度匹配类别。
   - 设计动机：叕模态 prompt 只能调整一侧，双模态同时调整两侧更有效。

3. **零样本图分类原型**:
   - 训练后，对于未见类别，只需提供类别名文本 → LLM 编码得类别表征 → 与图表征 cosine 匹配。
   - 这是 GNN 领域首个 CLIP 式零样本分类原型。

## 实验关键数据

### 主实验

| 设置 | Morpher vs 基线 | 说明 |
|------|:---:|------|
| Few-shot | 显著优于 | 双模态 prompt 比单模态更有效 |
| 多任务层级 | 显著优于 | 节点/边/图级别任务都有效 |
| 跨域 | 显著优于 | 从一个学坟数据集迁移到另一个 |
| **零样本未见类** | **首次实现** | GNN 真正"学会了语言" |

### 关键发现
- **双模态 > 单模态 prompt**：在所有设置下一致优于仅图 prompt 或仅文本 prompt。
- **图 prompt 平衡设计关键**：机械地将所有 prompt 节点连到所有输入节点会导致不稳定优化，平衡是待解决的关键问题。
- **零样本可行**：证明了即使文本监督极弱（仅类别名），GNN 也能通过对齐学到语义关联。

## 亮点与洞察
- **"极弱文本监督下的跨模态对齐"是重要贡献**：证明了不需要大规模配对数据，只需几个词的类别名就能桥接两个独立预训练的编码器。
- **CLIP 范式向图领域的推广**：开辟了图数据的泛化能力提升的新方向。
- **LLM 嵌入空间作为通用语义锶点**：无需联合预训练，直接用 LLM 的嵌入空间作为对齐目标。

## 局限性 / 可改进方向
- 仅在小规模图数据集上验证，大规模图上的可扩展性未测试。
- 零样本能力受限于 LLM 语义空间的质量和类别名的语义丰富度。
- 图 prompt 的节点数 $n_g$ 和连接策略需手动调参。

## 相关工作与启发
- **vs MoleculeSTM / MoMu**: 分子领域图-文对齐有充足配对数据，可联合预训练；Morpher 在无配对数据时通过 prompt learning 解决。
- **vs CoOp/MaPLe (VLM prompt learning)**: 视觉领域的多模态 prompt 有成功先例；Morpher 是图领域的对应探索，面临更大挑战（非欧压数据 + 极弱监督）。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个图-文多模态 prompt learning + 首个 GNN 零样本分类
- 实验充分度: ⭐⭐⭐⭐ few-shot + 多任务 + 跨域 + 零样本
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，图 prompt 分析深入
- 价值: ⭐⭐⭐⭐⭐ 为图数据引入语言理解开辟新方向
