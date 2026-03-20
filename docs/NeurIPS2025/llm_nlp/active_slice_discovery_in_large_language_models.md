# Active Slice Discovery in Large Language Models

**会议**: NeurIPS 2025 (Workshop: Reliable ML from Unreliable Data)  
**arXiv**: [2511.20713](https://arxiv.org/abs/2511.20713)  
**代码**: 待发布（论文承诺公开）  
**领域**: human_understanding  
**关键词**: slice discovery, active learning, LLM interpretability, toxicity classification, sparse autoencoder  

## 一句话总结

提出 **Active Slice Discovery** 问题框架，将主动学习引入 LLM 错误切片发现，利用不确定性采样 + LLM 内部表征（原始 embedding 或 SAE 特征）在仅使用 2-10% 标注的情况下达到接近全标注的切片检测精度。

## 研究背景与动机

1. **LLM 存在系统性错误模式**：LLM 在特定数据子集（error slices）上会表现出一致性的失败，例如在毒性分类中对某些人口统计群体的评论识别不佳。发现这些 error slices 对模型审计和改进至关重要。
2. **传统 slice discovery 完全无监督**：现有方法（Domino、Spotlight 等）在无任何切片标注的情况下对错误样本进行聚类，但无监督设定本身难度很大，效果有限。
3. **人工逐个标注成本极高**：完整标注数据集中每个样本属于哪个 slice 需要大量人力，实际场景中不可行。
4. **主动学习可降低标注需求**：通过有策略地选择最有价值的样本请求人工判断其 slice 归属，可以用极少标注达到较好效果。
5. **LLM 内部表征可能蕴含 slice 信息**：模型隐层 embedding 以及 Sparse Autoencoder（SAE）激活值可能编码了足够的语义信息，帮助区分不同 error slices。
6. **缺乏系统性研究**：尽管 active learning 和 slice discovery 分别有大量工作，但将二者结合的 active slice discovery 尚未被正式研究过。

## 方法详解

### 整体框架

给定一个已训练的分类器 $f_\theta$、少量带 slice 标注的种子集 $\mathcal{D}_s$、以及大量仅有分类标签的数据集 $\mathcal{D}$，目标是学习一个 slice 成员函数 $\phi: \mathcal{X} \times \mathcal{Y} \to \{0,1\}^k$。迭代过程：

1. 用当前标注集训练切片分类器
2. 通过查询策略 $A$ 选择最有价值的未标注样本
3. 向 oracle（人工标注者）查询该样本的 slice 归属
4. 将新标注加入 $\mathcal{D}_s$，重复直到预算 $K$ 耗尽

### 关键设计 1：特征表征选择

- **原始层嵌入（Raw Layer Embeddings）**：取 Llama-3.1-8B 倒数第二层的隐层输出作为每个样本的表征向量
- **SAE 稀疏激活（Sparse Autoencoder Features）**：使用 Llama Scope SAE 在 Llama-3.1-8B 最后一层上训练的稀疏自编码器的激活值作为特征。SAE 特征更具可解释性且产生更稳定的训练曲线

### 关键设计 2：主动学习查询策略

论文对比了三类查询策略：
- **不确定性策略**（效果最佳）：Least Confidence、Prediction Entropy、Breaking Ties — 选择模型最不确定的样本进行标注
- **多样性策略**：Embedding K-Means、Discriminative Active Learning、Lightweight Coreset — 选择表征空间中最多样的样本
- **基线**：Random Sampling

### 关键设计 3：切片分类器

- **MLP（多层感知机）**：精度上限更高（85.8%），但需要仔细调参
- **线性 SVM**：无需复杂调参，配合 SAE 特征可达 83.0%，更易于部署

### 损失函数

切片分类本质上是二分类问题，用标准交叉熵损失训练 MLP 或 hinge loss 训练 SVM；评估指标为切片成员分类准确率：

$$\text{Acc}_j = \mathbb{E}_{x,y,\mathbf{s}}\left[\mathbf{1}\left[\phi_j(x,y) = s_j\right]\right]$$

## 实验关键数据

**数据集**：Jigsaw Toxicity Dataset  
**基座模型**：Llama-3.1-8B  
**主动学习库**：Small-Text

### 主要结果（Table 1："disagree" slice）

| 配置 | 分类器 | 最佳准确率 | 所需标注量 |
|------|--------|-----------|-----------|
| Raw Embedding | MLP + AL | **85.8%** | 250 / 12,504 (2%) |
| Raw Embedding | SVM + LC | 81.0% | 3,500 |
| SAE Features | MLP + AL | 82.2% | 1,460 / 12,416 |
| SAE Features | SVM + LC | 83.0% | 1,000 |

### 关键发现

1. **标注效率惊人**：MLP + Embedding + AL 仅用 2% 标注（250 个样本）即达到 85.8% 准确率，相比全监督节省 98% 标注成本
2. **Slice 类型影响难度**：身份类 slices（female, christian）仅需几百个标注即可高精度检测；反应类 slices（disagree, sad）需要更多标注（>1000），因为词汇线索更弱
3. **SAE 特征 vs 原始 embedding**：SAE 让训练曲线更平滑稳定，且对查询策略选择不太敏感。disagree slice 的检测率从 embedding 的 0.80 提升到 SAE 的 0.83
4. **不确定性策略一致胜出**：Least Confidence、Prediction Entropy、Breaking Ties 在两种表征下均显著优于多样性策略和随机采样

## 亮点与洞察

- **问题定义新颖**：首次正式化 active slice discovery 问题，将"发现 LLM 何处出错"从被动观察转为主动探索，具有很强的实际意义
- **98% 标注节省**：最佳配置下仅需 2% 标注就能接近全监督效果，说明 error slices 在表征空间中确实存在可识别的结构
- **SAE 的可解释性优势**：引入稀疏自编码器作为中间表征，兼顾性能与可解释性，是机械可解释性工具在下游任务中的有效应用
- **灵活可组合的 pipeline**：框架支持不同 LLM、表征、分类器和查询策略的自由组合，方便实践者选择最适合自己场景的配置

## 局限性

1. **仅在单一数据集上验证**：所有实验局限于 Jigsaw 毒性分类，未在其他 NLP 任务（QA、摘要、代码生成等）上测试，泛化性存疑
2. **Slice 需预定义**：当前框架假设已知 slice 的数量和种子样本，未解决如何从零开始发现未知类型的 slice
3. **未与先进的无监督 slice discovery 方法直接对比**：如 Domino、DISCERN 等方法在相同数据上的表现如何？
4. **仅限于分类任务**：生成式任务中的 error slice 定义和检测更为复杂，论文未涉及
5. **Workshop paper 篇幅限制**：实验规模和分析深度受限，缺乏消融实验和更细致的分析

## 相关工作

- **Slice Discovery（无监督）**：Domino（Eyuboglu et al.）、Spotlight（D'Eon et al.）、SliceLine（Sagadeeva & Burtsev）— 本文在此基础上引入主动标注
- **LLM 可解释性**：Sparse Autoencoder（Cunningham et al.; Templeton et al.）— SAE 在本文中被用作特征提取器
- **主动学习**：Settles (2009) 综述；Desai et al. (2025) 文本分类中的不确定性策略 — 本文直接采用这些策略
- **最相关工作**：Hua et al. (2023) 用 slice discovery 指导 active learning 训练更好的分类器。与本文目标相反：他们 elicit 任务标签 $y$，本文 elicit slice 标签 $\mathbf{s}$

## 评分

- 新颖性: ⭐⭐⭐⭐ — 问题定义新颖且有实际意义，首次将 active learning 与 slice discovery 结合
- 实验充分度: ⭐⭐⭐ — 单数据集、单模型、workshop 篇幅，实验规模有限但覆盖了关键维度
- 写作质量: ⭐⭐⭐⭐ — 问题形式化清晰，实验结果呈现直观
- 价值: ⭐⭐⭐⭐ — 为 LLM 审计提供了高效实用的工具框架，98% 标注节省具有显著实践价值
